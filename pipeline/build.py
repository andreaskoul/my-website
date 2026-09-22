"""Turn data/raw/*.jsonl into site/data/<TICKER>.json.

  python pipeline/build.py            # daily: reuse each firm's fitted stories
  python pipeline/build.py --refit    # weekly: refit stories (also runs if no state yet)

Per firm, in order:
  1. clean + de-duplicate (exact headline across sources, then near-duplicate rewrites, cos > .95)
  2. embed headline + summary with gte-small (cached by text hash)
  3. RELEVANCE, judged per group, never per article (one passing mention is noise, a group is not).
     Each firm gets K1 coarse anchors; an anchor is kept only if it passes BOTH tests
     (Beta posteriors, 95% lower bound of the ratio > 1):
       name lift  -- it names the firm more often than the other firms' articles do
                     (rules out groups where the firm only appears in passing)
       feed lift  -- its kind of content is over-represented in this firm's feed vs the other 19
                     (rules out genre: ETF lists, market wraps, earnings-call templates are
                      equally common in every feed)
     Either test alone fails: name lift alone keeps genre for rarely-mentioned firms (Boeing's base
     rate is 0.1%); feed lift alone drops heavily-covered firms' own core (Nvidia is tagged into
     many feeds). Tried and rejected: a pooled Gaussian mixture on both axes (set by niche firms,
     too strict for megacaps), content breadth, neighbourhood correlation.
  4. STORIES: refit on the relevant articles; K = the finest granularity that stays stable
     under resampling (ARI >= STABLE). No K is "true" -- this is the stable level of detail.
  5. IDENTITY: new stories are matched one-to-one to last week's by meaning; matches keep their
     id, name and history, so a story does not get renamed or recoloured by a refit.
  6. NAMES: only unmatched (new) stories are named -- by a free OpenRouter model if
     OPENROUTER_API_KEY is set, otherwise by their most distinctive keywords.
  7. EXPORT: this week's articles per story, weekly share series, one representative
     headline per story per week.
"""
import argparse, collections, hashlib, json, math, os, re, sys, urllib.request
from datetime import datetime, timedelta, timezone
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

ap = argparse.ArgumentParser(); ap.add_argument("--refit", action="store_true"); ap.add_argument("--only", default="")
args = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(f"{ROOT}/config/firms.json"))
MODEL, K1, BETA, STABLE, NEARDUP, MATCH = "thenlper/gte-small", 8, 20.0, 0.85, 0.95, 0.5
OR_KEY = os.environ.get("OPENROUTER_API_KEY")
OR_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
now = datetime.now(timezone.utc)
start = now - timedelta(days=CFG["window_days"])
week_cut = now - timedelta(days=7)
norm = lambda t: re.sub(r"[^a-z0-9]+", "", t.lower())[:120]
unit = lambda A: A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)
ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
monday = lambda d: (d - timedelta(days=d.weekday())).date()

# ---- load every firm's feed (the other firms' feeds are the relevance base rate)
feeds = {}
for f in CFG["firms"]:
    p = f"{ROOT}/data/raw/{f['ticker']}.jsonl"
    rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    rows = [r for r in rows if r["t"] and ts(r["published"]) >= start]
    for r in rows:                                       # some providers put a bare URL in the summary field
        if re.match(r"\s*https?://\S+\s*$", r["desc"] or ""): r["desc"] = ""
    by_head = {}
    for r in sorted(rows, key=lambda r: (r["src"] != "polygon", r["published"])):   # prefer polygon's longer summary on ties
        k = norm(r["t"])
        if len(k) < 15: continue
        if k in by_head:
            if len(r["desc"]) > len(by_head[k]["desc"]): by_head[k]["desc"] = r["desc"]
            continue
        by_head[k] = r
    feeds[f["ticker"]] = sorted(by_head.values(), key=lambda r: r["published"])
    for r in feeds[f["ticker"]]: r["text"] = r["t"] + (". " + r["desc"] if r["desc"] else "")

# ---- embeddings, cached by text hash (the cache is not committed; CI keeps it in actions/cache)
os.makedirs(f"{ROOT}/data/cache", exist_ok=True)
cp = f"{ROOT}/data/cache/emb.npz"
cache = dict(np.load(cp)) if os.path.exists(cp) else {}
todo = {}
for rows in feeds.values():
    for r in rows:
        r["h"] = hashlib.sha1(r["text"].encode()).hexdigest()[:16]
        if r["h"] not in cache: todo[r["h"]] = r["text"]
if todo:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL)
    keys = list(todo)
    V = m.encode([todo[k] for k in keys], normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    for k, v in zip(keys, V): cache[k] = v.astype(np.float16)
    np.savez(cp, **cache)
print(f"embedded {len(todo)} new texts, cache {len(cache)}", flush=True)

# ---- names for new stories
STOP = set("""the a an and or of to in on for with at by from as is are was were be been it its this that these those has
have had will would can could not but than into over after about more most new says said stock stocks shares inc corp company
companies year years today week why what how here heres your you buy should just now up down billion million report reports""".split())
def name_story(texts, firm):
    if OR_KEY:
        prompt = (f"These news headlines about {firm} belong to one ongoing story. Give it a short, plain name (2-5 words, "
                  f"no quotes) and a one-sentence description of what the story is about. Reply as JSON "
                  f'{{"name":"...","blurb":"..."}}.\n\n' + "\n".join("- " + t for t in texts[:10]))
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps({"model": OR_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}).encode(),
                headers={"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"})
            out = json.loads(urllib.request.urlopen(req, timeout=60).read())["choices"][0]["message"]["content"]
            j = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
            return j["name"].strip()[:60], j["blurb"].strip()[:200], "llm"
        except Exception as e:
            print(f"   naming via OpenRouter failed ({e}); using keywords", flush=True)
    c = collections.Counter(w for t in texts for w in re.findall(r"[a-z][a-z\-]{2,}", t.lower())
                            if w not in STOP and w not in firm.lower())
    top = [w for w, _ in c.most_common(4)]
    return " · ".join(top[:3]).title() if top else "Untitled", texts[0][:160], "keywords"

# ---- per firm: near-duplicate rewrites (syndication that survives exact-headline de-dup)
XS, NAMED, PATS = {}, {}, {f["ticker"]: re.compile(f["match"], re.I) for f in CFG["firms"]}
for f in CFG["firms"]:
    tk = f["ticker"]; rows = feeds[tk]
    X = np.array([cache[r["h"]] for r in rows], dtype=np.float64).reshape(-1, 384)
    keep = []
    for i in range(len(X)):
        if keep and float((X[keep[-400:]] @ X[i]).max()) > NEARDUP: continue
        keep.append(i)
    feeds[tk] = [rows[i] for i in keep]; XS[tk] = X[keep]
    NAMED[tk] = np.array([bool(PATS[tk].search(r["text"])) for r in feeds[tk]])

# ---- relevance model (refit, or any firm without a fitted state)
need = args.refit or any(not os.path.exists(f"{ROOT}/data/state/{f['ticker']}.json") for f in CFG["firms"])
P1 = {}
if need:
    g = np.random.default_rng(0)
    lo95 = lambda a, n, b, m: float(np.percentile(g.beta(1 + a, 1 + n - a, 20000) / g.beta(1 + b, 1 + m - b, 20000), 2.5))
    for f in CFG["firms"]:
        tk = f["ticker"]; X = XS[tk]
        if len(X) < 60: continue
        mine = {r["h"] for r in feeds[tk]}
        oth = [(t2, i) for t2 in XS if t2 != tk for i, r in enumerate(feeds[t2]) if r["h"] not in mine]
        Xo = np.array([XS[t2][i] for t2, i in oth])
        no = int(sum(PATS[tk].search(feeds[t2][i]["text"]) is not None and tk not in (feeds[t2][i].get("tickers") or [])
                     for t2, i in oth))
        mu1 = X.mean(0); C1 = unit(KMeans(K1, n_init=4, random_state=0).fit(unit(X - mu1)).cluster_centers_)
        h1, ho = (unit(X - mu1) @ C1.T).argmax(1), (unit(Xo - mu1) @ C1.T).argmax(1)
        keep1 = []
        for k in range(K1):
            n_k, o_k = int((h1 == k).sum()), int((ho == k).sum())
            name_ok = lo95(int(NAMED[tk][h1 == k].sum()), n_k, no, len(oth)) > 1     # names the firm beyond chance
            feed_ok = lo95(n_k, len(h1), o_k, len(ho)) > 1                            # not genre shared by every feed
            keep1.append(bool(n_k >= 15 and name_ok and feed_ok))
        P1[tk] = dict(mu1=mu1, C1=C1, h1=h1, keep1=keep1)

index = []
for f in CFG["firms"]:
    tk, firm = f["ticker"], f["name"]
    if args.only and tk not in args.only.split(","): continue
    rows, X = feeds[tk], XS[tk]
    if len(rows) < 60: print(f"{tk}: only {len(rows)} articles, skipped"); continue

    sp = f"{ROOT}/data/state/{tk}.json"
    state = json.load(open(sp)) if os.path.exists(sp) else None
    if args.refit or state is None:
        mu1, C1, h1, keep1 = P1[tk]["mu1"], P1[tk]["C1"], P1[tk]["h1"], P1[tk]["keep1"]
        rel = np.array(keep1)[h1]
        if rel.sum() < 40: print(f"{tk}: too few relevant articles ({rel.sum()}), skipped"); continue
        mu2 = X[rel].mean(0); X2 = unit(X[rel] - mu2)
        K, rng = 4, np.random.default_rng(1)
        for k in range(4, 10):
            base = KMeans(k, n_init=4, random_state=0).fit(X2).labels_
            ari = np.mean([adjusted_rand_score(base, KMeans(k, n_init=2, random_state=s + 1)
                           .fit(X2[rng.choice(len(X2), int(.8 * len(X2)), replace=False)]).predict(X2)) for s in range(3)])
            if ari >= STABLE: K = k
        C2 = unit(KMeans(K, n_init=8, random_state=0).fit(X2).cluster_centers_)
        h2 = (X2 @ C2.T).argmax(1)
        raw_c = unit(np.array([X[rel][h2 == k].mean(0) for k in range(K)]))
        # identity: match to last fit's stories by meaning (in the new centred space)
        old = state["stories"] if state else []
        ids, nxt = [None] * K, (state["next_id"] if state else 0)
        if old:
            Oc = unit(np.array([s["centroid"] for s in old]) - mu2)
            Sim = C2 @ Oc.T; r_i, c_i = linear_sum_assignment(-Sim)
            for a, b in zip(r_i, c_i):
                if Sim[a, b] >= MATCH: ids[a] = old[b]["id"]
        stories = []
        for k in range(K):
            prev = next((s for s in old if s["id"] == ids[k]), None)
            if prev: nm, bl, src = prev["name"], prev["blurb"], prev["name_src"]
            else:
                idx = np.where(h2 == k)[0]; top = idx[np.argsort(-(X2[idx] @ C2[k]))[:10]]
                nm, bl, src = name_story([rows[np.where(rel)[0][i]]["t"] for i in top], firm)
                ids[k] = nxt; nxt += 1
            stories.append(dict(id=ids[k], name=nm, blurb=bl, name_src=src, centroid=np.round(raw_c[k], 5).tolist(),
                                born=prev["born"] if prev else now.date().isoformat()))
        state = dict(fitted=now.isoformat(), K=K, next_id=nxt, mu1=np.round(mu1, 5).tolist(),
                     C1=np.round(C1, 5).tolist(), keep1=keep1, mu2=np.round(mu2, 5).tolist(),
                     C2=np.round(C2, 5).tolist(), stories=stories)
        json.dump(state, open(sp, "w"))
        print(f"{tk}: refit  K={K}  kept anchors {sum(keep1)}/{K1}", flush=True)

    # ---- assign every article in the window with the stored model
    mu1, C1, keep1 = np.array(state["mu1"]), np.array(state["C1"]), np.array(state["keep1"])
    mu2, C2, stories = np.array(state["mu2"]), np.array(state["C2"]), state["stories"]
    rel = keep1[(unit(X - mu1) @ C1.T).argmax(1)]
    R = [rows[i] for i in np.where(rel)[0]]; X2 = unit(X[rel] - mu2)
    S = X2 @ C2.T; hard = S.argmax(1)
    P = np.exp(BETA * (S - S.max(1, keepdims=True))); P /= P.sum(1, keepdims=True)
    wk = np.array([monday(ts(r["published"])) for r in R]); weeks = sorted(set(wk))
    recent = np.array([ts(r["published"]) >= week_cut for r in R])
    out = []
    for k, s in enumerate(stories):
        series = [dict(week=w.isoformat(), n=int(((hard == k) & (wk == w)).sum()),
                       share=round(float(P[wk == w, k].sum() / max((wk == w).sum(), 1)), 4)) for w in weeks]
        evo = []
        for w in weeks:
            m = np.where((hard == k) & (wk == w))[0]
            if len(m) < 3: continue
            c = unit(X2[m].mean(0)); j = m[np.argmax(X2[m] @ c)]
            evo.append(dict(week=w.isoformat(), n=len(m), h=R[j]["t"], p=R[j]["publisher"], u=R[j]["url"]))
        lat = sorted(np.where((hard == k) & recent)[0], key=lambda i: -S[i, k])[:10]
        out.append(dict(id=s["id"], name=s["name"], blurb=s["blurb"], n=int((hard == k).sum()), latest_n=int(((hard == k) & recent).sum()),
                        series=series, evolution=evo,
                        latest=[dict(t=R[i]["t"], d=R[i]["desc"][:260], p=R[i]["publisher"],
                                     date=ts(R[i]["published"]).strftime("%b %d"), u=R[i]["url"]) for i in lat]))
    doc = dict(ticker=tk, name=firm, updated=now.strftime("%Y-%m-%d %H:%M UTC"), weeks=[w.isoformat() for w in weeks],
               latest_from=week_cut.strftime("%b %d"), latest_to=now.strftime("%b %d"),
               n_articles=len(rows), n_relevant=len(R), stories=out)
    json.dump(doc, open(f"{ROOT}/site/data/{tk}.json", "w"), ensure_ascii=False)
    top = max(out, key=lambda s: s["latest_n"]) if out else None
    index.append(dict(ticker=tk, name=firm, top=top["name"] if top else ""))
    print(f"{tk}: {len(rows)} articles, {len(R)} relevant, {len(out)} stories", flush=True)

if not args.only:
    json.dump(dict(updated=now.strftime("%Y-%m-%d %H:%M UTC"), firms=index), open(f"{ROOT}/site/data/index.json", "w"))
