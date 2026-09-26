"""Turn data/raw/*.jsonl into site/data/<TICKER>.json.

  python pipeline/build.py            # daily: reuse each firm's fitted stories
  python pipeline/build.py --refit    # weekly: refit stories (also runs if no state yet)

Per firm, in order:
  1. clean + de-duplicate (exact headline across sources, then near-duplicate rewrites, cos > .95)
  2. embed headline + summary with EMBED_MODEL (default openrouter:google/gemini-embedding-2 via the
     OpenRouter embeddings API; a plain model id such as thenlper/gte-small runs locally), cached by text hash, one cache per model. Stories fitted with another
     model are refit from scratch, since centroids from two models are not comparable.
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
     Stories that lose their match are kept in state["past"], so a story that comes back gets its
     old id and name back.
  6. EMERGING (every run, not only on refit days): the last EMERGE_DAYS of the firm's feed, minus
     what the current stories already explain, is grouped; a group becomes a new story when it is
     at least as tight as a typical story, runs over 2+ days, and passes the same two relevance
     tests as the anchors (against the other feeds' articles of the same days), plus a third: it is
     far more common in those days than in the weeks before (so a topic that was always there but
     not split out by the fit does not count). At most MAX_LIVE per firm. It stays a story until
     the next refit decides whether it has become one of the fitted stories.
  7. EVENTS (every run): short, dated happenings inside the stories -- an earnings call, a deal, a
     listing. Articles are close when they say the same thing AND appear within days of each other;
     groups of EV_MIN+ such articles are candidates. The LLM keeps real events (drops recurring
     commentary: buy/sell opinion, predictions, explainers), names them and merges groups that are the
     same event. Known events are recognised by their articles, so a growing event keeps its id and
     name and is not sent to the LLM again.
  8. NAMES: only new stories and new events are named -- by OPENROUTER_MODEL if it and
     OPENROUTER_API_KEY are set; stories fall back to their most distinctive keywords.
  9. EXPORT: this week's articles per story, weekly share series, one representative
     headline per story per week, and the events with their articles.
"""
import argparse, collections, hashlib, json, math, os, re, sys, time, urllib.request
from datetime import datetime, timedelta, timezone
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score

ap = argparse.ArgumentParser(); ap.add_argument("--refit", action="store_true"); ap.add_argument("--only", default="")
args = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(f"{ROOT}/config/firms.json"))
K1, BETA, STABLE, NEARDUP, MATCH = 8, 20.0, 0.85, 0.95, 0.5
EMERGE_DAYS, MIN_EMERGE, OUT_Q, MAX_LIVE = 14, 10, 0.10, 3
EV_TAU, EV_CUT, EV_MIN, EV_SAME = 6.0, 0.30, 8, 0.5     # days, similarity cut, min articles, overlap to be the same event
EMBED = os.environ.get("EMBED_MODEL") or "openrouter:google/gemini-embedding-2"   # the committed state was fitted with this
OR_KEY = os.environ.get("OPENROUTER_API_KEY")
OR_MODEL = os.environ.get("OPENROUTER_MODEL") or "deepseek/deepseek-v4.1-flash"   # names stories, judges and names events
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
    for r in rows: r["dt"] = ts(r["published"])
    rows = [r for r in rows if r["t"] and r["dt"] >= start]
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

# ---- embeddings, cached by text hash, one cache file per model (not committed; CI keeps it in actions/cache)
os.makedirs(f"{ROOT}/data/cache", exist_ok=True)
cp = f"{ROOT}/data/cache/emb-{re.sub(r'[^A-Za-z0-9]+', '-', EMBED).strip('-')}.npz"
cache = {}
if os.path.exists(cp):
    z = np.load(cp); cache = dict(zip(z["h"].tolist(), z["V"]))
todo = {}
for rows in feeds.values():
    for r in rows:
        r["h"] = hashlib.sha1(r["text"].encode()).hexdigest()[:16]
        if r["h"] not in cache: todo[r["h"]] = r["text"]
live = {r["h"] for rows in feeds.values() for r in rows}          # texts that fell out of the window are dropped
st_model = None
def embed(texts):
    global st_model
    if EMBED.startswith("openrouter:"):
        from concurrent.futures import ThreadPoolExecutor
        def batch(i):
            for attempt in range(8):
                try:
                    req = urllib.request.Request("https://openrouter.ai/api/v1/embeddings",
                        data=json.dumps({"model": EMBED.split(":", 1)[1], "input": texts[i:i + 100]}).encode(),
                        headers={"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"})
                    d = json.loads(urllib.request.urlopen(req, timeout=120).read())["data"]
                    return [x["embedding"] for x in sorted(d, key=lambda x: x["index"])]
                except Exception as e:
                    if attempt == 7: raise SystemExit(f"OpenRouter embeddings failed: {e}")
                    time.sleep(3 * (attempt + 1))                  # Google answers 429 now and then under load
        with ThreadPoolExecutor(32) as ex:                        # Gemini takes at most 100 texts per request; throughput
                                                                  # tops out near 32 requests in flight (~900 texts/s)
            out = [v for b in ex.map(batch, range(0, len(texts), 100)) for v in b]
        return unit(np.array(out, dtype=np.float32))
    if st_model is None:
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(EMBED)
    return st_model.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
keys = list(todo)
for i in range(0, len(keys), 8000):                            # save as we go: a first run embeds ~40k texts
    for k, v in zip(keys[i:i + 8000], embed([todo[k] for k in keys[i:i + 8000]])): cache[k] = v.astype(np.float16)
    ks = [k for k in cache if k in live]
    np.savez(cp, h=np.array(ks), V=np.array([cache[k] for k in ks]))
    print(f"embedded {min(i + 8000, len(keys))}/{len(keys)}", flush=True)
print(f"embedded {len(todo)} new texts with {EMBED}, cache {len(cache)}", flush=True)
DIM = len(next(iter(cache.values()))) if cache else 384

# ---- names for new stories
STOP = set("""the a an and or of to in on for with at by from as is are was were be been it its this that these those has
have had will would can could not but than into over after about more most new says said stock stocks shares inc corp company
companies year years today week why what how here heres your you buy should just now up down billion million report reports""".split())
def chat(prompt):
    """One OpenRouter chat completion with OR_MODEL; returns the reply text."""
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps({"model": OR_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2,
                         "reasoning": {"enabled": False}}).encode(),       # naming and sorting need no thinking; with it
        headers={"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"})   # DeepSeek is ~6x slower
    return json.loads(urllib.request.urlopen(req, timeout=180).read())["choices"][0]["message"]["content"]

def name_story(texts, firm):
    if OR_KEY and OR_MODEL:
        prompt = (f"These news headlines about {firm} belong to one ongoing story. Give it a short, plain name (2-5 words, "
                  f"no quotes) and a one-sentence description of what the story is about. Reply as JSON "
                  f'{{"name":"...","blurb":"..."}}.\n\n' + "\n".join("- " + t for t in texts[:10]))
        try:
            j = json.loads(re.search(r"\{.*\}", chat(prompt), re.S).group(0))
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
    X = np.array([cache[r["h"]] for r in rows], dtype=np.float64).reshape(-1, DIM)
    keep = []
    for i in range(len(X)):
        if keep and float((X[keep[-400:]] @ X[i]).max()) > NEARDUP: continue
        keep.append(i)
    feeds[tk] = [rows[i] for i in keep]; XS[tk] = X[keep]
    NAMED[tk] = np.array([bool(PATS[tk].search(r["text"])) for r in feeds[tk]])

# ---- relevance tests (Beta posteriors: 95% lower bound of the ratio of two rates)
g = np.random.default_rng(0)
lo95 = lambda a, n, b, m: float(np.percentile(g.beta(1 + a, 1 + n - a, 20000) / g.beta(1 + b, 1 + m - b, 20000), 2.5))
def others(tk, since):
    """The other feeds' articles since `since` (minus those also in this feed), and how many of them name the firm."""
    mine = {r["h"] for r in feeds[tk]}
    oth = [(t2, i) for t2 in XS if t2 != tk for i, r in enumerate(feeds[t2]) if r["h"] not in mine and r["dt"] >= since]
    no = int(sum(PATS[tk].search(feeds[t2][i]["text"]) is not None and tk not in (feeds[t2][i].get("tickers") or [])
                 for t2, i in oth))
    return np.array([XS[t2][i] for t2, i in oth]).reshape(-1, DIM), no

# ---- relevance model (refit, or any firm without a state fitted with this embedding model)
fitted_with = lambda st: st.get("embed", "thenlper/gte-small")
STATES = {}
for f in CFG["firms"]:
    sp = f"{ROOT}/data/state/{f['ticker']}.json"
    STATES[f["ticker"]] = json.load(open(sp)) if os.path.exists(sp) else None
need = args.refit or any(st is None or fitted_with(st) != EMBED for st in STATES.values())
P1 = {}
if need:
    for f in CFG["firms"]:
        tk = f["ticker"]; X = XS[tk]
        if len(X) < 60: continue
        Xo, no = others(tk, start)
        mu1 = X.mean(0); C1 = unit(KMeans(K1, n_init=4, random_state=0).fit(unit(X - mu1)).cluster_centers_)
        h1, ho = (unit(X - mu1) @ C1.T).argmax(1), (unit(Xo - mu1) @ C1.T).argmax(1)
        keep1 = []
        for k in range(K1):
            n_k, o_k = int((h1 == k).sum()), int((ho == k).sum())
            name_ok = lo95(int(NAMED[tk][h1 == k].sum()), n_k, no, len(Xo)) > 1     # names the firm beyond chance
            feed_ok = lo95(n_k, len(h1), o_k, len(ho)) > 1                            # not genre shared by every feed
            keep1.append(bool(n_k >= 15 and name_ok and feed_ok))
        P1[tk] = dict(mu1=mu1, C1=C1, h1=h1, keep1=keep1)

ip = f"{ROOT}/site/data/index.json"
prev_index = {x["ticker"]: x for x in json.load(open(ip))["firms"]} if os.path.exists(ip) else {}
index = []
for f in CFG["firms"]:
    tk, firm = f["ticker"], f["name"]
    if args.only and tk not in args.only.split(","): continue
    rows, X = feeds[tk], XS[tk]
    if len(rows) < 60:                                   # a thin week must not drop the firm from the site
        print(f"{tk}: only {len(rows)} articles, skipped (site keeps its last build)")
        index += [prev_index[tk]] if tk in prev_index else []; continue

    sp = f"{ROOT}/data/state/{tk}.json"
    state = STATES[tk]
    if args.refit or state is None or fitted_with(state) != EMBED:
        mu1, C1, h1, keep1 = P1[tk]["mu1"], P1[tk]["C1"], P1[tk]["h1"], P1[tk]["keep1"]
        rel = np.array(keep1)[h1]
        if rel.sum() < 40:
            print(f"{tk}: too few relevant articles ({rel.sum()}), skipped (site keeps its last build)")
            index += [prev_index[tk]] if tk in prev_index else []; continue
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
        # identity: match to last fit's stories by meaning (in the new centred space), then to retired ones
        same = state is not None and fitted_with(state) == EMBED    # centroids from another model are not comparable
        old, past = (state["stories"], state.get("past", [])) if same else ([], [])
        ids, nxt = [None] * K, (state["next_id"] if state else 0)
        if old:
            Oc = unit(np.array([s["centroid"] for s in old]) - mu2)
            Sim = C2 @ Oc.T; r_i, c_i = linear_sum_assignment(-Sim)
            for a, b in zip(r_i, c_i):
                if Sim[a, b] >= MATCH: ids[a] = old[b]["id"]
        for k in range(K):
            if ids[k] is None and past:
                sims = unit(np.array([p["centroid"] for p in past]) - mu2) @ C2[k]; b = int(sims.argmax())
                if sims[b] >= MATCH and past[b]["id"] not in ids: ids[k] = past[b]["id"]
        stories = []
        for k in range(K):
            prev = next((s for s in old + past if s["id"] == ids[k]), None)
            if prev: nm, bl, src = prev["name"], prev["blurb"], prev["name_src"]
            else:
                idx = np.where(h2 == k)[0]; top = idx[np.argsort(-(X2[idx] @ C2[k]))[:10]]
                nm, bl, src = name_story([rows[np.where(rel)[0][i]]["t"] for i in top], firm)
                ids[k] = nxt; nxt += 1
            stories.append(dict(id=ids[k], name=nm, blurb=bl, name_src=src, centroid=np.round(raw_c[k], 5).tolist(),
                                born=prev["born"] if prev else now.date().isoformat()))
        past = [{k: s[k] for k in ("id", "name", "blurb", "name_src", "centroid", "born")}
                for s in old + past if s["id"] not in ids][:100]
        state = dict(fitted=now.isoformat(), embed=EMBED, K=K, next_id=nxt, mu1=np.round(mu1, 5).tolist(),
                     C1=np.round(C1, 5).tolist(), keep1=keep1, mu2=np.round(mu2, 5).tolist(),
                     C2=np.round(C2, 5).tolist(), stories=stories, past=past,
                     events=(state or {}).get("events", []), next_event=(state or {}).get("next_event", 0))
        json.dump(state, open(sp, "w"))
        print(f"{tk}: refit  K={K}  kept anchors {sum(keep1)}/{K1}", flush=True)

    mu1, C1, keep1 = np.array(state["mu1"]), np.array(state["C1"]), np.array(state["keep1"])
    mu2, stories = np.array(state["mu2"]), state["stories"]
    def claims():
        """Similarity of every article to every story, and which articles count as relevant: those in a kept
        coarse anchor, plus those within an emerging story's radius (its topic may sit in a dropped anchor)."""
        Sa = unit(X - mu2) @ np.array(state["C2"]).T
        rl = keep1[(unit(X - mu1) @ C1.T).argmax(1)]
        for k, s in enumerate(stories):
            if s.get("radius"): rl = rl | (Sa[:, k] >= s["radius"])
        return Sa, rl

    # ---- EMERGING: group what the current stories leave unexplained in the last EMERGE_DAYS.
    #      "Unexplained" is measured against the fitted stories only, so the bar does not move as
    #      emerging stories are added; articles an emerging story already claims are not candidates.
    Sall, rel = claims()
    Xc = unit(X - mu2)
    fit = np.array([not s.get("radius") for s in stories]); rel0 = keep1[(unit(X - mu1) @ C1.T).argmax(1)]
    bestf, own = Sall[:, fit].max(1), Sall[:, fit].argmax(1)
    claimed = np.zeros(len(X), bool)
    for k in np.where(~fit)[0]: claimed |= Sall[:, k] >= stories[k]["radius"]
    since = now - timedelta(days=EMERGE_DAYS)
    fresh = np.array([r["dt"] >= since for r in rows])
    cand = np.where(fresh & ~claimed & (bestf < np.quantile(bestf[rel0], OUT_Q)))[0]   # further from every story than 90% of the firm's own
    tight = []                                                               # mean pairwise cosine of each fitted story
    for k in range(fit.sum()):
        m = np.where(rel0 & (own == k))[0]
        if len(m) > 1: v = Xc[m].mean(0); tight.append((len(m) * float(v @ v) - 1) / (len(m) - 1))
    found = []
    if len(cand) >= MIN_EMERGE and tight and (~fit).sum() < MAX_LIVE:
        lab = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                      distance_threshold=1 - float(np.median(tight))).fit(Xc[cand]).labels_
        Xo, no = others(tk, since); Xoc = unit(Xo - mu2)
        for l in set(lab):
            gi = cand[lab == l]
            if len(gi) < MIN_EMERGE or len({rows[i]["dt"].date() for i in gi}) < 2: continue   # a group, over 2+ days
            c = unit(Xc[gi].mean(0)); rad = float(np.quantile(Xc[gi] @ c, 0.1))
            name_ok = lo95(int(NAMED[tk][gi].sum()), len(gi), no, len(Xo)) > 1
            near = Xc @ c >= rad
            feed_ok = lo95(int(near[fresh].sum()), int(fresh.sum()), int((Xoc @ c >= rad).sum()), len(Xo)) > 1
            new_ok = lo95(int(near[fresh].sum()), int(fresh.sum()), int(near[~fresh].sum()), int((~fresh).sum())) > 1
            if name_ok and feed_ok and new_ok: found.append((len(gi), gi, c, rad))   # new_ok: far more common lately than before
    for _, gi, c, rad in sorted(found, key=lambda x: -x[0])[:MAX_LIVE - (~fit).sum()]:
        past = state.setdefault("past", [])
        sims = unit(np.array([p["centroid"] for p in past]).reshape(-1, DIM) - mu2) @ c
        if len(sims) and sims.max() >= MATCH:            # a story coming back keeps its id and name
            p = past.pop(int(sims.argmax())); sid, nm, bl, src, born = p["id"], p["name"], p["blurb"], p["name_src"], p["born"]
        else:
            top = gi[np.argsort(-(Xc[gi] @ c))[:10]]
            nm, bl, src = name_story([rows[i]["t"] for i in top], firm)
            sid, born = state["next_id"], now.date().isoformat(); state["next_id"] += 1
        stories.append(dict(id=sid, name=nm, blurb=bl, name_src=src, centroid=np.round(unit(X[gi].mean(0)), 5).tolist(),
                            born=born, radius=round(rad, 4)))
        state["C2"].append(np.round(c, 5).tolist())
        print(f"{tk}: new story '{nm}' ({len(gi)} articles over {len({rows[i]['dt'].date() for i in gi})} days)", flush=True)
    if found: json.dump(state, open(sp, "w"))

    # ---- assign every article in the window with the stored model
    Sall, rel = claims()
    R = [rows[i] for i in np.where(rel)[0]]; X2 = unit(X[rel] - mu2)
    S = Sall[rel]; hard = S.argmax(1)
    P = np.exp(BETA * (S - S.max(1, keepdims=True))); P /= P.sum(1, keepdims=True)
    wk = np.array([monday(r["dt"]) for r in R]); weeks = sorted(set(wk))
    recent = np.array([r["dt"] >= week_cut for r in R])
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
                        new=s["name_src"] != "manual" and s["born"] >= since.date().isoformat(),
                        series=series, evolution=evo,
                        latest=[dict(t=R[i]["t"], d=R[i]["desc"][:260], p=R[i]["publisher"],
                                     date=R[i]["dt"].strftime("%b %d"), u=R[i]["url"]) for i in lat]))

    # ---- EVENTS: similarity = cosine x exp(-days apart / EV_TAU); average-linkage groups of EV_MIN+ articles.
    #      A group sharing at least EV_SAME of its articles with a known event (state["events"], kept or
    #      rejected) takes that one's verdict, id and name; only the rest go to the LLM, in one call per 40.
    day = np.array([(r["dt"] - start).total_seconds() / 86400 for r in R])
    lab = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=1 - EV_CUT
                                  ).fit(1 - (X2 @ X2.T) * np.exp(-np.abs(day[:, None] - day[None]) / EV_TAU)).labels_
    groups = [g for g in (np.where(lab == l)[0] for l in set(lab)) if len(g) >= EV_MIN]
    live_h = {r["h"] for r in R}
    known = [dict(e, h=[h for h in e["h"] if h in live_h]) for e in state.get("events", [])]
    known = [e for e in known if e["h"]]                                      # events that left the window are dropped
    gh = [{R[i]["h"] for i in g} for g in groups]
    owner = [next((e for e in known if len(hs & set(e["h"])) >= EV_SAME * len(hs)), None) for hs in gh]
    todo = [j for j in range(len(groups)) if owner[j] is None]
    if todo and OR_KEY and OR_MODEL:
        central = lambda g: g[np.argsort(-(X2[g] @ unit(X2[g].mean(0))))]
        for b in range(0, len(todo), 40):
            js = todo[b:b + 40]
            kn = [e for e in known if e["event"]][-60:]
            prompt = (f"Below are groups of news articles about {firm} (or its industry), each group published within a few days. "
                "For each group decide whether it covers a specific EVENT -- something that happened: a report or earnings, a deal, "
                "a listing, a lawsuit or ruling, a launch, an executive statement, a big price move on given days -- or recurring "
                "COMMENTARY with no new fact (buy/sell opinion, predictions, explainers, lists of stocks). For events give a short "
                "headline-style name of 3-7 words stating the concrete fact (e.g. \"Q2 revenue hits record $96B\", \"SK Hynix lists "
                "on Nasdaq\"), not a theme. If a group is the same event as a known event or as another group, set same_as to that "
                "id. Reply with JSON only: a list of {\"id\": \"g..\", \"event\": true|false, \"name\": \"...\", \"same_as\": \"e..\"|\"g..\"|null}.\n\n"
                + ("Known events:\n" + "\n".join(f"[e{e['id']}] {e['name']} ({e['start']})" for e in kn) + "\n\n" if kn else "")
                + "\n\n".join(f"[g{j}] {len(groups[j])} articles, {R[groups[j][day[groups[j]].argmin()]]['dt']:%b %d}-"
                                f"{R[groups[j][day[groups[j]].argmax()]]['dt']:%b %d}\n"
                                + "\n".join(f"  - {R[i]['dt']:%b %d}: {R[i]['t']}" for i in central(groups[j])[:6]) for j in js))
            try:
                j = json.loads(re.search(r"\[.*\]|\{.*\}", chat(prompt), re.S).group(0))       # a list, or one object for one group
                ans = {str(v["id"]).strip("[]"): v for v in (j if isinstance(j, list) else [j])}
            except Exception as e:
                print(f"   events via OpenRouter failed ({e}); left for the next run", flush=True); continue
            by_e = {f"e{e['id']}": e for e in known}
            for j in sorted(js, key=lambda j: (ans.get(f"g{j}") or {}).get("same_as") is not None):   # new events first
                v = ans.get(f"g{j}")
                if not v: continue
                sa = str(v.get("same_as") or "")
                tgt = by_e.get(sa) or (owner[int(sa[1:])] if sa[1:].isdigit() and sa.startswith("g") and int(sa[1:]) < len(owner) else None)
                if tgt is None:
                    d0 = R[groups[j][day[groups[j]].argmin()]]["dt"].date().isoformat()
                    tgt = dict(id=state.get("next_event", 0), name=str(v.get("name") or "")[:80].strip(), event=bool(v.get("event")),
                               start=d0, h=[]); state["next_event"] = tgt["id"] + 1
                    known.append(tgt); by_e[f"e{tgt['id']}"] = tgt
                owner[j] = tgt
    for j, e in enumerate(owner):
        if e is not None: e["h"] = sorted(set(e["h"]) | gh[j])
    state["events"] = known
    json.dump(state, open(sp, "w"))
    pos = {r["h"]: i for i, r in enumerate(R)}
    events = []
    for e in known:
        m = np.array([pos[h] for h in e["h"]])
        if not e["event"] or len(m) < EV_MIN: continue
        m = m[np.argsort(-(X2[m] @ unit(X2[m].mean(0))))]                    # most typical first
        d = sorted(R[i]["dt"] for i in m)
        events.append(dict(id=e["id"], name=e["name"], story=stories[int(np.bincount(hard[m]).argmax())]["id"], n=len(m),
                           latest_n=int(recent[m].sum()), start=d[0].date().isoformat(), end=d[-1].date().isoformat(),
                           weeks={w.isoformat(): int((wk[m] == w).sum()) for w in weeks if (wk[m] == w).any()},
                           top=[dict(t=R[i]["t"], p=R[i]["publisher"], date=R[i]["dt"].strftime("%b %d"), u=R[i]["url"]) for i in m[:3]],
                           h=[R[i]["h"] for i in m]))
    events.sort(key=lambda e: e["end"], reverse=True)
    print(f"{tk}: {len(groups)} candidate groups ({len(todo)} new), {len(events)} events", flush=True)

    doc = dict(ticker=tk, name=firm, updated=now.strftime("%Y-%m-%d %H:%M UTC"), weeks=[w.isoformat() for w in weeks],
               latest_from=week_cut.strftime("%b %d"), latest_to=now.strftime("%b %d"),
               n_articles=len(rows), n_relevant=len(R), stories=out, events=events)
    json.dump(doc, open(f"{ROOT}/site/data/{tk}.json", "w"), ensure_ascii=False)
    top = max(out, key=lambda s: s["latest_n"]) if out else None
    index.append(dict(ticker=tk, name=firm, top=top["name"] if top else ""))
    print(f"{tk}: {len(rows)} articles, {len(R)} relevant, {len(out)} stories", flush=True)

if not args.only:
    json.dump(dict(updated=now.strftime("%Y-%m-%d %H:%M UTC"), firms=index), open(ip, "w"))
