"""Real-data map for the narrative-map mockups: every relevant article of one firm, placed in 2D by meaning.

  pip install umap-learn                        # mockup only, not in requirements.txt
  python pipeline/map_mockup.py NVDA            # -> site/mockups/NVDA.html (from site/mockups/template.html)

Reproduces build.py's daily path for one firm (clean, de-duplicate, embed with the fitted model, relevance
and story assignment from data/state), then:
  - EVENTS: groups of articles that say the same thing within days (cosine x exp(-days apart / 6)), kept,
    named and merged by one LLM call (OPENROUTER_MODEL, default anthropic/claude-sonnet-5; cached per group).
  - LAYOUT: one region per story, placed near its MDS position on story-centroid distance, area by article
    count, title reserved above it so no two regions or titles overlap; inside a region, UMAP of its articles.
"now" is pinned to the fitted state's timestamp so the numbers match the committed site/data build.
Run it in a checkout whose data/state was fitted with the model to compare (EMBED_MODEL=... build.py --refit).
"""
import hashlib, json, os, re, sys, urllib.request
from datetime import datetime, timedelta
import numpy as np
import umap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tk = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
CFG = json.load(open(f"{ROOT}/config/firms.json")); f = next(x for x in CFG["firms"] if x["ticker"] == tk)
state = json.load(open(f"{ROOT}/data/state/{tk}.json")); site = json.load(open(f"{ROOT}/site/data/{tk}.json"))
EMBED = state.get("embed", "thenlper/gte-small")
now = datetime.fromisoformat(state["fitted"]); start = now - timedelta(days=CFG["window_days"]); week_cut = now - timedelta(days=7)
norm = lambda t: re.sub(r"[^a-z0-9]+", "", t.lower())[:120]
unit = lambda A: A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)
monday = lambda d: (d - timedelta(days=d.weekday())).date()

# ---- same cleaning and de-duplication as build.py
rows = [json.loads(l) for l in open(f"{ROOT}/data/raw/{tk}.jsonl")]
for r in rows: r["dt"] = datetime.fromisoformat(r["published"].replace("Z", "+00:00"))
rows = [r for r in rows if r["t"] and start <= r["dt"] <= now]
for r in rows:
    if re.match(r"\s*https?://\S+\s*$", r["desc"] or ""): r["desc"] = ""
by_head = {}
for r in sorted(rows, key=lambda r: (r["src"] != "polygon", r["published"])):
    k = norm(r["t"])
    if len(k) < 15: continue
    if k in by_head:
        if len(r["desc"]) > len(by_head[k]["desc"]): by_head[k]["desc"] = r["desc"]
        continue
    by_head[k] = r
rows = sorted(by_head.values(), key=lambda r: r["published"])
for r in rows: r["text"] = r["t"] + (". " + r["desc"] if r["desc"] else ""); r["h"] = hashlib.sha1(r["text"].encode()).hexdigest()[:16]

# ---- embeddings: build.py's cache if present, else embed now
cp = f"{ROOT}/data/cache/emb-{re.sub(r'[^A-Za-z0-9]+', '-', EMBED).strip('-')}.npz"
cache = {}
if os.path.exists(cp): z = np.load(cp); cache = dict(zip(z["h"].tolist(), z["V"]))
todo = [r for r in rows if r["h"] not in cache]
if todo:
    from sentence_transformers import SentenceTransformer
    V = SentenceTransformer(EMBED).encode([r["text"] for r in todo], normalize_embeddings=True, batch_size=128)
    for r, v in zip(todo, V): cache[r["h"]] = v.astype(np.float16)
    os.makedirs(os.path.dirname(cp), exist_ok=True); np.savez(cp, h=np.array(list(cache)), V=np.array(list(cache.values())))
X = np.array([cache[r["h"]] for r in rows], dtype=np.float64)
keep = []
for i in range(len(X)):
    if keep and float((X[keep[-400:]] @ X[i]).max()) > 0.95: continue
    keep.append(i)
rows, X = [rows[i] for i in keep], X[keep]

# ---- relevance and stories from the fitted state
mu1, C1, keep1, mu2 = np.array(state["mu1"]), np.array(state["C1"]), np.array(state["keep1"]), np.array(state["mu2"])
C2, stories = np.array(state["C2"]), state["stories"]
Sa = unit(X - mu2) @ C2.T
rel = keep1[(unit(X - mu1) @ C1.T).argmax(1)]
for k, s in enumerate(stories):
    if s.get("radius"): rel = rel | (Sa[:, k] >= s["radius"])
R = [rows[i] for i in np.where(rel)[0]]; X2 = unit(X[rel] - mu2); S = Sa[rel]; hard = S.argmax(1)
from sklearn.metrics import silhouette_score
top2 = np.sort(S, 1)[:, -2:]
print(f"{tk}: {len(rows)} articles, {len(R)} relevant (site build says {site['n_articles']}, {site['n_relevant']}); "
      f"silhouette {silhouette_score(X2, hard, metric='cosine'):.3f}, median margin to 2nd story {np.median(top2[:, 1] - top2[:, 0]):.3f}")

# ---- EVENTS: short, dated happenings, found across all of the firm's relevant articles (an earnings call
#      touches several themes). Two articles are close when they say the same thing AND appear within days
#      of each other: similarity = cosine x exp(-|days apart| / TAU). Average-linkage groups of MIN_EV+
#      articles are candidates; an LLM then keeps real events (drops recurring commentary such as "should
#      you buy X?"), names them and merges groups that are the same event. Names are cached per group.
from sklearn.cluster import AgglomerativeClustering
TAU, EV_CUT, MIN_EV = 6.0, 0.30, 8
days = np.array([(r["dt"] - start).total_seconds() / 86400 for r in R])
Sim = (X2 @ X2.T) * np.exp(-np.abs(days[:, None] - days[None]) / TAU)
lab = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                              distance_threshold=1 - EV_CUT).fit(1 - Sim).labels_
del Sim
groups = sorted([np.where(lab == l)[0] for l in set(lab) if (lab == l).sum() >= MIN_EV], key=len, reverse=True)
tops = [g[np.argsort(-(X2[g] @ unit(X2[g].mean(0))))[:6]] for g in groups]
OR_KEY, OR_MODEL = os.environ.get("OPENROUTER_API_KEY"), os.environ.get("OPENROUTER_MODEL") or "anthropic/claude-sonnet-5"
ecp = f"{ROOT}/data/cache/events-{tk}-{re.sub(r'[^A-Za-z0-9]+', '-', EMBED).strip('-')}.json"
ecache = json.load(open(ecp)) if os.path.exists(ecp) else {}
gkey = lambda g: hashlib.sha1("|".join(sorted(R[i]["h"] for i in g)).encode()).hexdigest()[:16]
todo = [j for j, g in enumerate(groups) if gkey(g) not in ecache]
if todo and OR_KEY:
    listing = "\n\n".join(f"[{j}] {len(groups[j])} articles, {R[groups[j][0]]['dt']:%b %d}-{max(R[i]['dt'] for i in groups[j]):%b %d}\n"
                          + "\n".join(f"  - {R[i]['dt']:%b %d}: {R[i]['t']}" for i in tops[j]) for j in todo)
    prompt = (f"Below are groups of news articles about {f['name']}, each published within a few days. For each group decide "
              "whether it covers a specific EVENT (something that happened: a report, a deal, a listing, a ruling, a launch, a "
              "price move on a given day) or is recurring COMMENTARY (buy/sell opinion, predictions, explainers with no new fact). "
              "For events, give a short event-centric headline name of 3-7 words that states the concrete fact (e.g. \"Q2 revenue "
              "hits record $96B\", \"SK Hynix lists on Nasdaq\"), not a theme. If two groups are the same event, give the later one "
              "same_as = the other's id. Reply with JSON only: a list of {\"id\": int, \"event\": bool, \"name\": str, \"same_as\": int|null}.\n\n" + listing)
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps({"model": OR_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0}).encode(),
        headers={"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["message"]["content"]
    for v in json.loads(re.search(r"\[.*\]", out, re.S).group(0)):
        j = int(v["id"]); sa = v.get("same_as")
        ecache[gkey(groups[j])] = dict(event=bool(v["event"]), name=v["name"].strip()[:70],
                                       same_as=gkey(groups[int(sa)]) if sa is not None and 0 <= int(sa) < len(groups) else None)
    os.makedirs(os.path.dirname(ecp), exist_ok=True); json.dump(ecache, open(ecp, "w"), indent=1)
elif todo: print("no OPENROUTER_API_KEY: events named by their most typical headline")
ev_of = -np.ones(len(R), int); events = []
root = lambda key: root(ecache[key]["same_as"]) if ecache.get(key, {}).get("same_as") in ecache and ecache[key]["same_as"] != key else key
by_root = {}
for j, g in enumerate(groups):
    v = ecache.get(gkey(g), dict(event=True, name=R[tops[j][0]]["t"][:70]))
    if not v["event"]: continue
    by_root.setdefault(root(gkey(g)), []).append(j)
for rk, js in by_root.items():
    g = np.concatenate([groups[j] for j in js]); ev_of[g] = len(events)
    dts = sorted(R[i]["dt"] for i in g)
    events.append(dict(name=ecache.get(rk, {}).get("name") or R[tops[js[0]][0]]["t"][:70], n=len(g),
                       k=int(np.bincount(hard[g]).argmax()), start=dts[0].date().isoformat(), end=dts[-1].date().isoformat(),
                       peak=max(set(d.date() for d in dts), key=[d.date() for d in dts].count).isoformat()))
print(f"events: {len(groups)} candidate groups, {len(events)} events after the LLM pass, covering {(ev_of >= 0).mean():.0%} of articles")

# ---- 2D: one region per story, placed by MDS on story-centroid distance, area proportional to article count,
#      pushed apart until no two overlap; inside each region its own articles laid out by UMAP (cosine).
#      Layout runs in the page's pixel space (map W x H) so regions stay round, then maps to [0,1].
MW, PADX, PADY = 1000, 60, 42
MH = 540 + 80 * max(0, len(stories) - 6)                         # more stories, taller map
C = unit(np.array([X2[hard == k].mean(0) for k in range(len(stories))]))
Dm = 1 - C @ C.T; n = len(C); J = np.eye(n) - 1 / n
w, V = np.linalg.eigh(-0.5 * J @ (Dm ** 2) @ J); P = V[:, -2:] * np.sqrt(np.maximum(w[-2:], 1e-9))
P = (P - P.mean(0)) / (np.abs(P).max() + 1e-9) * [(MW - 2 * PADX) / 2, (MH - 2 * PADY) / 2] + [MW / 2, MH / 2]
cnt = np.bincount(hard, minlength=n); target = P.copy()
# each story is a box: its region plus its title above (title size as the page draws it at the story's busiest week).
# Largest story first, each takes the free spot nearest its MDS position; if one finds no room, all regions shrink.
fs = np.array([15 + np.sqrt(max([v["n"] for v in s["series"]] + [s["latest_n"]])) * 0.8 for s in site["stories"]])
tw = np.array([len(s["name"]) for s in site["stories"]]) * fs * 0.5
gx, gy = np.meshgrid(np.arange(0, MW + 1, 8.0), np.arange(0, MH + 1, 8.0)); G = np.c_[gx.ravel(), gy.ravel()]
for area in (0.26, 0.22, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06):
    rad = np.sqrt(cnt / cnt.sum() * MW * MH * area / np.pi)
    hw, top, bot = np.maximum(rad, tw / 2 + 6), rad + fs + 22, rad + 6      # half width, extent above / below centre
    P, placed = np.zeros((n, 2)), []
    for k in np.argsort(-cnt):
        ok = (G[:, 0] - hw[k] >= 4) & (G[:, 0] + hw[k] <= MW - 4) & (G[:, 1] - top[k] >= 4) & (G[:, 1] + bot[k] <= MH - 4)
        for j in placed:
            ok &= (np.abs(G[:, 0] - P[j, 0]) >= hw[k] + hw[j] + 14) | (G[:, 1] - P[j, 1] >= bot[j] + top[k] + 10) | (P[j, 1] - G[:, 1] >= bot[k] + top[j] + 10)
        if not ok.any(): break
        c = G[ok]; P[k] = c[((c - target[k]) ** 2).sum(1).argmin()]; placed.append(k)
    if len(placed) == n: break
print(f"layout: map {MW}x{MH}, regions {area:.0%} of it, {len(placed)}/{n} placed")
Y = np.zeros((len(R), 2))
for k in range(n):
    m = np.where(hard == k)[0]
    if len(m) >= 20: Z = umap.UMAP(n_neighbors=min(30, len(m) - 1), min_dist=0.1, metric="cosine", random_state=0).fit_transform(X2[m])
    else: Z = np.random.default_rng(k).normal(size=(len(m), 2))
    Z = Z - np.median(Z, 0); r = np.hypot(*Z.T) + 1e-9               # keep each article's direction, even out the radius
    Z = Z / r[:, None] * np.sqrt((np.argsort(np.argsort(r)) + 0.5) / len(r))[:, None]   # so a dense core fills its region
    Y[m] = P[k] + Z * rad[k]
tx = lambda X_: (X_ - PADX) / (MW - 2 * PADX); ty = lambda Y_: (Y_ - PADY) / (MH - 2 * PADY)   # inverse of the page's px/py
for e_i, e in enumerate(events):
    m = ev_of == e_i; e["x"], e["y"] = round(float(tx(np.median(Y[m, 0]))), 4), round(float(ty(np.median(Y[m, 1]))), 4)

weeks = site["weeks"]
order = np.lexsort((-S[np.arange(len(R)), hard], hard))       # per story, most typical first (the tooltips list the top ones)
arts = [dict(k=int(hard[i]), e=int(ev_of[i]), t=R[i]["t"], d=R[i]["desc"][:170], p=R[i]["publisher"], date=R[i]["dt"].strftime("%b %d"),
             week=monday(R[i]["dt"]).isoformat(), u=R[i]["url"], now=int(R[i]["dt"] >= week_cut),
             x=round(float(tx(Y[i, 0])), 4), y=round(float(ty(Y[i, 1])), 4)) for i in order]
links = [[i, j, round(float(C[i] @ C[j]), 2)] for i in range(n) for j in range(i + 1, n)]
doc = {k: site[k] for k in ("name", "ticker", "weeks", "updated", "latest_from", "latest_to", "n_articles", "n_relevant")}
doc["embed"] = EMBED; doc["mapH"] = MH
doc["stories"] = [dict({k: s[k] for k in ("name", "blurb", "n", "latest_n", "series")},
                       cx=round(float(tx(P[k, 0])), 4), cy=round(float(ty(P[k, 1])), 4), r=round(float(rad[k]), 1),
                       lx=round(float(tx(P[k, 0])), 4), ly=round(float(ty(P[k, 1] - rad[k] - 4)), 4)) for k, s in enumerate(site["stories"])]
doc.update(arts=arts, links=links, events=events)
tpl = open(f"{ROOT}/site/mockups/template.html").read()
open(f"{ROOT}/site/mockups/{tk}.html", "w").write(tpl.replace("__DATA__", json.dumps(doc, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")))
print(f"wrote site/mockups/{tk}.html with {len(arts)} articles, {len(events)} events")
