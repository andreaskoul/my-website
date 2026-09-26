"""Fetch company news from Finnhub + Polygon into data/raw/<TICKER>.jsonl.

  python pipeline/fetch.py --days 2     # daily run (2 days = overlap for safety; reaches further back
                                        # for a firm whose newest stored article is older, so missed runs leave no gap)
  python pipeline/fetch.py --days 91    # backfill

Appends new articles only (keyed on source + id) and trims each file to the
display window plus a margin, so the repo stays small. Keys come from the
environment: FINNHUB_API_KEY, POLYGON_API_KEY. A missing key skips that source.

Rate limits on the free tiers: Finnhub 60 calls/min (queried one day at a time,
because its per-call cap is undocumented), Polygon 5 calls/min.
"""
import argparse, html, json, os, re, time, urllib.parse, urllib.request
from datetime import datetime, timedelta, timezone

ap = argparse.ArgumentParser(); ap.add_argument("--days", type=int, default=2); ap.add_argument("--only", default="")
args = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(f"{ROOT}/config/firms.json"))
FK, PK = os.environ.get("FINNHUB_API_KEY"), os.environ.get("POLYGON_API_KEY")
now = datetime.now(timezone.utc)
keep_after = (now - timedelta(days=CFG["window_days"] + 30)).isoformat()
firms = [f for f in CFG["firms"] if not args.only or f["ticker"] in args.only.split(",")]
if not (FK or PK): raise SystemExit("no FINNHUB_API_KEY or POLYGON_API_KEY in the environment (repo secrets): nothing to fetch")
if not (FK and PK): print(f"warning: {'POLYGON' if FK else 'FINNHUB'}_API_KEY missing, that source is skipped", flush=True)

def clean(t):
    t = html.unescape(html.unescape(t or ""))
    if re.search("[\u00c2\u00c3\u00e2][\u0080-\u00bf\u20ac\u2122\u0153\u201c\u201d\u2018\u2019]", t):
        for enc in ("cp1252", "latin-1"):                  # UTF-8 that was decoded as Latin-1 upstream ("Palantirâ€™s")
            try: t = t.encode(enc).decode("utf-8"); break
            except UnicodeError: pass
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[​-‏﻿­]", "", t)
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return "" if re.match(r"https?://\S+$", t) else t      # a bare URL is not a summary

def get(url, headers=None):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429: time.sleep(15 * (attempt + 1)); continue
            if e.code in (401, 403): raise SystemExit(f"key refused: HTTP {e.code} for {url.split('?')[0]}")
            return None
        except Exception:
            time.sleep(3)
    return None

for f in firms:
    tk = f["ticker"]; path = f"{ROOT}/data/raw/{tk}.jsonl"
    rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    seen = {(r["src"], r["id"]) for r in rows}
    new = []
    last = max((r["published"] for r in rows), default=None)       # catch up after missed runs, within the window
    gap = (now - datetime.fromisoformat(last.replace("Z", "+00:00"))).days + 1 if last else CFG["window_days"]
    days = min(max(args.days, gap), CFG["window_days"])

    if PK:
        since = (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = "https://api.polygon.io/v2/reference/news?" + urllib.parse.urlencode(
            {"ticker": tk, "published_utc.gte": since, "limit": 1000, "order": "asc", "sort": "published_utc"})
        while url:
            d = get(url, {"Authorization": f"Bearer {PK}"}); time.sleep(12.5)
            if not d: break
            for a in d.get("results", []):
                if ("polygon", a["id"]) in seen: continue
                seen.add(("polygon", a["id"]))
                new.append(dict(src="polygon", id=a["id"], t=clean(a.get("title")), desc=clean(a.get("description")),
                                tickers=a.get("tickers", []), publisher=(a.get("publisher") or {}).get("name"),
                                published=a["published_utc"], url=a.get("article_url")))
            url = d.get("next_url")

    if FK:
        for i in range(days, -1, -1):
            day = (now.date() - timedelta(days=i)).isoformat()
            d = get("https://finnhub.io/api/v1/company-news?" + urllib.parse.urlencode(
                {"symbol": tk, "from": day, "to": day, "token": FK})) or []
            time.sleep(1.1)
            for a in d:
                k = ("finnhub", str(a.get("id")))
                if k in seen: continue
                seen.add(k)
                h, s = clean(a.get("headline")), clean(a.get("summary"))
                new.append(dict(src="finnhub", id=k[1], t=h, desc=(s if s != h else ""),
                                tickers=[x for x in (a.get("related") or "").split(",") if x], publisher=a.get("source"),
                                published=datetime.fromtimestamp(a.get("datetime", 0), timezone.utc).isoformat(),
                                url=a.get("url")))

    rows = [r for r in rows + new if r["published"] >= keep_after]
    rows.sort(key=lambda r: r["published"])
    with open(path, "w") as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{tk:<6} +{len(new):>5} new   {len(rows):>6} stored", flush=True)
