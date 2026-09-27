# Narratives — project brief

A minimal public website: pick a company from a dropdown, see **which stories dominate its news
this week** (with the actual articles), and **how each story evolved** over the last ~13 weeks.
Runs itself: GitHub Actions fetches, rebuilds and deploys daily to GitHub Pages. Hard constraint:
**zero running cost.**

This brief was written at the end of a long design session. Everything below was tested on real
data unless marked otherwise. Read "Decisions" before changing any method — most of them replaced
an earlier idea that failed on real data, and the reasons are recorded here.

## Layout

```
config/firms.json          20 firms: ticker, display name, regex used to test "does this text name the firm"
pipeline/fetch.py          Finnhub + Polygon -> data/raw/<TICKER>.jsonl (append, dedup, trim to window)
pipeline/build.py          raw -> stories -> site/data/<TICKER>.json (+ index.json)
data/raw/                  committed; ~13 weeks per firm
data/state/<TICKER>.json   committed; the fitted story model per firm (centroids, names, ids)
data/cache/emb-<model>.npz NOT committed; embedding cache per model, kept in actions/cache
site/narratives.html       the narrative landscape page (vanilla JS, no build step)
site/index.html, cv.html, thesis.html   the personal site around it: shared header (name + links), blank for now
.github/workflows/update.yml   daily at 06:15 UTC; refits stories on Mondays; manual run with days=91 = backfill
```

Run locally:
```
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
export FINNHUB_API_KEY=... POLYGON_API_KEY=...        # never commit keys
python pipeline/fetch.py --days 2          # or --days 91 for a full backfill (~40 min, rate limits)
python pipeline/build.py --refit           # first run / weekly; plain `build.py` reuses fitted stories
cd site && python -m http.server           # open http://localhost:8000/narratives.html#NVDA
```

## Running it (live since 2026-09-26)

Site: https://andreaskoul.github.io/my-website/ — repo `andreaskoul/my-website` (public), branch `main`
(the default: scheduled workflows only run from the default branch).

- Workflow `update`: daily 06:15 UTC; Mondays refit all firms; also on a push to main that changes site/
  (outside site/data), so page edits and uploads such as site/thesis.pdf go live without waiting. Two jobs: `update` (fetch, build, commit
  data to the branch) and `deploy` (publish `site/` to Pages from the branch tip). Kept apart so a Pages
  problem never stops the data update: the first run on `main` was refused whole by the github-pages
  environment rule, which still named the old branch.
- Repo secrets: `FINNHUB_API_KEY`, `POLYGON_API_KEY`, `OPENROUTER_API_KEY` (all required; fetch and
  build stop with a clear message when one is missing). Optional repo variables: `OPENROUTER_MODEL` (unset
  = deepseek/deepseek-v4.1-flash) and `EMBED_MODEL` (unset = openrouter:google/gemini-embedding-2; a plain
  id such as thenlper/gte-small runs locally and needs sentence-transformers + torch). Changing
  `EMBED_MODEL` refits every firm from scratch and story names are not carried over.
- Settings that must stay: Pages source = GitHub Actions; environment github-pages allows `main`.
- Fetch catches each firm up to its newest stored article, so missed runs leave no gap.
- Measured on the first runs: daily ~8 min (fetch ~5-7 min, Polygon's 5 calls/min; embedding ~1 min on a
  cold cache for 41k texts, ~$0.40; events: one LLM call per firm with new groups). Mondays refit all
  firms, ~45 min on 4 cores with 3072-d vectors (job timeout 150 min) — watch the first Monday run.

## Decisions (and why)

**Data: Finnhub company-news + Polygon news, merged, de-duplicated.** GDELT (DOC API and GKG)
was tried at length and discarded: the DOC API matches article bodies but returns only titles
(~80% of results didn't mention the objective); GKG's entity tagger misses ambiguous names
(Apple: 0 of 25 headlines tagged) and tags Facebook/Google as the *venue* of thousands of
unrelated stories. Polygon alone is clean but tiny (NVDA ~90/week) and has 3 publishers (Motley
Fool 57%). Finnhub gives ~1,000/week for NVDA and 36+ for every firm, headline + human summary.
~568 articles per quarter appear in both sources for NVDA → de-dup by normalised headline, keep
the longer summary. Caveat: 78% of Finnhub items arrive as "Yahoo" with the original outlet
hidden; resolving those redirects is the main open data improvement.

**Text embedded: headline + summary (~35 words median), English.** Cleaning: HTML entities
(twice), tags, zero-width chars, space-before-punctuation. Near-duplicate rewrites (cos > .95)
are collapsed after embedding.

**Embedding: google/gemini-embedding-2 via OpenRouter (3072-d, ~$0.20 per million tokens).** Chosen
2026-09-23 by rebuilding NVDA end to end (relevance, story fit, names, map) with three models and
inspecting the maps (site/mockups/NVDA-*.html, embedding-comparison.png):
- thenlper/gte-small (the earlier default, best of 8 local models): 5 broad themes, all on topic, but
  four of them overlap in meaning — only SpaceX separates.
- gemini-embedding-2: 9 stories, all about Nvidia or its orbit, and finer (memory boom vs SK Hynix-style
  volatility; stock outlook vs ecosystem deals). Adopted.
- qwen3-embedding-8b: 11 stories, but 3 were genre (Buffett advice, Vanguard ETFs, dividend picks —
  ~2,000 articles) that passed both lift tests. Rejected.
OpenRouter limits, measured: Gemini takes at most 100 texts per request (Google's cap; larger batches get
HTTP 400); throughput plateaus at ~900 texts/s from ~32 requests in flight (100+ in flight is no faster);
Google returns an occasional 429 under load, which the retries absorb. The key itself has no rate limit.
The near-duplicate cut (cos > .95) is model-specific: under Gemini it collapses fewer rewrites than
under gte-small (NVDA keeps ~575 more articles). Two traps found earlier:
(1) an earlier benchmark ran English-only models on a 96%-non-English corpus — its conclusions
are void; (2) an instruction prefix ("represent this headline for grouping by narrative")
looked like a win on that broken corpus and *hurt* on the clean one.

**Pre-processing of vectors: mean-centring only.** Raw anisotropy is ~0.80 (the uniform summary
register makes everything look alike); centring takes it to ~0. Removing further principal
directions (all-but-the-top, k>0) was tested and lost for every model. (An earlier
implementation forgot to subtract the mean and made anisotropy *rise* — fixed, recorded here.)

**Stories = anchors fitted on the whole window, not clusters per week.** Per-week clustering on
thin windows was unstable (ARI 0.4–0.5) and its K grew with n (no canonical K — Kleinberg's
impossibility showing up empirically). Fitting once on the quarter and soft-assigning each week
gives ARI 0.94 at K=6 for NVDA. Evolution = each story's weekly share, a time series. K per
firm = finest K with resampling ARI ≥ 0.85 (in the committed Gemini fit most firms sit at the floor, K=4,
because their relevant sets are a few hundred articles; NVDA, AAPL, MSFT and LLY 6, META and AVGO 5; plus
21 emerging stories across firms, 111 stories in all). Single linkage was also tried and rejected: it
chains into one giant cluster on real news embeddings (its high "stability" was the
stability of a degenerate partition — stability must never be read without cluster balance).

**Relevance is judged per group, not per article** (the user's principle: one indirect article
is noise, a group of them is signal). Finnhub's NVDA feed names Nvidia in only ~24% of items.
Method (build.py step 3): 8 coarse anchors per firm; keep an anchor only if it passes BOTH tests,
each a ratio of Beta posteriors whose 95% lower bound must exceed 1:
- *name lift*: the anchor names the firm more often than the other 19 feeds' articles do;
- *feed lift*: this kind of content is over-represented in this firm's feed vs the other 19.
Name lift alone kept genre content for rarely-mentioned firms (Boeing's base rate is 0.1%, so
any market wrap clears it). Feed lift alone dropped heavily-covered firms' own core coverage
(Finnhub tags Nvidia news into many feeds). Together: 3-7 of 8 anchors kept per firm, 30-80%
of each feed judged relevant, and peer narratives survive (Micron under Nvidia, Novo Nordisk
under Lilly, Paramount-Warner under Netflix, Chevron under Exxon).
Tried and rejected for this step: a pooled 2-component Gaussian mixture on (lift, name share)
— its "specific" component is set by niche firms at lift 100+, so megacaps kept 1 anchor;
content breadth (effective number of feeds); correlation with firm-neighbourhood similarity
(too noisy with 19 feeds); 16 coarse anchors instead of 8.

**Identity across refits.** Weekly refits match new stories to old ones (Hungarian on centroid
cosine, threshold 0.5); matches keep id/name/colour/history. Only new stories get named.

**Naming.** New stories are named by the OpenRouter model in `OPENROUTER_MODEL` (default
deepseek/deepseek-v4.1-flash since 2026-09-26; Claude Sonnet 5 named the stories of the Gemini refit) from
their 10 most typical headlines; keyword fallback when no key is set. Calls run with reasoning off:
DeepSeek v4.1 Flash reasons by default (~300 hidden tokens even to name one event, ~6x slower) and the
names were as good without it.
Names are kept across refits by the identity matching. (The hand-written names of the gte-small fit were
dropped with the switch to Gemini: the committed state was refit and LLM-named on 2026-09-23.)

**Emerging stories (build.py step 6, every run).** Between Monday refits the fitted stories are fixed, so
each run also looks for new ones: the firm's last 14 days minus what the fitted stories explain (further
from every story than 90% of the firm's relevant articles, and not already claimed by an emerging story)
is grouped by average linkage, cut at the typical fitted story's mean pairwise cosine. A group of 10+
articles over 2+ days becomes a story if it passes name lift, feed lift (against the other feeds' same
14 days) and a novelty test (far more common in those 14 days than in the weeks before). At most 3 live
per firm. An emerging story claims every article within its radius, even one in a dropped coarse anchor.
At the next refit it either matches a fitted story (and keeps its id and name) or is retired to
`state.past`; retired stories that come back get their id and name back. **Crash-tested only** (a
stand-in embedder, all paths: daily, refit, daily after refit): thresholds still need a real-data check.

**Events (build.py step 7, every run).** Stories are themes that run for weeks; events are the dated
happenings inside them (an earnings call, a deal, a listing, a probe). Found on the firm's relevant
articles over the whole window, not per story, because one event (earnings) touches several themes.
Two articles are close when they say the same thing AND appear within days of each other: similarity =
cosine x exp(-days apart / 6). Average-linkage groups (cut 0.30) of 8+ articles are candidates. Plain
semantic clustering (HDBSCAN within a story) was tried first and found no short-lived groups at all —
the uniform summary register dominates; the time factor is what makes events appear. About half the
candidates are recurring commentary ("should you buy SpaceX?", predictions), so one LLM call per 40 new
groups sorts event vs commentary, names events (3-7 words stating the fact) and merges groups that are the
same event as each other or as a known one. Identity: state["events"] keeps every judged group (kept or
rejected) with its article hashes; a group sharing half its articles with a known one takes its verdict,
id and name, so growing events are never re-sent or renamed (a rerun sends 0 groups). Events whose
articles all leave the window are dropped. On 2026-09-26: 253 events across the 20 firms (NVDA 32, AAPL
34, GOOGL 31; TSM, JPM, WMT, BRK.B 1-3, since 8 articles is a lot for a small feed — a size-scaled
minimum is untested). Stored as article hashes, not centroids: a 3072-d centroid per event would add
megabytes to the committed state daily. The site shows this week's events (then the latest before
them), marks each story's events on its chart line (◇) and in its week-by-week timeline.

**Story lifecycle (2026-09-27).** The landscape showed the same stories in every week, for three reasons:
one fit over the whole 13-week window with 4-7 stories per firm gives broad themes that do run all quarter;
every relevant article was forced into its nearest story (argmax), so each story collected stray articles in
every week (TSLA "Q2 Earnings Slump" had 114 articles in the first week of July: Q2 deliveries, but also
"Cathie Wood buys…" at cosine 0.08); and the site redrew all history from today's stories, so a story retired
by a refit vanished from the whole timeline. Now: (1) membership has a cut — an article joins its nearest
story only if at least as close as that story's 25%-quantile member (emerging stories: their radius), else no
story (~25-29% of relevant articles for NVDA/TSLA); (2) it is written once to state["assign"] and never
rewritten, so a retired story keeps its articles and name and is exported marked retired ("ended" on the
site); only articles in no story can be claimed later; (3) a story is alive in a week with 3+ articles and
20%+ of its peak weekly share (share, not count: news volume dips in some weeks, e.g. late August, and a
count rule made every story "die" at once); the map shows a region's title and black dots only then.
(4) At a refit a matched story whose centroid moved (cosine to last week's < 0.8) is renamed from its current
headlines; the old name goes to its "was" list, shown on the site as "formerly …". Tested on TSLA: the refit
renamed "Cybercab Rollout & Scrutiny" (0.79) and "EV Market Q2 2026" (0.75); "Q2 Earnings Slump" is alive
in July then ends; Roadster, Vietnam, Semi are born in September. Not done: replaying the 13 weeks of fits
to reconstruct past lifecycles (hours of compute, and the themes are mostly persistent anyway); history
accumulates from the first weekly refit (2026-09-28). A refit keeps an unmatched emerging story while it is still active (10+ articles in its radius over the
last 14 days, the bar it passed to be born): a K=4 fit had retired TSLA "Semi production" with 21 articles
that week; now Semi and Roadster are kept and the fading "Enters Vietnam" (3) is retired.

**Story relevance (2026-09-27).** NVDA showed "SpaceX Stock Volatility" as a top story: only 5% of its
articles named Nvidia (1 of 38 that week) — Finnhub files "should you buy SpaceX?" pieces under NVDA. The
coarse-anchor test (step 3) had passed a mixed group from which the fit split an off-topic story, never tested
again. Now every story must pass two tests, every run: (1) its own articles name the firm >= 3x as often as the
other feeds do (95% bound; a plain "> 1" was too weak — Nvidia is named in only 2.9% of other feeds, so 5% cleared
it); across all firms this hid 6 of 144 stories (NVDA SpaceX, Vanguard ETFs, Bitcoin; MSFT enterprise-software
valuations; GOOGL quantum; AVGO AI-bubble debate) and kept every genuine neighbour story (NVDA memory 16%,
4.2x); (2) the LLM, asked once per story (new, reborn or renamed), agrees it bears on the firm — about it, its
competitors, suppliers, customers, partners, market, or a company linked by ownership, leadership or a
possible merger (without that clause it hid TSLA's SpaceX story, which names Tesla 70% of the time and covers
merger talk). Its one-line reason is shown under the story on the site when fewer than half its articles
name the firm ("Micron and Sandisk are key HBM suppliers…"). Events are kept if most of their articles are
in shown stories or they pass the name test themselves: requiring a shown story dropped real Nvidia events
("Amkor and Nvidia $1.5B deal"); requiring the name test dropped neighbour events ("AMD crosses $1T").

**The landscape (build.py step 10, site "The landscape").** Every relevant article is a dot; each story
is a circled region with its title above it. A shared 2D map (one UMAP over all articles) was tried first:
under gte-small four of five stories overlapped and their titles collided; supervising UMAP with the story
labels (target_weight 0.05-0.5) collapsed each story into a speck. So the layout is two-level: regions are
placed near their MDS position on story-centroid distance, area by article count, the title's box reserved
above each region; largest first, each takes the nearest free spot, all shrink if one does not fit (the map
grows 80px per story beyond 6). Inside a region, UMAP of its articles, each article's distance from the
centre replaced by its rank so a dense core fills the region. Event labels sit at their articles' median,
moved to the nearest free row when they would cover a title. Written to site/data/map/<TICKER>.json (NVDA
~1.3 MB, ~300 KB gzipped), loaded after the page; not committed (it would grow the history by MBs a day) —
the update job hands the built site to the deploy job as the Pages artifact. The layout is recomputed every run, so regions can move
between days. Known issue: at phone width the 1000px map shrinks until the titles are unreadable.
Snapshots from the design phase (site/mockups/, including the embedding comparison) were made by a
since-removed pipeline/map_mockup.py and are not updated.

**Design (user's explicit spec):** dead simple — white background, black and red text only,
browser-default fonts (Times New Roman, default monospace), no cards/shadows/rounded corners.
Red marks the selected story and positive momentum. Keep it that way.

## Tried and rejected — do not reintroduce without new evidence

- κ (vMF concentration) as "consensus" or as the emerging-story test: κ measures textual
  homogeneity; templated boilerplate scores highest of all (earnings-call summaries κ≈265).
- Emerging-story tests on κ vs nulls: uniform null and random-article null pass everything
  (k-means manufactures tight clusters); a selection-matched null passes nothing.
- Hawkes branching ratio alone: every group self-excites on raw timestamps because publishers
  post in batches. A two-timescale Hawkes (fast = minutes, slow = hours–days) did separate
  templates (fast only) from backdrop (no slow excitation, e.g. macro/Fed) from narratives
  (slow excitation), but hour-scale excitation still mixes in editorial cadence. Promising for a
  "story vitality" measure later; not in the pipeline yet.
- Divisive spectral clustering, Sinkhorn lineage between per-week clusters, SVD discourse
  removal with k=20, instruction prefixes, GDELT in any form, Polygon as the sole source.

## Open questions / next steps

0. **Genre leakage** (known, visible on the site): 8 of 111 stories in the Gemini fit are generic
   content that passed both lift tests because that firm's feed carries more of it than average —
   ETF pieces (NVDA "Vanguard ETF Investing", AAPL, LLY, TSM, BRK.B) and market roundups (CRM "Dow
   Jones Daily Movers", INTC "Tech Stocks Roundup") and one emerging story (NVDA "Bitcoin & Crypto Market
   Trends"). Two more emerging ones appeared on the 2026-09-26 run: GOOGL "Alluvium Capital
   Portfolio" (a fund's holdings) and AMZN "Big Tech Stock Picks". The fit is also sensitive to the window: NVDA
   refit 16 hours apart kept 3 vs 4 of 8 anchors, and only the later fit has the Vanguard story. A third test is needed; untested idea: genre is uniformly
   present in nearly every feed while a theme concentrates in a few.

1. Rank a story's articles by how *typical* they are (current) or how *new* — newness surfaces
   the actual developments (e.g. a specific disclosure) over repeat explainers.
2. Sectors in the dropdown: composite of member firms' articles (agreed design, not built).
3. Resolve Finnhub "Yahoo" redirects to the original outlet → true outlet diversity, and
   cross-outlet Hawkes excitation as a clean contagion measure.
4. Emerging stories: articles far from every anchor, forming a group that starts generating
   its own follow-ups (slow-kernel Hawkes) — replaces the failed κ tests.
5. Phase 2: SSTorytime knowledge graph + RAG chatbot under the chart. Needs a server; out of
   scope for the static site.
6. Countries: parked (news is multilingual; the relevance regexes and names are English-only).

## Conventions

- User prefers minimal code: flat scripts, direct inline computation, few wrapper functions.
- User dislikes AI-sounding prose; explanations should build from intuition before formalism.
- Test method changes on real data (one firm end to end) before adopting them.
