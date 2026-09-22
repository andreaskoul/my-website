# Narratives

What is being said about a company this week, and how each story got here.

Pick a company, see its dominant news stories for the past seven days with the articles behind
them, and follow any story week by week over the last quarter. Updated daily, automatically.

**How it works.** Company news from Finnhub and Polygon is de-duplicated and embedded. Articles
that mean the same thing are grouped into stories, keeping only groups that discuss the company
far more often than chance. Each story's share of coverage is tracked week by week, and its most
typical headline each week tells the story in its own words.

**Running it.** A GitHub Actions workflow (`.github/workflows/update.yml`) fetches, rebuilds and
deploys to GitHub Pages every day at 06:15 UTC and refits the stories on Mondays. Needs repo
secrets `FINNHUB_API_KEY` and `POLYGON_API_KEY` (free tiers); `OPENROUTER_API_KEY` is optional
and only used to name new stories. See `DESIGN.md` for the full design record.
