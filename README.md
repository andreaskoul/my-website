# Andreas Koulopoulos

My personal website, served from `site/` on GitHub Pages.

- `index.html`: home
- `cv.html`: CV
- `thesis.html`: thesis
- `narratives.html`: Narratives, what is being said about a company this week, updated daily from `pipeline/` (see `DESIGN.md`)

A GitHub Actions workflow (`.github/workflows/update.yml`) refreshes the Narratives data and deploys the site. It needs the repo secrets `FINNHUB_API_KEY`, `POLYGON_API_KEY` and `OPENROUTER_API_KEY`.
