# Andreas (Apollo) Koulopoulos

My personal website: https://andreaskoul.github.io/my-website/

Served from `site/` on GitHub Pages.

- `index.html`: home
- `cv.html`: CV
- `thesis.html`: thesis
- `narratives.html`: Narratives, what is being said about a company this week, updated daily from `pipeline/` (see `DESIGN.md`)

`docs/narratives-pipeline.html`: a conceptual flow diagram of the Narratives pipeline (source of the claude.ai artifact https://claude.ai/artifact/Geotc4UYBdAWub2CK3eSAH; not deployed).

A GitHub Actions workflow (`.github/workflows/update.yml`) refreshes the Narratives data and deploys the site. It needs the repo secrets `FINNHUB_API_KEY`, `POLYGON_API_KEY` and `OPENROUTER_API_KEY`.
