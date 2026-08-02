# Sprint 5: CI, Docker, Health

**Goal:** Raise coverage discipline, add full-NLP compose profile, harden spaCy image build, complete scrape sample URLs, fix README counts.

## Tasks
1. `.coveragerc`: cover `ui/*`; keep omit `app.py` / `streamlit_app.py`; CI `--cov-fail-under=65`
2. `docker-compose.yml` profile `full-nlp` (`ALLOW_HEAVY_NLP=1`, `mem_limit: 4g`); Dockerfile spaCy download must fail the build
3. `SCRAPE_SAMPLE_URLS` covers all `NEWS_SOURCES` (+ test)
4. README: test count ~270, coverage 65%, full-nlp docs
