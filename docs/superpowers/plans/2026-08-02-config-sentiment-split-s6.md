# Sprint 6: Config / Sentiment Decomposition

**Goal:** Typed media registry module; sentiment split into ≤~250 LOC files; single HF env source.

## Layout
- `media_sources.py` — TypedDict + NEWS_SOURCES + helpers + SCRAPE_SAMPLE_URLS
- `config.py` — re-exports + NLP function lists / caps / UI strings
- `nlp/resource_guard.py` — HF cache, RAM, quantize
- `nlp/sentiment_models.py` — load COSMUS / emotions
- `nlp/sentiment_inference.py` — classify_* 
- `nlp/sentiment.py` — facade re-exports

## Cleanup
- Remove unused `NLP_FUNCTIONS`
- Drop noop `show_lda_toggle` (LDA already gated in trends via `get_cloud_light`)
- `resource_guard` calls `apply_runtime_env()` instead of duplicating HF setdefaults
