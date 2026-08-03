#!/bin/sh
set -e

if [ "${PRELOAD_MODELS}" = "true" ] || [ "${PRELOAD_MODELS}" = "1" ]; then
  echo "Preloading spaCy and transformer models..."
  python - <<'PY'
from nlp.preprocessing import load_spacy_model
from nlp.sentiment import load_cosmus_pipeline, load_emotions_model

load_spacy_model()
try:
    load_cosmus_pipeline()
except Exception as exc:
    print("COSMUS preload skipped:", exc)
try:
    load_emotions_model()
except Exception as exc:
    print("Emotions preload skipped:", exc)
print("Preload done.")
PY
fi

# Durable corpus: fail fast when DATABASE_URL is set but migrations cannot run.
if [ -n "${DATABASE_URL:-}" ]; then
  echo "Running alembic upgrade head..."
  python -m alembic upgrade head
fi

exec "$@"
