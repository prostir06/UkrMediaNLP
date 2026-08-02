"""
Sentiment and emotion analysis with Ukrainian transformer models.

Facade over ``nlp.resource_guard``, ``nlp.sentiment_constants``,
``nlp.sentiment_models``, and ``nlp.sentiment_inference`` for stable imports.
"""

from nlp.resource_guard import (
    _available_ram_mb,
    _ensure_writable_hf_cache,
    _maybe_quantize,
    _quantize_model_if_cpu,
    _require_ram_for_transformers,
)
from nlp.sentiment_constants import (
    COSMUS_BASE_MODEL,
    COSMUS_ID2LABEL,
    COSMUS_LABEL2ID,
    COSMUS_MODEL,
    COSMUS_WEIGHTS_FILE,
    EMOTION_LABELS_UA,
    EMOTION_THRESHOLDS,
    EMOTIONS_MODEL,
    MAX_INPUT_CHARS,
    SENTIMENT_BATCH_SIZE,
    SENTIMENT_COLORS,
    SENTIMENT_LABELS_UA,
    _format_load_error,
    _label_to_ua,
    _truncate,
)
from nlp.sentiment_inference import (
    _dominant_emotion,
    _get_cosmus_pipeline,
    _get_emotions_model,
    _probs_to_emotions,
    _run_emotion_inference,
    classify_emotions,
    classify_emotions_batch,
    classify_sentiment_batch,
    classify_sentiment_cosmus,
)
from nlp.sentiment_models import (
    _remap_state_dict_keys,
    load_cosmus_pipeline,
    load_emotions_model,
)

__all__ = [
    "COSMUS_BASE_MODEL",
    "COSMUS_ID2LABEL",
    "COSMUS_LABEL2ID",
    "COSMUS_MODEL",
    "COSMUS_WEIGHTS_FILE",
    "EMOTION_LABELS_UA",
    "EMOTION_THRESHOLDS",
    "EMOTIONS_MODEL",
    "MAX_INPUT_CHARS",
    "SENTIMENT_BATCH_SIZE",
    "SENTIMENT_COLORS",
    "SENTIMENT_LABELS_UA",
    "_available_ram_mb",
    "_dominant_emotion",
    "_ensure_writable_hf_cache",
    "_format_load_error",
    "_get_cosmus_pipeline",
    "_get_emotions_model",
    "_label_to_ua",
    "_maybe_quantize",
    "_probs_to_emotions",
    "_quantize_model_if_cpu",
    "_remap_state_dict_keys",
    "_require_ram_for_transformers",
    "_run_emotion_inference",
    "_truncate",
    "classify_emotions",
    "classify_emotions_batch",
    "classify_sentiment_batch",
    "classify_sentiment_cosmus",
    "load_cosmus_pipeline",
    "load_emotions_model",
]
