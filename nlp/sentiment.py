"""
Sentiment and emotion analysis with Ukrainian transformer models.

Public facade over ``nlp.resource_guard``, ``nlp.sentiment_constants``,
``nlp.sentiment_models``, and ``nlp.sentiment_inference``.

Private helpers are not re-exported; import them from their defining modules.
"""

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
)
from nlp.sentiment_inference import (
    classify_emotions,
    classify_emotions_batch,
    classify_sentiment_batch,
    classify_sentiment_cosmus,
)
from nlp.sentiment_models import (
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
    "classify_emotions",
    "classify_emotions_batch",
    "classify_sentiment_batch",
    "classify_sentiment_cosmus",
    "load_cosmus_pipeline",
    "load_emotions_model",
]
