"""Sentiment/emotion label maps and small text helpers."""

from __future__ import annotations

COSMUS_MODEL = "YShynkarov/ukr-roberta-cosmus-sentiment"


COSMUS_BASE_MODEL = "youscan/ukr-roberta-base"


COSMUS_WEIGHTS_FILE = "ukrroberta_cosmus_sentiment.safetensors"


EMOTIONS_MODEL = "ukr-detect/ukr-emotions-classifier"


MAX_INPUT_CHARS = 2000


SENTIMENT_BATCH_SIZE = 8


COSMUS_ID2LABEL = {
    0: "mixed",
    1: "negative",
    2: "neutral",
    3: "positive",
}


COSMUS_LABEL2ID = {label: idx for idx, label in COSMUS_ID2LABEL.items()}


SENTIMENT_LABELS_UA = {
    "positive": "Позитивна",
    "negative": "Негативна",
    "neutral": "Нейтральна",
    "mixed": "Змішана",
    "LABEL_0": "Змішана",
    "LABEL_1": "Негативна",
    "LABEL_2": "Нейтральна",
    "LABEL_3": "Позитивна",
}


EMOTION_LABELS_UA = {
    "Joy": "Радість",
    "Anger": "Гнів",
    "Fear": "Страх",
    "Disgust": "Огида",
    "Surprise": "Подив",
    "Sadness": "Сум",
    "None": "Без емоцій",
}


EMOTION_THRESHOLDS = {key: 0.5 for key in EMOTION_LABELS_UA}


SENTIMENT_COLORS = {
    "Позитивна": "#2ecc71",
    "Негативна": "#e74c3c",
    "Нейтральна": "#95a5a6",
    "Змішана": "#f39c12",
}


def _truncate(text: str) -> str:
    text = (text or "").strip()
    return text[:MAX_INPUT_CHARS] if len(text) > MAX_INPUT_CHARS else text


def _label_to_ua(label: str) -> str:
    return SENTIMENT_LABELS_UA.get(
        label,
        SENTIMENT_LABELS_UA.get(label.lower(), label),
    )


def format_load_error(model_name: str, exc: Exception) -> str:
    """Build a user-facing message that includes the underlying cause."""
    cause = str(exc).strip() or type(exc).__name__
    hint = ""
    lowered = cause.lower()
    if "transformers" in lowered or "no module named" in lowered:
        hint = " Встановіть: pip install torch transformers"
    elif "out of memory" in lowered or "oom" in lowered:
        hint = " Недостатньо RAM — спробуйте Docker/VPS (2 GB+)."
    elif "permission" in lowered or "lock" in lowered:
        hint = " Помилка прав або блокування кешу Hugging Face (.cache/huggingface). Видаліть застарілі .lock файли або перезапустіть додатку."
    elif "connection" in lowered or "timed out" in lowered or "network" in lowered:
        hint = " Перевірте інтернет і доступ до huggingface.co."
    return f"Не вдалося завантажити модель {model_name}: {cause}.{hint}"


_format_load_error = format_load_error
