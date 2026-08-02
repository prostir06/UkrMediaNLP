"""Sentiment / emotion inference helpers (no Streamlit)."""

from __future__ import annotations

import logging

from exceptions import NLPAnalysisError
from nlp.sentiment_constants import (
    EMOTION_LABELS_UA,
    EMOTION_THRESHOLDS,
    SENTIMENT_BATCH_SIZE,
    _label_to_ua,
    _truncate,
    format_load_error,
)
from nlp.sentiment_models import (
    COSMUS_MODEL,
    EMOTIONS_MODEL,
    load_cosmus_pipeline,
    load_emotions_model,
)

logger = logging.getLogger(__name__)

_COSMUS_PIPELINE = None
_EMOTIONS_MODEL = None


def _get_cosmus_pipeline():
    """
    Resolve COSMUS via model_registry / Streamlit cache, with process fallback.

    Broad ``except`` is intentional: Streamlit cache may be unavailable outside
    a script run (CLI / unit tests). Failures fall through to a process-local
    singleton so inference can still proceed.
    """
    global _COSMUS_PIPELINE
    try:
        from nlp.model_registry import resolve_cosmus_pipeline

        return resolve_cosmus_pipeline()
    except Exception as exc:
        logger.debug("COSMUS registry/cache unavailable (%s); using process singleton", exc)
        try:
            if _COSMUS_PIPELINE is None:
                _COSMUS_PIPELINE = load_cosmus_pipeline()
            return _COSMUS_PIPELINE
        except NLPAnalysisError:
            raise
        except Exception as load_exc:
            raise NLPAnalysisError(
                format_load_error(COSMUS_MODEL, load_exc),
                step="cosmus_load",
            ) from load_exc


def _get_emotions_model():
    """
    Resolve emotions model via model_registry / Streamlit cache, with fallback.

    Same rationale as ``_get_cosmus_pipeline``: outside Streamlit the cache
    import path often fails; keep a process-local singleton for tests/CLI.
    """
    global _EMOTIONS_MODEL
    try:
        from nlp.model_registry import resolve_emotions_model

        return resolve_emotions_model()
    except Exception as exc:
        logger.debug("Emotions registry/cache unavailable (%s); using process singleton", exc)
        try:
            if _EMOTIONS_MODEL is None:
                _EMOTIONS_MODEL = load_emotions_model()
            return _EMOTIONS_MODEL
        except NLPAnalysisError:
            raise
        except Exception as load_exc:
            raise NLPAnalysisError(
                format_load_error(EMOTIONS_MODEL, load_exc),
                step="emotions_load",
            ) from load_exc


def classify_sentiment_cosmus(text: str) -> str:
    """Single-headline COSMUS label with soft fallback for callers/tests."""
    try:
        results = classify_sentiment_batch([text])
        return results[0] if results else "Нейтральна"
    except Exception as exc:
        logger.warning("COSMUS single-text classification failed: %s", exc)
        return "Нейтральна"


def classify_sentiment_batch(texts: list[str]) -> list[str]:
    if texts is None or len(texts) == 0:
        return []
    try:
        classifier = _get_cosmus_pipeline()
        truncated = [_truncate(text) for text in texts]
        outputs = classifier(truncated, batch_size=SENTIMENT_BATCH_SIZE)
        return [_label_to_ua(item["label"]) for item in outputs]
    except NLPAnalysisError:
        raise
    except Exception as exc:
        logger.warning("Batch sentiment failed: %s", exc)
        raise NLPAnalysisError(
            f"Пакетний аналіз тональності не вдався: {exc}",
            step="classify_sentiment_batch",
        ) from exc


def _probs_to_emotions(probs: list[float], id2label: dict) -> tuple[list[str], str]:
    labels = [id2label[i] for i in range(len(probs))]
    detected = [
        EMOTION_LABELS_UA.get(label, label)
        for label, prob in zip(labels, probs)
        if prob >= EMOTION_THRESHOLDS.get(label, 0.5)
    ]
    if not detected:
        detected = [EMOTION_LABELS_UA.get("None", "Без емоцій")]

    best_label, best_prob = max(zip(labels, probs), key=lambda item: item[1])
    if best_prob < EMOTION_THRESHOLDS.get(best_label, 0.5):
        dominant = EMOTION_LABELS_UA.get("None", "Без емоцій")
    else:
        dominant = EMOTION_LABELS_UA.get(best_label, best_label)
    return detected, dominant


def classify_emotions_batch(texts: list[str]) -> list[tuple[list[str], str]]:
    """Batched emotion inference — one forward pass per batch."""
    if texts is None or len(texts) == 0:
        return []

    try:
        tokenizer, model, torch = _get_emotions_model()
    except NLPAnalysisError:
        raise
    except Exception as exc:
        raise NLPAnalysisError(
            format_load_error(EMOTIONS_MODEL, exc),
            step="emotions_load",
        ) from exc

    truncated = [_truncate(text) for text in texts]
    results: list[tuple[list[str], str]] = []

    try:
        for start in range(0, len(truncated), SENTIMENT_BATCH_SIZE):
            batch = truncated[start : start + SENTIMENT_BATCH_SIZE]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            with torch.no_grad():
                probs_batch = torch.sigmoid(model(**inputs).logits).tolist()

            id2label = model.config.id2label
            for probs in probs_batch:
                # HF may store id2label keys as strings — normalise to int index.
                normalised = {
                    int(key) if not isinstance(key, int) and str(key).isdigit() else key: value
                    for key, value in id2label.items()
                }
                results.append(_probs_to_emotions(probs, normalised))
    except NLPAnalysisError:
        raise
    except Exception as exc:
        logger.exception("Emotion batch inference failed")
        raise NLPAnalysisError(
            f"Інференс емоцій не вдався: {exc}",
            step="emotions_infer",
        ) from exc
    return results


def _run_emotion_inference(text: str) -> tuple[list[str], str]:
    """Single-text emotion inference via the batched path."""
    return classify_emotions_batch([text])[0]


def classify_emotions(text: str) -> list[str]:
    try:
        detected, _ = _run_emotion_inference(text)
        return detected
    except Exception as exc:
        # Soft-fail for single-text callers; UI batch path surfaces NLPAnalysisError.
        logger.warning("Emotion inference failed: %s", exc)
        return [EMOTION_LABELS_UA.get("None", "Без емоцій")]


def _dominant_emotion(text: str) -> str:
    try:
        _, dominant = _run_emotion_inference(text)
        return dominant
    except Exception as exc:
        logger.warning("Dominant emotion inference failed: %s", exc)
        return EMOTION_LABELS_UA.get("None", "Без емоцій")

