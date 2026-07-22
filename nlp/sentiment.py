"""
Sentiment and emotion analysis with Ukrainian transformer models.

COSMUS note
-----------
``YShynkarov/ukr-roberta-cosmus-sentiment`` on Hugging Face ships only a
custom-named ``.safetensors`` weight file (no ``config.json`` / tokenizer).
We therefore load the tokenizer and architecture from
``youscan/ukr-roberta-base`` and overlay the fine-tuned COSMUS weights.
"""

import logging
import os
from collections import Counter

# Disable Hugging Face Hub file locking to prevent PermissionError in shared/containerized environments.
os.environ.setdefault("HF_HUB_DISABLE_DISK_LOCK", "1")

import matplotlib.pyplot as plt

from exceptions import NLPAnalysisError

logger = logging.getLogger(__name__)


COSMUS_MODEL = "YShynkarov/ukr-roberta-cosmus-sentiment"
COSMUS_BASE_MODEL = "youscan/ukr-roberta-base"
COSMUS_WEIGHTS_FILE = "ukrroberta_cosmus_sentiment.safetensors"
EMOTIONS_MODEL = "ukr-detect/ukr-emotions-classifier"
MAX_INPUT_CHARS = 2000
SENTIMENT_BATCH_SIZE = 8

# Label order from sklearn LabelEncoder (alphabetical) used in COSMUS training.
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


def _format_load_error(model_name: str, exc: Exception) -> str:
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


def load_cosmus_pipeline():
    """
    Load COSMUS sentiment as a transformers text-classification pipeline.

    The HF repo currently lacks ``config.json`` / tokenizer files, so we load
    ``youscan/ukr-roberta-base`` and overlay the published safetensors weights.
    If the repo is completed later, the direct ``pipeline(model=...)`` path is tried first.
    """
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise NLPAnalysisError(
            _format_load_error(COSMUS_MODEL, exc),
            step="cosmus_load",
        ) from exc

    try:
        return pipeline(
            "text-classification",
            model=COSMUS_MODEL,
            truncation=True,
            max_length=512,
        )
    except Exception as direct_exc:
        logger.info(
            "Direct COSMUS load failed (%s); loading base + weights",
            direct_exc,
        )
        try:
            return _build_cosmus_pipeline_from_weights()
        except Exception as overlay_exc:
            raise NLPAnalysisError(
                _format_load_error(COSMUS_MODEL, overlay_exc),
                step="cosmus_load",
            ) from overlay_exc


def _build_cosmus_pipeline_from_weights():
    """
    Build a pipeline from ``youscan/ukr-roberta-base`` + COSMUS safetensors.

    The published HF model card only contains
    ``ukrroberta_cosmus_sentiment.safetensors`` (no config/tokenizer).
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline,
    )

    tokenizer = AutoTokenizer.from_pretrained(COSMUS_BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        COSMUS_BASE_MODEL,
        num_labels=len(COSMUS_ID2LABEL),
        id2label=COSMUS_ID2LABEL,
        label2id=COSMUS_LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    weights_path = hf_hub_download(
        repo_id=COSMUS_MODEL,
        filename=COSMUS_WEIGHTS_FILE,
    )
    state_dict = load_file(weights_path)

    # Remap common key prefixes if the checkpoint was saved without HF layout.
    remapped = _remap_state_dict_keys(state_dict, model.state_dict().keys())
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        logger.warning("COSMUS weights missing keys: %s", missing[:8])
    if unexpected:
        logger.warning("COSMUS weights unexpected keys: %s", unexpected[:8])

    model.eval()
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=512,
    )


def _remap_state_dict_keys(state_dict: dict, model_keys) -> dict:
    """
    Align checkpoint key names with the Hugging Face model layout.

    Some training exports omit the ``roberta.`` / ``classifier.`` prefixes
    or wrap weights under ``model.``.
    """
    model_key_set = set(model_keys)
    if set(state_dict.keys()) <= model_key_set or set(state_dict.keys()) & model_key_set:
        # Already compatible enough for strict=False load.
        if any(key in model_key_set for key in state_dict):
            return state_dict

    candidates = []
    for prefix in ("", "model.", "roberta."):
        remapped = {
            (key if key.startswith(prefix) or not prefix else f"{prefix}{key}"): value
            for key, value in state_dict.items()
        }
        # Also try stripping a leading "module." from DataParallel exports.
        remapped = {
            key.removeprefix("module."): value for key, value in remapped.items()
        }
        overlap = len(set(remapped) & model_key_set)
        candidates.append((overlap, remapped))

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_overlap, best = candidates[0]
    if best_overlap == 0:
        logger.warning("Could not remap COSMUS weight keys; loading as-is")
        return state_dict
    return best


def _quantize_model_if_cpu(model):
    """Apply dynamic INT8 quantization on Linear layers for CPU inference."""
    try:
        import torch

        if hasattr(torch, "ao") and hasattr(torch.ao, "quantization"):
            return torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        if hasattr(torch, "quantization"):
            return torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
    except Exception as exc:
        logger.debug("CPU dynamic quantization skipped: %s", exc)
    return model


def load_emotions_model():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(EMOTIONS_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(EMOTIONS_MODEL)
        model = _quantize_model_if_cpu(model)
        model.eval()
        return tokenizer, model, torch
    except Exception as exc:
        raise NLPAnalysisError(
            _format_load_error(EMOTIONS_MODEL, exc),
            step="emotions_load",
        ) from exc



_COSMUS_PIPELINE = None
_EMOTIONS_MODEL = None


def _get_cosmus_pipeline():
    global _COSMUS_PIPELINE
    try:
        from nlp.model_registry import resolve_cosmus_pipeline

        return resolve_cosmus_pipeline()
    except Exception:
        if _COSMUS_PIPELINE is None:
            _COSMUS_PIPELINE = load_cosmus_pipeline()
        return _COSMUS_PIPELINE


def _get_emotions_model():
    global _EMOTIONS_MODEL
    try:
        from nlp.model_registry import resolve_emotions_model

        return resolve_emotions_model()
    except Exception:
        if _EMOTIONS_MODEL is None:
            _EMOTIONS_MODEL = load_emotions_model()
        return _EMOTIONS_MODEL


def classify_sentiment_cosmus(text: str) -> str:
    results = classify_sentiment_batch([text])
    return results[0] if results else "Нейтральна"


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
        return ["Нейтральна"] * len(texts)


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


    tokenizer, model, torch = _get_emotions_model()
    truncated = [_truncate(text) for text in texts]
    results: list[tuple[list[str], str]] = []

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
            results.append(_probs_to_emotions(probs, id2label))
    return results


def _run_emotion_inference(text: str) -> tuple[list[str], str]:
    """Single-text emotion inference via the batched path."""
    return classify_emotions_batch([text])[0]


def classify_emotions(text: str) -> list[str]:
    try:
        detected, _ = _run_emotion_inference(text)
        return detected
    except NLPAnalysisError:
        raise
    except Exception as exc:
        logger.warning("Emotion inference failed: %s", exc)
        return [EMOTION_LABELS_UA.get("None", "Без емоцій")]


def _dominant_emotion(text: str) -> str:
    try:
        _, dominant = _run_emotion_inference(text)
        return dominant
    except NLPAnalysisError:
        raise
    except Exception as exc:
        logger.warning("Dominant emotion inference failed: %s", exc)
        return EMOTION_LABELS_UA.get("None", "Без емоцій")


def build_sentiment_figure(texts, method: str = "cosmus"):
    if method == "cosmus":
        labels_list = classify_sentiment_batch(list(texts))
    elif method == "emotions":
        batch = classify_emotions_batch([str(t) for t in texts])
        labels_list = [dominant for _, dominant in batch]
    elif method == "news_rules":
        from nlp.news_sentiment import classify_news_sentiment_batch

        labels_list = classify_news_sentiment_batch([str(t) for t in texts])
    else:
        raise ValueError(f"Unknown sentiment method: {method}")

    import pandas as pd

    counts = pd.Series(labels_list).value_counts()
    if counts.empty:
        return None

    colors = [SENTIMENT_COLORS.get(label, "#3498db") for label in counts.index]
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    ax.set_ylabel("Кількість")
    ax.tick_params(axis="x", rotation=25)
    return fig


def build_emotion_figure(texts):
    counter: Counter[str] = Counter()
    try:
        batch = classify_emotions_batch([str(t) for t in texts])
        for detected, _ in batch:
            counter.update(detected)
    except NLPAnalysisError:
        raise
    except Exception as exc:
        logger.warning("Emotion batch failed: %s", exc)
        return None

    if not counter:
        return None

    labels, counts = zip(*counter.most_common())
    fig, ax = plt.subplots()
    ax.barh(labels, counts, color="#8e44ad", edgecolor="white")
    ax.set_xlabel("Кількість згадувань")
    return fig
