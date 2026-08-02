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
import tempfile
from pathlib import Path

from exceptions import NLPAnalysisError

# Disable Hugging Face Hub file locking to prevent PermissionError in shared/containerized environments.
# Disable Xet transfers — without hf_xet they warn and often stall for many minutes on first download.
os.environ.setdefault("HF_HUB_DISABLE_DISK_LOCK", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
logger = logging.getLogger(__name__)


def _ensure_writable_hf_cache() -> None:
    """Ensure Hugging Face cache directory is writable, falling back to /tmp/hf_cache if needed."""
    current_hf_home = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
    if not current_hf_home:
        current_hf_home = str(Path.home() / ".cache" / "huggingface")

    try:
        path = Path(current_hf_home)
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / f".write_test_{os.getpid()}"
        test_file.touch()
        test_file.unlink()
    except (OSError, PermissionError) as exc:
        logger.warning(
            "HF cache directory %s is not writable (%s); switching HF_HOME to /tmp/hf_cache",
            current_hf_home,
            exc,
        )
        fallback = Path(tempfile.gettempdir()) / "hf_cache"
        try:
            fallback.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(fallback)
            os.environ["TRANSFORMERS_CACHE"] = str(fallback)
            os.environ["HF_HUB_CACHE"] = str(fallback)
        except Exception as fallback_exc:
            logger.warning("Failed setting fallback HF cache directory: %s", fallback_exc)


_ensure_writable_hf_cache()



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
    _ensure_writable_hf_cache()
    _require_ram_for_transformers("cosmus_load")
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise NLPAnalysisError(
            _format_load_error(COSMUS_MODEL, exc),
            step="cosmus_load",
        ) from exc

    try:
        pipe = pipeline(
            "text-classification",
            model=COSMUS_MODEL,
            truncation=True,
            max_length=512,
        )
        if hasattr(pipe, "model"):
            pipe.model = _maybe_quantize(pipe.model)
        return pipe
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

    try:
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
    except (PermissionError, OSError) as exc:
        logger.warning(
            "PermissionError loading COSMUS model (%s); retrying with temp HF cache",
            exc,
        )
        tmp_dir = str(Path(tempfile.gettempdir()) / "hf_cache")
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = tmp_dir
        os.environ["TRANSFORMERS_CACHE"] = tmp_dir
        os.environ["HF_HUB_CACHE"] = tmp_dir
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

    model = _maybe_quantize(model)
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


def _available_ram_mb() -> int | None:
    """Best-effort free physical RAM in MiB (Windows / Linux)."""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys // (1024 * 1024))
    except Exception as exc:
        logger.debug("Windows RAM probe failed: %s", exc)

    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception as exc:
        logger.debug("Linux RAM probe failed: %s", exc)
    return None


def _require_ram_for_transformers(step: str, min_mb: int | None = None) -> None:
    """
    Refuse to load heavy models when free RAM is too low.

    Prevents the common failure mode where torch allocates until the OS kills
    the Streamlit process (browser then shows WebSocket / health ERR_EMPTY_RESPONSE).
    """
    threshold = min_mb
    if threshold is None:
        try:
            threshold = int(os.environ.get("MIN_TRANSFORMERS_RAM_MB", "1536"))
        except ValueError:
            threshold = 1536

    available = _available_ram_mb()
    if available is None:
        logger.debug("RAM check skipped (unavailable)")
        return
    if available < threshold:
        raise NLPAnalysisError(
            f"Замало вільної RAM для transformers-моделі "
            f"(~{available} МБ вільно, потрібно ≥{threshold} МБ). "
            f"Закрийте інші програми або використайте «Тональність (новини)». "
            f"Якщо сторінка «відвалилась» — перезапустіть: streamlit run streamlit_app.py",
            step=step,
        )


def _maybe_quantize(model):
    """Apply CPU INT8 quantization only when QUANTIZE_CPU is explicitly enabled."""
    if os.environ.get("QUANTIZE_CPU", "0").lower() in {"1", "true", "yes"}:
        return _quantize_model_if_cpu(model)
    return model


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
    """
    Load the Ukrainian multi-label emotions classifier.

    First call downloads weights from Hugging Face (often 1–3 minutes).
    Uses a writable HF cache and optional CPU INT8 quantization.
    """
    _ensure_writable_hf_cache()
    _require_ram_for_transformers("emotions_load")
    try:
        import gc

        gc.collect()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Limit thread oversubscription — reduces hangs on Windows / small VMs.
        try:
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        except Exception as exc:
            logger.debug("torch.set_num_threads skipped: %s", exc)

        load_kwargs = {"low_cpu_mem_usage": True}

        def _load_pair():
            tok = AutoTokenizer.from_pretrained(EMOTIONS_MODEL)
            try:
                mdl = AutoModelForSequenceClassification.from_pretrained(
                    EMOTIONS_MODEL,
                    **load_kwargs,
                )
            except Exception as load_exc:
                logger.debug("low_cpu_mem_usage load failed (%s); retrying plain", load_exc)
                mdl = AutoModelForSequenceClassification.from_pretrained(EMOTIONS_MODEL)
            return tok, mdl

        try:
            tokenizer, model = _load_pair()
        except (PermissionError, OSError) as exc:
            logger.warning(
                "PermissionError loading emotions model (%s); retrying with temp HF cache",
                exc,
            )
            tmp_dir = str(Path(tempfile.gettempdir()) / "hf_cache")
            Path(tmp_dir).mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = tmp_dir
            os.environ["TRANSFORMERS_CACHE"] = tmp_dir
            os.environ["HF_HUB_CACHE"] = tmp_dir
            tokenizer, model = _load_pair()

        # Quantization is off by default (QUANTIZE_CPU=0): it adds latency and can
        # hard-crash Streamlit on some Windows / CPU torch builds.
        model = _maybe_quantize(model)
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
                _format_load_error(COSMUS_MODEL, load_exc),
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
                _format_load_error(EMOTIONS_MODEL, load_exc),
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
            _format_load_error(EMOTIONS_MODEL, exc),
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
