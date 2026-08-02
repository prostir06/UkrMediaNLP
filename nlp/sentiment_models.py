"""Load COSMUS and emotions transformer models (no Streamlit)."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from exceptions import NLPAnalysisError
from nlp.resource_guard import (
    ensure_writable_hf_cache,
    maybe_quantize,
    require_ram_for_transformers,
)
from nlp.sentiment_constants import (
    COSMUS_BASE_MODEL,
    COSMUS_ID2LABEL,
    COSMUS_LABEL2ID,
    COSMUS_MODEL,
    COSMUS_WEIGHTS_FILE,
    EMOTIONS_MODEL,
    format_load_error,
)

logger = logging.getLogger(__name__)


def load_cosmus_pipeline():
    """
    Load COSMUS sentiment as a transformers text-classification pipeline.

    The HF repo currently lacks ``config.json`` / tokenizer files, so we load
    ``youscan/ukr-roberta-base`` and overlay the published safetensors weights.
    If the repo is completed later, the direct ``pipeline(model=...)`` path is tried first.
    """
    ensure_writable_hf_cache()
    require_ram_for_transformers("cosmus_load")
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
            pipe.model = maybe_quantize(pipe.model)
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

    model = maybe_quantize(model)
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


def load_emotions_model():
    """
    Load the Ukrainian multi-label emotions classifier.

    First call downloads weights from Hugging Face (often 1–3 minutes).
    Uses a writable HF cache and optional CPU INT8 quantization.
    """
    ensure_writable_hf_cache()
    require_ram_for_transformers("emotions_load")
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
        model = maybe_quantize(model)
        model.eval()
        return tokenizer, model, torch
    except Exception as exc:
        raise NLPAnalysisError(
            _format_load_error(EMOTIONS_MODEL, exc),
            step="emotions_load",
        ) from exc


_COSMUS_PIPELINE = None


_EMOTIONS_MODEL = None


_format_load_error = format_load_error
