"""HF cache, RAM checks, and optional CPU quantization for transformers."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from exceptions import NLPAnalysisError
from runtime_env import apply_runtime_env

logger = logging.getLogger(__name__)

# Single source of truth for HF / torch process defaults.
apply_runtime_env()


def ensure_writable_hf_cache() -> None:
    """Ensure Hugging Face cache directory is writable, falling back to temp."""
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
            "HF cache directory %s is not writable (%s); switching HF_HOME to temp",
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


ensure_writable_hf_cache()


def available_ram_mb() -> int | None:
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


def require_ram_for_transformers(step: str, min_mb: int | None = None) -> None:
    """Refuse to load heavy models when free RAM is too low."""
    threshold = min_mb
    if threshold is None:
        try:
            threshold = int(os.environ.get("MIN_TRANSFORMERS_RAM_MB", "1536"))
        except ValueError:
            threshold = 1536

    available = available_ram_mb()
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


def quantize_model_if_cpu(model):
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


def maybe_quantize(model):
    """Apply CPU INT8 quantization only when QUANTIZE_CPU is explicitly enabled."""
    if os.environ.get("QUANTIZE_CPU", "0").lower() in {"1", "true", "yes"}:
        return quantize_model_if_cpu(model)
    return model


# Compatibility aliases used by older tests / call sites.
_ensure_writable_hf_cache = ensure_writable_hf_cache
_available_ram_mb = available_ram_mb
_require_ram_for_transformers = require_ram_for_transformers
_maybe_quantize = maybe_quantize
_quantize_model_if_cpu = quantize_model_if_cpu
