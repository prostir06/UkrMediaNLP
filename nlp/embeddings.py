"""
Optional semantic embeddings for corpus search (no Streamlit).

HTML5 / CSS3 / StandardJS do not apply — this module is pure Python (PEP 8).

Design
------
* Disabled by default on Streamlit Cloud (RAM / light deps). Enable with
  ``ALLOW_EMBEDDINGS=1``.
* Public API raises ``NLPAnalysisError`` so UI layers can catch typed failures.
* Default encoder is a deterministic hashing trick (no torch /
  sentence-transformers required for unit tests). A heavier backend can be
  swapped in later behind the same ``embed_texts`` façade.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from collections.abc import Sequence

from exceptions import NLPAnalysisError

logger = logging.getLogger(__name__)

# Default vector width for the hashing encoder. Keep small for tests / smoke.
_DEFAULT_DIM = 64

# Env values accepted as "enabled" (case-insensitive after strip).
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def embeddings_enabled() -> bool:
    """
    Return True when semantic embeddings are explicitly allowed.

    Soft-fails to False if the environment cannot be read (unusual, but
    keeps Cloud imports safe).
    """
    try:
        raw = os.environ.get("ALLOW_EMBEDDINGS", "0")
        return str(raw).strip().lower() in _TRUTHY
    except Exception as exc:  # pragma: no cover - env edge cases
        logger.warning("embeddings_enabled: cannot read ALLOW_EMBEDDINGS: %s", exc)
        return False


def embed_texts(texts: Sequence[str], *, dim: int = _DEFAULT_DIM) -> list[list[float]]:
    """
    Embed texts into fixed-size L2-normalised vectors.

    Args:
        texts: Iterable of strings (None / non-str coerced to empty).
        dim: Output dimensionality; must be a positive integer.

    Returns:
        One float vector per input text (same order).

    Raises:
        NLPAnalysisError: When embeddings are disabled, ``dim`` is invalid,
            ``texts`` is not iterable, or encoding fails unexpectedly.
    """
    if not embeddings_enabled():
        raise NLPAnalysisError(
            "Semantic embeddings вимкнені (задайте ALLOW_EMBEDDINGS=1).",
            step="embeddings",
        )

    try:
        dim_int = int(dim)
    except (TypeError, ValueError) as exc:
        raise NLPAnalysisError(
            f"dim має бути цілим числом: {exc}",
            step="embeddings",
        ) from exc

    if dim_int <= 0:
        raise NLPAnalysisError("dim must be positive", step="embeddings")

    try:
        # Materialise once so a bad iterator fails here with a typed error.
        items = list(texts)
    except TypeError as exc:
        raise NLPAnalysisError(
            f"texts має бути ітерованим: {exc}",
            step="embeddings",
        ) from exc

    try:
        return [_hash_embed(_coerce_text(text), dim_int) for text in items]
    except NLPAnalysisError:
        raise
    except Exception as exc:
        logger.exception("embed_texts failed (n=%s dim=%s)", len(items), dim_int)
        raise NLPAnalysisError(
            f"Не вдалося побудувати embeddings: {exc}",
            step="embeddings",
        ) from exc


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Return cosine similarity of two equal-length vectors.

    Returns 0.0 on shape mismatch, empty inputs, zero norms, or soft failures
    (never raises — callers use this in ranking loops).
    """
    try:
        if len(a) != len(b) or not a:
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b, strict=True):
            fx = float(x)
            fy = float(y)
            dot += fx * fy
            na += fx * fx
            nb += fy * fy
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(dot / (math.sqrt(na) * math.sqrt(nb)))
    except (TypeError, ValueError) as exc:
        logger.debug("cosine_similarity soft-fail: %s", exc)
        return 0.0
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("cosine_similarity unexpected error: %s", exc)
        return 0.0


def _coerce_text(value: object) -> str:
    """Normalise one embedding input to a plain string."""
    try:
        if value is None:
            return ""
        return str(value)
    except Exception as exc:
        logger.debug("_coerce_text failed for %r: %s", value, exc)
        return ""


def _hash_embed(text: str, dim: int) -> list[float]:
    """
    Deterministic bag-of-token hashing encoder (unit / fallback backend).

    Each whitespace token contributes ±1 to a SHA-256-derived bucket; the
    result is L2-normalised so cosine similarity is meaningful.
    """
    try:
        vec = [0.0] * dim
        tokens = text.casefold().split()
        if not tokens:
            return vec
        for token in tokens:
            try:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
            except Exception as exc:
                # Skip pathological tokens rather than failing the whole batch.
                logger.debug("skip token hash for %r: %s", token[:32], exc)
                continue
            idx = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm:
            vec = [v / norm for v in vec]
        return vec
    except NLPAnalysisError:
        raise
    except Exception as exc:
        logger.exception("_hash_embed failed")
        raise NLPAnalysisError(
            f"Хеш-енкодер embeddings не вдався: {exc}",
            step="embeddings",
        ) from exc
