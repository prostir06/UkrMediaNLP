"""
Optional semantic embeddings for corpus search (no Streamlit).

Disabled by default — set ``ALLOW_EMBEDDINGS=1`` to enable. On Streamlit Cloud
keep this off (light deps / RAM). Heavy backends (sentence-transformers) are
loaded lazily only when enabled.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Sequence

from exceptions import NLPAnalysisError

logger = logging.getLogger(__name__)

_DEFAULT_DIM = 64


def embeddings_enabled() -> bool:
    """Return True when semantic embeddings are explicitly allowed."""
    return os.environ.get("ALLOW_EMBEDDINGS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def embed_texts(texts: Sequence[str], *, dim: int = _DEFAULT_DIM) -> list[list[float]]:
    """
    Embed texts into fixed-size vectors.

    Without a heavy model installed, uses a deterministic hashing encoder so
    unit tests and local smoke runs stay dependency-light. Callers must check
    ``embeddings_enabled()`` (or catch ``NLPAnalysisError``).
    """
    if not embeddings_enabled():
        raise NLPAnalysisError(
            "Semantic embeddings вимкнені (задайте ALLOW_EMBEDDINGS=1).",
            step="embeddings",
        )
    if dim <= 0:
        raise NLPAnalysisError("dim must be positive", step="embeddings")

    try:
        return [_hash_embed(str(text or ""), dim) for text in texts]
    except NLPAnalysisError:
        raise
    except Exception as exc:
        logger.exception("embed_texts failed")
        raise NLPAnalysisError(
            f"Не вдалося побудувати embeddings: {exc}",
            step="embeddings",
        ) from exc


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return cosine similarity of two equal-length vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


def _hash_embed(text: str, dim: int) -> list[float]:
    """Deterministic bag-of-token hashing trick (unit / fallback encoder)."""
    vec = [0.0] * dim
    tokens = text.casefold().split()
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = sum(v * v for v in vec) ** 0.5
    if norm:
        vec = [v / norm for v in vec]
    return vec
