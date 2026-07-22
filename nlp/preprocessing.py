"""
Text preprocessing and spaCy model loading (no Streamlit dependency).
"""

import logging
import os

import pandas as pd

from exceptions import NLPAnalysisError
from nlp.text_utils import as_text_list, normalise_whitespace

logger = logging.getLogger(__name__)

SPACY_MODEL_NAME = os.environ.get("SPACY_MODEL", "uk_core_news_sm")

# Re-export label maps for backward compatibility
POS_LABELS_UA = {
    "NOUN": "іменник",
    "VERB": "дієслово",
    "ADJ": "прикметник",
    "ADV": "прислівник",
    "PROPN": "власна назва",
    "PRON": "займенник",
    "DET": "детермінатив",
    "ADP": "прийменник",
    "CONJ": "союз",
    "CCONJ": "союз",
    "SCONJ": "підрядний союз",
    "PART": "частка",
    "NUM": "числівник",
    "INTJ": "вигук",
    "AUX": "допоміжне дієслово",
    "X": "інше",
    "PUNCT": "пунктуація",
    "SYM": "символ",
}

NER_LABELS_UA = {
    "PER": "Особа",
    "ORG": "Організація",
    "LOC": "Локація",
}


def load_spacy_model(model_name: str | None = None):
    """Load Ukrainian spaCy pipeline (``SPACY_MODEL`` env overrides default)."""
    name = model_name or SPACY_MODEL_NAME
    try:
        import spacy

        return spacy.load(name)
    except OSError as exc:
        logger.exception("spaCy model %s is not installed", name)
        raise NLPAnalysisError(
            f"Модель spaCy '{name}' не встановлена. "
            f"Встановіть: python -m spacy download {name}",
            step="spacy_load",
        ) from exc


def preprocess_texts(texts) -> pd.Series:
    """Remove HTML artefacts and normalise whitespace in titles."""
    try:
        cleaned = pd.Series(as_text_list(texts))
        cleaned = cleaned.str.replace(r"<[^>]+>", " ", regex=True)
        cleaned = cleaned.str.replace("&amp;", " ", regex=False)
        cleaned = cleaned.str.replace("&lt;", " ", regex=False)
        cleaned = cleaned.str.replace("&gt;", " ", regex=False)
        cleaned = cleaned.str.replace("&nbsp;", " ", regex=False)
        cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
        return cleaned.str.strip()
    except (AttributeError, TypeError, ValueError) as exc:
        logger.warning("Preprocessing failed, returning raw text: %s", exc)
        return pd.Series(as_text_list(texts))


__all__ = [
    "NER_LABELS_UA",
    "POS_LABELS_UA",
    "SPACY_MODEL_NAME",
    "load_spacy_model",
    "normalise_whitespace",
    "preprocess_texts",
]
