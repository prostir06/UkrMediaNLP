"""
NLP analysis facade — pure compute re-exports only (no Streamlit / UI).

UI plotting lives in ``ui.renderers``; feature screens import renderers
directly. Prefer ``from nlp.<module> import ...`` for new code.
"""

from nlp.keywords import extract_keywords
from nlp.ngrams import get_top_n_bigram, get_top_n_trigram, get_top_n_words
from nlp.preprocessing import preprocess_texts
from nlp.textstat_ua import aggregate_corpus_metrics, get_textstat_metrics

preprocess = preprocess_texts

__all__ = [
    "aggregate_corpus_metrics",
    "extract_keywords",
    "get_textstat_metrics",
    "get_top_n_bigram",
    "get_top_n_trigram",
    "get_top_n_words",
    "preprocess",
]
