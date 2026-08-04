"""
Deprecated NLP façade — prefer ``from nlp.<module> import ...``.

Kept for backward-compatible imports in older scripts/tests. New code must
not import from here. Streamlit-free (no ``ui``).
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
