"""
NLP analysis facade for the Streamlit UI.

Re-exports pure NLP helpers and UI renderers. Compute modules in ``nlp/*``
do not import Streamlit; plotting lives in ``ui.renderers``.
"""

from nlp.ngrams import get_top_n_bigram, get_top_n_trigram, get_top_n_words
from nlp.preprocessing import NER_LABELS_UA, preprocess_texts
from nlp.sentiment import SENTIMENT_LABELS_UA
from nlp.textstat_ua import aggregate_corpus_metrics, get_textstat_metrics
from nlp.topics import run_topic_modeling
from ui.renderers import (
    display_topic_modeling,
    plot_emotion_distribution,
    plot_most_common_named_entity_barchart,
    plot_parts_of_speech_barchart,
    plot_sentiment_barchart,
    render_wordclouds,
    run_text_summarization,
)

preprocess = preprocess_texts

__all__ = [
    "NER_LABELS_UA",
    "SENTIMENT_LABELS_UA",
    "aggregate_corpus_metrics",
    "display_topic_modeling",
    "get_textstat_metrics",
    "get_top_n_bigram",
    "get_top_n_trigram",
    "get_top_n_words",
    "plot_emotion_distribution",
    "plot_most_common_named_entity_barchart",
    "plot_parts_of_speech_barchart",
    "plot_sentiment_barchart",
    "preprocess",
    "render_wordclouds",
    "run_text_summarization",
    "run_topic_modeling",
]
