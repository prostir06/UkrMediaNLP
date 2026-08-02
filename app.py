"""
Streamlit UI for Ukrainian news RSS analysis.

Run locally:
    streamlit run app.py
"""

import logging

import streamlit as st

from cache import load_articles
from config import (
    MEDIA_CATEGORIES,
    NLP_FUNCTIONS_FULL,
    NLP_FUNCTIONS_LIGHT,
    get_cloud_light,
    get_source_config,
    source_category,
    sources_for_category,
)
from exceptions import DataLoaderError
from nlp_analysis import preprocess
from ui.corpus_controls import (
    CORPUS_FUNCTIONS,
    load_corpus_into_session,
    render_corpus_sidebar,
)
from ui.features.compare import render_compare_media
from ui.features.corpus_search import render_corpus_search
from ui.features.corpus_trends import render_topic_trends
from ui.features.intro import render_intro
from ui.features.ner import render_ner
from ui.features.ngrams import (
    render_bigrams,
    render_keywords,
    render_trigrams,
    render_unigrams,
    render_wordcloud,
)
from ui.features.pos import render_pos
from ui.features.sentiment_ui import (
    render_sentiment_cosmus,
    render_sentiment_emotions,
    render_sentiment_news,
)
from ui.features.snapshot import render_snapshot
from ui.features.summarization import render_summarization
from ui.features.textstat import render_text_stat
from ui.features.topics import render_topic_modeling
from ui.session_corpus import (
    commit_corpus_load,
    invalidate_stale_corpus,
    load_source,
)
from ui.widgets import sample_size_slider

# Compatibility aliases for tests and older imports.
_sample_size_slider = sample_size_slider
_load_source = load_source
_commit_corpus_load = commit_corpus_load
_invalidate_stale_corpus = invalidate_stale_corpus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(source_name: str, nlp_function: str) -> None:
    if nlp_function == "Пошук у корпусі":
        try:
            render_corpus_search()
        except Exception as exc:
            logger.exception("Corpus search render failed")
            st.error(f"Пошук у корпусі не вдався: {exc}")
        return

    if nlp_function == "Тренди тем":
        try:
            render_topic_trends()
        except Exception as exc:
            logger.exception("Topic trends render failed")
            st.error(f"Не вдалося показати тренди тем: {exc}")
        return

    try:
        config = get_source_config(source_name)
    except KeyError:
        st.error(f"Невідоме джерело: {source_name}")
        return

    if nlp_function == "Вступ":
        try:
            render_intro(config.get("intro", ""))
        except Exception as exc:
            logger.exception("Intro render failed for %s", source_name)
            st.error(f"Не вдалося показати вступ: {exc}")
        return

    if nlp_function == "Порівняння медіа":
        try:
            render_compare_media(source_name)
        except Exception as exc:
            logger.exception("Compare media failed")
            st.error(f"Порівняння не вдалося: {exc}")
        return

    try:
        df = _load_source(source_name)
    except DataLoaderError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        logger.exception("Failed to load articles")
        st.error(f"Не вдалося завантажити статті: {exc}")
        return

    if df.empty:
        st.warning("Статті для обраного джерела не знайдено.")
        return

    titles = preprocess(df["title"])
    content = df["content"].fillna("").astype(str)

    handlers = {
        "Огляд статей": lambda: render_snapshot(df),
        "Уніграми": lambda: render_unigrams(titles),
        "Біграми": lambda: render_bigrams(titles),
        "Триграми": lambda: render_trigrams(titles),
        "Ключові слова": lambda: render_keywords(titles),
        "Хмара слів": lambda: render_wordcloud(titles),
        "Статистика тексту": lambda: render_text_stat(df),
        "Розпізнавання сутностей": lambda: render_ner(df, titles),
        "Частини мови": lambda: render_pos(content),
        "Тематичне моделювання": lambda: render_topic_modeling(content),
        "Тональність (RoBERTa)": lambda: render_sentiment_cosmus(titles),
        "Тональність (Емоції)": lambda: render_sentiment_emotions(titles),
        "Тональність (новини)": lambda: render_sentiment_news(titles),
        "Сумаризація": lambda: render_summarization(df),
    }

    handler = handlers.get(nlp_function)
    if handler is None:
        st.error(f"Невідома функція: {nlp_function}")
        return

    try:
        handler()
    except MemoryError:
        logger.exception("NLP function '%s' OOM", nlp_function)
        st.error(
            "Недостатньо пам'яті для цієї функції. "
            "На Streamlit Cloud free tier спробуйте n-грами, NER або LDA. "
            "Для transformers використовуйте Docker/VPS (2 GB+ RAM)."
        )
    except Exception as exc:
        logger.exception("NLP function '%s' failed", nlp_function)
        st.error(f"Аналіз не вдався: {exc}")


def _select_sidebar_source() -> str | None:
    """
    Render Категорія → Медіа controls.

    Returns the selected media name, or ``None`` when the category is empty
    or Streamlit widgets fail unexpectedly.
    """
    try:
        selected_category = st.sidebar.selectbox("Категорія", MEDIA_CATEGORIES)
        media_options = sources_for_category(selected_category)
    except Exception as exc:
        logger.exception("Sidebar category selection failed")
        st.error(f"Не вдалося побудувати список категорій: {exc}")
        return None

    if not media_options:
        st.sidebar.warning("У цій категорії ще немає медіа.")
        st.info(
            f"Категорія «{selected_category}» поки порожня. "
            "Оберіть іншу категорію зі списку."
        )
        return None

    try:
        return st.sidebar.selectbox("Медіа", media_options)
    except Exception as exc:
        logger.exception("Sidebar media selection failed")
        st.error(f"Не вдалося обрати медіа: {exc}")
        return None


def main() -> None:
    from runtime_env import apply_runtime_env

    try:
        apply_runtime_env()
    except Exception as exc:
        logger.warning("runtime_env apply failed: %s", exc)

    st.set_page_config(
        page_title="UkrMediaNLP",
        page_icon="📰",
        layout="wide",
    )
    st.title("Аналіз новин українських медіа")
    st.caption("Збір новин з RSS, скрейпінг та NLP-аналіз українською мовою.")

    st.sidebar.header("Налаштування")
    selected_source = _select_sidebar_source()
    if not selected_source:
        return

    try:
        nlp_functions = NLP_FUNCTIONS_LIGHT if get_cloud_light() else NLP_FUNCTIONS_FULL
        selected_function = st.sidebar.selectbox("Функція", nlp_functions)
    except Exception as exc:
        logger.exception("NLP function select failed")
        st.error(f"Не вдалося показати функції NLP: {exc}")
        return

    if selected_function in CORPUS_FUNCTIONS:
        category = source_category(selected_source)
        if category is None:
            st.error("Не вдалося визначити категорію для корпусу.")
            return
        try:
            corpus_controls = render_corpus_sidebar(category)
        except Exception as exc:
            logger.exception("Corpus sidebar render failed")
            st.error(f"Не вдалося показати налаштування корпусу: {exc}")
            return

        _invalidate_stale_corpus(
            category=category,
            current_sources=corpus_controls["sources"],
            all_category=corpus_controls["all_category"],
        )

        if corpus_controls["load_clicked"]:
            sources = corpus_controls["sources"]
            if not sources:
                st.warning("Оберіть хоча б одне медіа для корпусу.")
            else:
                try:
                    with st.spinner("Завантаження корпусу..."):
                        corpus_df, warnings = load_corpus_into_session(
                            sources=sources,
                            date_from=corpus_controls["date_from"],
                            date_to=corpus_controls["date_to"],
                            include_missing=corpus_controls["include_missing"],
                            category=category,
                            load_articles_fn=load_articles,
                        )
                    _commit_corpus_load(corpus_df, warnings, list(sources), category)
                except Exception as exc:
                    logger.exception("Corpus load failed")
                    st.error(f"Не вдалося завантажити корпус: {exc}")

    load_data(selected_source, selected_function)


if __name__ == "__main__":
    main()
