"""
Streamlit UI for Ukrainian news RSS analysis.

Run locally:
    streamlit run app.py
"""

import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from cache import load_articles
from config import (
    MAX_ARTICLES,
    MAX_POS_ARTICLES,
    MAX_SENTIMENT_TITLES,
    MAX_SUMMARY_ARTICLES,
    NEWS_SOURCES,
    NGRAM_DESCRIPTION,
    NLP_FUNCTIONS,
    WORDCLOUD_DESCRIPTION,
    get_cloud_light,
)
from exceptions import DataLoaderError
from nlp.news_sentiment import classify_news_sentiment_batch
from nlp.preprocessing import NER_LABELS_UA
from nlp_analysis import (
    aggregate_corpus_metrics,
    display_topic_modeling,
    extract_keywords,
    get_textstat_metrics,
    get_top_n_bigram,
    get_top_n_trigram,
    get_top_n_words,
    plot_emotion_distribution,
    plot_most_common_named_entity_barchart,
    plot_parts_of_speech_barchart,
    plot_sentiment_barchart,
    preprocess,
    render_wordclouds,
    run_text_summarization,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

POS_DESCRIPTIONS = [
    ("NOUN", "іменник", "Київ, уряд, закон"),
    ("VERB", "дієслово", "заявив, ухвалив, працює"),
    ("ADJ", "прикметник", "новий, важливий, економічний"),
    ("ADV", "прислівник", "швидко, сьогодні, дуже"),
    ("PROPN", "власна назва", "Україна, Зеленський, NATO"),
    ("ADP", "прийменник", "у, на, для, про"),
    ("CCONJ", "союз", "і, але, або"),
]


def _sample_size_slider(label: str, default: int, max_value: int, key: str) -> int:
    """
    Clamp and render a Streamlit sample-size slider.

    Guards against empty corpora (``max_value < 1``) and invalid defaults so
    Streamlit never receives an out-of-range ``value``.
    """
    try:
        upper = max(1, int(max_value))
        value = min(max(1, int(default)), upper)
        return int(st.slider(label, min_value=1, max_value=upper, value=value, key=key))
    except (TypeError, ValueError) as exc:
        logger.warning("Sample size slider fallback (%s): %s", key, exc)
        return max(1, int(default) if str(default).isdigit() else 1)
    except Exception as exc:
        logger.exception("Unexpected slider failure for key=%s", key)
        raise RuntimeError(f"Не вдалося показати слайдер: {exc}") from exc


def _load_source(source_name: str) -> pd.DataFrame:
    """
    Load articles with a live scrape progress bar (N/M).

    Progress is only meaningful on cache miss; SQLite hits return immediately.
    The bar is always cleared in ``finally`` so a failed load does not leave
    a stuck widget on the page.
    """
    progress = None
    try:
        progress = st.progress(0, text="Завантаження статей...")

        def on_progress(done: int, total: int) -> None:
            """Update the Streamlit progress bar; ignore callback UI errors."""
            if total <= 0 or progress is None:
                return
            try:
                progress.progress(
                    min(done / total, 1.0),
                    text=f"Скрейпінг статей: {done}/{total}",
                )
            except Exception as exc:
                logger.debug("Progress UI update skipped: %s", exc)

        return load_articles(source_name, progress_callback=on_progress)
    except DataLoaderError:
        raise
    except Exception as exc:
        logger.exception("Failed to load source %s", source_name)
        raise DataLoaderError(
            f"Не вдалося завантажити статті: {exc}",
            source_name=source_name,
        ) from exc
    finally:
        if progress is not None:
            try:
                progress.empty()
            except Exception as exc:
                logger.debug("Progress cleanup failed: %s", exc)


def render_intro(intro_text: str) -> None:
    st.markdown(intro_text)


def render_snapshot(df: pd.DataFrame) -> None:
    st.subheader("Таблиця статей")
    total_in_feed = df.attrs.get("total_in_feed", len(df))
    if total_in_feed > MAX_ARTICLES:
        st.caption(
            f"Завантажено {len(df)} з {total_in_feed} статей RSS "
            f"(обмеження MAX_ARTICLES={MAX_ARTICLES})."
        )

    if df.attrs.get("from_cache"):
        st.caption("Дані з SQLite-кешу статей.")

    st.dataframe(
        df[["title", "published", "category", "scraped_ok", "link"]],
        use_container_width=True,
        hide_index=True,
    )

    total = len(df)
    scraped = int(df["scraped_ok"].sum()) if total else 0
    col1, col2, col3 = st.columns(3)
    col1.metric("Кількість статей", total)
    col2.metric("З повним текстом", scraped)
    col3.metric("Успішність скрейпінгу", f"{scraped / total:.0%}" if total else "—")

    with st.expander("Посилання на статті"):
        for _, row in df.iterrows():
            st.markdown(f"- [{row['title']}]({row['link']})")

    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Завантажити CSV",
        data=csv_bytes,
        file_name="articles.csv",
        mime="text/csv",
    )


def _render_ngram_table(
    titles: pd.Series,
    extractor,
    label: str,
    chart_type: str = "scatter",
) -> None:
    st.markdown(NGRAM_DESCRIPTION)
    common = extractor(titles, 10)
    if not common:
        st.warning(f"{label} не знайдено в заголовках.")
        return

    df_ngrams = pd.DataFrame(common, columns=[label, "Кількість"])
    st.table(df_ngrams)

    if chart_type == "bar":
        fig = px.bar(df_ngrams, x=label, y="Кількість", color="Кількість", height=500)
    else:
        fig = px.scatter(
            df_ngrams,
            x=label,
            y="Кількість",
            color="Кількість",
        )
        fig.update_layout(xaxis_title=label, yaxis_title="Кількість")
    st.plotly_chart(fig, use_container_width=True)


def render_unigrams(titles: pd.Series) -> None:
    st.subheader("Уніграми")
    _render_ngram_table(titles, get_top_n_words, "Слово")


def render_bigrams(titles: pd.Series) -> None:
    st.subheader("Біграми")
    _render_ngram_table(titles, get_top_n_bigram, "Біграма", chart_type="bar")


def render_trigrams(titles: pd.Series) -> None:
    st.subheader("Триграми")
    _render_ngram_table(titles, get_top_n_trigram, "Триграма")


def render_keywords(titles: pd.Series) -> None:
    st.subheader("Ключові слова")
    st.markdown("Виділення найбільш частотних лем у заголовках новин.")
    keywords = extract_keywords(titles, top_n=15, lemmatize=True)
    if not keywords:
        st.warning("Ключових слів не знайдено.")
        return
    df_kw = pd.DataFrame(keywords, columns=["Ключове слово", "Частота"])
    st.table(df_kw)
    fig = px.bar(df_kw, x="Ключове слово", y="Частота", color="Частота", height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_wordcloud(titles: pd.Series) -> None:
    st.subheader("Хмара слів")
    st.markdown(WORDCLOUD_DESCRIPTION)
    render_wordclouds(titles)


def render_text_stat(df: pd.DataFrame) -> None:
    st.subheader("Статистика тексту")
    st.markdown(
        "Базові метрики читабельності та структури тексту для українських статей. "
        "Індекси Flesch/SMOG недоступні — застосовано власні метрики."
    )

    content = df["content"].fillna("").astype(str)
    sample_text = next((text for text in content if text.strip()), "")
    if not sample_text:
        st.warning("Немає тексту статей для аналізу.")
        return

    st.markdown("**Перша стаття з повним текстом**")
    for name, value in get_textstat_metrics(sample_text):
        st.write(f"**{name}:** {value}")

    st.divider()
    st.markdown("**Увесь завантажений корпус**")
    for name, value in aggregate_corpus_metrics(content):
        st.write(f"**{name}:** {value}")


def render_ner(df: pd.DataFrame, titles: pd.Series) -> None:
    st.subheader("Розпізнавання сутностей (NER)")
    st.markdown(
        "Named Entity Recognition — виділення іменованих сутностей у текстах статей "
        "(модель spaCy). За замовчуванням аналізується вибірка контенту."
    )
    entity = st.selectbox(
        "Тип сутності",
        options=list(NER_LABELS_UA.keys()),
        format_func=lambda key: NER_LABELS_UA[key],
    )
    source_mode = st.radio(
        "Джерело тексту",
        ["Контент статей", "Лише заголовки"],
        horizontal=True,
    )
    if source_mode == "Лише заголовки":
        texts = titles
    else:
        content = df["content"].fillna("").astype(str)
        nonempty = content[content.str.strip().astype(bool)]
        texts = nonempty if len(nonempty) else titles
        sample_n = _sample_size_slider(
            "Скільки статей для NER",
            default=min(15, len(texts)),
            max_value=max(1, len(texts)),
            key="ner_sample",
        )
        texts = texts.head(sample_n)
    plot_most_common_named_entity_barchart(texts, entity=entity)


def render_pos(content: pd.Series) -> None:
    st.subheader("Частини мови")
    st.markdown("Розподіл частин мови у текстах статей (spaCy).")
    for tag, name_ua, example in POS_DESCRIPTIONS:
        st.markdown(f"- **{name_ua} ({tag})** — {example}")
    sample_n = _sample_size_slider(
        "Скільки статей для POS",
        default=min(MAX_POS_ARTICLES, max(1, len(content))),
        max_value=max(1, len(content)),
        key="pos_sample",
    )
    plot_parts_of_speech_barchart(content.head(sample_n))


def _render_sentiment_table(titles: pd.Series, labels: list[str]) -> None:
    """
    Show a per-article sentiment table and offer CSV download.

    Length mismatches are truncated to the shorter sequence so Streamlit
    never receives a ragged DataFrame.
    """
    try:
        title_list = [str(t) for t in titles]
        label_list = [str(label) for label in labels]
        size = min(len(title_list), len(label_list))
        if size == 0:
            st.warning("Немає рядків для таблиці тональності.")
            return

        table = pd.DataFrame(
            {
                "Заголовок": title_list[:size],
                "Тональність": label_list[:size],
            }
        )
        st.dataframe(table, use_container_width=True, hide_index=True)
        csv_bytes = table.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Завантажити тональність CSV",
            data=csv_bytes,
            file_name="sentiment_per_article.csv",
            mime="text/csv",
        )
    except Exception as exc:
        logger.exception("Sentiment table render failed")
        st.error(f"Не вдалося побудувати таблицю тональності: {exc}")


def render_sentiment_cosmus(titles: pd.Series) -> None:
    st.subheader("Тональність (RoBERTa-COSMUS)")
    st.markdown(
        "Класифікація тональності заголовків моделлю "
        "`YShynkarov/ukr-roberta-cosmus-sentiment` "
        "(позитивна / негативна / нейтральна / змішана)."
    )
    st.info(
        "Модель навчена переважно на соцмережах і відгуках. "
        "На новинних заголовках точність може бути нижчою — "
        "порівняйте з «Тональність (новини)»."
    )
    sample_n = _sample_size_slider(
        "Скільки заголовків",
        default=min(MAX_SENTIMENT_TITLES, max(1, len(titles))),
        max_value=max(1, len(titles)),
        key="cosmus_sample",
    )
    sample = titles.head(sample_n)
    with st.spinner("Аналіз тональності..."):
        plot_sentiment_barchart(sample, method="cosmus")


def render_sentiment_emotions(titles: pd.Series) -> None:
    st.subheader("Тональність (Емоції)")
    st.markdown(
        "Мультиміткова модель `ukr-detect/ukr-emotions-classifier`: "
        "радість, гнів, страх, огида, подив, сум."
    )
    st.info(
        "Класифікатор емоцій орієнтований на розмовний текст. "
        "Нейтральні новинні заголовки часто отримують мітку «Без емоцій»."
    )
    sample_n = _sample_size_slider(
        "Скільки заголовків",
        default=min(MAX_SENTIMENT_TITLES, max(1, len(titles))),
        max_value=max(1, len(titles)),
        key="emotions_sample",
    )
    sample = titles.head(sample_n)
    with st.spinner("Аналіз емоцій..."):
        plot_emotion_distribution(sample)


def render_sentiment_news(titles: pd.Series) -> None:
    st.subheader("Тональність (новини)")
    st.markdown(
        "Правиловий baseline для новинних заголовків (лексикон позитивних/негативних маркерів). "
        "Легкий для Streamlit Cloud, без transformers."
    )
    sample_n = _sample_size_slider(
        "Скільки заголовків",
        default=min(MAX_SENTIMENT_TITLES, max(1, len(titles))),
        max_value=max(1, len(titles)),
        key="news_sample",
    )
    sample = titles.head(sample_n)
    with st.spinner("Аналіз тональності..."):
        labels = classify_news_sentiment_batch([str(t) for t in sample])
        plot_sentiment_barchart(sample, method="news_rules")
        _render_sentiment_table(sample, labels)


def render_compare_media(primary_source: str) -> None:
    st.subheader("Порівняння медіа")
    st.markdown("Порівняння топ-уніграм і новинної тональності двох джерел.")
    other = st.selectbox(
        "Друге медіа",
        [name for name in NEWS_SOURCES if name != primary_source],
    )
    try:
        df_a = _load_source(primary_source)
        df_b = _load_source(other)
    except DataLoaderError as exc:
        st.error(str(exc))
        return

    titles_a = preprocess(df_a["title"])
    titles_b = preprocess(df_b["title"])
    top_a = dict(get_top_n_words(titles_a, 15))
    top_b = dict(get_top_n_words(titles_b, 15))
    shared = sorted(set(top_a) & set(top_b), key=lambda w: top_a[w] + top_b[w], reverse=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{primary_source}**")
        st.table(pd.DataFrame(list(top_a.items())[:10], columns=["Слово", "N"]))
        labels_a = classify_news_sentiment_batch([str(t) for t in titles_a.head(30)])
        dist_a = pd.Series(labels_a).value_counts()
        st.markdown("Тональність (новини)")
        st.table(dist_a.rename("N"))
    with col2:
        st.markdown(f"**{other}**")
        st.table(pd.DataFrame(list(top_b.items())[:10], columns=["Слово", "N"]))
        labels_b = classify_news_sentiment_batch([str(t) for t in titles_b.head(30)])
        dist_b = pd.Series(labels_b).value_counts()
        st.markdown("Тональність (новини)")
        st.table(dist_b.rename("N"))

    if shared:
        st.markdown("**Спільні слова**")
        st.write(", ".join(shared[:20]))
    else:
        st.caption("Спільних топ-слів не знайдено.")


def render_topic_modeling(content: pd.Series) -> None:
    st.subheader("Тематичне моделювання (LDA)")
    st.markdown(
        "Latent Dirichlet Allocation — виявлення прихованих тем у корпусі статей. "
        "Потрібно щонайменше 3 тексти з достатньою кількістю слів."
    )
    display_topic_modeling(content)


def render_summarization(df: pd.DataFrame) -> None:
    st.subheader("Сумаризація тексту")
    st.markdown(
        "Екстрактивна сумаризація на основі LexRank: "
        "найважливіші речення з повного тексту статті (spaCy + TF-IDF)."
    )
    sample_n = _sample_size_slider(
        "Скільки статей сумаризувати",
        default=min(MAX_SUMMARY_ARTICLES, max(1, len(df))),
        max_value=max(1, len(df)),
        key="summary_sample",
    )
    run_text_summarization(df, sentence_count=3, max_articles=sample_n)


def load_data(source_name: str, nlp_function: str) -> None:
    config = NEWS_SOURCES.get(source_name)
    if config is None:
        st.error(f"Невідоме джерело: {source_name}")
        return

    if nlp_function == "Вступ":
        render_intro(config["intro"])
        return

    if nlp_function == "Порівняння медіа":
        render_compare_media(source_name)
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


def main() -> None:
    st.set_page_config(
        page_title="UkrMediaNLP",
        page_icon="📰",
        layout="wide",
    )
    st.title("Аналіз новин українських медіа")
    st.caption("Збір новин з RSS, скрейпінг та NLP-аналіз українською мовою.")
    if get_cloud_light():
        st.caption("Режим LIGHT_CLOUD: transformers-моделі приховані.")

    st.sidebar.header("Налаштування")
    selected_source = st.sidebar.selectbox("Медіа", list(NEWS_SOURCES.keys()))
    selected_function = st.sidebar.selectbox("Функція", NLP_FUNCTIONS)

    load_data(selected_source, selected_function)


if __name__ == "__main__":
    main()
