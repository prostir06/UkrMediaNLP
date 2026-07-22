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
)
from exceptions import DataLoaderError
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
    st.dataframe(
        df[["title", "published", "category", "scraped_ok", "link"]],
        use_container_width=True,
        hide_index=True,
    )

    total = len(df)
    scraped = int(df["scraped_ok"].sum()) if total else 0
    st.metric("Кількість статей", total)
    st.metric("З повним текстом", scraped)
    if total:
        st.caption(f"Успішність скрейпінгу: {scraped / total:.0%}")

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
    st.markdown("Виділення найбільш частотних та значущих слів у заголовках новин.")
    keywords = extract_keywords(titles, top_n=15)
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


def render_ner(titles: pd.Series) -> None:
    st.subheader("Розпізнавання сутностей (NER)")
    st.markdown(
        "Named Entity Recognition — виділення іменованих сутностей у заголовках: "
        "особи, організації, локації (модель spaCy `uk_core_news_sm`)."
    )
    entity = st.selectbox(
        "Тип сутності",
        options=list(NER_LABELS_UA.keys()),
        format_func=lambda key: NER_LABELS_UA[key],
    )
    plot_most_common_named_entity_barchart(titles, entity=entity)


def render_pos(content: pd.Series) -> None:
    st.subheader("Частини мови")
    st.markdown(
        f"Розподіл частин мови у текстах статей (spaCy). "
        f"Аналізуються перші {MAX_POS_ARTICLES} статей."
    )
    for tag, name_ua, example in POS_DESCRIPTIONS:
        st.markdown(f"- **{name_ua} ({tag})** — {example}")
    plot_parts_of_speech_barchart(content)


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
    sample = titles.head(MAX_SENTIMENT_TITLES)
    if len(titles) > MAX_SENTIMENT_TITLES:
        st.caption(f"Аналізуються перші {MAX_SENTIMENT_TITLES} заголовків.")
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
    sample = titles.head(MAX_SENTIMENT_TITLES)
    if len(titles) > MAX_SENTIMENT_TITLES:
        st.caption(f"Аналізуються перші {MAX_SENTIMENT_TITLES} заголовків.")
    with st.spinner("Аналіз емоцій..."):
        plot_emotion_distribution(sample)


def render_sentiment_news(titles: pd.Series) -> None:
    st.subheader("Тональність (новини)")
    st.markdown(
        "Правиловий baseline для новинних заголовків (лексикон позитивних/негативних маркерів). "
        "Легкий для Streamlit Cloud, без transformers."
    )
    sample = titles.head(MAX_SENTIMENT_TITLES)
    with st.spinner("Аналіз тональності..."):
        plot_sentiment_barchart(sample, method="news_rules")


def render_compare_media(primary_source: str) -> None:
    st.subheader("Порівняння медіа")
    st.markdown("Порівняння топ-уніграм двох джерел за заголовками.")
    other = st.selectbox(
        "Друге медіа",
        [name for name in NEWS_SOURCES if name != primary_source],
    )
    try:
        df_a = load_articles(primary_source)
        df_b = load_articles(other)
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
    with col2:
        st.markdown(f"**{other}**")
        st.table(pd.DataFrame(list(top_b.items())[:10], columns=["Слово", "N"]))

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
    st.caption(f"Сумаризуються перші {MAX_SUMMARY_ARTICLES} статей з текстом.")
    run_text_summarization(df, sentence_count=3, max_articles=MAX_SUMMARY_ARTICLES)


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
        df = load_articles(source_name)
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
        "Розпізнавання сутностей": lambda: render_ner(titles),
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

    st.sidebar.header("Налаштування")
    selected_source = st.sidebar.selectbox("Медіа", list(NEWS_SOURCES.keys()))
    selected_function = st.sidebar.selectbox("Функція", NLP_FUNCTIONS)

    load_data(selected_source, selected_function)


if __name__ == "__main__":
    main()
