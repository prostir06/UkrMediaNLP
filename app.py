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
    MAX_TREND_TERMS,
    MEDIA_CATEGORIES,
    NEWS_SOURCES,
    NGRAM_DESCRIPTION,
    NLP_FUNCTIONS_FULL,
    NLP_FUNCTIONS_LIGHT,
    WORDCLOUD_DESCRIPTION,
    get_cloud_light,
    get_source_config,
    source_category,
    sources_for_category,
)
from exceptions import DataLoaderError
from nlp.corpus import (
    aggregate_trends,
    aggregate_trends_by_source,
    parse_manual_terms,
    search_corpus,
    suggest_lda_labels,
    suggest_terms,
)
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
from ui.corpus_charts import (
    build_source_hit_bar,
    build_source_trends_line,
    build_trends_line,
)
from ui.corpus_controls import (
    CORPUS_FUNCTIONS,
    load_corpus_into_session,
    render_corpus_sidebar,
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


def _commit_corpus_load(
    corpus_df: pd.DataFrame,
    warnings: list[str],
    sources: list[str],
    category: str,
) -> bool:
    """Store a corpus load unless every requested source failed."""
    previous = st.session_state.get("corpus_df")
    has_previous = isinstance(previous, pd.DataFrame) and not previous.empty
    total_failure = bool(sources) and corpus_df.empty and len(warnings) >= len(sources)

    for warning in warnings:
        st.warning(f"Не вдалося завантажити джерело: {warning}")

    if total_failure and has_previous:
        st.error("Жодне джерело не завантажено. Попередній корпус збережено.")
        return False

    st.session_state["corpus_df"] = corpus_df
    st.session_state["corpus_sources"] = list(sources)
    st.session_state["corpus_category"] = category
    st.session_state["corpus_loaded_at"] = pd.Timestamp.now()

    if total_failure:
        st.error("Жодне джерело не завантажено.")
    elif corpus_df.empty:
        st.warning("Корпус порожній: статті за заданими умовами не знайдено.")
    else:
        st.success(f"Корпус завантажено: {len(corpus_df)} статей.")

    stats_list = corpus_df.attrs.get("scrape_stats_by_source") or []
    if isinstance(stats_list, list) and stats_list:
        parts = []
        for stats in stats_list:
            if not isinstance(stats, dict):
                continue
            src = stats.get("source", "?")
            ok = int(stats.get("ok", 0))
            total = int(stats.get("total", 0))
            elapsed = stats.get("elapsed_ms")
            rate = f"{ok / total:.0%}" if total else "—"
            timing = f", {elapsed} мс" if elapsed is not None else ""
            parts.append(f"{src}: {ok}/{total} ({rate}){timing}")
        if parts:
            st.caption("Успішність скрейпінгу — " + "; ".join(parts))
    return True


def _invalidate_stale_corpus(
    category: str,
    current_sources: list[str],
    all_category: bool,
) -> bool:
    """Soft-clear a loaded corpus that no longer matches sidebar controls."""
    corpus_df = st.session_state.get("corpus_df")
    if not isinstance(corpus_df, pd.DataFrame) or corpus_df.empty:
        return False

    stored_category = st.session_state.get("corpus_category")
    stored_sources = st.session_state.get("corpus_sources", [])
    category_changed = stored_category != category
    sources_changed = not all_category and sorted(stored_sources) != sorted(current_sources)
    if not category_changed and not sources_changed:
        return False

    st.session_state["corpus_df"] = corpus_df.iloc[0:0].copy()
    st.warning("Налаштування корпусу змінилися; попередній корпус більше не актуальний.")
    st.info("Завантажте корпус повторно для поточної категорії та вибраних медіа.")
    return True


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

    scrape_stats = df.attrs.get("scrape_stats")
    if isinstance(scrape_stats, dict) and scrape_stats.get("total"):
        ok = int(scrape_stats.get("ok", 0))
        total_scrape = int(scrape_stats.get("total", 0))
        elapsed = scrape_stats.get("elapsed_ms")
        rate = f"{ok / total_scrape:.0%}" if total_scrape else "—"
        timing = f", {elapsed} мс" if elapsed is not None else ""
        st.caption(f"Скрейпінг: {ok}/{total_scrape} успішно ({rate}){timing}.")

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
    st.caption(
        "Перше завантаження моделі з Hugging Face може зайняти 5–10 хвилин "
        "(залежить від мережі) і ~1 GB RAM. Не закривайте вкладку."
    )
    default_n = min(MAX_SENTIMENT_TITLES, max(1, len(titles)))
    sample_n = _sample_size_slider(
        "Скільки заголовків",
        default=default_n,
        max_value=max(1, len(titles)),
        key="cosmus_sample",
    )
    sample = titles.head(sample_n)

    col_run, col_reset = st.columns([2, 1])
    with col_run:
        if st.button("Запустити аналіз тональності", type="primary", key="run_cosmus"):
            st.session_state["cosmus_pending"] = True
            st.session_state["cosmus_sample_n"] = int(sample_n)
    with col_reset:
        if st.button("Скинути", key="reset_cosmus"):
            st.session_state.pop("cosmus_pending", None)
            st.session_state.pop("cosmus_done", None)

    if not st.session_state.get("cosmus_pending") and not st.session_state.get("cosmus_done"):
        st.caption("Натисніть кнопку, щоб завантажити модель і побудувати діаграму.")
        return

    # Prefer the sample size captured when the user clicked Run.
    stored_n = st.session_state.get("cosmus_sample_n", sample_n)
    try:
        sample = titles.head(int(stored_n))
    except (TypeError, ValueError) as exc:
        logger.debug("Invalid cosmus_sample_n (%s); using slider value", exc)
        sample = titles.head(sample_n)

    try:
        show_status = bool(st.session_state.get("cosmus_pending"))
        if show_status:
            with st.status("Завантаження моделі / аналіз тональності...", expanded=True) as status:
                status.write("Завантаження ваг з Hugging Face (перший раз довго)…")
                plot_sentiment_barchart(sample, method="cosmus")
                status.update(label="Готово", state="complete")
        else:
            # Model already cached — redraw without the long-load status chrome.
            plot_sentiment_barchart(sample, method="cosmus")
        st.session_state["cosmus_done"] = True
        st.session_state["cosmus_pending"] = False
    except MemoryError:
        logger.exception("COSMUS OOM in UI")
        st.session_state["cosmus_pending"] = False
        st.error("Недостатньо пам'яті для RoBERTa-COSMUS. Спробуйте «Тональність (новини)».")
    except Exception as exc:
        logger.exception("COSMUS UI failed")
        st.session_state["cosmus_pending"] = False
        st.error(f"Аналіз тональності не вдався: {exc}")


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
    st.warning(
        "Перше завантаження моделі з Hugging Face зазвичай займає **5–10 хвилин** "
        "і ~1 GB RAM. Не закривайте вкладку. "
        "Якщо в консолі з’являється `WebSocket onclose` / `ERR_EMPTY_RESPONSE` — "
        "процес Streamlit впав (часто OOM); у терміналі виконайте "
        "`streamlit run streamlit_app.py` знову. "
        "На Streamlit Cloud free tier краще «Тональність (новини)»."
    )
    default_n = min(15, max(1, len(titles)))
    sample_n = _sample_size_slider(
        "Скільки заголовків",
        default=default_n,
        max_value=max(1, len(titles)),
        key="emotions_sample",
    )
    sample = titles.head(sample_n)

    col_run, col_reset = st.columns([2, 1])
    with col_run:
        if st.button("Запустити аналіз емоцій", type="primary", key="run_emotions"):
            st.session_state["emotions_pending"] = True
            st.session_state["emotions_sample_n"] = int(sample_n)
    with col_reset:
        if st.button("Скинути", key="reset_emotions"):
            st.session_state.pop("emotions_pending", None)
            st.session_state.pop("emotions_done", None)

    if not st.session_state.get("emotions_pending") and not st.session_state.get("emotions_done"):
        st.caption("Натисніть кнопку, щоб завантажити модель і побудувати діаграму.")
        return

    # Re-use sample size chosen when the user clicked Run.
    stored_n = st.session_state.get("emotions_sample_n", sample_n)
    try:
        sample = titles.head(int(stored_n))
    except (TypeError, ValueError) as exc:
        logger.debug("Invalid emotions_sample_n (%s); using slider value", exc)
        sample = titles.head(sample_n)

    try:
        show_status = bool(st.session_state.get("emotions_pending"))
        if show_status:
            with st.status("Завантаження моделі емоцій…", expanded=True) as status:
                status.write(
                    "Крок 1/2: завантаження з Hugging Face "
                    "(перший раз 5–10 хв, далі з кешу секунди)."
                )
                status.write("Крок 2/2: інференс по заголовках…")
                plot_emotion_distribution(sample)
                status.update(label="Аналіз емоцій завершено", state="complete")
        else:
            # Model already in cache — redraw chart without the long-load status UI.
            plot_emotion_distribution(sample)
        st.session_state["emotions_done"] = True
        st.session_state["emotions_pending"] = False
    except MemoryError:
        logger.exception("Emotions OOM in UI")
        st.session_state["emotions_pending"] = False
        st.error(
            "Недостатньо пам'яті для моделі емоцій. "
            "Спробуйте менше заголовків або «Тональність (новини)»."
        )
    except Exception as exc:
        logger.exception("Emotions UI failed")
        st.session_state["emotions_pending"] = False
        st.error(f"Аналіз емоцій не вдався: {exc}")


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
    # Prefer peers from the same sidebar category (Новини / Спорт / …).
    primary_cat = source_category(primary_source)
    try:
        peers = [
            name
            for name, cfg in NEWS_SOURCES.items()
            if name != primary_source
            and isinstance(cfg, dict)
            and (primary_cat is None or cfg.get("category") == primary_cat)
        ]
    except (AttributeError, TypeError) as exc:
        logger.warning("Cannot build compare peers: %s", exc)
        peers = []
    if not peers:
        peers = [name for name in NEWS_SOURCES if name != primary_source]
    if not peers:
        st.warning("Немає іншого медіа для порівняння в цій категорії.")
        return

    other = st.selectbox("Друге медіа", peers)
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


def render_corpus_search() -> None:
    """Render search controls and results for the loaded multi-source corpus."""
    corpus_df = st.session_state.get("corpus_df")
    if not isinstance(corpus_df, pd.DataFrame) or corpus_df.empty:
        st.info("Спочатку завантажте корпус у бічній панелі.")
        return

    st.subheader("Пошук у корпусі")
    query = st.text_input("Пошуковий запит", key="corpus_search_query")
    field_mode = st.radio(
        "Шукати в",
        ("Заголовках і текстах", "Лише заголовках", "Лише текстах"),
        horizontal=True,
        key="corpus_search_fields",
    )
    field_map = {
        "Заголовках і текстах": ("title", "content"),
        "Лише заголовках": ("title",),
        "Лише текстах": ("content",),
    }
    col_word, col_lemma = st.columns(2)
    with col_word:
        whole_word = st.checkbox(
            "Лише ціле слово",
            value=False,
            key="corpus_search_whole_word",
        )
    with col_lemma:
        use_lemmas = st.checkbox(
            "Враховувати леми",
            value=False,
            key="corpus_search_lemmas",
        )

    if not str(query).strip():
        st.caption("Введіть слово або фразу для пошуку.")
        return

    try:
        results = search_corpus(
            corpus_df,
            query=str(query),
            fields=field_map.get(field_mode, ("title", "content")),
            whole_word=bool(whole_word),
            use_lemmas=bool(use_lemmas),
        )
    except Exception as exc:
        logger.exception("Corpus search failed")
        st.error(f"Пошук у корпусі не вдався: {exc}")
        return

    source_count = results["source"].nunique() if "source" in results.columns else 0
    metric_hits, metric_sources = st.columns(2)
    metric_hits.metric("Знайдено статей", len(results))
    metric_sources.metric("Медіа зі знахідками", int(source_count))

    if results.empty:
        st.warning("За вашим запитом нічого не знайдено.")
        return

    if "source" in results.columns:
        figure = build_source_hit_bar(results["source"].fillna("").value_counts())
        if figure is not None:
            st.plotly_chart(figure, use_container_width=True)

    date_column = "published_dt" if "published_dt" in results.columns else "published"
    visible_columns = [
        column
        for column in ("title", "link", date_column, "source", "snippet")
        if column in results.columns
    ]
    st.dataframe(
        results[visible_columns],
        use_container_width=True,
        hide_index=True,
    )


def render_topic_trends() -> None:
    """Render topic trends for the loaded multi-source corpus."""
    corpus_df = st.session_state.get("corpus_df")
    if not isinstance(corpus_df, pd.DataFrame) or corpus_df.empty:
        st.info("Спочатку завантажте корпус у бічній панелі.")
        return

    st.subheader("Тренди тем")
    automatic_terms = suggest_terms(corpus_df, 15)
    lda_terms: list[str] = []
    if not get_cloud_light():
        with st.expander("Поглиблено (LDA)"):
            try:
                lda_terms = suggest_lda_labels(corpus_df)
            except Exception as exc:
                logger.warning("LDA topic suggestions failed: %s", exc)
                lda_terms = []
            if not lda_terms:
                st.warning("Не вдалося запропонувати теми за допомогою LDA.")
            else:
                st.caption("LDA: " + ", ".join(lda_terms))

    manual_text = st.text_area(
        "Власні теми (по одній у рядку)",
        key="topic_trends_manual_terms",
    )
    manual_terms = parse_manual_terms(manual_text)

    candidates: list[str] = []
    seen: set[str] = set()
    for term in [*automatic_terms, *lda_terms, *manual_terms]:
        normalized = str(term).strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            candidates.append(normalized)

    selected_terms = st.multiselect(
        "Теми для порівняння",
        candidates,
        max_selections=MAX_TREND_TERMS,
        key="topic_trends_terms",
    )
    frequency_label = st.radio(
        "Групування",
        ("День", "Тиждень"),
        horizontal=True,
        key="topic_trends_frequency",
    )
    frequency = "D" if frequency_label == "День" else "W-MON"

    if not selected_terms:
        st.caption("Оберіть хоча б одну тему.")
        return

    trend_figure = build_trends_line(
        aggregate_trends(corpus_df, list(selected_terms), freq=frequency)
    )
    if trend_figure is None:
        st.warning("Для обраних тем не знайдено даних із датами.")
    else:
        st.plotly_chart(trend_figure, use_container_width=True)

    comparison_term = st.selectbox(
        "Тема для порівняння медіа",
        selected_terms,
        key="topic_trends_source_term",
    )
    source_figure = build_source_trends_line(
        aggregate_trends_by_source(corpus_df, comparison_term, freq=frequency)
    )
    if source_figure is None:
        st.warning("Немає даних для порівняння медіа за цією темою.")
    else:
        st.plotly_chart(source_figure, use_container_width=True)


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
            corpus_controls = render_corpus_sidebar(category, show_lda_toggle=False)
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
