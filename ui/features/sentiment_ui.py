"""Sentiment analysis feature screens."""

import logging

import pandas as pd
import streamlit as st

from config import MAX_SENTIMENT_TITLES
from nlp.news_sentiment import classify_news_sentiment_batch
from ui.renderers import plot_emotion_distribution, plot_sentiment_barchart
from ui.widgets import sample_size_slider

logger = logging.getLogger(__name__)

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
    sample_n = sample_size_slider(
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
    sample_n = sample_size_slider(
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
    sample_n = sample_size_slider(
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
