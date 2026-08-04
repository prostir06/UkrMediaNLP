"""Article snapshot / table feature screen."""

import pandas as pd
import streamlit as st

from config import MAX_ARTICLES


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
        titles = df["title"].fillna("").astype(str).tolist()
        links = df["link"].fillna("").astype(str).tolist()
        for title, link in zip(titles, links, strict=False):
            st.markdown(f"- [{title}]({link})")

    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Завантажити CSV",
        data=csv_bytes,
        file_name="articles.csv",
        mime="text/csv",
    )
