"""
Word cloud generation for Ukrainian headlines (no Streamlit).

Uses optional spaCy lemmatization so declined forms collapse onto lemmas.
Font resolution prefers bundled DejaVu, then system fonts.
"""

import logging
from pathlib import Path

from matplotlib import font_manager
from wordcloud import WordCloud

from nlp.text_utils import UKRAINIAN_TOKEN_PATTERN, load_stopwords

logger = logging.getLogger(__name__)

FONT_CANDIDATES = [
    Path(__file__).resolve().parent.parent / "data" / "fonts" / "DejaVuSans.ttf",
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
]

DEFAULT_STYLE = {
    "background_color": "white",
    "max_words": 5000,
    "colormap": "Blues",
    "width": 2000,
    "height": 1200,
}


def _resolve_font_path() -> str | None:
    """
    Pick a TTF that can render Cyrillic glyphs.

    Returns:
        Absolute font path, or ``None`` to let WordCloud use its default.
    """
    for path in FONT_CANDIDATES:
        try:
            if path.exists():
                return str(path)
        except OSError as exc:
            logger.debug("Font path check failed for %s: %s", path, exc)

    try:
        return font_manager.findfont("DejaVu Sans", fallback_to_default=True)
    except Exception as exc:
        logger.debug("Matplotlib font lookup failed: %s", exc)
        return None


def build_wordcloud_images(
    titles,
    styles: list[dict] | None = None,
    lemmatize: bool = True,
):
    """
    Build word-cloud image arrays (NumPy) for UI display.

    Args:
        titles: Iterable or pandas Series of headlines / short texts.
        styles: Optional list of WordCloud kwargs; defaults to ``DEFAULT_STYLE``.
        lemmatize: When True, run spaCy lemmatization before tokenization.

    Returns:
        List of image arrays; empty list when there is no usable text.
    """
    try:
        if hasattr(titles, "fillna"):
            text_list = titles.fillna("").astype(str).tolist()
        else:
            text_list = [str(t or "") for t in titles]
    except (TypeError, ValueError, AttributeError) as exc:
        logger.warning("Cannot normalise titles for word cloud: %s", exc)
        return []

    if lemmatize:
        try:
            from nlp.model_registry import resolve_spacy_nlp
            from nlp.text_utils import lemmatize_texts

            nlp = resolve_spacy_nlp()
            text_list = lemmatize_texts(text_list, nlp)
        except Exception as exc:
            # spaCy may be missing in light Cloud installs — keep raw tokens.
            logger.debug("Wordcloud lemmatization skipped: %s", exc)

    long_string = " ".join(text_list)
    if not long_string.strip():
        return []

    try:
        stopwords = set(load_stopwords())
    except Exception as exc:
        logger.warning("Stopwords load failed, continuing without them: %s", exc)
        stopwords = set()

    font_path = _resolve_font_path()
    selected = styles if styles is not None else [DEFAULT_STYLE]

    images = []
    for style in selected:
        try:
            wordcloud = WordCloud(
                font_path=font_path,
                stopwords=stopwords,
                regexp=UKRAINIAN_TOKEN_PATTERN,
                **style,
            ).generate(long_string)
            images.append(wordcloud.to_array())
        except Exception as exc:
            # One broken style must not abort the whole render.
            logger.warning("WordCloud style failed (%s): %s", style, exc)
    return images
