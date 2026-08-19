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
    "width": 2000,
    "height": 1200,
    # Prefer color_func over sequential "Blues" — pale ends are unreadable on white.
    "color_func": None,  # filled at build time via _contrast_color_func
}

# Dark, saturated hues that stay readable on a white background.
_CONTRAST_PALETTE = (
    "rgb(15, 56, 110)",   # deep navy
    "rgb(0, 90, 100)",    # dark teal
    "rgb(120, 20, 40)",   # burgundy
    "rgb(35, 35, 35)",    # near-black
    "rgb(70, 30, 120)",   # deep purple
    "rgb(0, 85, 55)",     # forest green
    "rgb(140, 60, 0)",    # burnt orange
    "rgb(20, 70, 140)",   # strong blue
)


def _contrast_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    """Pick a dark palette color so small words stay visible on white."""
    try:
        rng = random_state
        if rng is None:
            import random as _random

            idx = _random.randint(0, len(_CONTRAST_PALETTE) - 1)
        else:
            idx = int(rng.randint(0, len(_CONTRAST_PALETTE) - 1))
        return _CONTRAST_PALETTE[idx]
    except (IndexError, TypeError, ValueError):
        return _CONTRAST_PALETTE[0]


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
    except (OSError, ValueError, TypeError) as exc:
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
        except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
            # spaCy may be missing in light Cloud installs — keep raw tokens.
            logger.debug("Wordcloud lemmatization skipped: %s", exc)

    long_string = " ".join(text_list)
    if not long_string.strip():
        return []

    try:
        stopwords = set(load_stopwords())
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Stopwords load failed, continuing without them: %s", exc)
        stopwords = set()

    font_path = _resolve_font_path()
    selected = styles if styles is not None else [DEFAULT_STYLE]

    images = []
    for style in selected:
        try:
            style_kwargs = dict(style)
            # Inject contrast palette unless the caller supplied colormap/color_func.
            if "color_func" not in style_kwargs and "colormap" not in style_kwargs:
                style_kwargs["color_func"] = _contrast_color_func
            elif style_kwargs.get("color_func") is None and "colormap" not in style_kwargs:
                style_kwargs["color_func"] = _contrast_color_func

            wordcloud = WordCloud(
                font_path=font_path,
                stopwords=stopwords,
                regexp=UKRAINIAN_TOKEN_PATTERN,
                **style_kwargs,
            ).generate(long_string)
            images.append(wordcloud.to_array())
        except (ValueError, TypeError, RuntimeError, OSError) as exc:
            # One broken style must not abort the whole render.
            logger.warning("WordCloud style failed (%s): %s", style, exc)
    return images
