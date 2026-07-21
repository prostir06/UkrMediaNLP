"""Word cloud generation for Ukrainian headlines (no Streamlit)."""

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

EXTRA_STYLES = [
    {
        "width": 2000,
        "height": 1200,
        "random_state": 1,
        "background_color": "salmon",
        "colormap": "Pastel1",
        "collocations": False,
    },
    {
        "width": 2000,
        "height": 1200,
        "random_state": 1,
        "background_color": "black",
        "colormap": "Set2",
        "collocations": False,
    },
]


def _resolve_font_path() -> str | None:
    for path in FONT_CANDIDATES:
        if path.exists():
            return str(path)
    try:
        return font_manager.findfont("DejaVu Sans", fallback_to_default=True)
    except Exception:
        return None


def build_wordcloud_images(titles, styles: list[dict] | None = None):
    """
    Build word-cloud image arrays.

    By default returns a single cloud. Pass ``styles=EXTRA_STYLES`` or a custom
    list for additional variants (UI shows them as tabs).
    """
    if hasattr(titles, "fillna"):
        long_string = " ".join(titles.fillna("").astype(str).tolist())
    else:
        long_string = " ".join(str(t or "") for t in titles)

    if not long_string.strip():
        return []

    stopwords = set(load_stopwords())
    font_path = _resolve_font_path()
    selected = styles if styles is not None else [DEFAULT_STYLE]

    images = []
    for style in selected:
        wordcloud = WordCloud(
            font_path=font_path,
            stopwords=stopwords,
            regexp=UKRAINIAN_TOKEN_PATTERN,
            **style,
        ).generate(long_string)
        images.append(wordcloud.to_array())
    return images
