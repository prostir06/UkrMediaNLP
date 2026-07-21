"""Extra n-gram coverage for bigrams/trigrams."""

import pandas as pd

from nlp.ngrams import get_top_n_bigram, get_top_n_trigram, get_top_n_words


def test_get_top_n_bigram_and_trigram():
    corpus = pd.Series(
        [
            "уряд ухвалив новий закон про енергетику",
            "уряд ухвалив новий план відновлення",
            "новий закон про енергетику ухвалив уряд",
        ]
    )
    assert isinstance(get_top_n_words(corpus, 5), list)
    assert isinstance(get_top_n_bigram(corpus, 5), list)
    assert isinstance(get_top_n_trigram(corpus, 5), list)
