"""
Latent Dirichlet Allocation topic modeling for Ukrainian texts.
"""

import logging
import warnings

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from exceptions import NLPAnalysisError
from nlp.text_utils import UKRAINIAN_TOKEN_PATTERN, single_token_stopwords

logger = logging.getLogger(__name__)


def _as_documents(content) -> list[str]:
    if hasattr(content, "fillna"):
        documents = content.fillna("").astype(str).tolist()
    else:
        documents = [str(item) for item in content]
    return [doc.strip() for doc in documents if doc.strip()]


def run_topic_modeling(
    content,
    number_topics: int = 8,
    number_words: int = 6,
    lemmatize: bool = True,
) -> list[str]:
    documents = _as_documents(content)
    if len(documents) < 3:
        return []

    if lemmatize:
        try:
            from nlp.model_registry import resolve_spacy_nlp
            from nlp.text_utils import lemmatize_texts

            nlp = resolve_spacy_nlp()
            documents = lemmatize_texts(documents, nlp)
        except (NLPAnalysisError, ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
            logger.warning("LDA lemmatization skipped: %s", exc)

    try:
        warnings.simplefilter("ignore", DeprecationWarning)
        vectorizer = CountVectorizer(
            stop_words=single_token_stopwords(),
            token_pattern=UKRAINIAN_TOKEN_PATTERN,
            max_features=5000,
            min_df=2,
            lowercase=True,
        )
        matrix = vectorizer.fit_transform(documents)
        if matrix.shape[1] == 0:
            return []

        n_topics = min(number_topics, max(2, len(documents) // 3))
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            n_jobs=-1,
        )
        lda.fit(matrix)

        feature_names = vectorizer.get_feature_names_out()
        topics: list[str] = []
        for index, topic in enumerate(lda.components_, start=1):
            top_indices = topic.argsort()[:-number_words - 1:-1]
            words = " ".join(feature_names[i] for i in top_indices)
            topics.append(f"Тема {index}: {words}")
        return topics
    except ValueError as exc:
        logger.warning("LDA vectorisation failed: %s", exc)
        return []
    except Exception as exc:
        logger.exception("Topic modeling failed")
        raise NLPAnalysisError("Тематичне моделювання не вдалося.", step="lda") from exc

