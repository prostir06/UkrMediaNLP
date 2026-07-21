"""
Named entity recognition for Ukrainian news titles.
"""

import logging
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from exceptions import NLPAnalysisError
from nlp.model_registry import resolve_spacy_nlp
from nlp.preprocessing import NER_LABELS_UA

logger = logging.getLogger(__name__)


def extract_entities_batch(texts: list[str], entity: str, nlp=None) -> list[list[str]]:
    """Extract named entities of one type from multiple texts using spaCy pipe."""
    if nlp is None:
        nlp = resolve_spacy_nlp()

    results: list[list[str]] = []
    for doc in nlp.pipe(texts, batch_size=32):
        results.append([ent.text for ent in doc.ents if ent.label_ == entity])
    return results


def build_ner_figure(texts, entity: str = "PER"):
    try:
        text_list = [str(t or "") for t in texts]
        entity_lists = extract_entities_batch(text_list, entity)
        flat_entities = [item for sublist in entity_lists for item in sublist]

        counter = Counter(flat_entities)
        if not counter:
            return None, NER_LABELS_UA.get(entity, entity)

        labels, counts = map(list, zip(*counter.most_common(10)))
        title = NER_LABELS_UA.get(entity, entity)
        fig, ax = plt.subplots()
        sns.barplot(x=counts, y=labels, ax=ax).set_title(title)
        return fig, title
    except NLPAnalysisError:
        raise
    except Exception as exc:
        logger.exception("NER chart failed")
        raise RuntimeError("NER chart failed") from exc
