"""Unit tests for sentiment helper functions (no model download)."""

import pytest

import nlp.sentiment as sentiment
import nlp.sentiment_inference as sentiment_inference
import nlp.sentiment_models as sentiment_models
from nlp.sentiment_constants import _format_load_error, _label_to_ua, _truncate


def test_truncate_limits_length():
    assert len(_truncate("а" * 5000)) == 2000


def test_truncate_handles_none():
    assert _truncate("") == ""


def test_label_to_ua_maps_known_labels():
    assert _label_to_ua("positive") == "Позитивна"
    assert _label_to_ua("negative") == "Негативна"
    assert _label_to_ua("LABEL_3") == "Позитивна"
    assert _label_to_ua("LABEL_0") == "Змішана"


def test_classify_sentiment_cosmus_fallback(monkeypatch):
    sentiment_inference._COSMUS_PIPELINE = None

    def broken_pipeline():
        raise IndexError("bad output")

    monkeypatch.setattr(sentiment_inference, "load_cosmus_pipeline", broken_pipeline)
    assert sentiment.classify_sentiment_cosmus("Тестовий заголовок") == "Нейтральна"


def test_classify_emotions_fallback(monkeypatch):
    sentiment_inference._EMOTIONS_MODEL = None

    def broken_model():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(sentiment_inference, "load_emotions_model", broken_model)
    assert sentiment.classify_emotions("Тест") == ["Без емоцій"]


def test_dominant_emotion_fallback(monkeypatch):
    sentiment_inference._EMOTIONS_MODEL = None

    def broken_model():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(sentiment_inference, "load_emotions_model", broken_model)
    assert sentiment_inference._dominant_emotion("Тест") == "Без емоцій"


def test_classify_sentiment_batch_with_mock(monkeypatch):
    class MockPipeline:
        def __call__(self, texts, batch_size=8):
            return [{"label": "positive"} for _ in texts]

    monkeypatch.setattr(
        sentiment_inference,
        "_get_cosmus_pipeline",
        lambda: MockPipeline(),
    )
    results = sentiment.classify_sentiment_batch(["Good", "Bad"])
    assert results == ["Позитивна", "Позитивна"]


def test_classify_sentiment_batch_raises_on_failure(monkeypatch):
    from exceptions import NLPAnalysisError

    def boom():
        raise RuntimeError("pipeline down")

    monkeypatch.setattr(sentiment_inference, "_get_cosmus_pipeline", boom)
    with pytest.raises(NLPAnalysisError, match="тональності"):
        sentiment.classify_sentiment_batch(["Текст"])


def test_format_load_error_includes_install_hint():
    message = _format_load_error(
        "demo-model",
        ModuleNotFoundError("No module named 'transformers'"),
    )
    assert "demo-model" in message
    assert "pip install" in message


def test_remap_state_dict_keys_strips_module_prefix():
    model_keys = {"roberta.embeddings.weight", "classifier.out_proj.weight"}
    state = {
        "module.roberta.embeddings.weight": 1,
        "module.classifier.out_proj.weight": 2,
    }
    remapped = sentiment_models._remap_state_dict_keys(state, model_keys)
    assert "roberta.embeddings.weight" in remapped
    assert "classifier.out_proj.weight" in remapped
