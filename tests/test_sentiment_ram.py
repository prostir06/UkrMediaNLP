"""Unit tests for transformers memory / quantize helpers and emotion batch."""

from types import SimpleNamespace

import pytest
import torch

from exceptions import NLPAnalysisError
from nlp import resource_guard, sentiment, sentiment_inference, sentiment_models


def test_require_ram_passes_when_enough(monkeypatch):
    monkeypatch.setattr(resource_guard, "available_ram_mb", lambda: 4096)
    resource_guard.require_ram_for_transformers("emotions_load", min_mb=1536)


def test_require_ram_raises_when_low(monkeypatch):
    monkeypatch.setattr(resource_guard, "available_ram_mb", lambda: 200)
    with pytest.raises(NLPAnalysisError) as exc_info:
        resource_guard.require_ram_for_transformers("emotions_load", min_mb=1536)
    assert exc_info.value.step == "emotions_load"
    assert "RAM" in str(exc_info.value) or "МБ" in str(exc_info.value)


def test_require_ram_skips_when_unknown(monkeypatch):
    monkeypatch.setattr(resource_guard, "available_ram_mb", lambda: None)
    resource_guard.require_ram_for_transformers("emotions_load", min_mb=99999)


def test_require_ram_invalid_env_threshold(monkeypatch):
    monkeypatch.setenv("MIN_TRANSFORMERS_RAM_MB", "not-a-number")
    monkeypatch.setattr(resource_guard, "available_ram_mb", lambda: 200)
    with pytest.raises(NLPAnalysisError):
        resource_guard.require_ram_for_transformers("emotions_load")


def test_maybe_quantize_skips_by_default(monkeypatch):
    monkeypatch.delenv("QUANTIZE_CPU", raising=False)
    sentinel = object()
    called = {"n": 0}

    def boom(model):
        called["n"] += 1
        return model

    monkeypatch.setattr(resource_guard, "quantize_model_if_cpu", boom)
    assert resource_guard.maybe_quantize(sentinel) is sentinel
    assert called["n"] == 0


def test_maybe_quantize_runs_when_enabled(monkeypatch):
    monkeypatch.setenv("QUANTIZE_CPU", "1")
    monkeypatch.setattr(
        resource_guard,
        "quantize_model_if_cpu",
        lambda model: "quantized",
    )
    assert resource_guard.maybe_quantize(object()) == "quantized"


def test_quantize_model_soft_fails(monkeypatch):
    import types

    fake_torch = types.ModuleType("torch")

    class Boom:
        def __getattr__(self, name):
            raise RuntimeError("boom")

    fake_torch.ao = Boom()
    fake_torch.quantization = Boom()
    fake_torch.nn = types.SimpleNamespace(Linear=object)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    sentinel = object()
    assert resource_guard.quantize_model_if_cpu(sentinel) is sentinel


def test_probs_to_emotions_none_and_dominant():
    id2label = {0: "Joy", 1: "Anger", 2: "None"}
    detected, dominant = sentiment_inference._probs_to_emotions([0.1, 0.1, 0.1], id2label)
    assert detected == ["Без емоцій"]
    assert dominant == "Без емоцій"


def test_probs_to_emotions_detects_joy(monkeypatch):
    monkeypatch.setitem(sentiment.EMOTION_THRESHOLDS, "Joy", 0.3)
    id2label = {0: "Joy", 1: "Anger"}
    detected, dominant = sentiment_inference._probs_to_emotions([0.8, 0.1], id2label)
    assert "Радість" in detected
    assert dominant == "Радість"


def test_classify_emotions_batch_empty():
    assert sentiment.classify_emotions_batch([]) == []
    assert sentiment.classify_emotions_batch(None) == []  # type: ignore[arg-type]


def test_classify_emotions_batch_with_mock_model(monkeypatch):
    class FakeConfig:
        id2label = {0: "Joy", 1: "Anger", 2: "None"}

    class FakeModel:
        config = FakeConfig()

        def __call__(self, **inputs):
            logits = torch.tensor([[5.0, -2.0, -2.0], [-2.0, 5.0, -2.0]])
            return SimpleNamespace(logits=logits)

    class FakeTokenizer:
        def __call__(self, batch, **kwargs):
            return {"input_ids": torch.tensor([[1, 2], [3, 4]])}

    monkeypatch.setattr(
        sentiment_inference,
        "_get_emotions_model",
        lambda: (FakeTokenizer(), FakeModel(), torch),
    )
    monkeypatch.setitem(sentiment.EMOTION_THRESHOLDS, "Joy", 0.5)
    monkeypatch.setitem(sentiment.EMOTION_THRESHOLDS, "Anger", 0.5)

    results = sentiment.classify_emotions_batch(["радість", "гнів"])
    assert len(results) == 2
    detected0, dominant0 = results[0]
    assert "Радість" in detected0
    assert dominant0 == "Радість"


def test_load_emotions_model_respects_ram_gate(monkeypatch):
    monkeypatch.setattr(
        sentiment_models,
        "require_ram_for_transformers",
        lambda step, min_mb=None: (_ for _ in ()).throw(
            NLPAnalysisError("low ram", step=step)
        ),
    )
    with pytest.raises(NLPAnalysisError) as exc_info:
        sentiment.load_emotions_model()
    assert exc_info.value.step == "emotions_load"
