"""Settings, factories (registry selection, missing extras) and the lazy proxy."""

import importlib.util

import pytest

from app.core import factories
from app.core.config import Settings
from app.core.factories import MissingEngineError, NERServiceFactory, OCRServiceFactory
from app.core.lazy import Lazy


class TestSettings:
    def test_defaults_are_the_spike_choices(self):
        s = Settings(_env_file=None)
        assert (s.ocr_engine, s.ner_engine, s.embedding_engine) == ("rapidocr", "gliner", "fastembed")
        assert s.embedding_model == "BAAI/bge-small-en-v1.5" and s.embedding_dim == 384
        assert s.similarity_threshold == 90 and s.gliner_threshold == 0.3

    def test_environment_overrides(self, monkeypatch):
        monkeypatch.setenv("OCR_ENGINE", "tesseract")
        monkeypatch.setenv("QDRANT_PORT", "7000")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        s = Settings(_env_file=None)
        assert s.ocr_engine == "tesseract" and s.qdrant_port == 7000
        assert s.llm_api_key.get_secret_value() == "sk-test"
        assert "sk-test" not in repr(s)  # secrets never print

    def test_rejects_unknown_engines_and_bad_values(self):
        with pytest.raises(ValueError):
            Settings(_env_file=None, ner_engine="regex")
        with pytest.raises(ValueError):
            Settings(_env_file=None, similarity_threshold=150)

    def test_settings_are_immutable_and_hashable(self):
        s = Settings(_env_file=None)
        with pytest.raises(ValueError):
            s.ocr_dpi = 300
        assert hash(s) == hash(Settings(_env_file=None))


class TestFactories:
    def test_registry_selects_the_configured_builder(self, monkeypatch):
        calls = []
        monkeypatch.setitem(OCRServiceFactory.builders, "tesseract", lambda s: calls.append(s.ocr_dpi) or "ocr")
        assert OCRServiceFactory.create(Settings(_env_file=None, ocr_engine="tesseract", ocr_dpi=200)) == "ocr"
        assert calls == [200]

    def test_every_engine_literal_has_a_builder(self):
        assert set(OCRServiceFactory.builders) == {"rapidocr", "tesseract"}
        assert set(NERServiceFactory.builders) == {"gliner", "spacy"}
        assert set(factories.EmbeddingServiceFactory.builders) == {"fastembed", "sentence-transformers"}

    def test_missing_extra_fails_fast_with_guidance(self, monkeypatch):
        real_find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util, "find_spec", lambda name, *a: None if name == "pytesseract" else real_find_spec(name, *a)
        )
        with pytest.raises(MissingEngineError, match="fallback"):
            OCRServiceFactory.create(Settings(_env_file=None, ocr_engine="tesseract"))


class TestLazy:
    def test_builds_once_on_first_use_and_forwards(self):
        built = []
        lazy = Lazy(lambda: built.append(1) or (lambda x: x * 2))
        assert built == []
        assert lazy(21) == 42 and lazy(1) == 2
        assert built == [1]

    def test_forwards_attributes(self):
        assert Lazy(lambda: "abc").upper() == "ABC"
