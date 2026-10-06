"""Composition root: factories that build services from Settings.

Each interface has a factory with a registry of builders keyed by engine name
(`Settings.*_engine`). Builders import their libraries lazily, so an install with
only the `chosen` extra never imports a fallback library; selecting an engine
whose extra isn't installed fails fast with a clear message. Heavy objects
(models, clients) are constructed here — services only receive them.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ClassVar

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from spacy.language import Language

    from app.services.embedding_service import EmbeddingService
    from app.services.ner_service import NERService
    from app.services.ocr_service import OCRService
    from app.services.rag_service import LLMClient
    from app.services.vector_service import VectorStore

log = get_logger(__name__)


class MissingEngineError(RuntimeError):
    """The selected engine's optional dependencies are not installed."""


def _require(module: str, extra: str, engine: str) -> None:
    import importlib.util

    if importlib.util.find_spec(module) is None:
        raise MissingEngineError(f"engine '{engine}' needs the `{extra}` extra (uv sync --extra {extra})")


# --------------------------------------------------------------------------- #
# Cached model/client constructors (one instance per process and arguments)
# --------------------------------------------------------------------------- #
@lru_cache
def spacy_model(name: str) -> "Language":
    _require("spacy", "fallback", "spacy")
    import spacy

    return spacy.load(name)


@lru_cache
def sentence_transformer(name: str):
    _require("sentence_transformers", "fallback", "sentence-transformers")
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


@lru_cache
def fastembed_model(name: str, cache_dir: str):
    _require("fastembed", "chosen", "fastembed")
    from fastembed import TextEmbedding

    return TextEmbedding(name, cache_dir=cache_dir)


@lru_cache
def rapidocr_engine(model_dir: str):
    _require("rapidocr", "chosen", "rapidocr")
    from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR

    # Same models and settings measured in spike 01. Models are downloaded into MODELS_DIR
    # (not into site-packages) so images can bake them and run as a non-root user.
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    return RapidOCR(params={
        "Global.use_cls": False,
        "Global.log_level": "warning",
        "Global.model_root_dir": model_dir,
        "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE, "Det.lang_type": LangDet.CH,
        "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE, "Rec.lang_type": LangRec.EN,
    })


def gliner_onnx_dir(settings: Settings) -> Path:
    return Path(settings.models_dir) / settings.gliner_model.replace("/", "--")


def ensure_gliner_onnx(settings: Settings) -> Path:
    """Export GLiNER to ONNX (and dynamic int8) once; later runs reuse the artifacts."""
    out = gliner_onnx_dir(settings)
    needed = out / ("model_quantized.onnx" if settings.gliner_quantized else "model.onnx")
    if not needed.exists():
        from gliner import GLiNER

        log.info("gliner.export.start", model=settings.gliner_model, out=str(out),
                 quantized=settings.gliner_quantized)
        GLiNER.from_pretrained(settings.gliner_model).export_to_onnx(out, quantize=settings.gliner_quantized)
        if settings.gliner_quantized:
            (out / "model.onnx").unlink(missing_ok=True)  # only the int8 model is served (~580 MB saved)
        log.info("gliner.export.done", out=str(out))
    return needed


@lru_cache
def gliner_model(settings: Settings):
    _require("gliner", "chosen", "gliner")
    from gliner import GLiNER

    onnx_file = ensure_gliner_onnx(settings)
    return GLiNER.from_pretrained(str(onnx_file.parent), load_onnx_model=True, onnx_model_file=onnx_file.name)


@lru_cache
def qdrant_client(host: str, port: int, timeout_s: float) -> "QdrantClient":
    from qdrant_client import QdrantClient

    return QdrantClient(host=host, port=port, timeout=timeout_s)


# --------------------------------------------------------------------------- #
# Service factories
# --------------------------------------------------------------------------- #
class OCRServiceFactory:
    @staticmethod
    def _rapidocr(s: Settings) -> "OCRService":
        from app.services.ocr_service import RapidOCRService

        engine = rapidocr_engine(str(Path(s.models_dir) / "rapidocr"))
        return RapidOCRService(engine, dpi=s.ocr_dpi, tighten_boxes=s.ocr_tighten_boxes)

    @staticmethod
    def _tesseract(s: Settings) -> "OCRService":
        _require("pytesseract", "fallback", "tesseract")
        from app.services.ocr_service import TesseractOCRService

        return TesseractOCRService(dpi=s.ocr_dpi)

    builders: ClassVar[dict[str, Callable[[Settings], "OCRService"]]] = {
        "rapidocr": _rapidocr, "tesseract": _tesseract,
    }

    @classmethod
    def create(cls, settings: Settings) -> "OCRService":
        return cls.builders[settings.ocr_engine](settings)


class NERServiceFactory:
    @staticmethod
    def _gliner(s: Settings) -> "NERService":
        from app.services.ner_service import GLiNERNERService

        return GLiNERNERService(gliner_model(s), threshold=s.gliner_threshold)

    @staticmethod
    def _spacy(s: Settings) -> "NERService":
        from app.services.ner_service import SpacyNERService

        return SpacyNERService(spacy_model(s.spacy_model))

    builders: ClassVar[dict[str, Callable[[Settings], "NERService"]]] = {"gliner": _gliner, "spacy": _spacy}

    @classmethod
    def create(cls, settings: Settings) -> "NERService":
        return cls.builders[settings.ner_engine](settings)


class EmbeddingServiceFactory:
    @staticmethod
    def _fastembed(s: Settings) -> "EmbeddingService":
        from app.services.embedding_service import FastEmbedEmbeddingService

        model = fastembed_model(s.embedding_model, str(Path(s.models_dir) / "fastembed"))
        return FastEmbedEmbeddingService(model, s.embedding_query_prefix, s.embedding_document_prefix)

    @staticmethod
    def _sentence_transformers(s: Settings) -> "EmbeddingService":
        from app.services.embedding_service import SentenceTransformerEmbeddingService

        model = sentence_transformer(s.embedding_model)
        return SentenceTransformerEmbeddingService(model, s.embedding_query_prefix, s.embedding_document_prefix)

    builders: ClassVar[dict[str, Callable[[Settings], "EmbeddingService"]]] = {
        "fastembed": _fastembed, "sentence-transformers": _sentence_transformers,
    }

    @classmethod
    def create(cls, settings: Settings) -> "EmbeddingService":
        return cls.builders[settings.embedding_engine](settings)


class VectorStoreFactory:
    @staticmethod
    def create(settings: Settings) -> "VectorStore":
        from app.services.vector_service import QdrantVectorStore

        return QdrantVectorStore(
            default_qdrant_client(settings), collection=settings.qdrant_collection,
            vector_size=settings.embedding_dim, score_threshold=settings.retrieval_score_threshold,
        )


class LLMClientFactory:
    @staticmethod
    def create(settings: Settings) -> "LLMClient":
        from app.services.rag_service import OpenAICompatibleChatClient

        key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
        return OpenAICompatibleChatClient(key, settings.llm_base_url, settings.llm_model, settings.llm_timeout_s)


# --------------------------------------------------------------------------- #
# Defaults used by the backwards-compatible functional API
# --------------------------------------------------------------------------- #
def default_spacy_model(settings: Settings | None = None) -> "Language":
    return spacy_model((settings or get_settings()).spacy_model)


def default_embedding_service(settings: Settings | None = None) -> "EmbeddingService":
    return EmbeddingServiceFactory.create(settings or get_settings())


def default_qdrant_client(settings: Settings | None = None) -> "QdrantClient":
    s = settings or get_settings()
    return qdrant_client(s.qdrant_host, s.qdrant_port, s.qdrant_timeout_s)
