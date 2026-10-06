"""Composition root for heavy models and clients.

Everything expensive to build (ML models, network clients) is constructed here,
once per process, and injected into services — services never build their own.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from app.core.config import Settings, get_settings

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    from spacy.language import Language


@lru_cache
def spacy_model(name: str) -> "Language":
    import spacy

    return spacy.load(name)


@lru_cache
def sentence_transformer(name: str) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


@lru_cache
def qdrant_client(host: str, port: int, timeout_s: float) -> "QdrantClient":
    from qdrant_client import QdrantClient

    return QdrantClient(host=host, port=port, timeout=timeout_s)


def default_spacy_model(settings: Settings | None = None) -> "Language":
    return spacy_model((settings or get_settings()).spacy_model)


def default_sentence_transformer(settings: Settings | None = None) -> "SentenceTransformer":
    return sentence_transformer((settings or get_settings()).embedding_model)


def default_qdrant_client(settings: Settings | None = None) -> "QdrantClient":
    s = settings or get_settings()
    return qdrant_client(s.qdrant_host, s.qdrant_port, s.qdrant_timeout_s)
