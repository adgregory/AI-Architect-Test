"""Text embeddings."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from app.core.factories import default_embedding_service
from app.core.lazy import Lazy


@runtime_checkable
class EmbeddingService(Protocol):
    """Swappable embedding model. Vectors are L2-normalised (cosine == dot product)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


class FastEmbedEmbeddingService:
    """fastembed TextEmbedding on ONNX Runtime, no torch (chosen in spike 03). Model injected."""

    def __init__(self, model: Any, query_prefix: str = "", document_prefix: str = ""):
        self._model = model
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix

    @staticmethod
    def _normalize(vectors) -> np.ndarray:
        arr = np.asarray(list(vectors), dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._normalize(self._model.embed([self._document_prefix + t for t in texts])).tolist()

    def embed_query(self, query: str) -> list[float]:
        return self._normalize(self._model.embed([self._query_prefix + query]))[0].tolist()


class SentenceTransformerEmbeddingService:
    """sentence-transformers model (fallback). Model injected."""

    def __init__(self, model: Any, query_prefix: str = "", document_prefix: str = ""):
        self._model = model
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode([self._document_prefix + t for t in texts], normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self._model.encode([self._query_prefix + query], normalize_embeddings=True)[0].tolist()


# --------------------------------------------------------------------------- #
# Functional API (backwards compatible), backed by the embedding service the
# factory builds from Settings — loaded once, lazily, never per call.
# --------------------------------------------------------------------------- #
model = Lazy(default_embedding_service)


class _ModuleEmbeddings:
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        return model.get().embed_documents(texts)

    def get_query_embedding(self, query: str) -> list[float]:
        return model.get().embed_query(query)


_module_embeddings = _ModuleEmbeddings()
get_embeddings = _module_embeddings.get_embeddings
get_query_embedding = _module_embeddings.get_query_embedding
