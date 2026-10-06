"""Text embeddings."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.core.factories import default_sentence_transformer
from app.core.lazy import Lazy


@runtime_checkable
class EmbeddingService(Protocol):
    """Swappable embedding model. Vectors are L2-normalised (cosine == dot product)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


class SentenceTransformerEmbeddingService:
    """sentence-transformers model, injected. Optional prefixes for models that need them."""

    def __init__(self, model: Any, query_prefix: str = "", document_prefix: str = ""):
        self._model = model
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode([self._document_prefix + t for t in texts], normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self._model.encode([self._query_prefix + query], normalize_embeddings=True)[0].tolist()


# --------------------------------------------------------------------------- #
# Functional API (backwards compatible). The model is loaded once, lazily, by
# the composition root — not on every call.
# --------------------------------------------------------------------------- #
model = Lazy(default_sentence_transformer)
_default_embeddings = SentenceTransformerEmbeddingService(model)

get_embeddings = _default_embeddings.embed_documents
get_query_embedding = _default_embeddings.embed_query
