"""Vector store for document chunks."""

from __future__ import annotations

import uuid
from typing import Any, Protocol, runtime_checkable

from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import get_settings
from app.core.factories import default_qdrant_client
from app.core.lazy import Lazy
from app.services.embedding_service import get_embeddings

_settings = get_settings()
COLLECTION_NAME = _settings.qdrant_collection
VECTOR_SIZE = _settings.embedding_dim


@runtime_checkable
class VectorStore(Protocol):
    """Swappable vector backend (Qdrant today; hybrid or other engines later)."""

    def ensure_collection(self) -> None: ...

    def upsert(
        self,
        texts: list[str],
        vectors: list[list[float]],
        document_id: str | None = None,
        metadata: list[dict] | None = None,
    ) -> str: ...

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]: ...


class QdrantVectorStore:
    """Qdrant collection with cosine distance. The client is injected."""

    def __init__(
        self,
        client: Any,
        collection: str = COLLECTION_NAME,
        vector_size: int = VECTOR_SIZE,
        score_threshold: float = _settings.retrieval_score_threshold,
    ):
        self._client = client
        self._collection = collection
        self._vector_size = vector_size
        self._score_threshold = score_threshold

    def ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
            )

    def upsert(self, texts, vectors, document_id=None, metadata=None) -> str:
        """Point IDs are UUID5(document_id, chunk index): unique across documents and stable
        for a document, so re-ingesting it overwrites its own points instead of duplicating."""
        document_id = document_id or str(uuid.uuid4())
        points = []
        for i, (text, vector) in enumerate(zip(texts, vectors, strict=True)):
            payload = {"text": text, "document_id": document_id, "chunk_index": i}
            if metadata and i < len(metadata):
                payload.update(metadata[i])
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{i}")),
                    vector=vector,
                    payload=payload,
                )
            )
        self._client.upsert(collection_name=self._collection, points=points)
        return document_id

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=self._score_threshold,
        )
        return [{"text": hit.payload["text"], "score": hit.score} for hit in hits]


# --------------------------------------------------------------------------- #
# Functional API (backwards compatible). `client` is a lazy proxy (no connection
# at import); the adapter resolves `client` and `get_embeddings` at call time.
# --------------------------------------------------------------------------- #
client = Lazy(default_qdrant_client)


class _ModuleVectorStore:
    def init_collection(self) -> None:
        QdrantVectorStore(client).ensure_collection()

    def store_document_chunks(
        self, chunks: list[str], metadata: list[dict] | None = None, document_id: str | None = None
    ) -> str:
        return QdrantVectorStore(client).upsert(chunks, get_embeddings(chunks), document_id, metadata)

    def search_similar(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        return QdrantVectorStore(client).search(query_embedding, top_k)


_module_store = _ModuleVectorStore()
init_collection = _module_store.init_collection
store_document_chunks = _module_store.store_document_chunks
search_similar = _module_store.search_similar
