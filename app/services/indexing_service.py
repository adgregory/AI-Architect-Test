"""Indexing for RAG: chunk → embed → store, in one place.

Used inline by /api/ingest, and split across the workflow's CPU activity (chunk + embed,
which needs the embedding model) and IO activity (store, which needs the vector store).
Documents are identified by their content (app.storage.document_id_for), so re-indexing the
same PDF overwrites its chunks.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.embedding_service import EmbeddingService
from app.services.rag_service import TextChunker
from app.services.vector_service import VectorStore


@dataclass(frozen=True)
class EmbeddedChunks:
    texts: list[str]
    vectors: list[list[float]]

    def to_dict(self) -> dict:
        return {"texts": self.texts, "vectors": self.vectors}

    @classmethod
    def from_dict(cls, data: dict) -> EmbeddedChunks:
        return cls(data["texts"], data["vectors"])


class DocumentIndexer:
    """Each process injects only what its half needs: CPU workers have no vector-store
    client, IO workers load no embedding model."""

    def __init__(
        self,
        chunker: TextChunker,
        chunk_size: int,
        embeddings: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
    ):
        self._chunker = chunker
        self._chunk_size = chunk_size
        self._embeddings = embeddings
        self._vector_store = vector_store

    def embed(self, text: str) -> EmbeddedChunks:
        if self._embeddings is None:
            raise RuntimeError("this indexer has no embedding model")
        texts = [c for c in self._chunker.chunk(text, self._chunk_size) if c.strip()]
        return EmbeddedChunks(texts, self._embeddings.embed_documents(texts))

    def store(self, document_id: str, chunks: EmbeddedChunks, source: str) -> int:
        if self._vector_store is None:
            raise RuntimeError("this indexer has no vector store")
        if not chunks.texts:
            return 0
        self._vector_store.ensure_collection()
        self._vector_store.upsert(
            chunks.texts, chunks.vectors, document_id=document_id, metadata=[{"source": source}] * len(chunks.texts)
        )
        return len(chunks.texts)

    def index(self, document_id: str, text: str, source: str) -> int:
        return self.store(document_id, self.embed(text), source)
