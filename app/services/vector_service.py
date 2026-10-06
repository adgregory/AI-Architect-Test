import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.services.embedding_service import get_embeddings


client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "pdf_documents"
VECTOR_SIZE = 384


def init_collection():
    """Create the vector collection if it doesn't exist."""
    collections = client.get_collections().collections
    existing = [c.name for c in collections]

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )


def store_document_chunks(
    chunks: list[str], metadata: list[dict] = None, document_id: str | None = None
) -> str:
    """Store text chunks with their embeddings in the vector database.

    Point IDs are UUID5(document_id, chunk index): unique across documents, and
    stable for a document, so re-ingesting it overwrites its own points instead
    of duplicating them. Returns the document ID.
    """
    document_id = document_id or str(uuid.uuid4())
    embeddings = get_embeddings(chunks)

    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        payload = {"text": chunk, "document_id": document_id, "chunk_index": i}
        if metadata and i < len(metadata):
            payload.update(metadata[i])

        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{i}")),
                vector=embedding,
                payload=payload,
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return document_id


def search_similar(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Search for similar text chunks."""
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        score_threshold=0.5,
    )

    return [
        {
            "text": hit.payload["text"],
            "score": hit.score,
        }
        for hit in results
    ]
