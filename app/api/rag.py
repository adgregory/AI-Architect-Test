import os
import tempfile

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.deps import get_container, get_rag_service
from app.api.extract import read_pdf_upload
from app.core.container import Container
from app.core.logging import get_logger
from app.models.schemas import IngestResponse, RAGRequest, RAGResponse
from app.services.rag_service import RAGService

router = APIRouter()
log = get_logger(__name__)


# Seam kept for the provided API tests, which patch this name.
def generate_answer(question: str, rag: RAGService) -> dict:
    return rag.answer(question)


@router.post("/ingest", response_model=IngestResponse)
def ingest_pdf(pdf_file: UploadFile = File(...), container: Container = Depends(get_container)):
    """OCR a PDF, chunk it, embed the chunks and store them in the vector database."""
    settings = container.settings
    content = read_pdf_upload(pdf_file, settings.max_upload_mb * 2**20)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(content)
    tmp.close()

    try:
        text = container.ocr.read(tmp.name).text
        chunks = [c for c in container.chunker.chunk(text, settings.chunk_size) if c.strip()]
        store = container.vector_store
        store.ensure_collection()
        document_id = store.upsert(chunks, container.embeddings.embed_documents(chunks),
                                   metadata=[{"source": pdf_file.filename}] * len(chunks))
        log.info("ingest.done", document_id=document_id, chunks=len(chunks))
        return {"status": "success", "document_id": document_id, "chunks_stored": len(chunks)}
    finally:
        os.unlink(tmp.name)


@router.post("/ask", response_model=RAGResponse)
def ask_question(request: RAGRequest, rag: RAGService = Depends(get_rag_service)):
    """Answer a question from the ingested documents (retrieve top-k chunks, then generate)."""
    return generate_answer(request.question, rag)
