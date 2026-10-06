import json
import os
import tempfile

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import StreamingResponse

from app.api.deps import get_container, get_rag_service
from app.api.extract import read_pdf_upload
from app.core.container import Container
from app.core.logging import get_logger
from app.models.schemas import IngestResponse, RAGRequest, RAGResponse

router = APIRouter()
log = get_logger(__name__)


# Seam kept for the provided API tests, which patch this name. `rag` is the configured
# answering backend (agent service client or in-process RAGService).
def generate_answer(question: str, rag) -> dict:
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
def ask_question(request: RAGRequest, rag=Depends(get_rag_service)):
    """Answer a question from the ingested documents (retrieve top-k chunks, then generate)."""
    return generate_answer(request.question, rag)


@router.post("/ask/stream")
async def ask_question_stream(request: RAGRequest, rag=Depends(get_rag_service)) -> StreamingResponse:
    """Server-sent events: `sources`, then `token`s as the model generates, then `done`.

    Passes the agent service's stream through; the backend keeps the API contract, auth and
    rate limiting. The in-process backend emits its full answer as a single token."""

    async def events():
        if hasattr(rag, "stream"):
            async for event in rag.stream(request.question):
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
        else:
            result = rag.answer(request.question)
            for event in ({"type": "sources", "sources": result["sources"]},
                          {"type": "token", "text": result["answer"]},
                          {"type": "done", **result, "cached": False}):
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
