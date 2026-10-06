from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import StreamingResponse

from app.api.deps import get_container, get_rag_service
from app.api.sse import SSE_HEADERS, sse_event
from app.api.uploads import read_pdf_upload, temporary_pdf
from app.core.container import Container
from app.core.logging import get_logger
from app.models.schemas import IngestResponse, RAGRequest, RAGResponse
from app.storage import document_id_for

router = APIRouter()
log = get_logger(__name__)


# Seam kept for the provided API tests, which patch this name. `rag` is the configured
# answering backend (agent service client or in-process RAGService).
def generate_answer(question: str, rag) -> dict:
    return rag.answer(question)


@router.post("/ingest", response_model=IngestResponse)
def ingest_pdf(pdf_file: UploadFile = File(...), container: Container = Depends(get_container)):
    """OCR a PDF and index it for /api/ask (chunk, embed, store) — synchronously.
    The document ID is content-addressed: ingesting the same PDF again overwrites it."""
    content = read_pdf_upload(pdf_file, container.settings.max_upload_bytes)
    with temporary_pdf(content) as path:
        text = container.ocr.read(path).text
    document_id = document_id_for(content)
    chunks_stored = container.indexer.index(document_id, text, source=pdf_file.filename or "document.pdf")
    log.info("ingest.done", document_id=document_id, chunks=chunks_stored)
    return {"status": "success", "document_id": document_id, "chunks_stored": chunks_stored}


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
                yield sse_event(event["type"], event)
        else:
            result = rag.answer(request.question)
            for event in (
                {"type": "sources", "sources": result["sources"]},
                {"type": "token", "text": result["answer"]},
                {"type": "done", **result, "cached": False},
            ):
                yield sse_event(event["type"], event)

    return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)
