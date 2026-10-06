from dataclasses import asdict

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile

from app.api.deps import get_app_settings, get_container, get_extraction_session
from app.api.uploads import parse_names, read_pdf_upload, temporary_pdf
from app.core.config import Settings
from app.core.container import Container
from app.core.logging import get_logger
from app.models.schemas import ExtractionResponse
from app.services.extraction_service import ExtractionResultBuilder, ExtractionSession
from app.services.ocr_service import OCRResult
from app.storage import JobArtifacts, document_id_for

router = APIRouter()
log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Pipeline steps. The route calls the injected session through these module-level
# names — the provided API tests patch exactly these names, so they are kept as
# the route's seams.
# --------------------------------------------------------------------------- #
def extract_text_from_pdf(pdf_path: str, session: ExtractionSession) -> str:
    return session.extract_text(pdf_path)


def find_name_bounding_boxes(pdf_path: str, text: str, session: ExtractionSession) -> list[dict]:
    return session.find_name_boxes(pdf_path, text)


def fuzzy_match_names(extracted: list[str], query: list[dict], session: ExtractionSession) -> list[dict]:
    return session.match(extracted, query)


async def index_extracted_document(container: Container, document_id: str, filename: str, ocr: OCRResult) -> None:
    """After the response: store the OCR result and start the same indexing workflow the async
    jobs use, so documents processed here are also answerable by /api/ask. Best effort — a
    failure here never affects the extraction response."""
    try:
        if container.orchestrator is None:
            raise RuntimeError("orchestrator not configured")
        art = JobArtifacts(container.storage, document_id)
        art.put_json(art.page(0), asdict(ocr))  # the whole document as one page: indexing needs text only
        await container.orchestrator.start_indexing(document_id, filename, page_count=1)
        log.info("extract.index_scheduled", document_id=document_id)
    except Exception as exc:  # noqa: BLE001 - background, best effort
        log.warning("extract.index_failed", document_id=document_id, error=str(exc))


@router.post("/extract", response_model=ExtractionResponse)
def extract_names_from_pdf(
    background: BackgroundTasks,
    pdf_file: UploadFile = File(...),
    names: str = Form(...),
    session: ExtractionSession = Depends(get_extraction_session),
    settings: Settings = Depends(get_app_settings),
    container: Container = Depends(get_container),
):
    """Extract person names and their bounding boxes from a scanned PDF and fuzzy-match
    them (≥ 90%) against the requested name pairs. The document is also indexed for
    /api/ask in the background, after the response is sent."""
    content = read_pdf_upload(pdf_file, settings.max_upload_bytes)
    query_names = parse_names(names)

    with temporary_pdf(content) as path:
        text = extract_text_from_pdf(path, session)
        name_boxes = find_name_bounding_boxes(path, text, session)
        matches = fuzzy_match_names([nb["name"] for nb in name_boxes], [q.model_dump() for q in query_names], session)
        ocr = session.cached(path)

    log.info("extract.done", names=len(name_boxes), matches=len(matches), query_names=len(query_names))
    if settings.index_on_extract and container.orchestrator is not None and ocr is not None:
        background.add_task(
            index_extracted_document, container, document_id_for(content), pdf_file.filename or "document.pdf", ocr
        )
    return ExtractionResultBuilder.build(name_boxes, matches)
