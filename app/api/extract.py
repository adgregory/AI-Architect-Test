import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import TypeAdapter, ValidationError

from app.api.deps import get_app_settings, get_extraction_session
from app.core.config import Settings
from app.core.logging import get_logger
from app.models.schemas import ExtractionResponse, NamePair
from app.services.extraction_service import ExtractionResultBuilder, ExtractionSession

router = APIRouter()
log = get_logger(__name__)

PDF_SIGNATURE = b"%PDF-"
_NAME_PAIRS = TypeAdapter(list[NamePair])


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


def read_pdf_upload(pdf_file: UploadFile, max_bytes: int) -> bytes:
    """Reject anything that is not a PDF (filename and file signature) or is too large."""
    filename = (pdf_file.filename or "").lower()
    content = pdf_file.file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"PDF exceeds the {max_bytes // 2**20} MB upload limit")
    if not filename.endswith(".pdf") or not content.startswith(PDF_SIGNATURE):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file must be a PDF document")
    return content


def _parse_names(names: str) -> list[NamePair]:
    try:
        return _NAME_PAIRS.validate_json(names)
    except ValidationError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "names must be a JSON list of {first_name, last_name} objects",
        ) from exc


@router.post("/extract", response_model=ExtractionResponse)
def extract_names_from_pdf(
    pdf_file: UploadFile = File(...),
    names: str = Form(...),
    session: ExtractionSession = Depends(get_extraction_session),
    settings: Settings = Depends(get_app_settings),
):
    """Extract person names and their bounding boxes from a scanned PDF and fuzzy-match
    them (≥ 90%) against the requested name pairs."""
    content = read_pdf_upload(pdf_file, settings.max_upload_mb * 2**20)
    query_names = _parse_names(names)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(content)
    tmp.close()

    try:
        text = extract_text_from_pdf(tmp.name, session)
        name_boxes = find_name_bounding_boxes(tmp.name, text, session)
        matches = fuzzy_match_names(
            [nb["name"] for nb in name_boxes], [q.model_dump() for q in query_names], session
        )
        log.info("extract.done", names=len(name_boxes), matches=len(matches), query_names=len(query_names))

        return ExtractionResultBuilder.build(name_boxes, matches)
    finally:
        os.unlink(tmp.name)
