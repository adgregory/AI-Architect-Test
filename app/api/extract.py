import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import TypeAdapter, ValidationError

from app.models.schemas import ExtractionResponse, NamePair
from app.services.ocr_service import extract_text_from_pdf
from app.services.bbox_service import find_name_bounding_boxes
from app.services.fuzzy_service import fuzzy_match_names

router = APIRouter()

PDF_SIGNATURE = b"%PDF-"
_NAME_PAIRS = TypeAdapter(list[NamePair])


def _read_pdf_upload(pdf_file: UploadFile) -> bytes:
    """Reject anything that is not a PDF by filename and file signature."""
    filename = (pdf_file.filename or "").lower()
    content = pdf_file.file.read()
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
):
    """Extract names from PDF and perform fuzzy matching."""
    content = _read_pdf_upload(pdf_file)
    query_names = _parse_names(names)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(content)
    tmp.close()

    try:
        text = extract_text_from_pdf(tmp.name)

        name_boxes = find_name_bounding_boxes(tmp.name, text)

        extracted_name_strings = [nb["name"] for nb in name_boxes]
        matches = fuzzy_match_names(
            extracted_name_strings, [q.model_dump() for q in query_names]
        )

        return {
            "extracted_names": [
                {
                    "name": nb["name"],
                    "bounding_box": {
                        "page_number": nb["page"],
                        "x": nb["x"],
                        "y": nb["y"],
                        "width": nb["width"],
                        "height": nb["height"],
                    },
                }
                for nb in name_boxes
            ],
            "fuzzy_matches": matches,
        }
    finally:
        os.unlink(tmp.name)
