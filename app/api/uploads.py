"""Upload handling shared by every route that accepts a PDF."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import HTTPException, UploadFile, status
from pydantic import TypeAdapter, ValidationError

from app.models.schemas import NamePair

PDF_SIGNATURE = b"%PDF-"
_NAME_PAIRS = TypeAdapter(list[NamePair])


def read_pdf_upload(pdf_file: UploadFile, max_bytes: int) -> bytes:
    """Reject anything that is not a PDF (filename and file signature) or is too large."""
    filename = (pdf_file.filename or "").lower()
    content = pdf_file.file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"PDF exceeds the {max_bytes // 2**20} MB upload limit")
    if not filename.endswith(".pdf") or not content.startswith(PDF_SIGNATURE):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file must be a PDF document")
    return content


def parse_names(names: str) -> list[NamePair]:
    try:
        return _NAME_PAIRS.validate_json(names)
    except ValidationError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "names must be a JSON list of {first_name, last_name} objects"
        ) from exc


@contextmanager
def temporary_pdf(content: bytes) -> Iterator[str]:
    """A file path for libraries that need one; always removed afterwards."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
    try:
        yield tmp.name
    finally:
        os.unlink(tmp.name)
