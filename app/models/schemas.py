from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Box in PDF coordinate space (points, origin top-left of the page)."""

    page_number: int = Field(ge=0, description="Zero-based page index")
    x: float
    y: float
    width: float
    height: float


class ExtractedName(BaseModel):
    name: str
    bounding_box: BoundingBox


class FuzzyMatch(BaseModel):
    extracted_name: str
    matched_name: str
    score: float = Field(ge=0.0, le=1.0)


class ExtractionResponse(BaseModel):
    extracted_names: list[ExtractedName]
    fuzzy_matches: list[FuzzyMatch]


class NamePair(BaseModel):
    first_name: str = Field(min_length=1)
    last_name: str = Field(min_length=1)


class RAGRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class RAGResponse(BaseModel):
    answer: str
    sources: list[str]


class IngestResponse(BaseModel):
    status: str
    document_id: str
    chunks_stored: int


class HealthResponse(BaseModel):
    status: str
