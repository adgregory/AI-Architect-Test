# Baseline Test Failures

Snapshot of the test suite **before any application code was changed**. Every row is now
resolved; each one feeds the findings tables in `TASK.md`.

- **Commit:** `f2e9ac9` (uv migration + `numpy<2` pin, app code untouched)
- **Environment:** Python 3.12.13, Tesseract 5.5.3 (Homebrew), macOS arm64
- **Command:** `uv run pytest -v`
- **Result:** 30 failed, 19 passed, 0 errors

> Environment issues fixed before this snapshot (not app bugs):
> NumPy 2 ABI mismatch with spaCy 3.7 (`fbf3763`), and Tesseract not installed.

"Suspected cause" is a first reading of the code, not a decision on the fix.

**Current status (2026-10-05): all 30 resolved — 49 passed, 0 failed.**

## 1. Bugs

| # | Test | Symptom | Suspected cause | Status |
|---|------|---------|-----------------|--------|
| B1 | `test_ocr.py::TestExtractTextFromPDF::test_extracts_text_from_all_pages` | One page missing from extracted text | `ocr_service.extract_text_from_pdf` iterates `range(1, len(doc))`, skipping page 0 | fixed in `0cc6b52` |
| B2 | `test_ocr.py::TestExtractTextFromPDF::test_document_is_properly_closed` | `doc.close()` never called | PDF documents opened with `fitz.open` are never closed (resource leak) | fixed in `0cc6b52` |
| B3 | `test_ocr.py::TestGetWordBoundingBoxes::test_coordinates_in_pdf_space` | X coordinate in 150 DPI image space, not 72 DPI PDF space | `get_word_bounding_boxes` renders at `dpi=150` and returns raw pixel coords without scaling by 72/150 | fixed in `0cc6b52` |
| B4 | `test_ner.py::TestExtractNames::test_extracts_only_person_entities` | ORG entities returned as names | `ner_service` keeps `ent.label_ in ("PERSON", "ORG")` instead of PERSON only | fixed in `b6a7c2e` |
| B5 | `test_ner.py::TestExtractNames::test_handles_multiple_entity_types` | `['Sarah Williams', 'MIT']` returned | Same as B4 | fixed in `b6a7c2e` |
| B6 | `test_ner.py::TestExtractNamesWithPositions::test_returns_positions_for_person_only` | Non-PERSON entities with positions | Same as B4, in the positions variant | fixed in `b6a7c2e` |
| B7 | `test_bbox.py::TestFindNameBoundingBoxes::test_case_insensitive_matching` | `John` not matched to `JOHN` | Word comparison in `bbox_service` is case-sensitive | fixed in `0b97a82` |
| B8 | `test_fuzzy.py::TestFuzzyMatchNames::test_threshold_is_90_percent` | Threshold is 70, expected 90 | `SIMILARITY_THRESHOLD = 70` in `fuzzy_service` | fixed in `7bee72e` |
| B9 | `test_fuzzy.py::TestFuzzyMatchNames::test_close_match_above_threshold` | `John Smth` vs `John Smith` scores 0.89 | `fuzz.partial_ratio` is the wrong scorer for full-name comparison | fixed in `7bee72e` |
| B10 | `test_rag.py::TestGenerateAnswer::test_prompt_includes_question` | Literal `{question}` in prompt | `rag_service` prompt f-string uses escaped `{{question}}`, so it renders literally | fixed in `4e5235a` |
| B11 | `test_vector.py::TestStoreDocumentChunks::test_generates_unique_ids` | Point IDs `[0, 1]` reused across documents | Point IDs derived from chunk index; second document overwrites the first | fixed in `f5fcb60` |

### Integration tests caused by the bugs above

| # | Test | Symptom | Root cause | Status |
|---|------|---------|------------|--------|
| I1 | `test_integration.py::TestOCRWithSamplePDFs::test_memo_extracts_text_from_single_page` | Empty text for single-page memo | B1 | fixed in `0cc6b52` |
| I2 | `test_integration.py::TestOCRWithSamplePDFs::test_memo_contains_key_words` | `MEMORANDUM` missing | B1 | fixed in `0cc6b52` |
| I3 | `test_integration.py::TestOCRWithSamplePDFs::test_minutes_extracts_both_pages` | `Richard Hernandez` (page 1) missing | B1 | fixed in `0cc6b52` |
| I4 | `test_integration.py::TestNERWithSamplePDFs::test_memo_finds_person_names` | 0 names extracted | B1 (no text → no names) | fixed in `0cc6b52` |
| I5 | `test_integration.py::TestNERWithSamplePDFs::test_ner_does_not_return_organizations` | `Acme Corporation`, `Google` returned | B4 | fixed in `b6a7c2e` |
| I6 | `test_integration.py::TestEndToEndExtraction::test_extract_endpoint_with_memo_pdf` | No names extracted end-to-end | B1 (possibly B4 too) | fixed in `0cc6b52` |

## 2. Missing features

| # | Test | Symptom | Needed | Status |
|---|------|---------|--------|--------|
| F1 | `test_api.py::TestExtractEndpoint::test_rejects_non_pdf_files` | `fitz.FileDataError` raised on non-PDF upload | Validate upload type/content and return a 4xx | fixed in `c9e70ee` |
| F2 | `test_api.py::TestExtractEndpoint::test_response_includes_fuzzy_matches` | No `fuzzy_matches` in response | Return fuzzy match results from `/extract` | fixed in `c9e70ee` |
| F3 | `test_api.py::TestExtractEndpoint::test_response_includes_page_number` | Bounding box lacks `page_number` | Add page number to bounding box schema | fixed in `c9e70ee` |
| F4 | `test_api.py::TestRAGEndpoints::test_ask_returns_sources` | No `sources` in RAG response | Return retrieved chunks with the answer | fixed in `c9e70ee` |
| F5 | `test_api.py::TestRAGEndpoints::test_health_endpoint_exists` | `/health` returns 404 | Add a health endpoint | fixed in `e4af849` |

## 3. Architecture

| # | Test | Symptom | Needed | Status |
|---|------|---------|--------|--------|
| A1 | `test_architecture.py::TestOOPDesign::test_services_use_classes` | 0 service files use classes | Implement services as classes | fixed in `6cc7e78` |
| A2 | `test_architecture.py::TestOOPDesign::test_no_bare_module_level_functions_in_services` | Bare functions in `vector_service`, `rag_service`, … | Move logic into service classes | fixed in `6cc7e78` |
| A3 | `test_architecture.py::TestAbstractions::test_services_implement_protocols_or_abcs` | No Protocols/ABCs | Interfaces for OCR, NER, embedding, vector store | fixed in `6cc7e78` |
| A4 | `test_architecture.py::TestAbstractions::test_embedding_service_has_interface` | No classes in `embedding_service.py` | `EmbeddingService` interface + implementation | fixed in `6cc7e78` |
| A5 | `test_architecture.py::TestAbstractions::test_vector_service_has_interface` | No classes in `vector_service.py` | `VectorStore` interface + `QdrantVectorStore` | fixed in `6cc7e78` |
| A6 | `test_architecture.py::TestAbstractions::test_ocr_service_has_interface` | No classes in `ocr_service.py` | `OCRService` interface + `TesseractOCRService` | fixed in `6cc7e78` |
| A7 | `test_architecture.py::TestDependencyInjection::test_no_hardcoded_model_loading_in_functions` | `SentenceTransformer` created inside `get_embeddings` / `get_query_embedding` | Load the model once and inject it | fixed in `6cc7e78` |
| A8 | `test_architecture.py::TestDependencyInjection::test_no_global_client_instantiation` | Global clients in `vector_service.py`, `ner_service.py` | Manage lifecycle via dependency injection | fixed in `6cc7e78` |

## Passing at baseline (19)

The remaining 19 tests passed at this snapshot, including
`test_bbox.py::test_merges_multiword_name_boxes` and
`test_bbox.py::test_handles_duplicate_names`, which only failed while Tesseract
was missing.
