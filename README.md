# ML-TechTest

> **Start here: read [TASK.md](TASK.md) for full instructions.** The repo contains a partially implemented application with bugs and missing features. Your job is to fix, refactor, and complete it.

## Running the solution

> Submission notes: **[overview slides (PDF)](docs/presentation/pdf-name-extractor-overview.pdf)** ·
> [DESIGN.md](DESIGN.md) (design) · [TASK.md](TASK.md#findings-report) (findings tables) ·
> [docs/adr/0001-architecture.md](docs/adr/0001-architecture.md) (decisions) · [spikes/](spikes/) (model benchmarks).

**Requirements:** Docker with Compose v2. No cloud account or API key is needed.

```bash
docker compose up -d --build        # postgres, temporal (+UI), qdrant, migrations, api, agent, workers
docker compose ps                   # wait until api and agent are healthy (the first build bakes the models in)
```

| URL | What |
|---|---|
| http://localhost:8000/docs | API (Swagger UI) |
| http://localhost:8233 | Temporal UI: async jobs and their workflows |
| http://localhost:6333/dashboard | Qdrant |

```bash
# 1. Extract names + bounding boxes and fuzzy-match (≥ 90%); the PDF is also indexed for /api/ask
#    in the background, so allow a few seconds before asking about it
curl -s -X POST localhost:8000/api/extract \
  -F "pdf_file=@sample_pdfs/meeting_minutes.pdf;type=application/pdf" \
  -F 'names=[{"first_name":"Richard","last_name":"Hernandez"}]'

# 2. Ask a question (RAG over everything extracted or ingested)
curl -s -X POST localhost:8000/api/ask -H 'content-type: application/json' \
  -d '{"question":"Who introduced the NovaTech partnership proposal?"}'

# Async job for large PDFs: 202 + job id, then stream the result
curl -s -X POST localhost:8000/api/jobs -F "pdf_file=@sample_pdfs/research_report.pdf;type=application/pdf" -F 'names=[]'
curl -sN localhost:8000/api/jobs/<job_id>/events

# Index without extracting; health
curl -s -X POST localhost:8000/api/ingest -F "pdf_file=@sample_pdfs/company_memo.pdf;type=application/pdf"
curl -s localhost:8000/health

docker compose down                 # add -v to drop the data volumes
```

**LLM for /api/ask.** Without credentials the agent answers with the retrieved sentence closest to the
question, marked `[extract — no LLM configured]`. Retrieval and sources work the same either way.
For generated answers with Gemini on Vertex AI, run `gcloud auth application-default login` and start
with `GOOGLE_CLOUD_PROJECT=<project> docker compose -f docker-compose.yml -f docker-compose.gcp.yml up -d --build`.
See [.env.example](.env.example) for every setting.

**Tests and tooling** use [uv](https://docs.astral.sh/uv/), which replaces `requirements.txt` and the
venv steps in TASK.md. [go-task](https://taskfile.dev) wraps the common commands (`task --list`).

```bash
uv sync --all-extras && uv run python scripts/prepare_models.py   # or: task setup
uv run pytest                                                     # or: task test (task test:fast: no models needed)
```

## Task Description:
The task is to build a system that can extract names and last names from a scanned
PDF document, identify their bounding box locations, and provide an API endpoint to perform fuzzy
matching between the extracted names and the names provided in the request.

Create a Branch and make a pull request to this repository.

## Requirements:

1. Use an open-source OCR library to extract text from the scanned PDF document.
2. Implement a named entity recognition (NER) or a large language model (LLM) approach to
identify and extract names and last names from the extracted text.
3. Parse the extracted text and OCR results to find the bounding box coordinates of the identified
names and last names.
4. Create a RESTful API using FastAPI or Flask that accepts a scanned PDF file and a set of
name-last name pairs as input.
5. Perform fuzzy matching (with a similarity threshold of 90%) between the extracted names and
the provided names in the API request.
6. Return the extracted names, their bounding box coordinates, and the fuzzy matching results as a
JSON response.
7. Implement a vector database (e.g., Chroma, Qdrant) to store the extracted text from the PDF
documents.
8. Use an open-source embedding model (e.g., Sentence-BERT, Universal Sentence Encoder) to
convert the extracted text into vector representations and store them in the vector database.
9. Create another RESTful API endpoint that accepts a question as input and returns an answer
based on the RAG strategy.
10. Create a Docker container image for the entire application, including the text extraction, named
entity recognition, bounding box parsing, vector database, and RAG components. Ensure that
the container can be built and run using Docker commands or a Docker Compose file. Include
instructions for running the containerized application and accessing the API endpoints.
11. Include a DESIGN.md document (max 3 pages) that covers:
    a. **Component diagram** of your solution showing all services, databases, and external dependencies and how they connect (ASCII, Mermaid, or image).
    b. **Sequence diagram** showing the request flow for both API endpoints: (1) PDF upload with name extraction and fuzzy matching, and (2) RAG question-answering (ASCII, Mermaid, or image).
    c. Justify your choice of OCR library, NER approach, embedding model, and vector database.
    d. Explain tradeoffs you considered (accuracy vs speed, cost vs quality).
    e. Describe how you would scale this system to handle 1000+ PDFs per hour.
    f. Identify potential failure modes and how you would mitigate them.
    g. If you were to add LLM routing (selecting different models based on query complexity),
       how would you implement it?

## Test Steps:

1. **Set up the Development Environment:**
   - Create a new Python virtual environment and install the required libraries (e.g., PyMuPDF, Tesseract OCR, spaCy, Hugging Face Transformers, FastAPI or Flask).
   - Obtain a sample scanned PDF document containing names and last names for testing purposes.

2. **Implement Text Extraction from PDF:**
   - Use PyMuPDF or a similar library to extract text from the scanned PDF document.
   - Optionally, you can preprocess the extracted text (e.g., remove newlines, clean up formatting) for better NER or LLM performance.

3. **Implement Named Entity Recognition (NER) or LLM Approach:**
   - Use spaCy's pre-trained NER model or a custom NER model trained on a relevant dataset to identify names and last names in the extracted text.
   - Alternatively, use a pre-trained NER (e.g., BERT, RoBERTa) from the Hugging Face Transformers library for the named entity recognition task. It is easier to use a Generative LLM for zero-shot NER and return the response as a usable JSON.

4. **Parse Bounding Box Coordinates:**
   - Integrate the OCR library (e.g., Tesseract OCR) to obtain the bounding box coordinates of the recognized words in the PDF document.
   - Match the extracted names and last names with their corresponding bounding box coordinates from the OCR results.

5. **Implement the RESTful API:**
   - Use FastAPI or Flask to create a RESTful API endpoint that accepts a scanned PDF file and a set of name-last name pairs as input.
   - Perform the text extraction, named entity recognition, and bounding box parsing steps on the uploaded PDF file.
   - Implement fuzzy matching (e.g., using the python-fuzzy library) between the extracted names and the provided names in the API request, with a similarity threshold of 90%.
   - Return the extracted names, their bounding box coordinates, and the fuzzy matching results as a JSON response.

6. **Set up a Vector Database:**
   - Choose and install an open-source vector database solution (e.g., Chroma, Qdrant, Milvus) compatible with your development environment. If you are using a docker image of an open-source database, make sure to include it in the docker compose file.
   - Create a new database or collection within the vector database to store the extracted text from the PDF documents.

7. **Implement Text Embedding:**
   - Use an open-source embedding model like Sentence-BERT or Universal Sentence Encoder to convert the extracted text into vector representations.
   - Integrate the embedding model with the vector database to store the text embeddings along with their corresponding text snippets.

8. **Implement Retrieval-Augmented Generation (RAG):**
   - Create a new RESTful API endpoint that accepts a question as input.
   - Implement the RAG strategy to retrieve relevant text snippets from the vector database based on the similarity between the question embedding and the stored text embeddings.
   - Use a pre-trained language model (e.g., GPT-3.5, Gemini) to generate an answer based on the retrieved text snippets and the input question.
   - Return the generated answer as the response to the API request.

9. **Containerization:**
   - Create a new Dockerfile in the project directory with all the required dependencies to run the API server.
   - If required, generate a docker-compose file.

10. **Documentation and Submission:**
    - Document the code, and any external libraries or resources used.
    - Provide instructions for setting up and running the system, as well as any additional dependencies or requirements.
