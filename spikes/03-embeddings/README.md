# Spike 03 — Text embedding model

**Status:** planned

## Question

Which open-source English embedding model best separates text that means the
same from text that must differ (cosine similarity), and best retrieves the
right chunk of our documents for a question — at acceptable latency, vector
size and footprint?

## Candidates

| ID | Model | Dim | Notes |
|----|-------|-----|-------|
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | current app baseline |
| `mpnet` | `sentence-transformers/all-mpnet-base-v2` | 768 | classic SBERT |
| `bge-small` | `BAAI/bge-small-en-v1.5` | 384 | query instruction prefix |
| `bge-base` | `BAAI/bge-base-en-v1.5` | 768 | query instruction prefix |
| `e5-base` | `intfloat/e5-base-v2` | 768 | `query:` / `passage:` prefixes |
| `nomic` | `nomic-ai/nomic-embed-text-v1.5` | 768 (Matryoshka) | `search_query:` / `search_document:` prefixes |
| `arctic-m` | `Snowflake/snowflake-arctic-embed-m-v1.5` | 768 | query prefix |

Each model is used the way its authors specify (prefixes, normalisation).
Every model runs with sentence-transformers' PyTorch backend and its **ONNX
backend**, on CPU and on MPS (PyTorch).

## Dataset (hand-written, committed, reviewed before running)

Grounded in the sample documents' domain (memo, board minutes, research report).

### A. Graded pairs (~120)
Score 0–3: **3** paraphrase, **2** closely related, **1** same topic but a
different fact, **0** unrelated. Includes **hard negatives** that share most
words but must differ:

| Category | Example |
|----------|---------|
| Negation | "The motion was approved" / "The motion was not approved" |
| Role swap | "Robert Chen reports to James Anderson" / "James Anderson reports to Robert Chen" |
| Entity swap | "Dr. James Chen is from Stanford" / "Robert Chen is from Stanford" |
| Number change | "The NIH grant is $600K" / "The NIH grant is $750K" |
| Antonym / outcome | "Revenue increased 12%" / "Revenue decreased 12%" |

### B. Retrieval set (~30 questions)
The three documents chunked as the RAG pipeline will chunk them; each question
labelled with its gold chunk(s). Queries are asymmetric (short question vs
passage), which is the real RAG use.

## Metrics

Raw cosine values are **not comparable across models** (some, e.g. e5, score
almost every pair above 0.7), so ranking is judged with scale-free metrics and
raw values are reported for context only.

- **Spearman ρ** between cosine and the 0–3 grade.
- **ROC-AUC**, paraphrases (3) vs hard negatives.
- **Separation margin**: (mean cos of paraphrases − mean cos of hard negatives)
  ÷ pooled standard deviation; plus raw means.
- **Triplet accuracy per hard-negative category**: share of cases where
  cos(anchor, paraphrase) > cos(anchor, hard negative).
- **Retrieval**: Recall@1, Recall@3 (the app retrieves top 3), MRR@10.
- **Operational**: encode latency (p50/p95/p99 per query and per chunk batch)
  for PyTorch vs ONNX, CPU vs MPS; cold start; vector dim → Qdrant storage per
  1M chunks; max sequence length (bounds chunk size); model size on disk.

## Results

_Pending._

## Decision

_Pending._
