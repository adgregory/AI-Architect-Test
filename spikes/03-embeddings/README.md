# Spike 03 — Text embedding model

**Status:** results in — recommendation awaiting decision; dataset pending user review

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

7 models; quality measured on PyTorch CPU (identical on MPS and ONNX — verified for every model
that loaded on all three). Dataset: 113 graded pairs (35 hard-negative triplets + 43 graded
pairs) and 30 questions over 33 chunks. Full JSON in [`results/`](results/).

### Quality

| Model | Dim | Spearman ρ | AUC para vs hard-neg | Margin (SD) | Triplet acc. | Recall@1 | **Recall@3** | MRR@10 |
|-------|-----|-----------|----------------------|-------------|--------------|----------|------------|--------|
| minilm (current app) | 384 | 0.28 | 0.39 | −0.30 | 37% | 0.83 | 0.93 | 0.89 |
| mpnet | 768 | 0.45 | 0.57 | 0.56 | 57% | 0.77 | 0.93 | 0.85 |
| **bge-small** | 384 | 0.45 | 0.58 | 0.35 | 54% | **0.87** | **1.00** | **0.92** |
| bge-base | 768 | **0.50** | **0.63** | 0.51 | 60% | 0.83 | 0.97 | 0.89 |
| e5-base | 768 | 0.42 | 0.55 | 0.21 | **63%** | 0.87 | 0.97 | 0.92 |
| nomic | 768 | 0.17 | 0.28 | −0.27 | 23% | 0.87 | 0.93 | 0.90 |
| arctic-m | 768 | 0.13 | 0.24 | −0.79 | 26% | 0.87 | 0.97 | 0.92 |

### Triplet accuracy by hard-negative category (paraphrase scored above the hard negative)

| Model | Negation | Antonym | Entity swap | Number change | Role swap |
|-------|----------|---------|-------------|---------------|-----------|
| minilm | 57% | 57% | 71% | 0% | 0% |
| mpnet | 86% | 86% | 71% | 43% | 0% |
| bge-small | 86% | 71% | 86% | 29% | 0% |
| bge-base | **100%** | **86%** | 71% | 43% | 0% |
| e5-base | 71% | **86%** | **100%** | **57%** | 0% |
| nomic | 14% | 14% | 86% | 0% | 0% |
| arctic-m | 14% | 29% | 71% | 0% | 0% |

### Speed and footprint (query = one question; batch = all 33 chunks)

| Model | Query p50 torch-cpu / mps / **onnx-cpu** | Query p99 onnx-cpu | Batch p50 onnx-cpu / mps | Weights | Max tokens |
|-------|------------------------------------------|--------------------|--------------------------|---------|------------|
| minilm | 2.5 / 3.0 / **1.2 ms** | 1.5 ms | 47 / 14 ms | 86 MB | 256 |
| mpnet | 11.5 / 7.2 / **4.5 ms** | 5.5 ms | 254 / 60 ms | 416 MB | 384 |
| **bge-small** | 4.8 / 5.4 / **2.4 ms** | 3.4 ms | 88 / 24 ms | 127 MB | 512 |
| bge-base | 11.3 / 6.3 / **4.5 ms** | 5.1 ms | 256 / 56 ms | 416 MB | 512 |
| e5-base | 11.3 / 6.3 / **4.1 ms** | 5.0 ms | 242 / 56 ms | 416 MB | 512 |
| nomic | 13.6 / 5.3 / **5.5 ms** | 6.3 ms | 318 / 75 ms | 522 MB | 8192 |
| arctic-m | 11.4 / 6.4 / — (ONNX failed to load) | — | — / 56 ms | 415 MB | 512 |

Cold start (fresh process, model load + first query) is 4–7 s for every model and backend.

### Findings

1. **Hard negatives are the weak spot of every model.** Best AUC is 0.63 (bge-base); MiniLM,
   Nomic and Arctic-M score hard negatives *above* paraphrases on average (AUC < 0.5).
   **Role swaps fail for every model (0%)** — "A reports to B" vs "B reports to A" share all
   tokens, which a single sentence vector can't order. Number changes barely move vectors
   (0–57%). The "expand vs shrink" item fails for all models (cos 0.84–0.97 to the negative).
2. **Retrieval is good for every model** (Recall@3 ≥ 0.93): a polarity or number flip rarely
   matters for *finding* the right chunk — the LLM reading the chunk decides the fact.
   **bge-small is the only model with perfect Recall@3** and ties for the best Recall@1/MRR.
3. **Raw cosine is not comparable across models.** Mean cosine of *unrelated* pairs ranges from
   0.02 (MiniLM) to 0.68 (e5-base); paraphrases from 0.83 to 0.96. Any cosine threshold (semantic
   cache, "no relevant context" cut-off) must be calibrated per model.
4. **Small chunks retrieve badly.** The most-missed question (q-27, "Who reviewed the annual
   research report?", missed by four models) has its answer in a two-line chunk
   ("Reviewed by: Margaret Thompson, CEO / Date: …"). Chunking should merge very short blocks
   with their neighbours and carry the document title/section into each chunk.
5. **ONNX is the fastest single-query path for every model** (2–2.6× over PyTorch CPU, faster than
   MPS for single queries). For **batch** encoding (ingestion), MPS is 3.5–4.5× faster than ONNX CPU;
   ONNX's batch gain over PyTorch CPU is small (≈ 10–20%).
6. **Small models don't benefit from MPS** for single queries (MiniLM, bge-small are slower on MPS
   than CPU); base-size models do (1.6–2.5×).
7. **Arctic-M on ONNX** doesn't load through sentence-transformers out of the box (its repo config
   passes `add_pooling_layer`, which the ONNX loader rejects) — a packaging issue, not a model one.
8. **Nomic's weak pair results are not a prefix artefact:** re-run with its `clustering:` prefix
   it scored slightly worse (AUC 0.25 vs 0.28).

### Caveats

- **Small, hand-labelled dataset:** 30 questions (one question = 3.3 points of Recall@3) and 113
  pairs written and graded by the author; pending user review. Differences of one or two
  questions between models are within noise; the hard-negative pattern (role swap 0% everywhere,
  AUC < 0.65 everywhere) is robust.
- **Domain:** corporate memo / minutes / research-report English only.

## Recommendation (awaiting decision)

**`BAAI/bge-small-en-v1.5` on ONNX Runtime CPU**, behind the `EmbeddingModel` interface, with the
BGE query instruction on questions and no prefix on chunks:

- Best retrieval (Recall@3 1.00, MRR 0.92) — the job the embeddings actually do in RAG.
- 2.4 ms per query on CPU, 127 MB weights, 384 dims (half the Qdrant storage and memory of the
  768-dim models), 512-token context (comfortably above our chunk size).
- Hard-negative handling is middle of the pack (AUC 0.58); **no model** is good at it, so the
  design compensates explicitly rather than by picking a bigger embedding:
  - the LLM reads retrieved chunks and decides polarity, numbers and roles;
  - the semantic cache gets polarity / entity / number guards (spike 04);
  - a cross-encoder reranker (e.g. `bge-reranker-base`) is the upgrade path if precision on
    near-duplicate passages ever matters more than latency.
- **Alternative:** `bge-base-en-v1.5` (best hard-negative AUC 0.63, 100% on negation) if
  sentence-level discrimination matters more than retrieval — at 2× the latency, 2× the vector
  size and slightly lower Recall@3 (0.97).
- **Rejected:** MiniLM (current; weakest on hard negatives), Nomic and Arctic-M (score hard
  negatives above paraphrases), MPNet (lower Recall@1, no advantage over bge-base).
