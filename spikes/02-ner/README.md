# Spike 02 — Person-name NER

**Status:** planned

## Question

Which **local, pre-trained** NER model should find person names in OCR'd text,
given accuracy on clean and noisy (OCR) input, false positives on
organisations/places, first/last-name splitting, latency, footprint and ONNX?

## Why a pre-trained model, not an LLM call

| | Pre-trained NER (local) | LLM API call |
|---|---|---|
| Latency | ms per page on CPU | 0.5–several s per call |
| Cost | none per page | per-token, grows with volume |
| PII | text never leaves the service | names sent to a third party |
| Determinism | same input → same output | varies; needs JSON validation/retries |
| Availability | no external dependency | rate limits, outages, timeouts |
| Fit | PERSON detection is exactly what these models are trained for | generality mostly unused here |

An LLM stays a possible *fallback* for low-confidence documents (see
DESIGN.md routing), not the default path. No LLM is benchmarked here.

## Candidates

| ID | Model | Type | ONNX |
|----|-------|------|------|
| `spacy-sm` | `en_core_web_sm` 3.7 (current app) | CNN, OntoNotes | No (thinc) |
| `spacy-trf` | `en_core_web_trf` 3.7 | RoBERTa-base, OntoNotes | No (thinc) |
| `bert-ner` | `dslim/bert-base-NER` | BERT-base token classification, CoNLL-03 | Yes — tested with/without (optimum export) |
| `gliner` | `urchade/gliner_small-v2.1` | Zero-shot span model, label `person` | Yes — tested with/without |

Transformer candidates run on CPU and MPS; spaCy runs on CPU.

## Inputs

1. **Ground-truth text** — exact page text from the spike 01 dataset (upper bound).
2. **Real OCR output** — the per-page OCR text saved by spike 01
   (`spikes/01-ocr/results/raw/*.jsonl`), so NER is measured on the errors it
   will actually see (all 150 pages × each OCR configuration).

## Ground truth

The generator's expected-name lists (30 names, 3 documents), located in each
page's text: every mention, its character span, and its first/last split.
Negatives worth tracking are listed explicitly: Acme Corporation, NovaTech
Solutions, ETH Zurich, Stanford University, NIH, NSF, DARPA, AWS, London,
Atlas, Forbes, NeurIPS, ICML.

## Metrics

- **Name-level precision / recall / F1** per page (set of distinct names,
  titles like Dr./Prof. and trailing punctuation stripped, case-insensitive).
- **Partial matches** reported separately (e.g. only `Sarah` or `Williams`).
- **False-positive breakdown**: tracked organisations/places predicted as person.
- **First/last split accuracy** using the splitting rule the app would use.
- **Latency** per page: p50 / p90 / p95 / p99, cold start; CPU and MPS.
- **Resources and footprint**: peak RSS, model size on disk, ONNX vs native.

## Results

_Pending._

## Decision

_Pending._
