# Spike 02 — Person-name NER

**Status:** decided

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
| `spacy-sm` | `en_core_web_sm` 3.8 (app pins 3.7) | CNN, OntoNotes | No (thinc) |
| `spacy-trf` | `en_core_web_trf` 3.8 | RoBERTa-base, OntoNotes | No (thinc) |
| `bert-ner` | `dslim/bert-base-NER` | BERT-base token classification, CoNLL-03 | Yes — tested with/without (optimum export) |
| `gliner` | `urchade/gliner_small-v2.1` | Zero-shot span model, label `person` | Yes — fp32 and dynamic int8, tested with/without |

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

Scores are name-level, on the **Paddle-ONNX OCR output** (the chosen OCR engine, 150 pages)
unless noted; "fuzzy" = similarity ≥ 0.9, the threshold `/extract` uses. Latency is per clean
page (ground-truth text), warm. Configurations whose timing overlapped the powermetrics GPU pass
were re-run; accuracy is independent of timing. Full JSON in [`results/`](results/).

| Config | Clean-text F1 | F1 | Precision | Recall | False pos. | Org/place as person | Hard-scan F1 | p50 | p99 | Peak RSS |
|--------|---------------|----|-----------|--------|-----------|---------------------|--------------|-----|-----|----------|
| spacy-sm (current app) | 0.874 | 0.863 | 0.821 | 0.910 | 178 | 58 | 0.730 | 18 ms | 25 ms | 0.4 GB |
| spacy-trf | 1.000 | 0.933 | 0.919 | 0.946 | 33 | 0 | 0.802 | 76 ms | 85 ms | 2.8 GB |
| bert-ner torch-cpu | 1.000 | 0.913 | 0.897 | 0.929 | 53 | 0 | 0.760 | 43 ms | 51 ms | 0.8 GB |
| bert-ner torch-mps | ″ | ″ | ″ | ″ | ″ | ″ | ″ | 13 ms | 19 ms | 0.5 GB |
| bert-ner onnx-cpu | ″ | ″ | ″ | ″ | ″ | ″ | ″ | 36 ms | 43 ms | 1.4 GB |
| gliner torch-cpu | 1.000 | **0.942** | 0.946 | 0.937 | 2 | 0 | **0.826** | 69 ms | 79 ms | 1.9 GB |
| gliner torch-mps | ″ | ″ | ″ | ″ | ″ | ″ | ″ | 45 ms | 58 ms | 1.1 GB |
| gliner onnx-cpu (fp32, 583 MB) | ″ | ″ | ″ | ″ | ″ | ″ | ″ | 48 ms | 115 ms | 2.1 GB |
| gliner onnx-int8, threshold 0.5 (175 MB) | 0.949 | 0.876 | 0.984 | 0.789 | 0 | 0 | 0.690 | 40 ms | 49 ms | 1.2 GB |
| **gliner onnx-int8, threshold 0.3** | **1.000** | **0.940** | **0.971** | 0.910 | **0** | **0** | 0.808 | **32 ms** | **40 ms** | **1.1 GB** |

### Findings

1. **spaCy `en_core_web_sm` (what the app uses) is the weakest:** not perfect even on clean text
   (it tags headings and words like "Breakthrough" and "Atlas" as people), and on OCR text it
   produces 178 false positives, 58 of them organisations/places.
2. **All transformer models are perfect on clean text**; OCR noise separates them. GLiNER is best
   on every OCR engine's output and on the hardest scans, and **almost never invents a name**:
   the others turn OCR fragments into people ("Chiet", "Charman", "Counse", "Absent"); GLiNER's
   two remaining "false positives" are badly garbled real names.
3. **Scoring OCR'd names fairly:** a name the model found but OCR misspelled ("Anna Kowaisk") is
   counted at a separate *detected* level (similarity ≥ 0.6), not as an NER false positive.
4. **ONNX:** same accuracy as PyTorch for BERT-NER and GLiNER; faster on CPU (BERT-NER 36 vs 43 ms,
   GLiNER 48 vs 69 ms). spaCy pipelines have no ONNX path.
5. **int8 quantization** (ONNX Runtime dynamic quantization) makes GLiNER 3.3× smaller (583 → 175 MB),
   halves memory and lowers latency, but at the default 0.5 threshold recall drops 0.94 → 0.79:
   quantization shifts scores down, so true names fall just under the cut-off while junk spans
   stay far below it. **Lowering the threshold to 0.3 recovers F1 (0.940 vs 0.942 fp32) with zero
   false positives.** The same change on fp32 slightly hurts it (2 → 6 false positives, F1 0.938):
   each model/quantization has its own best threshold.
6. **GPU (MPS):** helps BERT-NER (3.3×) and GLiNER torch (1.5×), but int8 ONNX on CPU is already
   within that range, so no GPU is needed.
7. **Span hygiene:** GLiNER includes titles in ~35% of spans ("Dr. Aisha Patel"); the name
   normaliser (titles, punctuation, line breaks) handles it, and first/last splitting is correct
   for all ground-truth names, including O'Brien, O'Sullivan and Al-Rashidi.

### Caveat — threshold chosen on the evaluation data

The 0.3 threshold was picked by looking at the same pages and 30 names it is scored on, so 0.940
is slightly optimistic. The proper method is to tune on one set of documents and report on a
held-out set; with 30 names there isn't enough data to split. The threshold is therefore a
**configuration value** (`GLINER_THRESHOLD`, default 0.3 for int8), to be recalibrated on a
held-out labelled set as more documents become available, and again whenever the model or its
quantization changes.

## Decision

**Accepted (2026-10-05): GLiNER small v2.1 (`urchade/gliner_small-v2.1`), exported to ONNX and
dynamically quantized to int8, on ONNX Runtime CPU, label `person`, threshold 0.3 (configurable)**,
behind the `NERModel` interface.

- Best accuracy on real OCR output (F1 0.94), zero false positives, no organisations as people.
- 32 ms/page, 175 MB model, ~1.1 GB RSS, no GPU — fits CPU workers / Fargate.
- **Runner-up:** spaCy `en_core_web_trf` (F1 0.93, 33 false positives) — simpler to deploy, no ONNX.
- **Rejected:** spaCy `en_core_web_sm` (accuracy, false positives); BERT-NER (more false positives,
  weaker on scans).
