# Spikes

Short, throwaway experiments that inform technology decisions before they are
made in `app/`. Each spike answers one question with evidence (numbers,
outputs, observations) and ends with a recommendation that feeds `DESIGN.md`.

Spike code is not production code: it is not imported by `app/` and is not
part of the test suite.

## Layout

```
spikes/
└── NN-short-name/
    ├── README.md   # question, options compared, method, results, decision
    └── *.py        # runnable experiment(s)
```

Run a spike with the project environment:

```bash
uv run python spikes/NN-short-name/<script>.py
```

## Index

| # | Question | Status | Decision |
|---|----------|--------|----------|
| 01 | [OCR engine selection](01-ocr/README.md): Tesseract vs PaddleOCR (native/ONNX) vs Docling | done (sizes pending) | PaddleOCR PP-OCRv5 mobile on ONNX Runtime CPU |
| 02 | [Person-name NER](02-ner/README.md): spaCy sm/trf vs BERT-NER vs GLiNER (local, no LLM call) | planned | — |
| 03 | [Text embeddings](03-embeddings/README.md): 7 English models, semantic separation + retrieval | planned | — |
| 04 | Semantic answer cache: Qdrant similarity threshold, calibrated on spike 03 pairs | planned | — |
