# Spike 01 — OCR results

Host: macOS-26.5-arm64-arm-64bit, Python 3.12.13. 150 pages × 3 reps per configuration (accuracy from rep 0; latency over all reps, after warm-up).

## Accuracy (all pages)

| Config | CER | WER | Name recall | Name recall (case-insens.) | Name IoU word (mean / ≥0.5) | Name IoU line (mean / ≥0.5) | Name IoU segment (mean / ≥0.5) | Name IoU block (mean / ≥0.5) |
|---|---|---|---|---|---|---|---|---|
| tesseract-cpu | 14.9% | 24.5% | 76.3% | 76.6% | 0.64 / 67.2% | 0.21 / 8.4% | — | 0.05 / 0.0% |
| paddle-cpu | 1.5% | 7.0% | 90.1% | 90.1% | 0.50 / 63.8% | 0.19 / 3.1% | — | — |
| paddle-onnx-cpu | 2.4% | 9.1% | 87.9% | 88.2% | 0.44 / 38.9% | 0.15 / 1.8% | — | — |
| paddle-onnx-coreml | 2.4% | 9.1% | 87.9% | 88.2% | 0.44 / 38.9% | 0.15 / 1.8% | — | — |
| docling-cpu | 35.7% | 47.4% | 59.8% | 60.0% | — | — | 0.15 / 7.5% | 0.04 / 0.5% |
| docling-mps | 35.7% | 47.4% | 59.8% | 60.0% | — | — | 0.15 / 7.5% | 0.04 / 0.5% |


Name IoU: best overlap between each ground-truth name box and the engine's box for it at that granularity (union of matching words, or the containing line/segment/block). A name that isn't recognised scores 0.


## CER by degradation profile

| Config | clean | noise | skew | blur | jpeg | low_dpi | scan |
|---|---|---|---|---|---|---|---|
| tesseract-cpu | 0.0% | 3.5% | 0.3% | 8.7% | 0.3% | 4.6% | 43.1% |
| paddle-cpu | 0.0% | 0.9% | 0.0% | 0.4% | 0.2% | 0.2% | 4.1% |
| paddle-onnx-cpu | 0.0% | 0.9% | 0.1% | 0.4% | 0.2% | 0.2% | 7.1% |
| paddle-onnx-coreml | 0.0% | 0.9% | 0.1% | 0.4% | 0.2% | 0.2% | 7.1% |
| docling-cpu | 5.1% | 33.4% | 9.5% | 63.8% | 10.0% | 24.5% | 59.8% |
| docling-mps | 5.1% | 33.4% | 9.5% | 63.8% | 10.0% | 24.5% | 59.8% |


## Name recall by degradation profile

| Config | clean | noise | skew | blur | jpeg | low_dpi | scan |
|---|---|---|---|---|---|---|---|
| tesseract-cpu | 100.0% | 89.2% | 99.4% | 66.5% | 98.3% | 73.7% | 46.2% |
| paddle-cpu | 100.0% | 96.6% | 100.0% | 93.8% | 99.2% | 98.6% | 72.1% |
| paddle-onnx-cpu | 100.0% | 96.0% | 100.0% | 94.7% | 97.7% | 97.1% | 65.6% |
| paddle-onnx-coreml | 100.0% | 96.0% | 100.0% | 94.7% | 97.7% | 97.1% | 65.6% |
| docling-cpu | 97.5% | 59.2% | 95.8% | 33.3% | 83.9% | 64.9% | 31.7% |
| docling-mps | 97.5% | 59.2% | 95.8% | 33.3% | 83.9% | 64.9% | 31.7% |


## Latency per page (ms, warm)

| Config | mean | p50 | p90 | p95 | p99 | max | pages/min | cold start (s, median) |
|---|---|---|---|---|---|---|---|---|
| tesseract-cpu | 326.1 | 330.3 | 400.1 | 409.3 | 416.3 | 430.9 | 183.7 | 0.37 |
| paddle-cpu | 3183.8 | 3180.1 | 3631.0 | 3729.0 | 3799.6 | 3834.4 | 18.8 | 4.322 |
| paddle-onnx-cpu | 395.2 | 386.6 | 474.4 | 489.4 | 518.5 | 539.2 | 151.6 | 0.841 |
| paddle-onnx-coreml | 3319.4 | 3192.8 | 4290.0 | 4606.7 | 6186.2 | 6778.5 | 18.1 | 10.342 |
| docling-cpu | 6311.3 | 6144.8 | 8317.8 | 8759.6 | 10295.7 | 12863.7 | 9.5 | 12.374 |
| docling-mps | 1451.8 | 1363.7 | 1941.1 | 2137.5 | 3032.9 | 3287.5 | 41.3 | 6.454 |


## Resources during the warm run

| Config | CPU % mean (100 = 1 core) | CPU % peak | RSS peak (MB) | GPU mem peak (MB) | GPU active % (powermetrics) | GPU power mW |
|---|---|---|---|---|---|---|
| tesseract-cpu | 89.6 | 119.9 | 1306.0 | — | — | — |
| paddle-cpu | 102.3 | 107.9 | 2735.5 | — | — | — |
| paddle-onnx-cpu | 840.2 | 986.4 | 2462.8 | — | — | — |
| paddle-onnx-coreml | 102.4 | 148.5 | 9728.4 | — | 14.9 | 661.2 |
| docling-cpu | 144.4 | 424.2 | 13308.1 | — | — | — |
| docling-mps | 79.3 | 169.9 | 2422.0 | 12427.6 | 71.3 | 7856.5 |


## Engine details

### tesseract-cpu

- **Version:** tesseract 5.5.3
- **Models:** eng.traineddata (LSTM, --oem 1), --psm 3
- **ONNX:** False — Tesseract's LSTM models use its own traineddata format; no ONNX export/runtime.
- **Device used:** cpu
- **Positional output:** Axis-aligned boxes at block, paragraph, line and word level (image_to_data), pixel space of the input image, origin top-left, with per-word confidence.

### paddle-cpu

- **Version:** paddleocr 3.4.1, paddlepaddle 3.3.1
- **Models:** PP-OCRv5_mobile_det + en_PP-OCRv5_mobile_rec, rec batch 1; doc orientation/unwarping/textline cls off
- **ONNX:** False — Native Paddle inference. Same models via ONNX Runtime are benchmarked as paddle-onnx.
- **Device used:** cpu
- **Positional output:** Line-level quadrilaterals (rec_polys) and axis-aligned boxes (rec_boxes); with return_word_box=True, per-line pieces (text_word) with quads (text_word_region) where spaces/punctuation are separate pieces — merged here into words. Pixel space of the input image, origin top-left.

### paddle-onnx-cpu

- **Version:** rapidocr 3.9.2, onnxruntime 1.30.0
- **Models:** ch_PP-OCRv5_det_mobile + en_PP-OCRv5_rec_mobile (ONNX); cls off
- **ONNX:** True — PaddleOCR models exported to ONNX, run by ONNX Runtime.
- **Device used:** cpu (providers: ['CPUExecutionProvider'])
- **Positional output:** Line-level quadrilaterals (boxes); with return_word_box=True, per-word quads derived from CTC character positions (full line height). Pixel space of the input image, origin top-left.

### paddle-onnx-coreml

- **Version:** rapidocr 3.9.2, onnxruntime 1.30.0
- **Models:** ch_PP-OCRv5_det_mobile + en_PP-OCRv5_rec_mobile (ONNX); cls off
- **ONNX:** True — PaddleOCR models exported to ONNX, run by ONNX Runtime.
- **Device used:** coreml (providers: ['CoreMLExecutionProvider', 'CPUExecutionProvider'])
- **Positional output:** Line-level quadrilaterals (boxes); with return_word_box=True, per-word quads derived from CTC character positions (full line height). Pixel space of the input image, origin top-left.

### docling-cpu

- **Version:** docling 2.133.0, easyocr 1.7.2, torch 2.14.1
- **Models:** docling-layout-heron + TableFormer (docling-models) + EasyOCR craft_mlt_25k / latin_g2 (lang en,es,fr,de — Docling's EasyOCR default)
- **ONNX:** False — Default pipeline is PyTorch end to end. Docling also ships an ONNX layout model (docling-layout-heron-onnx) and can use RapidOCR on onnxruntime — not the default.
- **Device used:** accelerator=cpu, easyocr reader device=cpu
- **Positional output:** Layout items (doc.texts) with prov bbox in page coordinates (image treated as 72 DPI, origin BOTTOMLEFT) — here a block per paragraph; underlying OCR cells (page.predictions.layout.clusters[].cells) are EasyOCR phrase segments with TOPLEFT rects and confidence. No word-level boxes.

### docling-mps

- **Version:** docling 2.133.0, easyocr 1.7.2, torch 2.14.1
- **Models:** docling-layout-heron + TableFormer (docling-models) + EasyOCR craft_mlt_25k / latin_g2 (lang en,es,fr,de — Docling's EasyOCR default)
- **ONNX:** False — Default pipeline is PyTorch end to end. Docling also ships an ONNX layout model (docling-layout-heron-onnx) and can use RapidOCR on onnxruntime — not the default.
- **Device used:** accelerator=mps, easyocr reader device=mps
- **Positional output:** Layout items (doc.texts) with prov bbox in page coordinates (image treated as 72 DPI, origin BOTTOMLEFT) — here a block per paragraph; underlying OCR cells (page.predictions.layout.clusters[].cells) are EasyOCR phrase segments with TOPLEFT rects and confidence. No word-level boxes.
