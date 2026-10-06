"""Docling (standard pipeline: layout + table structure) with EasyOCR, its default OCR here.

Docling >= 2.1xx defaults to OcrAutoOptions, which on macOS tries ocrmac → nemotron →
rapidocr(onnxruntime) → easyocr. None of the first three are installed in this env, so
auto resolves to EasyOCR; it is set explicitly so the result doesn't depend on that.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # models are prefetched; avoid hub calls in timings

from PIL import Image  # noqa: E402

from common.engine import OCRResult, TextElement  # noqa: E402

ARTIFACTS = Path.home() / ".cache/docling/models"
EASYOCR_MODELS = Path.home() / ".EasyOCR/model"


class DoclingEngine:
    name = "docling"

    def __init__(self, device: str):
        if device not in ("cpu", "mps"):
            raise ValueError("device must be cpu or mps")
        self.device = device
        self._conv = None

    def load(self) -> None:
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
        from docling.document_converter import DocumentConverter, ImageFormatOption

        opts = PdfPipelineOptions(
            artifacts_path=ARTIFACTS,
            ocr_options=EasyOcrOptions(model_storage_directory=str(EASYOCR_MODELS)),
            accelerator_options=AcceleratorOptions(
                device=AcceleratorDevice.MPS if self.device == "mps" else AcceleratorDevice.CPU
            ),
        )
        self._conv = DocumentConverter(format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)})
        self._conv.initialize_pipeline(InputFormat.IMAGE)

    def ocr(self, image: Image.Image) -> OCRResult:
        from docling.datamodel.base_models import DocumentStream

        buf = io.BytesIO()
        image.save(buf, "PNG")
        buf.seek(0)
        result = self._conv.convert(DocumentStream(name="page.png", stream=buf))
        doc = result.document
        page_size = next(iter(doc.pages.values())).size
        scale = image.width / page_size.width  # docling treats an image page as 72 DPI

        def to_box(bbox) -> list[float]:
            b = bbox.to_top_left_origin(page_height=page_size.height)
            return [b.l * scale, b.t * scale, b.r * scale, b.b * scale]

        blocks = [TextElement(item.text, to_box(item.prov[0].bbox)) for item in doc.texts if item.prov]
        segments = []
        for page in result.pages:
            if page.predictions.layout is None:
                continue
            for cluster in page.predictions.layout.clusters:
                for cell in cluster.cells:
                    segments.append(TextElement(cell.text, to_box(cell.rect.to_bounding_box()),
                                                confidence=cell.confidence))
        return OCRResult(text=doc.export_to_text(), elements={"segment": segments, "block": blocks})

    def info(self) -> dict:
        from importlib.metadata import version

        import torch

        pipeline = next(iter(self._conv.initialized_pipelines.values()))
        reader = getattr(pipeline.ocr_model, "reader", None)
        reader_device = str(getattr(reader, "device", None))
        return {
            "engine_version": f"docling {version('docling')}, easyocr {version('easyocr')}, torch {torch.__version__}",
            "models": "docling-layout-heron + TableFormer (docling-models) + EasyOCR craft_mlt_25k / latin_g2 "
                      "(lang en,es,fr,de — Docling's EasyOCR default)",
            "onnx": False,
            "onnx_note": "Default pipeline is PyTorch end to end. Docling also ships an ONNX layout model "
                         "(docling-layout-heron-onnx) and can use RapidOCR on onnxruntime — not the default.",
            "device_used": f"accelerator={self.device}, easyocr reader device={reader_device}",
            "positional_output": "Layout items (doc.texts) with prov bbox in page coordinates (image treated as "
                                 "72 DPI, origin BOTTOMLEFT) — here a block per paragraph; underlying OCR cells "
                                 "(page.predictions.layout.clusters[].cells) are EasyOCR phrase segments with "
                                 "TOPLEFT rects and confidence. No word-level boxes.",
        }

    def gpu_memory_bytes(self) -> int | None:
        if self.device != "mps":
            return None
        import torch

        return torch.mps.driver_allocated_memory()


def create(device: str) -> DoclingEngine:
    return DoclingEngine(device)
