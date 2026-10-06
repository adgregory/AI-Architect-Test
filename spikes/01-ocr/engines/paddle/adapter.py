"""PaddleOCR 3.x, native PaddlePaddle inference (PP-OCRv5 mobile det + English rec)."""

from __future__ import annotations

import os

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from common.boxes import quad_to_box, words_from_pieces  # noqa: E402
from common.engine import OCRResult, TextElement  # noqa: E402

DET_MODEL = "PP-OCRv5_mobile_det"
REC_MODEL = "en_PP-OCRv5_mobile_rec"
# Spike finding: on Apple Silicon CPU, batch 1 is fastest (~3.3s/page vs ~4.6s at 6, ~5.5s at 32).
REC_BATCH_SIZE = 1


class PaddleEngine:
    name = "paddle"

    def __init__(self, device: str):
        if device != "cpu":
            raise ValueError("PaddlePaddle GPU builds require CUDA; only CPU is available on macOS")
        self.device = device
        self._ocr = None

    def load(self) -> None:
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            text_detection_model_name=DET_MODEL,
            text_recognition_model_name=REC_MODEL,
            text_recognition_batch_size=REC_BATCH_SIZE,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            return_word_box=True,
            device="cpu",
        )

    def ocr(self, image: Image.Image) -> OCRResult:
        bgr = np.asarray(image)[:, :, ::-1]
        res = self._ocr.predict(bgr)[0]
        lines, words = [], []
        for text, poly, pieces, regions in zip(
            res["rec_texts"], res["rec_polys"], res["text_word"], res["text_word_region"]
        ):
            poly = [[float(x), float(y)] for x, y in poly]
            lines.append(TextElement(text, quad_to_box(poly), polygon=poly))
            words.extend(words_from_pieces([(p, quad_to_box(r)) for p, r in zip(pieces, regions)]))
        return OCRResult(text="\n".join(l.text for l in lines), elements={"line": lines, "word": words})

    def info(self) -> dict:
        import paddle
        import paddleocr

        return {
            "engine_version": f"paddleocr {paddleocr.__version__}, paddlepaddle {paddle.__version__}",
            "models": f"{DET_MODEL} + {REC_MODEL}, rec batch {REC_BATCH_SIZE}; doc orientation/unwarping/textline cls off",
            "onnx": False,
            "onnx_note": "Native Paddle inference. Same models via ONNX Runtime are benchmarked as paddle-onnx.",
            "device_used": "cpu",
            "positional_output": "Line-level quadrilaterals (rec_polys) and axis-aligned boxes (rec_boxes); with "
                                 "return_word_box=True, per-line pieces (text_word) with quads (text_word_region) "
                                 "where spaces/punctuation are separate pieces — merged here into words. Pixel "
                                 "space of the input image, origin top-left.",
        }

    def gpu_memory_bytes(self) -> int | None:
        return None


def create(device: str) -> PaddleEngine:
    return PaddleEngine(device)
