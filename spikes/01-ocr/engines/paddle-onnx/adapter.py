"""PaddleOCR models on ONNX Runtime via RapidOCR (PP-OCRv5 mobile det + English rec).

Same models as the native `paddle` engine, so differences isolate the runtime.
Devices: cpu (CPUExecutionProvider) and coreml (CoreMLExecutionProvider: Apple GPU / ANE).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from common.boxes import quad_to_box, words_from_pieces
from common.engine import OCRResult, TextElement


class PaddleOnnxEngine:
    name = "paddle-onnx"

    def __init__(self, device: str):
        if device not in ("cpu", "coreml"):
            raise ValueError("device must be cpu or coreml")
        self.device = device
        self._ocr = None

    def load(self) -> None:
        from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR

        self._ocr = RapidOCR(params={
            "Global.use_cls": False,  # parity with native paddle (textline orientation off)
            "Global.log_level": "warning",
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Det.model_type": ModelType.MOBILE,
            "Det.lang_type": LangDet.CH,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.lang_type": LangRec.EN,
            "EngineConfig.onnxruntime.use_coreml": self.device == "coreml",
        })

    def _providers(self) -> list[str]:
        sess = self._ocr.text_det.session.session
        return sess.get_providers()

    def ocr(self, image: Image.Image) -> OCRResult:
        res = self._ocr(np.asarray(image), return_word_box=True)
        if res.txts is None:
            return OCRResult(text="", elements={"line": [], "word": []})
        lines, words = [], []
        for text, quad, word_res in zip(res.txts, res.boxes, res.word_results):
            poly = [[float(x), float(y)] for x, y in quad]
            lines.append(TextElement(text, quad_to_box(poly), polygon=poly))
            words.extend(words_from_pieces([(w[0] + " ", quad_to_box(w[2])) for w in word_res]))
        return OCRResult(text="\n".join(l.text for l in lines), elements={"line": lines, "word": words})

    def info(self) -> dict:
        from importlib.metadata import version

        import onnxruntime

        try:
            providers = self._providers()
        except AttributeError:
            providers = "unknown"
        return {
            "engine_version": f"rapidocr {version('rapidocr')}, onnxruntime {onnxruntime.__version__}",
            "models": "ch_PP-OCRv5_det_mobile + en_PP-OCRv5_rec_mobile (ONNX); cls off",
            "onnx": True,
            "onnx_note": "PaddleOCR models exported to ONNX, run by ONNX Runtime.",
            "device_used": self.device,
            "session_providers": providers,
            "positional_output": "Line-level quadrilaterals (boxes); with return_word_box=True, per-word quads "
                                 "derived from CTC character positions (full line height). Pixel space of the "
                                 "input image, origin top-left.",
        }

    def gpu_memory_bytes(self) -> int | None:
        return None


def create(device: str) -> PaddleOnnxEngine:
    return PaddleOnnxEngine(device)
