"""Tesseract 5 (LSTM) via pytesseract. One image_to_data call yields text and all box levels."""

from __future__ import annotations

import shutil
import subprocess

import pytesseract
from PIL import Image

from common.engine import OCRResult, TextElement

LEVELS = {2: "block", 4: "line", 5: "word"}


class TesseractEngine:
    name = "tesseract"

    def __init__(self, device: str):
        if device != "cpu":
            raise ValueError("Tesseract runs on CPU only")
        self.device = device

    def load(self) -> None:
        # No model to preload: the tesseract binary loads traineddata per call.
        pytesseract.get_tesseract_version()

    def ocr(self, image: Image.Image) -> OCRResult:
        d = pytesseract.image_to_data(image, config="--oem 1 --psm 3", output_type=pytesseract.Output.DICT)
        elements: dict[str, list[TextElement]] = {g: [] for g in LEVELS.values()}
        words_by_line: dict[tuple, list[str]] = {}
        for i, level in enumerate(d["level"]):
            if level not in LEVELS:
                continue
            box = [d["left"][i], d["top"][i], d["left"][i] + d["width"][i], d["top"][i] + d["height"][i]]
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            if level == 5:
                word = d["text"][i].strip()
                if not word:
                    continue
                elements["word"].append(TextElement(word, box, confidence=float(d["conf"][i])))
                words_by_line.setdefault(key, []).append(word)
            elif level == 4:
                elements["line"].append(TextElement("", box))
                elements["line"][-1].key = key
            else:
                elements["block"].append(TextElement("", box))
                elements["block"][-1].key = d["block_num"][i]

        for line in elements["line"]:
            line.text = " ".join(words_by_line.get(line.key, []))
        for block in elements["block"]:
            block.text = " ".join(l.text for l in elements["line"] if l.key[0] == block.key)
        for el in elements["line"] + elements["block"]:
            del el.key
        elements["line"] = [l for l in elements["line"] if l.text]
        elements["block"] = [b for b in elements["block"] if b.text]
        return OCRResult(text="\n".join(l.text for l in elements["line"]), elements=elements)

    def info(self) -> dict:
        binary = shutil.which("tesseract")
        version = subprocess.run([binary, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
        return {
            "engine_version": version,
            "pytesseract": pytesseract.__version__ if hasattr(pytesseract, "__version__") else None,
            "models": "eng.traineddata (LSTM, --oem 1), --psm 3",
            "onnx": False,
            "onnx_note": "Tesseract's LSTM models use its own traineddata format; no ONNX export/runtime.",
            "device_used": "cpu",
            "positional_output": "Axis-aligned boxes at block, paragraph, line and word level (image_to_data), "
                                 "pixel space of the input image, origin top-left, with per-word confidence.",
        }

    def gpu_memory_bytes(self) -> int | None:
        return None


def create(device: str) -> TesseractEngine:
    return TesseractEngine(device)
