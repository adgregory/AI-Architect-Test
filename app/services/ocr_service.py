import fitz
import pytesseract
from PIL import Image
import io

OCR_DPI = 150
PDF_POINTS_PER_INCH = 72


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a scanned PDF using OCR."""
    doc = fitz.open(pdf_path)
    try:
        full_text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=OCR_DPI)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)
            full_text += text + "\n"
        return full_text
    finally:
        doc.close()


def get_word_bounding_boxes(pdf_path: str) -> list[dict]:
    """Get bounding boxes for all words in the PDF, in PDF coordinate space (points)."""
    doc = fitz.open(pdf_path)
    scale = PDF_POINTS_PER_INCH / OCR_DPI
    results = []
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=OCR_DPI)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

            for i in range(len(ocr_data["text"])):
                word = ocr_data["text"][i].strip()
                if not word:
                    continue

                results.append({
                    "word": word,
                    "page": page_num,
                    "x": ocr_data["left"][i] * scale,
                    "y": ocr_data["top"][i] * scale,
                    "width": ocr_data["width"][i] * scale,
                    "height": ocr_data["height"][i] * scale,
                })
    finally:
        doc.close()

    return results
