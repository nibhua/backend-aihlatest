import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def extract_text_with_layout(pdf_path: str) -> list[dict]:
    """
    Extract text lines with layout info. If no text blocks found (e.g. scanned),
    falls back to Tesseract OCR on each page image.
    Returns a list of dicts with keys:
      text, font_size, font_name, bbox, page, page_width, page_height, is_bold
    """
    doc = fitz.open(pdf_path)
    extracted = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        w, h = page.rect.width, page.rect.height
        blocks = page.get_text("dict")["blocks"]

        if not blocks:
            # OCR fallback
            logger.debug(f"Page {page_num+1}: no text blocks, running OCR fallback")
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            n = len(ocr_data["text"])
            for i in range(n):
                txt = ocr_data["text"][i].strip()
                if not txt:
                    continue
                x, y, w0, h0 = (ocr_data["left"][i], ocr_data["top"][i],
                                ocr_data["width"][i], ocr_data["height"][i])
                extracted.append({
                    "text": txt,
                    "font_size": ocr_data["height"][i],    # approximate
                    "font_name": "OCR",
                    "bbox": [x, y, x + w0, y + h0],
                    "page": page_num + 1,
                    "page_width": w,
                    "page_height": h,
                    "is_bold": False
                })
            continue

        # Normal text extraction path
        for block in blocks:
            for line in block.get("lines", []):
                text = "".join(span["text"] for span in line["spans"]).strip()
                if not text:
                    continue
                bbox = line["bbox"]
                span0 = line["spans"][0]
                size = span0["size"]
                font = span0["font"]
                is_bold = any("bold" in span["font"].lower() for span in line["spans"])
                extracted.append({
                    "text": text,
                    "font_size": size,
                    "font_name": font,
                    "bbox": bbox,
                    "page": page_num + 1,  # 1-based
                    "page_width": w,
                    "page_height": h,
                    "is_bold": is_bold
                })

    logger.debug(f"Extracted {len(extracted)} lines from {pdf_path}")
    return extracted
