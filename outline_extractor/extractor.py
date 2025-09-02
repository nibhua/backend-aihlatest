from .pdf_processor import extract_text_with_layout
from .heading_extractor import extract_title, assign_levels

def extract_outline(pdf_path: str) -> dict:
    """
    Full pipeline: parse text/layout and detect title + headings.
    Returns:
      { "title": str,
        "outline": [ {level, text, page, bbox, confidence} ] }
    """
    lines = extract_text_with_layout(pdf_path)
    title = extract_title(lines)
    outline = assign_levels(lines)
    return {"title": title, "outline": outline}
