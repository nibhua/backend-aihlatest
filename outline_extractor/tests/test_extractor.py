import fitz
import pytest
from outline_extractor import extract_text_with_layout, extract_outline

@pytest.fixture
def simple_pdf(tmp_path):
    path = tmp_path / "simple.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello World", fontsize=30, fontname="helv")
    doc.save(str(path))
    return str(path)

@pytest.fixture
def numbered_pdf(tmp_path):
    path = tmp_path / "numbered.pdf"
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "1. Introduction", fontsize=24, fontname="helv")
    p2 = doc.new_page()
    p2.insert_text((72, 72), "This is body text.", fontsize=12, fontname="helv")
    doc.save(str(path))
    return str(path)

@pytest.fixture
def multiline_pdf(tmp_path):
    path = tmp_path / "multiline.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "just a body line", fontsize=10, fontname="helv")
    page.insert_text((72, 100), "Big Bold Heading", fontsize=28, fontname="helv")
    doc.save(str(path))
    return str(path)

@pytest.fixture
def footer_pdf(tmp_path):
    path = tmp_path / "footer.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 800), "Page 1", fontsize=12, fontname="helv")
    doc.save(str(path))
    return str(path)

@pytest.fixture
def mixed_pdf(tmp_path):
    path = tmp_path / "mixed.pdf"
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for size, text in [(36, "Top Level"), (24, "Second Level"), (18, "Third Level")]:
        page.insert_text((72, y), text, fontsize=size, fontname="helv")
        y += size + 10
    doc.save(str(path))
    return str(path)

@pytest.fixture
def scanned_pdf(tmp_path):
    pytest.skip("Skipping OCR fallback test; requires Tesseract installed")

def test_extract_layout(simple_pdf):
    lines = extract_text_with_layout(simple_pdf)
    assert any("Hello World" in l["text"] for l in lines)

def test_extract_outline(simple_pdf):
    result = extract_outline(simple_pdf)
    assert result["title"] == "Hello World"
    outlines = result["outline"]
    assert any(o["text"] == "Hello World" and o["level"] == "H1" for o in outlines)

def test_numbered_heading(numbered_pdf):
    result = extract_outline(numbered_pdf)
    assert result["title"].startswith("1. Introduction")
    # All headings must be on page 1
    assert all(o["page"] == 1 for o in result["outline"])

def test_multiline_heading(multiline_pdf):
    result = extract_outline(multiline_pdf)
    assert result["title"] == "Big Bold Heading"
    outlines = result["outline"]
    assert len(outlines) == 1
    assert outlines[0]["text"] == "Big Bold Heading"

def test_footer_not_heading(footer_pdf):
    result = extract_outline(footer_pdf)
    # Footer "Page 1" should be skipped
    assert result["title"] == ""
    assert result["outline"] == []

def test_mixed_levels(mixed_pdf):
    result = extract_outline(mixed_pdf)
    texts  = [o["text"]  for o in result["outline"]]
    levels = [o["level"] for o in result["outline"]]
    assert texts  == ["Top Level", "Second Level", "Third Level"]
    assert levels == ["H1",        "H2",           "H3"]

@pytest.mark.skip(reason="OCR fallback environment not guaranteed")
def test_scanned_fallback(scanned_pdf):
    pass
