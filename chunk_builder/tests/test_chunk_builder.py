# core/chunk_builder/tests/test_chunk_builder.py
import os
from pathlib import Path
import pytest

from outline_extractor import extract_outline
from chunk_builder import build_chunks_for_pdf

# This file lives at: core/chunk_builder/tests/test_chunk_builder.py
# So parents[2] == core/
ROOT = Path(__file__).resolve().parents[2]  # <-- core/

def _discover_pdfs() -> list[Path]:
    """Find PDFs via env override or common directories under core/."""
    # 1) Env override
    env_dir = os.getenv("CORE_INPUT_DIR")
    if env_dir:
        d = Path(env_dir)
        if d.is_dir():
            pdfs = [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]
            if pdfs:
                return sorted(set(pdfs), key=lambda p: p.as_posix())

    # 2) Common default locations under core/
    candidates = [ROOT / "uploads", ROOT / "input_pdfs", ROOT / "input_PDFS", ROOT / "input"]
    for c in candidates:
        if c.is_dir():
            pdfs = [p for p in c.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]
            if pdfs:
                return sorted(set(pdfs), key=lambda p: p.as_posix())

    return []

# Debug print so it's obvious what the test sees
print("\n[chunk_builder tests] ROOT:", ROOT)
print("[chunk_builder tests] CORE_INPUT_DIR:", os.getenv("CORE_INPUT_DIR"))
_found = _discover_pdfs()
print("[chunk_builder tests] PDFs found:", [str(p) for p in _found])

@pytest.fixture(scope="module")
def first_pdf():
    pdfs = _discover_pdfs()
    if not pdfs:
        pytest.skip(
            f"No PDFs found. Set CORE_INPUT_DIR "
            f"or place files under {ROOT/'uploads'} (or input_pdfs/input_PDFS/input)"
        )
    return str(pdfs[0])

def test_build_chunks_smoke(first_pdf):
    outline = extract_outline(first_pdf)
    chunks = build_chunks_for_pdf(first_pdf, outline)
    assert isinstance(chunks, list)
    if chunks:
        c0 = chunks[0]
        for k in ("doc_id", "heading_text", "page", "bbox", "text", "snippet"):
            assert k in c0
        assert isinstance(c0["snippet"], str)

def test_snippet_reasonable(first_pdf):
    outline = extract_outline(first_pdf)
    chunks = build_chunks_for_pdf(first_pdf, outline)
    if not chunks:
        pytest.skip("No chunks produced for this PDF (no headings found)")
    s = chunks[0]["snippet"]
    assert isinstance(s, str)
    assert 0 < len(s) <= 600

def test_bbox_shape(first_pdf):
    outline = extract_outline(first_pdf)
    chunks = build_chunks_for_pdf(first_pdf, outline)
    if not chunks:
        pytest.skip("No chunks to validate bbox")
    bbox = chunks[0]["bbox"]
    assert isinstance(bbox, (list, tuple))
    assert len(bbox) == 4
    for v in bbox:
        assert isinstance(v, (int, float))

def test_pages_nonzero(first_pdf):
    outline = extract_outline(first_pdf)
    chunks = build_chunks_for_pdf(first_pdf, outline)
    if not chunks:
        pytest.skip("No chunks to validate page numbers")
    assert all(isinstance(c["page"], int) and c["page"] >= 1 for c in chunks)

def test_doc_id_matches_filename(first_pdf):
    name = os.path.basename(first_pdf)
    outline = extract_outline(first_pdf)
    chunks = build_chunks_for_pdf(first_pdf, outline)
    if not chunks:
        pytest.skip("No chunks to validate doc_id")
    assert all(c["doc_id"] == name for c in chunks)
# chunk_builder/tests/test_chunk_builder.py
import os
import pytest
from outline_extractor import extract_outline
from chunk_builder import build_chunks_for_pdf

@pytest.mark.usefixtures("selected_pdf")
def test_build_chunks_smoke(selected_pdf):
    outline = extract_outline(selected_pdf)
    chunks = build_chunks_for_pdf(selected_pdf, outline)
    assert isinstance(chunks, list)
    if chunks:
        c0 = chunks[0]
        for k in ("doc_id", "heading_text", "page", "bbox", "text", "snippet"):
            assert k in c0
        assert isinstance(c0["snippet"], str)

def test_snippet_reasonable(selected_pdf):
    outline = extract_outline(selected_pdf)
    chunks = build_chunks_for_pdf(selected_pdf, outline)
    if not chunks:
        pytest.skip("No chunks produced for this PDF (no headings found)")
    s = chunks[0]["snippet"]
    assert isinstance(s, str) and 0 < len(s) <= 600

def test_bbox_shape(selected_pdf):
    outline = extract_outline(selected_pdf)
    chunks = build_chunks_for_pdf(selected_pdf, outline)
    if not chunks:
        pytest.skip("No chunks to validate bbox")
    bbox = chunks[0]["bbox"]
    assert isinstance(bbox, (list, tuple)) and len(bbox) == 4
    for v in bbox:
        assert isinstance(v, (int, float))

def test_pages_nonzero(selected_pdf):
    outline = extract_outline(selected_pdf)
    chunks = build_chunks_for_pdf(selected_pdf, outline)
    if not chunks:
        pytest.skip("No chunks to validate page numbers")
    assert all(isinstance(c["page"], int) and c["page"] >= 1 for c in chunks)

def test_doc_id_matches_filename(selected_pdf):
    name = os.path.basename(selected_pdf)
    outline = extract_outline(selected_pdf)
    chunks = build_chunks_for_pdf(selected_pdf, outline)
    if not chunks:
        pytest.skip("No chunks to validate doc_id")
    assert all(c["doc_id"] == name for c in chunks)
