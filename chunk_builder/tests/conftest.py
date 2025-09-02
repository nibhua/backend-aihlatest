# chunk_builder/tests/conftest.py
import pytest
from pathlib import Path

# Resolve the project root = "core"
# this file is at core/chunk_builder/tests/conftest.py
ROOT = Path(__file__).resolve().parents[2]  # tests -> chunk_builder -> core

def _find_input_dir():
    for name in ("uploads", "input_pdfs", "input_PDFS", "input"):
        p = ROOT / name
        if p.exists() and any(p.glob("*.pdf")):
            return p
    return None

def _list_pdfs(inp_dir: Path):
    return sorted(str(p) for p in inp_dir.glob("*.pdf"))

def pytest_addoption(parser):
    parser.addoption("--pdf", action="store", default=None,
                     help="PDF filename (basename) inside uploads to test, e.g. file02.pdf")
    parser.addoption("--pdf-index", action="store", type=int, default=None,
                     help="0-based index into the sorted list of PDFs in uploads")

@pytest.fixture(scope="session")
def input_dir():
    inp = _find_input_dir()
    if inp is None:
        pytest.skip("No PDFs found in core/uploads (or input_pdfs/input_PDFS/input)")
    return inp

@pytest.fixture(scope="session")
def selected_pdf(request, input_dir: Path):
    """Pick a single PDF based on --pdf or --pdf-index, otherwise the first."""
    pdf_opt = request.config.getoption("--pdf")
    idx_opt = request.config.getoption("--pdf-index")
    pdfs = _list_pdfs(input_dir)

    if pdf_opt:
        matches = [p for p in pdfs if Path(p).name.lower() == pdf_opt.lower()]
        if not matches:
            pytest.fail(f"--pdf '{pdf_opt}' not found under {input_dir}")
        return matches[0]

    if idx_opt is not None:
        if idx_opt < 0 or idx_opt >= len(pdfs):
            pytest.fail(f"--pdf-index {idx_opt} out of range (0..{len(pdfs)-1})")
        return pdfs[idx_opt]

    # default: first file
    return pdfs[0]
