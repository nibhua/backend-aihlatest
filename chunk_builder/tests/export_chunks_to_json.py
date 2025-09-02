# core/chunk_builder/tests/export_chunks_to_json.py
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# --- Make project imports work when run as a script ---
ROOT = Path(__file__).resolve().parents[2]  # -> core/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from outline_extractor import extract_outline
from chunk_builder import build_chunks_for_pdf

INPUT_DIR_CANDIDATES = ["uploads", "input_pdfs", "input_PDFS"]
OUTPUT_DIR_NAME = "output_JSN"  # will be created under core/ if missing


def find_input_dir() -> Path | None:
    for name in INPUT_DIR_CANDIDATES:
        p = ROOT / name
        if p.exists() and any(p.glob("*.pdf")):
            return p
    return None


def pick_pdfs(input_dir: Path, only: str | None) -> List[Path]:
    if only:
        # allow either a bare filename or a path
        p = Path(only)
        if not p.suffix.lower() == ".pdf":
            p = p.with_suffix(".pdf")
        if not p.is_absolute():
            p = input_dir / p.name
        return [p] if p.exists() else []
    return sorted(input_dir.glob("*.pdf"))


def chunk_record_to_export(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert builder chunk into a compact JSON record:
      {heading, page, bbox, snippet, text}
    Where 'text' is the subsequent body text (without the heading).
    """
    heading = rec.get("heading_text", "")
    full = rec.get("text", "") or ""
    body = full[len(heading):].lstrip() if heading and full.startswith(heading) else full
    return {
        "heading": heading,
        "page": rec.get("page"),
        "bbox": rec.get("bbox"),
        "snippet": rec.get("snippet"),
        "text": body,  # body-only for clarity
    }


def process_pdf(pdf_path: Path, out_dir: Path) -> Path:
    outline = extract_outline(str(pdf_path))
    chunks = build_chunks_for_pdf(str(pdf_path), outline)

    data = {
        "doc_id": pdf_path.name,
        "title": outline.get("title", ""),
        "num_chunks": len(chunks),
        "chunks": [chunk_record_to_export(c) for c in chunks],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (pdf_path.stem + ".json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_file


def main():
    parser = argparse.ArgumentParser(
        description="Export heading-anchored chunks for all PDFs in uploads → JSON files."
    )
    parser.add_argument(
        "--pdf",
        help="Optional: process a single PDF by filename (e.g. --pdf file02.pdf).",
        default=None,
    )
    parser.add_argument(
        "--out",
        help=f"Optional: output folder name under core/ (default: {OUTPUT_DIR_NAME})",
        default=OUTPUT_DIR_NAME,
    )
    args = parser.parse_args()

    inp = find_input_dir()
    if not inp:
        print(f"❌ No PDFs found in {', '.join(INPUT_DIR_CANDIDATES)} under {ROOT}")
        sys.exit(1)

    pdfs = pick_pdfs(inp, args.pdf)
    if not pdfs:
        print(f"❌ No matching PDFs found. Looked in: {inp}  (filter={args.pdf!r})")
        sys.exit(1)

    out_dir = ROOT / args.out
    print(f"📂 Input: {inp}")
    print(f"📝 Output: {out_dir}")
    print(f"📄 PDFs: {len(pdfs)} to process")

    written: List[Path] = []
    for i, pdf in enumerate(pdfs, 1):
        try:
            out_file = process_pdf(pdf, out_dir)
            written.append(out_file)
            print(f"  [{i}/{len(pdfs)}] ✅ {pdf.name} → {out_file.name}")
        except Exception as e:
            print(f"  [{i}/{len(pdfs)}] ❌ {pdf.name}: {e}")

    print(f"\n✅ Done. Wrote {len(written)} JSON file(s) to {out_dir}")


if __name__ == "__main__":
    main()
