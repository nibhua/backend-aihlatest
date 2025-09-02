
---

## 📂 `outline_extractor/README.md`

```markdown
# outline_extractor – Heading Discovery & Sanity Checks

Purpose: prototype/validate heading detection and collect geometry before wiring the full chunk builder.

## Responsibilities
- Parse PDFs (PyMuPDF/fitz).
- Identify headings (font size/weight/spacing) and write page-level records with **BL bbox**.
- Dump quick JSON/CSV for manual inspection or overlay renders.

## Inputs
- PDFs in `input_pdfs/` (or pass `--pdf`)

## Commands

### 1) Extract outline (JSON)
```bash
python -m outline_extractor.extract \
  --pdf "input_pdfs/South of France - Cities.pdf" \
  --out "tmp/outline.cities.json"
