
---

## 📂 `scripts/README.md`

```markdown
# scripts – End-to-End Entry Points (CLI)

These are the **only commands** you need for the full offline pipeline.

---

## 1) Build a collection

```bash
python -m scripts.build_collection \
  --input_dir input_pdfs \
  --out_root vector_store \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --min_heading_size 14 \
  --max_chunk_tokens 800 \
  --mmr_lambda 0.5
