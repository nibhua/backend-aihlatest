
---

## 📂 `chunk_builder/README.md`

```markdown
# chunk_builder – Build Chunks + Geometry + Embeddings

Creates the training/retrieval units ("chunks") that power search:
- **Heading detection** (using the heuristics validated in `outline_extractor`)
- **Body text grouping** under each heading
- **Geometry** for headings (and optionally body)
- **Embeddings** + vector index
- **chunks.json** and **meta.json**

## Inputs
- PDFs: `input_pdfs/`
- Model: any sentence-transformers model (default in scripts)
- Config flags: see below

## Outputs
- `vector_store/col_<id>/chunks.json`
- `vector_store/col_<id>/meta.json`
- Vector index files (faiss/npz/etc.)

## Run (direct)
```bash
python -m chunk_builder.builder \
  --input_dir input_pdfs \
  --out_root vector_store \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --min_heading_size 14 \
  --max_chunk_tokens 800
