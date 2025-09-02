# embeddings/

Tiny, offline-first wrapper around `sentence-transformers` to embed chunk text.

## What it does

- Loads a sentence embedding model from **local path** if available, otherwise by HF ID.
- Batch encodes texts on CPU.
- L2-normalizes vectors (cosine-ready).
- Adds **E5 prefixes** (`"passage: "` / `"query: "`) automatically when using the E5 family.

## Files

- `config.py` — env-driven settings and model path resolution.
- `model.py` — `EmbeddingModel` with `encode()`, `encode_one()`, `encode_chunks_texts()`.
- `__init__.py` — re-exports the main API.

## Env overrides

- `EMBEDDING_MODEL_ID` (default: `intfloat/e5-base-v2`)
- `EMBEDDING_MODEL_PATH` (prefer this local path if it exists)
- `EMBEDDING_DEVICE` (default: `cpu`)
- `EMBEDDING_BATCH_SIZE` (default: `128`)
- `EMBEDDING_NORMALIZE` (`1`/`0`, default `1`)

## Expected requirements

Add this (or compatible) to your `requirements.txt`:

