# core/scripts/build_collection.py
from __future__ import annotations

import argparse
import json
import time
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from embeddings import load_default_model
from indexer import NNStore, make_collection_id
from chunk_builder.builder import build_chunks_for_pdf

# NEW: snippet generation (CPU-only, fast)
from snippets.generator import generate_snippets

# NEW: lightweight TF-IDF to support persona compiler (CPU-only)
from sklearn.feature_extraction.text import TfidfVectorizer  # scikit-learn already in your stack

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _extract_outline(pdf_path: str) -> Dict:
    """
    Try multiple ways to obtain the outline dict for a single PDF:
      1) If a JSON with the same stem exists under core/output_JSN/, load it
      2) Else call outline_extractor (if importable)
    """
    pdf = Path(pdf_path)
    out_json = PROJECT_ROOT / "output_JSN" / f"{pdf.stem}.json"
    if out_json.exists():
        try:
            return json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    try:
        from outline_extractor.extractor import extract_outline  # type: ignore
        return extract_outline(str(pdf))
    except Exception:
        try:
            from outline_extractor.extractor import OutlineExtractor  # type: ignore
            ext = OutlineExtractor()
            return ext.extract(str(pdf))
        except Exception as e:
            raise RuntimeError(
                f"Could not obtain outline for {pdf_path}. "
                f"Provide a precomputed JSON in core/output_JSN/{pdf.stem}.json "
                f"or ensure outline_extractor exposes extract_outline()."
            ) from e


def _gather_chunks(pdf_paths: List[str], verbose: bool = True) -> List[Dict]:
    all_chunks: List[Dict] = []
    for p in pdf_paths:
        outline = _extract_outline(p)
        items = outline.get("outline", []) if isinstance(outline, dict) else []
        if verbose:
            print(f"  • {Path(p).name}: headings={len(items)}", end="")
        if not items:
            if verbose:
                print(" → 0 chunks (no headings)")
            continue
        chunks = build_chunks_for_pdf(p, outline)
        if verbose:
            print(f", chunks={len(chunks)}")
        all_chunks.extend(chunks)
    return all_chunks


def _resolve_input_dir(user_path: str | None) -> Path:
    if not user_path:
        return PROJECT_ROOT / "uploads"  # Changed default to uploads
    p = Path(user_path)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


def _build_persona_artifacts(chunks: List[Dict], out_dir: Path) -> None:
    """
    Build small, CPU-friendly artefacts used by the persona compiler:
      - tfidf.pkl: TfidfVectorizer fitted on chunk texts + headings
      - collection_vocab.pkl: (terms, idf_map) for fast high-IDF filtering
      - heading_graph.json: links between related headings (very small)
    """
    texts = [c.get("text") or "" for c in chunks]
    headings = [c.get("heading_text") or "" for c in chunks]

    if not any(texts) and not any(headings):
        print("  (Skip persona artefacts: no text/headings)")
        return

    # Fit TF-IDF on texts + headings (ngram (1,3) to capture phrases; small cap on features)
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 3), max_features=5000)
    tfidf.fit(texts + headings)
    vocab_terms = tfidf.get_feature_names_out()
    idf_map = dict(zip(vocab_terms, tfidf.idf_))

    (out_dir / "tfidf.pkl").write_bytes(pickle.dumps(tfidf))
    (out_dir / "collection_vocab.pkl").write_bytes(pickle.dumps((vocab_terms.tolist(), idf_map)))

    # Tiny heading graph: connect headings that share tokens or are neighbours
    heading_graph = defaultdict(set)
    for i, h in enumerate(headings):
        if not h:
            continue
        tokens = {t.lower() for t in h.split() if len(t) > 2}
        # neighbour links
        for j in (i - 1, i + 1):
            if 0 <= j < len(headings) and headings[j]:
                heading_graph[h].add(headings[j])
        # token overlap links (cheap heuristic)
        for j in range(i + 1, min(i + 6, len(headings))):
            other = headings[j]
            if not other:
                continue
            other_tokens = {t.lower() for t in other.split() if len(t) > 2}
            if tokens & other_tokens:
                heading_graph[h].add(other)
                heading_graph[other].add(h)

    # Serialize as lists
    heading_graph_json = {k: sorted(list(v)) for k, v in heading_graph.items()}
    (out_dir / "heading_graph.json").write_text(
        json.dumps(heading_graph_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("✔ Saved TF-IDF, vocab, and heading graph for persona routing")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a cosine store for a set of PDFs.")
    ap.add_argument(
        "--input_dir",
        type=str,
        default="uploads",
        help="Folder with PDFs. Relative paths are resolved against project root (core/). Default: uploads",
    )
    ap.add_argument("--collection_id", type=str, default="", help="If omitted, derived from filenames.")
    ap.add_argument("--batch_size", type=int, default=128, help="Embedding batch size")
    ap.add_argument("--dry_run", action="store_true", help="Only count chunks; do not write a store.")
    ap.add_argument("--quiet", action="store_true", help="Less per-PDF logging")
    args = ap.parse_args()

    input_dir = _resolve_input_dir(args.input_dir)
    pdfs = sorted(str(p) for p in input_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {input_dir}")

    coll_id = args.collection_id or make_collection_id(pdfs)

    print(f" Building collection {coll_id}")
    print(f"  PDFs: {len(pdfs)} from {input_dir}")

    t0 = time.time()
    chunks = _gather_chunks(pdfs, verbose=not args.quiet)
    print(f"  Extracted {len(chunks)} chunks in {time.time()-t0:.2f}s")

    # ---- Generate/refresh snippets BEFORE embedding (fast, CPU-only) ----
    # This writes/updates chunk["snippet"] in-memory; it will be persisted with the store.
    generate_snippets(chunks, max_chars=200)

    # ---- Early exit if nothing to embed or dry-run ----
    if args.dry_run or not chunks:
        if not chunks:
            print("  (No chunks produced. Check Phase-1 outlines or precompute core/output_JSN/<pdf>.json.)")
        return
    # ---------------------------------------------------

    model = load_default_model()
    t1 = time.time()
    embs = model.encode_chunks_texts(
        chunks, batch_size=args.batch_size, normalize=True, mode="passage", show_progress_bar=True
    )
    print(f"  Encoded {embs.shape[0]} chunks (dim={embs.shape[1]}) in {time.time()-t1:.2f}s")

    meta = {
        "collection_id": coll_id,
        "model_name_or_path": model.model_name_or_path,
        "dim": int(model.dim),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_chunks": int(embs.shape[0]),
    }
    store = NNStore.build(embs, chunks, meta)
    out_dir = store.save(coll_id)
    print(f"✔ Saved vector store to {out_dir}")

    # ---- Persona-compiler artefacts (tiny, CPU-only) ----
    _build_persona_artifacts(chunks, out_dir)


if __name__ == "__main__":
    main()
