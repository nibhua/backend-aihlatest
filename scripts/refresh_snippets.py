# core/scripts/refresh_snippets.py
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

from snippets.generator import generate_snippets

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _resolve_store_dir(user_supplied: str) -> Path:
    """
    Accept:
      - direct path to vector_store/<collection_id>
      - ID (resolve to core/vector_store/<id>)
    """
    p = Path(user_supplied)
    if p.exists():
        return p
    return (PROJECT_ROOT / "vector_store" / user_supplied).resolve()

def _load_chunks(store_dir: Path) -> List[Dict]:
    chunks_path = store_dir / "chunks.json"
    if not chunks_path.exists():
        raise SystemExit(f"chunks.json not found at {chunks_path}")
    return json.loads(chunks_path.read_text(encoding="utf-8"))

def _save_chunks(store_dir: Path, chunks: List[Dict]) -> None:
    chunks_path = store_dir / "chunks.json"
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

def _load_idf_map(store_dir: Path) -> Dict[str, float]:
    p = store_dir / "collection_vocab.pkl"
    if not p.exists():
        return {}
    try:
        terms, idf_map = pickle.loads(p.read_bytes())
        # normalize keys to lowercase
        return {k.lower(): float(v) for k, v in idf_map.items()}
    except Exception:
        return {}

def _touch_meta(store_dir: Path) -> None:
    meta_path = store_dir / "meta.json"
    if not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["snippets_refreshed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh precomputed snippets for a collection (no re-embedding).")
    ap.add_argument("--collection", required=True, help="Collection id OR path to the vector_store folder")
    ap.add_argument("--max_chars", type=int, default=200, help="Snippet max length (characters)")
    ap.add_argument("--no_vocab", action="store_true", help="Do not use collection IDF map (pure heuristic snippets)")
    args = ap.parse_args()

    store_dir = _resolve_store_dir(args.collection)
    chunks = _load_chunks(store_dir)
    idf_map = {} if args.no_vocab else _load_idf_map(store_dir)

    print(f"→ Refreshing snippets in {store_dir}")
    print(f"  Chunks: {len(chunks)}  |  Using IDF map: {'no' if args.no_vocab else 'yes'}")

    generate_snippets(chunks, idf_map=idf_map, max_chars=args.max_chars)
    _save_chunks(store_dir, chunks)
    _touch_meta(store_dir)
    print("✔ Snippets refreshed")

if __name__ == "__main__":
    main()
