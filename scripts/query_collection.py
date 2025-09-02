from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import shorten
from typing import List, Tuple

from embeddings import load_default_model
from indexer.nn_store import NNStore
from indexer.config import DEFAULT_K, RERANK_POOL
from indexer.rerank import rerank_candidates, explain_reason, extract_keywords
from persona.router import compile_recipe
from indexer.fusion import reciprocal_rank_fusion

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _fmt_snippet(s: str, width: int = 200) -> str:
    return shorten((s or "").replace("\n", " "), width=width, placeholder="…")

def _resolve_store_dir(user_supplied: str) -> Path:
    p = Path(user_supplied)
    if p.exists():
        return p
    return (PROJECT_ROOT / "vector_store" / user_supplied).resolve()

def _cosine_pool(store: NNStore, model, text: str, top_k: int) -> List[Tuple[int, float]]:
    qvec = model.encode_one(text, normalize=True, mode="query")
    idxs, scores = store.cosine_search(qvec, top_k=top_k)
    return list(zip([int(i) for i in idxs], [float(s) for s in scores]))

def main() -> None:
    ap = argparse.ArgumentParser(description="Query a built collection (cosine search).")
    ap.add_argument("--collection", required=True, help="Collection id OR path to the vector_store folder")
    ap.add_argument("--pool", type=int, default=RERANK_POOL, help="Candidate pool size to re-rank")

    sub = ap.add_subparsers(dest="mode", required=True)

    # -------- persona mode (no --query needed) --------
    p = sub.add_parser("persona", help="Persona + job only")
    p.add_argument("--persona", type=str, default="", help="Persona description (free text)")
    p.add_argument("--job", type=str, default="", help="Job-to-be-done description (free text)")
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--mmr", action="store_true", help="Diversify with MMR")
    p.add_argument("--no_overlap", action="store_true", help="Disable unigram overlap bonus")
    p.add_argument("--no_same_doc_boost", action="store_true", help="Disable same-doc boost")
    p.add_argument("--no_chapter_bonus", action="store_true", help="Disable chapter affinity bonus")
    p.add_argument("--no_phrases", action="store_true", help="Disable bigram/trigram phrase boost")

    # -------- free-text mode --------
    t = sub.add_parser("text", help="Free-text query")
    t.add_argument("--query", required=True)
    t.add_argument("--persona", type=str, default="", help="Persona description (free text)")
    t.add_argument("--job", type=str, default="", help="Job-to-be-done description (free text)")
    t.add_argument("--k", type=int, default=DEFAULT_K)
    t.add_argument("--mmr", action="store_true", help="Diversify with MMR")
    t.add_argument("--no_overlap", action="store_true", help="Disable unigram overlap bonus")
    t.add_argument("--no_same_doc_boost", action="store_true", help="Disable same-doc boost")
    t.add_argument("--no_chapter_bonus", action="store_true", help="Disable chapter affinity bonus")
    t.add_argument("--no_phrases", action="store_true", help="Disable bigram/trigram phrase boost")

    # -------- chunk mode --------
    c = sub.add_parser("chunk", help="Use an existing chunk as the query")
    c.add_argument("--chunk_id", type=int, required=True)
    c.add_argument("--k", type=int, default=DEFAULT_K)
    c.add_argument("--mmr", action="store_true", help="Diversify with MMR")
    c.add_argument("--no_overlap", action="store_true", help="Disable unigram overlap bonus")
    c.add_argument("--no_same_doc_boost", action="store_true", help="Disable same-doc boost")
    c.add_argument("--no_chapter_bonus", action="store_true", help="Disable chapter affinity bonus")
    c.add_argument("--no_phrases", action="store_true", help="Disable bigram/trigram phrase boost")

    args = ap.parse_args()

    store = NNStore.load(args.collection)
    model = load_default_model()

    mode = args.mode
    candidates: List[Tuple[int, float]] = []
    base_heading, base_doc = None, None
    qtext = ""

    if mode in ("persona", "text") and (getattr(args, "persona", "") or getattr(args, "job", "")):
        store_dir = _resolve_store_dir(args.collection)
        recipe = compile_recipe(getattr(args, "persona", ""), getattr(args, "job", ""), str(store_dir))
        queries: List[str] = recipe.get("queries", []) or ([getattr(args, "query", "")] if mode == "text" else [""])
        knobs = recipe.get("knobs", {})
        eff_k = int(knobs.get("k", args.k))
        eff_pool = max(eff_k * 5, int(knobs.get("pool", args.pool)))
        eff_mmr = bool(knobs.get("mmr", args.mmr))

        must_contain_terms = list(knobs.get("must_contain", []) or [])
        must_not_terms = list(knobs.get("must_not_contain", []) or [])
        boost_headings = list(knobs.get("boost_headings", []) or [])
        down_headings = list(knobs.get("downweight_headings", []) or [])

        all_ranked: List[List[Tuple[int, float]]] = []
        for q in queries:
            if not q.strip():
                continue
            if eff_mmr:
                mmr_res = store.search_texts([q], embedder=model, k=min(eff_pool, 50), mmr=True)[0]
                ranked = [(int(i), float(s)) for i, s in mmr_res]
            else:
                ranked = _cosine_pool(store, model, q, top_k=eff_pool)
            all_ranked.append(ranked)

        fused = reciprocal_rank_fusion(all_ranked, k=eff_pool) if all_ranked else []
        candidates = fused
        qtext = " | ".join([q for q in queries if q.strip()]) or getattr(args, "query", "")
        print(f'\nPersona="{getattr(args, "persona", "")}"  Job="{getattr(args, "job", "")}"')
        print(f"Expanded queries: {queries}")

        reranked = rerank_candidates(
            query_text=qtext,
            base_heading=base_heading,
            base_doc=base_doc,
            candidates=candidates,
            chunks=store.chunks,
            k=eff_k,
            use_overlap=not args.no_overlap,
            use_same_doc_boost=not args.no_same_doc_boost,
            use_chapter_bonus=not args.no_chapter_bonus,
            use_phrase_boost=not args.no_phrases,
            must_contain_terms=must_contain_terms,
            must_not_contain_terms=must_not_terms,
            boost_headings=boost_headings,
            downweight_headings=down_headings,
        )

    elif mode == "text":
        qtext = args.query
        qvec = model.encode_one(qtext, normalize=True, mode="query")
        pool_n = max(args.k * 5, args.pool)
        idxs, scores = store.cosine_search(qvec, top_k=pool_n)
        candidates = list(zip([int(i) for i in idxs], [float(s) for s in scores]))
        print(f'\nQuery = "{args.query}"')

        if args.mmr:
            mmr_res = store.search_texts([qtext], embedder=model, k=min(pool_n, 50), mmr=True)[0]
            if mmr_res:
                candidates = [(int(i), float(s)) for i, s in mmr_res]

        reranked = rerank_candidates(
            query_text=qtext,
            base_heading=None,
            base_doc=None,
            candidates=candidates,
            chunks=store.chunks,
            k=args.k,
            use_overlap=not args.no_overlap,
            use_same_doc_boost=not args.no_same_doc_boost,
            use_chapter_bonus=not args.no_chapter_bonus,
            use_phrase_boost=not args.no_phrases,
        )

    else:  # chunk mode
        if not (0 <= args.chunk_id < len(store.chunks)):
            raise SystemExit(f"--chunk_id out of range (0..{len(store.chunks)-1})")
        qch = store.chunks[args.chunk_id]
        qtext = qch.get("text") or qch.get("heading_text") or ""
        base_heading = qch.get("heading_text") or ""
        base_doc = qch.get("doc_id")
        qvec = model.encode_one(qtext, normalize=True, mode="passage")
        pool_n = max(args.k * 5, args.pool)
        idxs, scores = store.cosine_search(qvec, top_k=pool_n)
        candidates = list(zip([int(i) for i in idxs], [float(s) for s in scores]))
        print(f"\nQuery = CHUNK #{args.chunk_id} • {base_heading} (doc={base_doc}, page={qch.get('page')})")

        if args.mmr:
            mmr_res = store.search_texts([qtext], embedder=model, k=min(pool_n, 50), mmr=True)[0]
            if mmr_res:
                candidates = [(int(i), float(s)) for i, s in mmr_res]

        reranked = rerank_candidates(
            query_text=qtext,
            base_heading=base_heading,
            base_doc=base_doc,
            candidates=candidates,
            chunks=store.chunks,
            k=args.k,
        )

    print(f"\nTop {len(reranked)} results:")
    q_terms = extract_keywords(qtext)
    for rank, (idx, score) in enumerate(reranked, start=1):
        ch = store.chunks[idx]
        heading = ch.get("heading_text") or ch.get("heading") or ""
        page = ch.get("page")
        doc = ch.get("doc_id")
        snippet = ch.get("snippet") or _fmt_snippet(ch.get("text") or "")
        reason = explain_reason(q_terms, f"{ch.get('text','')} {ch.get('snippet','')}", heading, None)
        print(f"{rank:>2}. score={score:>6.3f}  p.{page:<3}  {heading}  ({doc})")
        if snippet:
            print(f"    {snippet}")
        print(f"    ↳ {reason}")

if __name__ == "__main__":
    main()
