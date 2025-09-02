# core/indexer/fusion.py
from __future__ import annotations

from typing import Dict, List, Tuple

def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[int, float]]],
    k: int = 10,
    const: int = 60,
) -> List[Tuple[int, float]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Args:
      result_lists: list of ranked lists. Each inner list is [(idx, score), ...] best-first.
      k: number of fused items to return.
      const: RRF constant (typical 60). Larger → flatter contribution from deeper ranks.

    Returns:
      List of (idx, fused_score), best-first, length ≤ k.
    """
    accum: Dict[int, float] = {}
    for rlist in result_lists:
        for rank, (idx, _score) in enumerate(rlist):
            accum[idx] = accum.get(idx, 0.0) + 1.0 / (const + rank + 1)

    fused = sorted(accum.items(), key=lambda kv: kv[1], reverse=True)
    return fused[:k]
