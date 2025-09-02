# core/snippets/generator.py
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_WHITESPACE_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+|\n+")
_PUNCT_END = (".", "!", "?")
# Common boilerplate headings we usually don't want as snippet text
_BOILER_HEADINGS = {"ingredients", "instructions", "notes", "method", "directions"}

def _clean(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "")).strip()

def _split_sentences(text: str) -> List[str]:
    text = _clean(text)
    if not text:
        return []
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    # Ensure each sentence ends with punctuation if original had none
    fixed = []
    for s in parts:
        if s and s[-1] not in _PUNCT_END:
            fixed.append(s)
        else:
            fixed.append(s)
    return fixed

def _score_sentence(
    sent: str,
    prefer_terms: Optional[Sequence[str]] = None,
    idf_map: Optional[Dict[str, float]] = None
) -> float:
    s = sent.lower()
    score = 0.0
    # Prefer sentences containing any of the “prefer” terms
    if prefer_terms:
        for t in prefer_terms:
            t = t.lower().strip()
            if t and t in s:
                score += 1.5
    # IDF-weight sum for tokens present (if provided)
    if idf_map:
        for tok in re.findall(r"[a-z][a-z0-9\-]+", s):
            score += idf_map.get(tok, 0.0) * 0.02
    # Favor mid-length sentences (short enough to be previewable)
    ln = len(sent)
    if 60 <= ln <= 240:
        score += 0.5
    return score

def make_snippet(
    chunk: Dict,
    idf_map: Optional[Dict[str, float]] = None,
    prefer_terms: Optional[Sequence[str]] = None,
    max_chars: int = 200,
    strip_boilerplate: bool = True,
) -> str:
    """
    Produce a compact snippet for a chunk.
    Heuristics:
      - Prefer a sentence that contains any prefer_terms (e.g., must_contain from router).
      - Otherwise pick the highest-scoring sentence by IDF-weight and preview length.
      - Skip boilerplate headings like 'Ingredients'/'Instructions' when used as the only text.
    """
    heading = _clean(chunk.get("heading_text") or chunk.get("heading") or "")
    text = _clean(chunk.get("text") or "")

    # If the entire chunk is mostly a boilerplate section title, try to pick from the text body
    if strip_boilerplate and heading.lower() in _BOILER_HEADINGS and text:
        sents = _split_sentences(text)
    else:
        # Some builders put key lines in 'text' already; rely on it
        sents = _split_sentences(text)

    # If no sentences, fallback to heading or any available text
    if not sents:
        candidate = heading or text
        return (candidate[: max_chars - 1] + "…") if len(candidate) > max_chars else candidate

    # Score sentences and pick best
    scored = [(s, _score_sentence(s, prefer_terms=prefer_terms, idf_map=idf_map)) for s in sents]
    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0][0].strip()

    # Trim to max_chars
    if len(best) > max_chars:
        best = best[: max_chars - 1].rstrip() + "…"
    return best

def generate_snippets(
    chunks: List[Dict],
    idf_map: Optional[Dict[str, float]] = None,
    prefer_terms: Optional[Sequence[str]] = None,
    max_chars: int = 200,
) -> None:
    """
    In-place snippet generation for a list of chunk dicts.
    Adds/overwrites `chunk["snippet"]`.
    """
    for ch in chunks:
        ch["snippet"] = make_snippet(
            ch,
            idf_map=idf_map,
            prefer_terms=prefer_terms,
            max_chars=max_chars,
            strip_boilerplate=True,
        )
