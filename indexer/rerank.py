from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Sequence, Tuple

from .config import (
    STOPWORDS,
    NOISE_HEADINGS,
    OVERLAP_WEIGHT,
    OVERLAP_CAP,
    SAME_DOC_BOOST,
    NOISE_PENALTY,
    CHAPTER_STRICT_BONUS,
    CHAPTER_COARSE_BONUS,
    PHRASE_WEIGHT,
    PHRASE_CAP,
    HEADING_BOOST,
    HEADING_PENALTY,
)

_NON_ALNUM = re.compile(r"[^A-Za-z0-9 \-']+")
_CHAPTER_RX = re.compile(r"^\s*(\d+(?:\.\d+)*)\b")

_GENERIC_CAP_SKIP = {
    "roman", "medieval", "local",
    "summer", "winter", "spring", "autumn",
    "french", "english", "spanish", "italian", "german",
}

def _fold(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()

def norm(s: str) -> str:
    s = _fold(s)
    s = re.sub(r"\s+", " ", s.strip())
    return s

def tokens(s: str) -> List[str]:
    s = _NON_ALNUM.sub(" ", (s or ""))
    s = _fold(s)
    return [t for t in s.split() if 2 < len(t) < 20 and t not in STOPWORDS]

def extract_keywords(text: str) -> List[str]:
    seen, out = set(), []
    for t in tokens(text):
        if t not in seen:
            out.append(t); seen.add(t)
    return out

def _extract_capitalized_words(orig: str) -> List[str]:
    out, seen = [], set()
    for m in re.finditer(r"\b[A-Z][A-Za-zÀ-ÿ'\-]+\b", orig or ""):
        w = m.group(0); lw = _fold(w)
        if lw in STOPWORDS or len(lw) <= 2:
            continue
        if m.start() == 0 and lw in _GENERIC_CAP_SKIP and "-" not in lw:
            continue
        if lw not in seen:
            out.append(lw); seen.add(lw)
    return out

def _ngram_phrases(text: str, n: int) -> List[str]:
    toks = tokens(text)
    phrases, seen = [], set()
    for i in range(len(toks) - n + 1):
        window = toks[i:i+n]
        if not any(len(t) >= 4 for t in window):
            continue
        p = " ".join(window)
        if p not in seen:
            phrases.append(p); seen.add(p)
    return phrases

def extract_strong_terms(query_text: str) -> List[str]:
    terms, seen = [], set()
    for w in _extract_capitalized_words(query_text):
        if w not in seen:
            terms.append(w); seen.add(w)
    for n in (2, 3):
        for p in _ngram_phrases(query_text, n):
            if p not in seen:
                terms.append(p); seen.add(p)
    return terms

def chapter_prefix(h: str) -> str:
    m = _CHAPTER_RX.match(h or "")
    return m.group(1) if m else ""

def same_chapter_bonus(h1: str, h2: str) -> float:
    p1, p2 = chapter_prefix(h1), chapter_prefix(h2)
    if not p1 or not p2:
        return 0.0
    if p1 == p2 or p1.startswith(p2 + ".") or p2.startswith(p1 + "."):
        return CHAPTER_STRICT_BONUS
    if p1.split(".")[0] == p2.split(".")[0]:
        return CHAPTER_COARSE_BONUS
    return 0.0

def looks_noise_heading(h: str) -> bool:
    t = norm(h)
    if t in NOISE_HEADINGS:
        return True
    if re.fullmatch(r"page\s+\d+\s+of\s+\d+", t):
        return True
    return False

def overlap_bonus(query_terms: Sequence[str], cand_text: str,
                  weight: float = OVERLAP_WEIGHT, cap: float = OVERLAP_CAP) -> float:
    if not query_terms:
        return 0.0
    cand_terms = set(tokens(cand_text))
    hits = sum(1 for q in query_terms if q in cand_terms)
    return min(weight * hits, cap)

def phrase_boost(query_text: str, cand_text: str,
                 weight: float = PHRASE_WEIGHT, cap: float = PHRASE_CAP) -> float:
    c = norm(cand_text).replace("-", " ")
    phrases = _ngram_phrases(query_text, 2) + _ngram_phrases(query_text, 3)
    hits = 0
    for p in phrases:
        if " " in p and p in c:
            hits += 1
    return min(weight * hits, cap)

def _contains_any_terms(cand_text: str, terms: Sequence[str]) -> bool:
    if not terms:
        return False
    s = norm(cand_text).replace("-", " ")
    for t in terms:
        t = (t or "").strip().lower().replace("-", " ")
        if not t:
            continue
        if " " in t:
            if t in s:
                return True
        else:
            if re.search(rf"\b{re.escape(t)}\b", s):
                return True
    return False

def _heading_bias_score(heading: str, boost_headings: Sequence[str], down_headings: Sequence[str]) -> float:
    if not heading:
        return 0.0
    h = norm(heading)
    score = 0.0
    for b in boost_headings or []:
        b = (b or "").strip().lower()
        if not b:
            continue
        if re.search(rf"\b{re.escape(b)}\b", h):
            score += HEADING_BOOST
    for d in down_headings or []:
        d = (d or "").strip().lower()
        if not d:
            continue
        if re.search(rf"\b{re.escape(d)}\b", h):
            score -= HEADING_PENALTY
    return score

def explain_reason(query_terms: Sequence[str], cand_text: str, cand_heading: str,
                   base_heading: str | None) -> str:
    cand_terms = set(tokens(cand_text))
    words = sorted([w for w in set(query_terms) if w in cand_terms])[:5]
    chap = same_chapter_bonus(base_heading or "", cand_heading)
    if words and chap >= CHAPTER_STRICT_BONUS:
        return f"Shares terms like {', '.join(words)} and is in the same chapter family."
    if words:
        return f"Shares terms like {', '.join(words)}."
    if chap > 0:
        return "Same parent chapter."
    return "Semantically closest match."

def rerank_candidates(
    query_text: str,
    base_heading: str | None,
    base_doc: str | None,
    candidates: Sequence[Tuple[int, float]],
    chunks: List[Dict],
    k: int,
    use_overlap: bool = True,
    use_same_doc_boost: bool = True,
    use_chapter_bonus: bool = True,
    use_phrase_boost: bool = True,
    must_contain_terms: Sequence[str] = (),
    must_not_contain_terms: Sequence[str] = (),
    boost_headings: Sequence[str] = (),
    downweight_headings: Sequence[str] = (),
) -> List[Tuple[int, float]]:
    """
    Re-score & de-duplicate candidates using generic signals.
    - Filters out must_not_contain; backfills later if needed.
    - If must_contain_terms provided, prefer those, then backfill.
    - Applies heading boosts/penalties (planning-style tasks).
    """
    q_terms = extract_keywords(query_text)
    strong_terms = extract_strong_terms(query_text)

    seen_keys = set()
    scored: List[Tuple[int, float, bool, bool]] = []  # (idx, score, has_strong, excluded)
    excluded_for_backfill: List[Tuple[int, float]] = []

    for idx, base_score in candidates:
        ch = chunks[int(idx)]
        heading = ch.get("heading_text") or ch.get("heading") or ""
        key = (ch.get("doc_id", ""), int(ch.get("page", 0)), norm(heading))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        cand_text = f"{heading} {ch.get('text','')} {ch.get('snippet','')}"
        s = float(base_score)

        # Pre-check exclusions
        is_excluded = _contains_any_terms(cand_text, must_not_contain_terms)

        if looks_noise_heading(heading):
            s -= NOISE_PENALTY
        if use_same_doc_boost and base_doc and ch.get("doc_id") == base_doc:
            s += SAME_DOC_BOOST
        if use_chapter_bonus and base_heading:
            s += same_chapter_bonus(base_heading, heading)
        if use_overlap:
            s += overlap_bonus(q_terms, cand_text)
        if use_phrase_boost:
            s += phrase_boost(query_text, cand_text)
        s += _heading_bias_score(heading, boost_headings, downweight_headings)

        has_strong = _contains_any_terms(cand_text, strong_terms)

        if is_excluded:
            excluded_for_backfill.append((int(idx), s))
        else:
            scored.append((int(idx), s, has_strong, is_excluded))

    scored.sort(key=lambda t: t[1], reverse=True)
    excluded_for_backfill.sort(key=lambda t: t[1], reverse=True)

    def _has_must(item_idx: int) -> bool:
        ch = chunks[item_idx]
        cand_text = f"{ch.get('heading_text') or ch.get('heading') or ''} {ch.get('text','')} {ch.get('snippet','')}"
        return _contains_any_terms(cand_text, must_contain_terms)

    if must_contain_terms:
        primary = [(i, sc) for (i, sc, _hs, _ex) in scored if _has_must(i)]
        if len(primary) >= k:
            return primary[:k]
        rest = [(i, sc) for (i, sc, _hs, _ex) in scored if not _has_must(i)]
        combined = primary + rest
        if len(combined) < k and excluded_for_backfill:
            combined += excluded_for_backfill[: (k - len(combined))]
        return combined[:k]

    result = [(i, sc) for (i, sc, _hs, _ex) in scored[:k]]
    if len(result) < k and excluded_for_backfill:
        need = k - len(result)
        result += excluded_for_backfill[:need]
    return result
