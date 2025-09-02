from __future__ import annotations

import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

DEFAULT_K = 5
DEFAULT_POOL = 80
DEFAULT_MMR = True

# Intent → knob deltas (no YAML / lexicon usage)
TASK_RULES: List[Tuple[re.Pattern, Dict[str, Any]]] = [
    (re.compile(r"\b(compare|benchmark|versus|vs\.?|trade[- ]?offs?)\b", re.I),
     dict(k=9, mmr=True, pool=100)),
    (re.compile(r"\b(study|notes|checklist|exam|prepare|what to (learn|do))\b", re.I),
     dict(k=6, mmr=False, weights={"PHRASE_WEIGHT": 0.05, "SAME_DOC_BOOST": 0.08})),
    (re.compile(r"\b(analy[sz]e|analysis|metrics?|kpi|performance|revenue|profit|by (year|region)|table|trend)\b", re.I),
     dict(k=8, mmr=True, weights={"OVERLAP_WEIGHT": 0.08})),
    (re.compile(r"\b(overview|summary|introduction|intro|high[- ]?level)\b", re.I),
     dict(k=8, mmr=True, pool=90)),
]

# Small built-in fallbacks (kept; they don't depend on YAML)
FALLBACK_DIET = {
    "vegetarian": {"must_not": ["chicken", "beef", "pork", "fish", "shrimp", "lamb", "bacon", "sausage", "ham"]},
    "vegan": {"must_not": ["egg", "eggs", "milk", "cheese", "butter", "yogurt"]},
    "gluten-free": {"must_not": ["wheat", "flour", "breadcrumbs", "couscous", "pasta"]},
}
FALLBACK_TRAVEL_PLANNING = {
    "boost_headings": ["itinerary", "day", "accommodation", "transport", "booking", "reservations", "checklist"],
    "downweight_headings": ["introduction", "history", "background"],
}

# ---------- IO helpers (no YAML) ----------

def _load_collection_vocab(store_dir: Path) -> Tuple[set, Dict[str, float]]:
    p = store_dir / "collection_vocab.pkl"
    if not p.exists():
        return set(), {}
    try:
        terms, idf_map = pickle.loads(p.read_bytes())
        terms_l = [t.lower() for t in terms]
        idf_l = {k.lower(): float(v) for k, v in idf_map.items()}
        return set(terms_l), idf_l
    except Exception:
        return set(), {}

def _load_heading_graph(store_dir: Path) -> Dict[str, List[str]]:
    p = store_dir / "heading_graph.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {str(k): [str(x) for x in v] for k, v in data.items()}
    except Exception:
        return {}

# ---------- text helpers ----------

_BASIC_STOP = {
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "with", "by", "at",
    "is", "are", "was", "were", "be", "being", "been", "as", "that", "this", "these", "those",
    "from", "it", "its", "into", "about", "across", "over", "under", "their", "his", "her",
    "my", "our", "your", "you", "we", "they"
}
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")

def _tokens(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text) if len(t) > 2]

def _bigrams_trigrams(words: Sequence[str]) -> Iterable[str]:
    for i in range(len(words) - 1):
        yield f"{words[i]} {words[i+1]}"
    for i in range(len(words) - 2):
        yield f"{words[i]} {words[i+1]} {words[i+2]}"

def _candidate_terms(text: str) -> List[str]:
    ws = [w.lower() for w in _tokens(text)]
    ws = [w for w in ws if w not in _BASIC_STOP]
    grams = list(_bigrams_trigrams(ws))
    return ws + grams

def _collection_filter(terms: Iterable[str], vocab: set) -> List[str]:
    return [t for t in terms if t.lower() in vocab]

def _top_by_idf(terms: Iterable[str], idf_map: Dict[str, float], top_n: int = 6) -> List[str]:
    uniq = {t.lower() for t in terms}
    ranked = sorted(uniq, key=lambda t: idf_map.get(t, 0.0), reverse=True)
    return ranked[:top_n]

# ---------- constraint extraction (fallback-only; no YAML) ----------

_DIET_KEYS = re.compile(r"\b(vegetarian|vegan|gluten[- ]?free|dairy[- ]?free|nut[- ]?free)\b", re.I)
_TRAVEL_PLAN_KEYS = re.compile(r"\b(itinerary|day\s*[1-9]|accommodation|transport|booking|reservation|checklist)\b", re.I)

def _diet_constraints(text: str) -> Dict[str, Any]:
    knobs: Dict[str, Any] = {}
    m = _DIET_KEYS.findall(text or "")
    if not m:
        return knobs
    must_not: List[str] = []
    for tag in m:
        tag_l = tag.lower().replace(" ", "-")
        if tag_l in FALLBACK_DIET:
            must_not += FALLBACK_DIET[tag_l]["must_not"]
    knobs["must_not_contain"] = sorted({w.lower() for w in must_not})
    return knobs

def _planning_bias(text: str) -> Dict[str, Any]:
    if not _TRAVEL_PLAN_KEYS.search(text or ""):
        return {}
    boost = list(FALLBACK_TRAVEL_PLANNING["boost_headings"])
    down = list(FALLBACK_TRAVEL_PLANNING["downweight_headings"])
    return {"boost_headings": boost, "downweight_headings": down}

# ---------- main API ----------

def compile_recipe(persona: str, job: str, store_path: str) -> Dict[str, Any]:
    """
    Compile persona + job into multi-query + knobs for retrieval/rerank.
    No YAML/lexicon dependencies.
    """
    base_text = f"{persona.strip()} {job.strip()}".strip()
    knobs: Dict[str, Any] = dict(
        k=DEFAULT_K, pool=DEFAULT_POOL, mmr=DEFAULT_MMR,
        must_contain=[], must_not_contain=[], boost_headings=[], downweight_headings=[],
        weights={}
    )

    # Intent-based tweaks
    for pat, changes in TASK_RULES:
        if pat.search(base_text):
            for k, v in changes.items():
                if k == "weights":
                    knobs["weights"].update(v)
                else:
                    knobs[k] = v
            break

    store_dir = Path(store_path)
    vocab, idf_map = _load_collection_vocab(store_dir)
    heading_graph = _load_heading_graph(store_dir)

    # Constraints (fallback-only)
    knobs.update(_diet_constraints(base_text))
    knobs.update(_planning_bias(base_text))

    # Distil key terms
    cand = _candidate_terms(base_text)
    key_terms = _collection_filter(cand, vocab) if vocab else cand[:8]
    key_terms = _top_by_idf(key_terms, idf_map, top_n=6) if idf_map else key_terms[:6]

    # Build queries
    queries: List[str] = []
    if job.strip():
        queries.append(job.strip())
    elif persona.strip():
        queries.append(persona.strip())
    else:
        queries.append(base_text or "document")

    if key_terms:
        queries.append(" ".join(key_terms))
        neighbours: List[str] = []
        for term in key_terms:
            neighbours.extend(heading_graph.get(term, []))
        neighbours = [n for n, _ in Counter(neighbours).most_common() if n not in key_terms][:5]
        if neighbours:
            queries.append(" ".join(key_terms + neighbours))
        knobs["must_contain"] = key_terms[:2]

    return {"queries": queries, "knobs": knobs}
