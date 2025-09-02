# chunker.py
from __future__ import annotations

import os
import re
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from functools import lru_cache

import fitz  # PyMuPDF

# --------------------------------
# Tunables
# --------------------------------
MAX_TOKENS = 512

# --------------------------------
# Precompiled regex (hot paths)
# --------------------------------
PAGE_OF_RE       = re.compile(r"(?mi)^page\s+\d+\s+of\s+\d+.*$")
VERSION_LINE_RE  = re.compile(r"(?mi)^version\s+\d{4}\s+page\s+\d+\s+of\s+\d+.*$")
COPYRIGHT_RE     = re.compile(r"(?mi)^©.*qualifications board.*$")
BULLET_ONLY_RE   = re.compile(r"\s*[•●▪■]\s*$")
TOO_SENTENCED_RE = re.compile(r"[.!?]{1}")
PAGE_FOOTER_RX   = re.compile(r"(?i)^Page\s*\d+$")

# number-like headings
_MONTHS = (
    r"JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUN(?:E)?|"
    r"JUL(?:Y)?|AUG(?:UST)?|SEP(?:T(?:EMBER)?)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?"
)
_DATE_HEADING_RX = re.compile(
    rf"^\s*(?:[0-3]?\d)\s+(?:{_MONTHS})\s+\d{{4}}\s*$",
    flags=re.IGNORECASE,
)

# --------------------------------
# Noise / furniture handling
# --------------------------------
NOISE_HEADINGS = {
    "table of contents",
}

def _is_noise_heading(t: str) -> bool:
    tnorm = re.sub(r"\s+", " ", (t or "")).strip().lower()
    return (
        tnorm in NOISE_HEADINGS
        or re.fullmatch(r"version\s+\d{4}", tnorm or "") is not None
        or re.fullmatch(r"page\s+\d+\s+of\s+\d+", tnorm or "") is not None
    )

@lru_cache(maxsize=512)
def _make_flexible_heading_pattern_cached(phrase: str) -> re.Pattern:
    """
    Case-insensitive regex for heading matching that tolerates:
      - arbitrary whitespace runs
      - hyphen/en-dash/em-dash variations
    Cached because headings repeat a lot across sections.
    """
    pat = re.escape(phrase or "")
    pat = pat.replace(r"\ ", r"\s+")
    pat = pat.replace(r"\–", r"[-–—]").replace(r"\—", r"[-–—]").replace(r"\-", r"[-–—]")
    return re.compile(pat, flags=re.IGNORECASE)

def _regex_find_span(haystack: str, phrase: str) -> Tuple[int, int]:
    """Fast exact match first; fall back to flexible cached regex."""
    if not haystack or not phrase:
        return (-1, -1)
    pos = haystack.find(phrase)
    if pos != -1:
        return (pos, pos + len(phrase))
    rx = _make_flexible_heading_pattern_cached(phrase)
    m = rx.search(haystack)
    return (m.start(), m.end()) if m else (-1, -1)

def _normalize_line(s: str) -> str:
    # compact spaces, strip bullets/punct, lowercase
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip("-–—•●▪■·.,:;|")
    return s.lower()

# --- table-ish detection helpers ---

_TABLE_BORDER_CHARS = "│┼┤┐┌└┘─━═─—–│┃┏┓┗┛"

def _looks_tabular_line(raw: str) -> bool:
    """Heuristic: row with multiple columns, pipes, or box-drawing/border chars."""
    if not raw:
        return False
    s = raw.rstrip("\n")
    if any(c in s for c in _TABLE_BORDER_CHARS) or s.count("|") >= 2:
        return True
    if s.count("\t") >= 1:
        return True
    if len(re.findall(r"  {1,}", s)) >= 2:
        return True
    if re.search(r"\bversion\b\s{2,}\bdate\b\s{2,}", s, flags=re.IGNORECASE):
        return True
    return False

def _strip_tabular_blocks(text: str) -> str:
    """
    Remove contiguous table-ish blocks (>=2 consecutive table-like lines).
    Optimized with a prepass of flags to avoid duplicate work.
    """
    if not text:
        return ""
    lines = text.splitlines()
    flags = [_looks_tabular_line(ln) for ln in lines]
    out: List[str] = []
    i, n = 0, len(lines)
    while i < n:
        if flags[i]:
            j = i + 1
            while j < n and flags[j]:
                j += 1
            if j - i >= 2:
                i = j
                if out and out[-1].strip() != "":
                    out.append("")
                continue
        out.append(lines[i])
        i += 1
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# --- repeating header/footer detection ---

def _collect_repeating_short_lines(
    page_texts: List[str],
    max_len: int = 80,
    min_pages_abs: int = 3,
    min_frac: float = 0.4,
) -> Set[str]:
    """
    Find short lines that repeat across many pages (likely headers/footers).
    Count presence-per-page to avoid duplicates within a page.
    """
    if not page_texts:
        return set()

    page_count = len(page_texts)
    thresh = max(min_pages_abs, int(math.ceil(min_frac * page_count)))

    counts: Dict[str, int] = {}
    for pg_txt in page_texts:
        seen_this_page: Set[str] = set()
        for raw in (pg_txt or "").splitlines():
            if not raw:
                continue
            ln = _normalize_line(raw)
            if not ln or len(ln) > max_len:
                continue
            if TOO_SENTENCED_RE.search(ln):
                continue
            if PAGE_OF_RE.fullmatch(ln):
                continue
            seen_this_page.add(ln)
        for ln in seen_this_page:
            counts[ln] = counts.get(ln, 0) + 1

    repeating = {ln for ln, c in counts.items() if c >= thresh}
    return repeating

def _clean_body_text(text: str, repeating_lines: Set[str]) -> str:
    """Strip page furniture and orphan bullets; normalize blank lines; drop table-ish blocks."""
    if not text:
        return ""

    text = VERSION_LINE_RE.sub("", text)
    text = COPYRIGHT_RE.sub("", text)

    cleaned_lines: List[str] = []
    for raw in text.splitlines():
        if _normalize_line(raw) in repeating_lines:
            continue
        if BULLET_ONLY_RE.fullmatch(raw or ""):
            continue
        cleaned_lines.append(raw)
    text = "\n".join(cleaned_lines)

    text = _strip_tabular_blocks(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# -------------------------------
# Basic helpers
# -------------------------------

def _read_page_texts_from_open_doc(doc: fitz.Document) -> List[str]:
    """Return plain text per page using PyMuPDF from an already-open doc."""
    return [doc[i].get_text("text") or "" for i in range(doc.page_count)]

def _count_tokens(txt: str) -> int:
    # Simple whitespace token proxy
    return len((txt or "").split())

def _split_paragraphs(txt: str) -> List[str]:
    # Split on blank lines, keep only non-empty trimmed paragraphs
    if not txt:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", txt) if p.strip()]
    paras = [p for p in paras if not BULLET_ONLY_RE.fullmatch(p)]
    return paras

def _first_sentences(text: str, n: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return " ".join(parts[:n]).strip()

# -------------------------------
# Section extraction & chunking
# -------------------------------

def _looks_like_date_heading(t: str) -> bool:
    return bool(_DATE_HEADING_RX.match(t or ""))

def _section_body_between(
    page_texts: List[str],
    heading: Dict[str, Any],
    next_heading: Optional[Dict[str, Any]],
    repeating_lines: Set[str],
) -> str:
    """
    Return text from current heading → right before the next heading.
    Start page: after heading. End page: before next heading.
    """
    start_page = int(heading["page"])  # 1-based
    end_inclusive = int(next_heading["page"]) if next_heading else len(page_texts)

    parts: List[str] = []
    for pg in range(start_page, end_inclusive + 1):
        text = page_texts[pg - 1]  # to 0-based

        if pg == start_page:
            _, end_idx = _regex_find_span(text, heading["text"])
            if end_idx != -1:
                text = text[end_idx:].lstrip()

        if next_heading and pg == end_inclusive:
            start_idx, _ = _regex_find_span(text, next_heading["text"])
            if start_idx != -1:
                text = text[:start_idx].rstrip()

        text = _clean_body_text(text, repeating_lines)
        if text:
            parts.append(text)

    return "\n\n".join(parts).strip()

def _chunk_section(
    doc_id: str,
    heading_text: str,
    page: int,
    bbox: Any,
    body_paragraphs: List[str],
) -> List[Dict[str, Any]]:
    """
    Pack heading + subsequent paragraphs into ≤MAX_TOKENS chunks (don’t split paragraphs).
    """
    chunks: List[Dict[str, Any]] = []
    cur: List[str] = []
    cur_tokens = _count_tokens(heading_text)

    for para in body_paragraphs:
        p_tokens = _count_tokens(para)
        if cur and cur_tokens + p_tokens > MAX_TOKENS:
            text = heading_text + "\n\n" + "\n\n".join(cur)
            chunks.append({
                "doc_id": doc_id,
                "heading_text": heading_text,
                "page": page,
                "bbox": bbox,
                "text": text,
                "snippet": _first_sentences(text, 2),
            })
            cur, cur_tokens = [], _count_tokens(heading_text)

        cur.append(para)
        cur_tokens += p_tokens

    if cur:
        text = heading_text + "\n\n" + "\n\n".join(cur)
        chunks.append({
            "doc_id": doc_id,
            "heading_text": heading_text,
            "page": page,
            "bbox": bbox,
            "text": text,
            "snippet": _first_sentences(text, 2),
        })

    if not chunks:
        text = heading_text
        chunks.append({
            "doc_id": doc_id,
            "heading_text": heading_text,
            "page": page,
            "bbox": bbox,
            "text": text,
            "snippet": _first_sentences(text, 2),
        })

    return chunks

def _sort_headings(outline_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Sort by (page, vertical y of bbox). bbox = (x0, y0, x1, y1)
    return sorted(
        outline_items,
        key=lambda h: (int(h["page"]), float(h["bbox"][1] if h.get("bbox") else 0.0))
    )

# -------------------------------
# Geometry helpers
# -------------------------------

def _xyxy(b):
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    x0, y0, x1, y1 = [float(v) for v in b]
    xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
    ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)
    return [xmin, ymin, xmax, ymax]

def _tl_to_bl_xyxy(b_xyxy, page_h: float):
    # incoming TL-origin (y down) -> BL-origin (y up)
    xmin, ymin, xmax, ymax = b_xyxy
    return [xmin, page_h - ymax, xmax, page_h - ymin]

def _rect_intersects_vert(r: List[float], y0: float, y1: float) -> bool:
    ry0, ry1 = float(r[1]), float(r[3])
    return not (ry1 <= y0 or ry0 >= y1)

def _clip_rect_to_y(r: List[float], y0: float, y1: float) -> Optional[List[float]]:
    """Clip rect to vertical band [y0, y1] (BL origin). Return None if empty."""
    x0, ry0, x1, ry1 = [float(v) for v in r]
    low = max(ry0, y0)
    high = min(ry1, y1)
    if high - low < 1.0:
        return None
    return [x0, low, x1, high]

def _union_rects(rects: List[List[float]]) -> Optional[List[float]]:
    if not rects:
        return None
    xs0 = min(r[0] for r in rects)
    ys0 = min(r[1] for r in rects)
    xs1 = max(r[2] for r in rects)
    ys1 = max(r[3] for r in rects)
    if ys1 - ys0 < 1.0 or xs1 - xs0 < 1.0:
        return None
    return [xs0, ys0, xs1, ys1]

# -------------------------------
# Text blocks (ink) collection
# -------------------------------

def _collect_page_blocks_bl(doc: fitz.Document) -> Dict[int, List[List[float]]]:
    """
    For each 1-based page index, collect rectangles (BL origin) for text blocks with non-empty text.
    """
    out: Dict[int, List[List[float]]] = {}
    for i in range(doc.page_count):
        page = doc[i]
        h = float(page.rect.height)
        blocks = page.get_text("blocks") or []
        rects: List[List[float]] = []
        for b in blocks:
            # PyMuPDF "blocks" entries: (x0, y0, x1, y1, text, block_no, ...)
            if len(b) < 5:
                continue
            x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
            if not (isinstance(txt, str) and txt.strip()):
                continue
            bb = _xyxy([x0, y0, x1, y1])
            if bb is None:
                continue
            rects.append(_tl_to_bl_xyxy(bb, h))
        if rects:
            out[i + 1] = rects
    return out

# -------------------------------
# Ink-aware per-section regions (BL origin)
# -------------------------------

def _compute_section_regions(
    start_page: int,
    start_bbox_bl: Optional[List[float]],
    next_page: Optional[int],
    next_bbox_bl: Optional[List[float]],
    page_sizes: List[Tuple[float, float]],
    blocks_by_page_bl: Dict[int, List[List[float]]],
) -> Dict[int, List[List[float]]]:
    """
    Returns a dict: { page_number (1-based): [ [x0,y0,x1,y1], ... ] } in BL origin.

    Uses real text blocks on each relevant page and unions them inside the
    vertical band for the section.
    """
    # Safe margins in PDF points
    MARGIN_LEFT = 54.0
    MARGIN_RIGHT = 54.0
    MARGIN_BOTTOM = 36.0
    MARGIN_TOP = 72.0
    PAD_BELOW_HEADING = 6.0   # gap under the heading (towards bottom)
    PAD_ABOVE_NEXT = 4.0      # gap above the next heading (towards top)

    n_pages = len(page_sizes)
    end_page = int(next_page) if next_page else n_pages
    start_page = max(1, int(start_page))
    end_page = max(start_page, min(end_page, n_pages))

    regions: Dict[int, List[List[float]]] = {}

    for p in range(start_page, end_page + 1):
        width, height = page_sizes[p - 1]
        # Default vertical band = text area between margins
        band_y0 = MARGIN_BOTTOM
        band_y1 = height - MARGIN_TOP

        # Start page: cap TOP of band to just below current heading
        if p == start_page and start_bbox_bl is not None:
            heading_ymin = float(start_bbox_bl[1])  # lower edge in BL origin
            band_y1 = min(band_y1, max(band_y0 + 1.0, heading_ymin - PAD_BELOW_HEADING))

        # Page with the next heading: raise BOTTOM to just above next heading
        if next_page == p and next_bbox_bl is not None:
            next_ymax = float(next_bbox_bl[3])  # upper edge
            band_y0 = max(band_y0, min(band_y1 - 1.0, next_ymax + PAD_ABOVE_NEXT))

        kept: List[List[float]] = []
        for r in blocks_by_page_bl.get(p, []):
            if _rect_intersects_vert(r, band_y0, band_y1):
                rc = _clip_rect_to_y(r, band_y0, band_y1)
                if rc:
                    kept.append(rc)

        if kept:
            u = _union_rects(kept)
            if u:
                regions.setdefault(p, []).append(u)

        if not regions.get(p):
            x0 = MARGIN_LEFT
            x1 = max(x0 + 1.0, width - MARGIN_RIGHT)
            if band_y1 - band_y0 >= 4.0:
                regions.setdefault(p, []).append([x0, band_y0, x1, band_y1])

    return regions

# -------------------------------
# Main builder
# -------------------------------

def build_chunks_for_pdf(pdf_path: str, outline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build heading-anchored chunks for a single PDF.
    Requires Phase-1 outline:
        outline = {"title": str, "outline": [{"level","text","page","bbox","confidence"}, ...]}
    Returns a list of chunks:
        {
          doc_id, heading_text, page, bbox, text, snippet,
          page_height_pt, bbox_origin,
          content_regions   # { "<page>": [[x0,y0,x1,y1], ...] } in BL origin
        }
    """
    doc_id = os.path.basename(pdf_path)
    items = outline.get("outline", [])
    if not items:
        return []

    # Drop obvious noise headings so they don't become chunks
    items = [h for h in items if not _is_noise_heading(h.get("text", ""))]
    items = _sort_headings(items)

    # Open once (faster; safe)
    with fitz.open(pdf_path) as _doc:
        page_texts = _read_page_texts_from_open_doc(_doc)
        page_sizes = [(_doc[i].rect.width, _doc[i].rect.height) for i in range(_doc.page_count)]
        page_heights = [sz[1] for sz in page_sizes]
        blocks_by_page_bl = _collect_page_blocks_bl(_doc)

        # Repeating short lines once per PDF (headers/footers)
        repeating_lines = _collect_repeating_short_lines(page_texts)

        # Inside "Revision History", ignore date-only pseudo-headings
        filtered_items: List[Dict[str, Any]] = []
        last_major_heading_text: Optional[str] = None
        for h in items:
            text = h.get("text", "") or ""
            if not _looks_like_date_heading(text):
                last_major_heading_text = text
                filtered_items.append(h)
                continue
            if last_major_heading_text and re.search(r"revision\s+history", last_major_heading_text, re.IGNORECASE):
                continue
            filtered_items.append(h)
        items = filtered_items

        chunks: List[Dict[str, Any]] = []
        seen_heading_once: Set[str] = set()   # normalized heading texts we've already emitted

        for i, h in enumerate(items):
            page = int(h.get("page") or 1)
            if not (1 <= page <= len(page_heights)):
                page = 1
            page_h = float(page_heights[page - 1])

            # TL -> BL conversion for heading bbox (if present)
            bbox_raw = h.get("bbox")
            bbox_bl = None
            bxy = _xyxy(bbox_raw) if bbox_raw is not None else None
            if bxy is not None:
                bbox_bl = _tl_to_bl_xyxy(bxy, page_h)

            next_h = items[i + 1] if i + 1 < len(items) else None

            # Robust body text
            body = _section_body_between(page_texts, h, next_h, repeating_lines)
            paras = _split_paragraphs(body) if body else []
            produced = _chunk_section(doc_id, h["text"], page, bbox_bl, paras)

            # Per-section content regions (ink-aware)
            next_page = int(next_h.get("page")) if next_h and next_h.get("page") is not None else None
            next_bbox_bl = None
            if next_h is not None and isinstance(next_h.get("bbox"), (list, tuple)) and next_page:
                n_page_h = float(page_heights[max(0, next_page - 1)])
                nbxy = _xyxy(next_h["bbox"])
                if nbxy is not None:
                    next_bbox_bl = _tl_to_bl_xyxy(nbxy, n_page_h)

            section_regions = _compute_section_regions(
                start_page=page,
                start_bbox_bl=bbox_bl,
                next_page=next_page,
                next_bbox_bl=next_bbox_bl,
                page_sizes=page_sizes,
                blocks_by_page_bl=blocks_by_page_bl,
            )
            content_regions_json: Dict[str, List[List[float]]] = {str(k): v for k, v in section_regions.items()}

            # annotate chunks
            h_norm = _normalize_line(h["text"])
            for ch in produced:
                ch["page_height_pt"] = page_h
                ch["bbox_origin"] = "BL" if bbox_bl is not None else None
                ch["content_regions"] = content_regions_json

                # conservative "leftover continuation" filter (tiny bodies skipped if already emitted this heading)
                body_only = (ch["text"] or "")[len(ch["heading_text"] or ""):].strip()
                is_tiny = len(body_only) < 25 or not re.search(r"[.!?;:]", body_only)
                if h_norm in seen_heading_once and is_tiny:
                    continue
                chunks.append(ch)

            if any(c for c in produced):
                seen_heading_once.add(h_norm)

        return chunks
