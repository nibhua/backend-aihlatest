# core/outline_extractor/heading_extractor.py

import os
import re
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import joblib

from .config import MODEL_PATH

logger = logging.getLogger(__name__)
# Set default level via env (DEBUG for dev, INFO for prod)
logger.setLevel(logging.DEBUG if os.environ.get("HEADING_LOG_DEBUG") == "1" else logging.INFO)

# -------------------------------------------------------
# Optional XGBoost booster fast-path (avoids sklearn wrap)
# -------------------------------------------------------
_XGB_SKLEARN = None     # sklearn-wrapped XGBClassifier (joblib)
_XGB_BOOSTER = None     # raw xgboost.Booster (json/ubj/binary)
_XGB_USE = os.environ.get("HEADING_ML", "1") != "0"   # allow disabling via env

# Try import xgboost lazily when needed
def _try_import_xgb():
    try:
        import xgboost as xgb  # type: ignore
        return xgb
    except Exception:
        return None

def get_model():
    """
    Lazy-load model once.
    Prefers native XGBoost Booster if `MODEL_PATH` ends with known booster formats.
    Otherwise tries joblib (sklearn wrapper). Falls back to None (heuristics only).
    """
    global _XGB_SKLEARN, _XGB_BOOSTER

    if not _XGB_USE:
        logger.info("HEADING_ML=0 — using heuristics only.")
        return None

    if _XGB_SKLEARN is not None or _XGB_BOOSTER is not None:
        return _XGB_SKLEARN or _XGB_BOOSTER

    path = MODEL_PATH
    if not path or not os.path.exists(path):
        logger.warning("MODEL_PATH not found (%s). Using heuristics.", path)
        return None

    # Prefer Booster if file looks like one
    booster_exts = (".json", ".ubj", ".bst", ".bin")
    if path.lower().endswith(booster_exts):
        xgb = _try_import_xgb()
        if xgb is not None:
            try:
                booster = xgb.Booster()
                booster.load_model(path)
                _XGB_BOOSTER = booster
                logger.info("Loaded XGBoost Booster from %s", path)
                return _XGB_BOOSTER
            except Exception as e:
                logger.warning("Could not load Booster at %s: %s. Will try joblib.", path, e)

    # Fallback: sklearn wrapper via joblib (mmap if possible)
    try:
        _XGB_SKLEARN = joblib.load(path, mmap_mode="r")
        logger.info("Loaded sklearn XGBoost model from %s", path)
        return _XGB_SKLEARN
    except Exception as e:
        logger.warning("Could not load model at %s: %s — falling back to heuristics", path, e)
        _XGB_SKLEARN = None
        _XGB_BOOSTER = None
        return None

# -------------------------------------------------------
# Utilities
# -------------------------------------------------------

PAGE_FOOTER_RX = re.compile(r"(?i)^Page\s*\d+$")
NUMBERING_L3   = re.compile(r"^\d+\.\d+\.\d+")
NUMBERING_L2   = re.compile(r"^\d+\.\d+")
NUMBERING_L1   = re.compile(r"^\d+\.")

def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()

def _to_features_matrix(lines: List[Dict[str, Any]]) -> np.ndarray:
    """
    Vectorized features: [font_size, is_bold, x0, y0, x1, y1, page_width, page_height]
    Returns float32 ndarray of shape (N, 8)
    """
    X = np.empty((len(lines), 8), dtype=np.float32)
    for i, l in enumerate(lines):
        X[i] = [
            float(l.get("font_size", 0)),
            1.0 if l.get("is_bold", False) else 0.0,
            float(l["bbox"][0]), float(l["bbox"][1]), float(l["bbox"][2]), float(l["bbox"][3]),
            float(l.get("page_width", 0)), float(l.get("page_height", 0)),
        ]
    return X

def _ml_predict_flags(lines: List[Dict[str, Any]]):
    """
    Returns (preds, probas, impl) where:
      - preds: np.ndarray[int32] shape (N,)
      - probas: np.ndarray[float32] shape (N,) probability of class 1 (heading)
      - impl: 'booster' | 'sklearn' | None
    If no model present, returns (None, None, None).
    """
    model = get_model()
    if model is None:
        return None, None, None

    X = _to_features_matrix(lines)

    # Booster fast path
    try:
        import xgboost as xgb  # type: ignore
    except Exception:
        xgb = None

    if xgb is not None and hasattr(model, "predict") and hasattr(model, "attributes"):
        # Heuristic check: Booster has attributes() method; sklearn wrapper won't
        try:
            dmat = xgb.DMatrix(X)
            # If trained with binary:logistic
            raw = model.predict(dmat)  # probability per sample
            probas = np.asarray(raw, dtype=np.float32).reshape(-1)
            preds = (probas >= 0.5).astype(np.int32)
            return preds, probas, "booster"
        except Exception as e:
            logger.debug("Booster predict failed, will try sklearn path: %s", e)

    # sklearn wrapper path
    if hasattr(model, "predict"):
        try:
            preds = model.predict(X)
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(X)[:, 1]
            else:
                # neutral fallback
                probas = np.where(preds == 1, 0.75, 0.25)
            return np.asarray(preds, dtype=np.int32), np.asarray(probas, dtype=np.float32), "sklearn"
        except Exception as e:
            logger.warning("Sklearn model predict failed: %s. Using heuristics only.", e)
            return None, None, None

    return None, None, None

# -------------------------------------------------------
# Public API
# -------------------------------------------------------

def extract_title(lines: List[Dict[str, Any]]) -> str:
    """
    Pick the most prominent line on page 1 as the title,
    skipping footers like "Page 1" and requiring >=2 words.
    """
    if not lines:
        return ""
    first_page = [l for l in lines if l.get("page") == 1]
    first_page.sort(key=lambda l: (-float(l.get("font_size", 0)), float(l["bbox"][1])))
    for l in first_page:
        txt = clean_text(l.get("text", ""))
        if PAGE_FOOTER_RX.match(txt):
            continue
        if len(txt.split()) >= 2 and not re.match(r"^\d+(\.\d+)*$", txt):
            return txt
    return ""

def assign_levels(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    From extracted lines, pick headings and assign H1/H2/H3.
    Returns items with {level, text, page, bbox, confidence}.
    """
    if not lines:
        return []

    # Group by page to compute median font sizes
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for l in lines:
        by_page.setdefault(int(l.get("page", 1)), []).append(l)
    medians = {p: float(np.median([float(l.get("font_size", 0)) for l in ls])) for p, ls in by_page.items()}

    # Batch ML predictions (if available)
    preds = probas = None
    impl = None
    if _XGB_USE:
        preds, probas, impl = _ml_predict_flags(lines)
        if impl:
            logger.info("Heading ML using %s backend", impl)

    candidates: List[Dict[str, Any]] = []

    for idx, l in enumerate(lines):
        txt = clean_text(l.get("text", ""))
        if not txt or PAGE_FOOTER_RX.match(txt):
            continue

        page = int(l.get("page", 1))
        is_hdr = False
        conf = 0.5

        # ML-based detection
        if preds is not None and probas is not None:
            pred = int(preds[idx])
            p = float(probas[idx])
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Line "%s" ML pred=%s conf=%.2f', txt, pred, p)
            if pred == 1:
                is_hdr, conf = True, p

        # Heuristics
        if not is_hdr:
            body_med = medians.get(page, 0.0)
            fs = float(l.get("font_size", 0.0))
            is_bold = bool(l.get("is_bold", False))
            if is_bold and fs >= body_med * 1.2:
                is_hdr, conf = True, max(conf, 0.6)
            elif NUMBERING_L3.match(txt) or NUMBERING_L2.match(txt) or NUMBERING_L1.match(txt):
                is_hdr, conf = True, max(conf, 0.6)
            elif fs > body_med * 1.5:
                is_hdr, conf = True, max(conf, 0.6)

        # Single-line page rule—only on page 1
        if (not is_hdr
            and page == 1
            and len(by_page.get(page, [])) == 1
            and re.search(r"[A-Za-z]", txt)
        ):
            is_hdr, conf = True, max(conf, 0.8)

        if not is_hdr:
            continue

        # Assign H1/H2/H3 by font-size rank on that page
        fonts = sorted(
            {float(x.get("font_size", 0.0)) for x in by_page.get(page, []) if float(x.get("font_size", 0.0)) >= medians.get(page, 0.0)},
            reverse=True,
        )
        lvl_map = {fonts[i]: f"H{i+1}" for i in range(min(3, len(fonts)))}
        level = lvl_map.get(float(l.get("font_size", 0.0)), "H3")

        # Numbering overrides
        if NUMBERING_L3.match(txt):
            level = "H3"
        elif NUMBERING_L2.match(txt):
            level = "H2"
        elif NUMBERING_L1.match(txt):
            level = "H1"

        candidates.append({
            "level": level,
            "text": txt,
            "page": page,
            "bbox": l["bbox"],
            "confidence": float(conf),
        })

    # Fallback ONLY on page 1 and ONLY if no real headings found there
    page1_lines = by_page.get(1, [])
    page1_cands = [c for c in candidates if c["page"] == 1]
    if not page1_cands and page1_lines:
        bodies = [l for l in page1_lines if not PAGE_FOOTER_RX.match(clean_text(l.get("text", "")))]
        if 1 <= len(bodies) <= 3:
            for idx, l in enumerate(sorted(bodies, key=lambda x: -float(x.get("font_size", 0)))):
                txt = clean_text(l.get("text", ""))
                level = f"H{idx+1}"
                candidates.append({
                    "level": level,
                    "text": txt,
                    "page": 1,
                    "bbox": l["bbox"],
                    "confidence": 0.5,
                })
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Page1 fallback: %s → %s", txt, level)

    # Deduplicate & sort
    seen = set()
    outline = []
    for h in sorted(candidates, key=lambda x: (x["page"], x["bbox"][1], -x["confidence"])):
        key = (h["level"], h["text"], h["page"])
        if key in seen:
            continue
        seen.add(key)
        outline.append(h)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Heading added: %s (conf=%.2f)", h["text"], h["confidence"])

    return outline
