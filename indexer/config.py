# core/indexer/config.py
from __future__ import annotations

import os
from pathlib import Path

# ---------- Tunables ----------
DEFAULT_K = int(os.getenv("NN_DEFAULT_K", "10"))

# MMR (diversity) controls
DEFAULT_MMR_LAMBDA = float(os.getenv("NN_MMR_LAMBDA", "0.5"))
DEFAULT_MMR_FETCH  = int(os.getenv("NN_MMR_FETCH",  "50"))

# Light re-ranker controls
RERANK_POOL          = int(os.getenv("NN_RERANK_POOL", "50"))
OVERLAP_WEIGHT       = float(os.getenv("NN_OVERLAP_WEIGHT", "0.05"))
OVERLAP_CAP          = float(os.getenv("NN_OVERLAP_CAP",    "0.20"))
SAME_DOC_BOOST       = float(os.getenv("NN_SAME_DOC_BOOST", "0.05"))
NOISE_PENALTY        = float(os.getenv("NN_NOISE_PENALTY",  "0.10"))
CHAPTER_STRICT_BONUS = float(os.getenv("NN_CHAP_STRICT", "0.10"))
CHAPTER_COARSE_BONUS = float(os.getenv("NN_CHAP_COARSE", "0.05"))

# Phrase boost
PHRASE_WEIGHT        = float(os.getenv("NN_PHRASE_WEIGHT", "0.03"))
PHRASE_CAP           = float(os.getenv("NN_PHRASE_CAP",    "0.10"))

# NEW: heading bias for planning/action tasks
HEADING_BOOST        = float(os.getenv("NN_HEADING_BOOST", "0.08"))
HEADING_PENALTY      = float(os.getenv("NN_HEADING_PENALTY", "0.06"))

# Legacy vector store base dir (for backward compatibility)
BASE_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", str(Path(__file__).resolve().parents[1] / "vector_store")))

def collection_dir(collection_id_or_path: str, *, create: bool = False) -> Path:
    """
    Get the vector store directory for a collection.
    
    This function now supports workspace-based storage:
    - If collection_id_or_path is a collection ID (starts with 'col_'), 
      it will use the workspace-based path
    - If it's an absolute path, it will use that path directly
    - Otherwise, it falls back to the legacy BASE_STORE_DIR structure
    """
    p = Path(collection_id_or_path)
    
    # If it's an absolute path, use it directly
    if p.is_absolute():
        d = p
    # If it's a collection ID (starts with 'col_'), use workspace-based path
    elif collection_id_or_path.startswith('col_'):
        try:
            # Import workspace manager
            from core.workspace_manager import workspace_manager
            d = workspace_manager.get_vector_store_path(collection_id_or_path)
        except ImportError:
            # Fallback to legacy path if workspace manager not available
            d = BASE_STORE_DIR / collection_id_or_path
    else:
        # Legacy behavior for other paths
        d = BASE_STORE_DIR / collection_id_or_path
    
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d

# Minimal stopwords (English)
STOPWORDS = set("""
a an and are as at be but by for from has have in into is it its of on or that the their there these this to with your
""".split())

# Headings treated as noise-ish
NOISE_HEADINGS = {
    "table of contents",
    "acknowledgements",
    "copyright notice",
}
