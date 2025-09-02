# core/embeddings/config.py
import os
from pathlib import Path

# -------- Defaults (override via env) --------
DEFAULT_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "intfloat/e5-base-v2")
DEFAULT_DEVICE   = os.getenv("EMBEDDING_DEVICE", "cpu")       # "cpu" only for now
BATCH_SIZE       = int(os.getenv("EMBEDDING_BATCH_SIZE", "128"))
NORMALIZE        = os.getenv("EMBEDDING_NORMALIZE", "1") != "0"

# Optional local bundle (preferred if present)
# You can drop your model files under: core/models/<whatever> and point to it here.
# Highest priority is EMBEDDING_MODEL_PATH if set.
LOCAL_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "").strip()

# A sensible fallback local directory (if you vendor the model):
# e.g., core/models/e5-base-v2
FALLBACK_LOCAL_DIR = (
    Path(__file__).resolve().parents[1] / "models" / "e5-base-v2"
)

def resolve_model_path(default_hf_id: str = DEFAULT_MODEL_ID) -> str:
    """
    Return the best-available model path:
    1) EMBEDDING_MODEL_PATH if it exists
    2) core/models/e5-base-v2 if it exists
    3) else the HF ID (requires internet at first run)
    """
    if LOCAL_MODEL_PATH and Path(LOCAL_MODEL_PATH).exists():
        return LOCAL_MODEL_PATH
    if FALLBACK_LOCAL_DIR.exists():
        return str(FALLBACK_LOCAL_DIR)
    return default_hf_id
