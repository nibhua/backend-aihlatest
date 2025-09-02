"""
One-time download to make the model fully offline.

Run:
  python -m scripts.download_e5_base_v2
"""

import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Get the absolute path to the project root (one level up from this file's folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Set the destination to <project_root>/models/e5-base-v2
DEST = PROJECT_ROOT / "models" / "e5-small-v2"

if __name__ == "__main__":
    os.makedirs(DEST, exist_ok=True)
    print(f"Downloading intfloat/e5-base-v2 → {DEST}")
    model = SentenceTransformer("intfloat/e5-small-v2")
    model.save(str(DEST))
    print("Done.")
