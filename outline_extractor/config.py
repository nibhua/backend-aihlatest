# core/outline_extractor/config.py
import os

# Environment variable override, fallback to bundled model
MODEL_PATH = os.getenv(
    "HEADING_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "xgboost_heading_model.joblib")
)
