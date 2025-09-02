import json
from typing import Any, Dict

ENVELOPE_KEYS = {"collection_id", "summary_type", "content"}

def unwrap_content_if_envelope(content_str: str) -> str:
    """
    If content_str is actually a JSON string of the full envelope
    (i.e., {"collection_id":..., "summary_type":..., "content": "..."}),
    return the inner 'content' string. Otherwise, return content_str.
    """
    try:
        obj = json.loads(content_str)
        if isinstance(obj, dict) and ENVELOPE_KEYS.issubset(obj.keys()):
            inner = obj.get("content")
            if isinstance(inner, str) and inner.strip():
                return inner
    except Exception:
        pass
    return content_str 