# core/embeddings/model.py
from __future__ import annotations

import re
from typing import List, Optional, Sequence
import numpy as np

from .config import (
    DEFAULT_DEVICE,
    BATCH_SIZE,
    NORMALIZE,
    resolve_model_path,
    DEFAULT_MODEL_ID,
)

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "sentence-transformers is required. "
        "Add `sentence-transformers>=2.6.0` to requirements.txt"
    ) from e


def _prep_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def _needs_e5_prefix(model_name_or_path: str) -> bool:
    return "e5" in (model_name_or_path or "").lower()


def _apply_prefixes(texts: Sequence[str], model_name_or_path: str, mode: str) -> List[str]:
    """
    E5 family expects 'query: ' vs 'passage: ' prefixes for best performance.
    mode: "query" | "passage" | "auto"
    """
    if mode not in ("query", "passage", "auto"):
        mode = "passage"
    if not _needs_e5_prefix(model_name_or_path):
        return list(texts)
    if mode == "auto":
        mode = "passage"
    pref = f"{mode}: "
    return [pref + t if t and not t.startswith(pref) else t for t in texts]


class EmbeddingModel:
    """Tiny wrapper around SentenceTransformer with batch/normalize + E5 prefixes."""

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: str = DEFAULT_DEVICE,
        normalize: bool = NORMALIZE,
    ) -> None:
        self.model_name_or_path = model_name_or_path or resolve_model_path(DEFAULT_MODEL_ID)
        self.device = device
        self.normalize = normalize

        self._model = SentenceTransformer(self.model_name_or_path, device=self.device)

        # Discover embedding dim
        try:
            self.dim = int(self._model.get_sentence_embedding_dimension())  # type: ignore[attr-defined]
        except Exception:
            probe = self._model.encode(["probe"], normalize_embeddings=True, convert_to_numpy=True)
            self.dim = int(probe.shape[1])

    def encode(
        self,
        texts: Sequence[str],
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None,
        mode: str = "passage",
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Encode a list/sequence of texts → (N, D) float32 array.
        Empty-safe: returns shape (0, D) if texts is empty.
        """
        # ---------- empty-safe short-circuit ----------
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # ---------------------------------------------

        if not isinstance(texts, (list, tuple)):
            texts = list(texts)

        cleaned = [_prep_text(t) for t in texts]
        prefixed = _apply_prefixes(cleaned, self.model_name_or_path, mode=mode)

        bs = batch_size or BATCH_SIZE
        do_norm = self.normalize if normalize is None else normalize

        vectors = self._model.encode(
            prefixed,
            batch_size=bs,
            normalize_embeddings=bool(do_norm),
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32, copy=False)
        return vectors

    def encode_one(self, text: str, normalize: Optional[bool] = None, mode: str = "passage") -> np.ndarray:
        return self.encode([text], batch_size=1, normalize=normalize, mode=mode)

    def encode_chunks_texts(
        self,
        chunks: Sequence[dict],
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None,
        mode: str = "passage",
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        texts = [(c.get("text") or "") for c in chunks]
        return self.encode(
            texts,
            batch_size=batch_size,
            normalize=normalize,
            mode=mode,
            show_progress_bar=show_progress_bar,
        )


def load_default_model() -> EmbeddingModel:
    return EmbeddingModel()
