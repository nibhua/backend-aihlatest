# core/indexer/nn_store.py
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from embeddings import EmbeddingModel, load_default_model
from .config import (
    collection_dir,
    DEFAULT_K,
    DEFAULT_MMR_FETCH,
    DEFAULT_MMR_LAMBDA,
)

# Adobe-themed adjectives for collection names
ADJECTIVES = [
    "masked", "gradient", "pixelated", "vectorized", "layered", "blurry",
    "clipping", "cropped", "rendered", "frosted", "hazy", "noisy", "undoable",
    "redoable", "transparent", "smart", "bezier", "baseline", "swatchy",
    "pantoned", "kerning", "shaded", "stroked", "blended", "liquified",
    "marquee", "gaussian", "opacity", "drop-shadowed", "animated",
    "outlined", "compressed", "exported", "flattened", "hidden", "revealed",
    "posterized", "stylized", "comic", "vivid", "contrasty", "saturated",
    "highlighted", "masked-out", "cropped-in", "snappy", "puppeted"
]

# Adobe-themed nouns for collection names
NOUNS = [
    "otter", "raccoon", "llama", "badger", "narwhal", "penguin", "puffin",
    "platypus", "beetle", "jaguar", "vulture", "panda", "koala", "dolphin",
    "hedgehog", "sloth", "alpaca", "sparrow", "moose", "goose", "parrot",
    "lobster", "capybara", "newt", "donkey", "orca", "gazelle", "tarantula",
    "gecko", "lynx", "toad", "bat", "owl", "peacock", "falcon", "stingray",
    "mole", "tortoise", "rhino", "shark", "eel", "camel", "sheep", "goat",
    "ferret", "crab", "otterling"
]

def _generate_adobe_collection_name() -> str:
    """
    Generates a random, Adobe-flavored collection name from a list of adjectives and nouns.
    
    Returns:
        A string representing the random collection name.
    """
    adj = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    num = random.randint(1000, 9999)
    return f"{adj}-{noun}-{num}"

def _sha1_of_names(paths: Sequence[str]) -> str:
    h = hashlib.sha1()
    for p in sorted([Path(x).name for x in paths]):
        h.update(p.encode("utf-8", errors="ignore"))
    return h.hexdigest()

def make_collection_id(pdf_paths: Sequence[str]) -> str:
    return "col_" + _generate_adobe_collection_name()

def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    eps = 1e-12
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + eps
    return mat / norms

@dataclass
class NNStore:
    """
    Lightweight cosine store over normalized embeddings.
    Files:
      - embeddings.npy  (float16)
      - chunks.json
      - meta.json
    """
    embeddings: np.ndarray  # (N, D) float32/float16, L2-normalized
    chunks: List[Dict]
    meta: Dict

    # ---------- Build / Save / Load ----------
    @staticmethod
    def build(embeddings: np.ndarray, chunks: List[Dict], meta: Dict) -> "NNStore":
        if embeddings.dtype not in (np.float32, np.float16):
            embeddings = embeddings.astype(np.float32, copy=False)
        embeddings = _l2_normalize(embeddings.astype(np.float32, copy=False))
        return NNStore(embeddings=embeddings, chunks=list(chunks), meta=dict(meta))

    def save(self, coll_id_or_path: str) -> Path:
        out_dir = collection_dir(coll_id_or_path, create=True)
        np.save(out_dir / "embeddings.npy", self.embeddings.astype(np.float16, copy=False))
        (out_dir / "chunks.json").write_text(json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_dir

    @staticmethod
    def load(coll_id_or_path: str) -> "NNStore":
        d = collection_dir(coll_id_or_path, create=False)
        emb = np.load(d / "embeddings.npy").astype(np.float32, copy=False)
        chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        return NNStore(embeddings=emb, chunks=chunks, meta=meta)

    # ---------- Search primitives ----------
    def cosine_search(self, qvec: np.ndarray, top_k: int = DEFAULT_K) -> Tuple[np.ndarray, np.ndarray]:
        """
        One query vector → top_k (indices, scores).
        Assumes qvec is normalized (we'll normalize just in case).
        """
        if qvec.ndim == 2:
            qvec = qvec[0]
        q = _l2_normalize(qvec[None, :].astype(np.float32, copy=False))[0]
        scores = self.embeddings @ q
        # Clamp to [-1, 1] to avoid >1 due to float16 round-trips (cosmetic)
        scores = np.clip(scores, -1.0, 1.0)
        if top_k < len(scores):
            idxs = np.argpartition(-scores, top_k)[:top_k]
            idxs = idxs[np.argsort(-scores[idxs])]
        else:
            idxs = np.argsort(-scores)
        return idxs, scores[idxs]

    def search_vectors(self, query_vecs: np.ndarray, k: int = DEFAULT_K) -> Tuple[np.ndarray, np.ndarray]:
        """Batch: (Q,D) queries → (Q,k) indices/scores."""
        if query_vecs.ndim == 1:
            query_vecs = query_vecs[None, :]
        q = _l2_normalize(query_vecs.astype(np.float32, copy=False))
        sims = q @ self.embeddings.T  # (Q, N)
        # Clamp for the same reason as above
        sims = np.clip(sims, -1.0, 1.0)
        idx = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
        row_sorted = np.take_along_axis(sims, idx, axis=1)
        order = np.argsort(-row_sorted, axis=1)
        top_idx = np.take_along_axis(idx, order, axis=1)
        top_scores = np.take_along_axis(sims, top_idx, axis=1)
        return top_idx, top_scores

    def search_texts(
        self,
        texts: Sequence[str],
        embedder: EmbeddingModel | None = None,
        k: int = DEFAULT_K,
        mmr: bool = False,
        mmr_lambda: float = DEFAULT_MMR_LAMBDA,
        mmr_fetch: int = DEFAULT_MMR_FETCH,
    ) -> List[List[Tuple[int, float]]]:
        """
        Encode queries with E5 'query:' prefix and search.
        If mmr=True, perform MMR re-ranking per query.
        Returns list per query of (chunk_index, score).
        """
        embedder = embedder or load_default_model()
        qvecs = embedder.encode(texts, mode="query", normalize=True)

        if not mmr:
            idx, sc = self.search_vectors(qvecs, k=k)
            return [[(int(i), float(s)) for i, s in zip(row_i, row_s)]
                    for row_i, row_s in zip(idx, sc)]

        # MMR
        pool_k = min(mmr_fetch, self.embeddings.shape[0])
        idx, sc = self.search_vectors(qvecs, k=pool_k)
        results: List[List[Tuple[int, float]]] = []

        for qi in range(qvecs.shape[0]):
            cand_idx = idx[qi].tolist()
            cand_scores = sc[qi].tolist()
            selected: List[int] = []
            selected_scores: List[float] = []

            while cand_idx and len(selected) < k:
                if not selected:
                    best = int(np.argmax(cand_scores))
                    selected.append(cand_idx.pop(best))
                    selected_scores.append(cand_scores.pop(best))
                else:
                    sel_vecs = self.embeddings[selected]                      # (S, D)
                    rest_vecs = self.embeddings[cand_idx]                     # (C, D)
                    penalty = np.max(rest_vecs @ sel_vecs.T, axis=1)          # (C,)
                    # Clamp penalty sims too (safety)
                    penalty = np.clip(penalty, -1.0, 1.0)
                    mmr_scores = mmr_lambda * np.array(cand_scores) - (1 - mmr_lambda) * penalty
                    pick = int(np.argmax(mmr_scores))
                    selected.append(cand_idx.pop(pick))
                    selected_scores.append(cand_scores.pop(pick))

            results.append(list(zip(selected, selected_scores)))

        return results
