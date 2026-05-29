from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

COLBERT_INDEX_ROOT_DEFAULT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "knowledge_build"
    / "data"
    / "pcf_policy_r18"
    / "colbert_index"
)
EMBEDDINGS_DIR_DEFAULT = COLBERT_INDEX_ROOT_DEFAULT / "embeddings"
INDEX_META_DEFAULT = COLBERT_INDEX_ROOT_DEFAULT / "index_meta.json"

_MAXSIM_WARM_BATCH = 64


def _index_available() -> bool:
    return INDEX_META_DEFAULT.exists() and EMBEDDINGS_DIR_DEFAULT.is_dir()


def _maxsim_score(
    query_emb: np.ndarray,
    doc_emb: np.ndarray,
) -> float:
    """Compute ColBERT MaxSim score between normalized token embeddings.

    query_emb: [Q, D]  — L2-normalized per-token query vectors
    doc_emb:   [T, D]  — L2-normalized per-token document vectors

    Returns sum over query tokens of max cosine similarity against all doc tokens.
    """
    similarities = doc_emb @ query_emb.T
    max_per_query = similarities.max(axis=0)
    return float(max_per_query.sum())


def _maxsim_batch(
    query_emb: np.ndarray,
    doc_embs: List[np.ndarray],
) -> np.ndarray:
    """Compute MaxSim scores for a batch of documents against a single query."""
    scores = np.zeros(len(doc_embs), dtype=np.float32)
    for i, doc_emb in enumerate(doc_embs):
        scores[i] = _maxsim_score(query_emb, doc_emb)
    return scores


class ColBERTRetriever:
    """MaxSim retrieval over a pre-built ColBERT per-token embedding index.

    Loads document embeddings from disk on demand and computes ColBERT-style
    late interaction scores for retrieval and re-ranking.
    """

    def __init__(
        self,
        index_root: Optional[Path] = None,
    ):
        root = Path(index_root) if index_root else COLBERT_INDEX_ROOT_DEFAULT
        self._index_root = root
        self._embeddings_dir = root / "embeddings"
        self._meta_path = root / "index_meta.json"

        if not self._meta_path.exists():
            raise FileNotFoundError(
                f"ColBERT index metadata not found at {self._meta_path}. "
                "Run `python knowledge_build/scripts/build_colbert_index.py` first."
            )

        with self._meta_path.open("r", encoding="utf-8") as f:
            self._meta: Dict[str, Any] = json.load(f)

        self._model_name: str = self._meta["model_name"]
        self._dim: int = self._meta["dim"]
        self._records_meta: Dict[str, Dict[str, Any]] = self._meta["records"]
        self._cache: Dict[str, np.ndarray] = {}

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def record_count(self) -> int:
        return len(self._records_meta)

    def record_ids(self) -> List[str]:
        return list(self._records_meta.keys())

    def _load_embedding(self, record_id: str) -> Optional[np.ndarray]:
        if record_id in self._cache:
            return self._cache[record_id]

        record_meta = self._records_meta.get(record_id)
        if record_meta is None:
            return None

        filepath = self._embeddings_dir / record_meta["file"]
        if not filepath.exists():
            return None

        with np.load(str(filepath)) as data:
            emb = data["tokens"].astype(np.float32)
        self._cache[record_id] = emb
        return emb

    def _load_embeddings_batch(
        self,
        record_ids: List[str],
    ) -> List[Optional[np.ndarray]]:
        return [self._load_embedding(rid) for rid in record_ids]

    def preload(self, record_ids: Optional[List[str]] = None) -> None:
        """Preload embeddings into memory for faster retrieval."""
        ids = record_ids if record_ids is not None else list(self._records_meta.keys())
        for rid in ids:
            self._load_embedding(rid)

    def retrieve(
        self,
        query_emb: np.ndarray,
        top_k: int = 20,
        domain_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Tuple[str, float, np.ndarray]]:
        """Retrieve top-k documents by MaxSim score.

        Args:
            query_emb: [Q, D] L2-normalized per-token query vectors
            top_k: Number of candidates to return
            domain_filter: Optional callable(record_metadata) → bool

        Returns list of (record_id, maxsim_score, doc_embedding).
        """
        candidate_ids: List[str] = []
        candidate_embs: List[np.ndarray] = []

        for record_id, record_meta in self._records_meta.items():
            if domain_filter is not None and not domain_filter(record_meta):
                continue
            doc_emb = self._load_embedding(record_id)
            if doc_emb is None:
                continue
            candidate_ids.append(record_id)
            candidate_embs.append(doc_emb)

        if not candidate_ids:
            return []

        scores = _maxsim_batch(query_emb, candidate_embs)
        ranked = sorted(
            zip(candidate_ids, scores, candidate_embs),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def rerank(
        self,
        query_emb: np.ndarray,
        candidate_ids: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Re-rank a pre-selected list of candidate documents by MaxSim score.

        Returns list of (record_id, maxsim_score).
        """
        embs = self._load_embeddings_batch(candidate_ids)
        ids_and_embs: List[Tuple[str, np.ndarray]] = []
        for rid, emb in zip(candidate_ids, embs):
            if emb is not None:
                ids_and_embs.append((rid, emb))

        if not ids_and_embs:
            return []

        scores = _maxsim_batch(query_emb, [e for _, e in ids_and_embs])
        scored = [
            (rid, float(score))
            for (rid, _), score in zip(ids_and_embs, scores)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


@lru_cache(maxsize=1)
def get_colbert_retriever() -> Optional[ColBERTRetriever]:
    if not _index_available():
        return None
    try:
        return ColBERTRetriever()
    except Exception:
        return None
