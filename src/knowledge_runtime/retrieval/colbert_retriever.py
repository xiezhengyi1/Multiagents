from __future__ import annotations

import json
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from knowledge_build.rag_release import COLBERT_INDEX_ROOT, CORPUS_ID, PROCESSED_ROOT, RELEASE

COLBERT_INDEX_ROOT_DEFAULT = COLBERT_INDEX_ROOT
EMBEDDINGS_DIR_DEFAULT = COLBERT_INDEX_ROOT_DEFAULT / "embeddings"
INDEX_META_DEFAULT = COLBERT_INDEX_ROOT_DEFAULT / "index_meta.json"
MANIFEST_PATH = PROCESSED_ROOT / "source_manifest.json"
PROCESSED_CORPUS_FILES = (
    PROCESSED_ROOT / "clauses.jsonl",
    PROCESSED_ROOT / "schema.jsonl",
    PROCESSED_ROOT / "glossary.jsonl",
)

_MAXSIM_WARM_BATCH = 64


def _index_available() -> bool:
    return INDEX_META_DEFAULT.exists() and EMBEDDINGS_DIR_DEFAULT.is_dir()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _processed_corpus_sha256() -> str:
    digest = hashlib.sha256()
    for path in PROCESSED_CORPUS_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"Processed corpus file not found: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _validate_index_metadata(meta: Dict[str, Any]) -> None:
    required_fields = (
        "release",
        "corpus_id",
        "source_manifest_sha256",
        "processed_corpus_sha256",
        "model_name",
        "dim",
        "total_records",
        "records",
    )
    missing = [field for field in required_fields if field not in meta]
    if missing:
        raise ValueError(f"ColBERT index metadata is incomplete; missing={missing}.")
    if meta["release"] != RELEASE or meta["corpus_id"] != CORPUS_ID:
        raise ValueError(
            f"ColBERT index release/corpus mismatch: expected R{RELEASE}/{CORPUS_ID}, "
            f"got R{meta['release']}/{meta['corpus_id']}."
        )
    if not isinstance(meta["records"], dict):
        raise ValueError("ColBERT index metadata records must be a dictionary.")
    if int(meta["total_records"]) != len(meta["records"]):
        raise ValueError("ColBERT index metadata total_records does not match records.")
    if not isinstance(meta["model_name"], str) or not meta["model_name"].strip():
        raise ValueError("ColBERT index metadata model_name is required.")
    if int(meta["dim"]) <= 0:
        raise ValueError("ColBERT index metadata dim must be positive.")
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Source manifest not found: {MANIFEST_PATH}")
    manifest_hash = _sha256_file(MANIFEST_PATH)
    if meta["source_manifest_sha256"] != manifest_hash:
        raise ValueError("ColBERT index source_manifest_sha256 does not match the canonical source manifest.")
    corpus_hash = _processed_corpus_sha256()
    if meta["processed_corpus_sha256"] != corpus_hash:
        raise ValueError("ColBERT index processed_corpus_sha256 does not match the canonical processed corpus.")


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

        _validate_index_metadata(self._meta)
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
        if emb.ndim != 2 or emb.shape[1] != self._dim:
            raise ValueError(f"ColBERT embedding shape for {record_id!r} does not match index metadata dim={self._dim}.")
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
    return ColBERTRetriever()
