"""Build ColBERT per-token embedding index from existing JSONL corpus.

Usage:
    python knowledge_build/scripts/build_colbert_index.py [--batch-size 32] [--device cpu|cuda]

The index is saved under knowledge_build/data/pcf_policy_r19/colbert_index/ as:
    embeddings/    — one .npz per record (compressed numpy)
    index_meta.json — record metadata map
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BUILD_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = KNOWLEDGE_BUILD_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

import numpy as np
from knowledge_build.rag_release import COLBERT_INDEX_ROOT, CORPUS_ID, PROCESSED_ROOT, RELEASE

CLAUSE_JSONL = PROCESSED_ROOT / "clauses.jsonl"
SCHEMA_JSONL = PROCESSED_ROOT / "schema.jsonl"
GLOSSARY_JSONL = PROCESSED_ROOT / "glossary.jsonl"
EMBEDDINGS_DIR = COLBERT_INDEX_ROOT / "embeddings"
INDEX_META = COLBERT_INDEX_ROOT / "index_meta.json"
MANIFEST_PATH = PROCESSED_ROOT / "source_manifest.json"
PROCESSED_CORPUS_FILES = (CLAUSE_JSONL, SCHEMA_JSONL, GLOSSARY_JSONL)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def processed_corpus_sha256(paths: tuple[Path, ...] = PROCESSED_CORPUS_FILES) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Processed corpus file not found: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build_colbert_index(
    batch_size: int = 32,
    device: Optional[str] = None,
) -> None:
    from knowledge_runtime.retrieval.colbert_encoder import ColBERTEncoder

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Source manifest not found: {MANIFEST_PATH}. Run build_pcf_policy_kb.py fetch first.")

    print(f"Loading corpus from {PROCESSED_ROOT}")
    records = (
        load_jsonl(CLAUSE_JSONL)
        + load_jsonl(SCHEMA_JSONL)
        + load_jsonl(GLOSSARY_JSONL)
    )
    print(f"Total records: {len(records)}")
    if not records:
        raise RuntimeError("Processed corpus is empty. Run build_pcf_policy_kb.py build before building the ColBERT index.")

    encoder = ColBERTEncoder(device=device)
    print(f"Encoder: {encoder.model_name}, device: {encoder._device}, dim: {encoder.dim}")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    texts = [record.get("page_content", "") for record in records]

    started = time.perf_counter()
    embeddings = encoder.encode_documents(texts, batch_size=batch_size, show_progress=True)
    elapsed = time.perf_counter() - started
    print(f"Encoding done in {elapsed:.2f}s ({len(records) / elapsed:.1f} docs/s)")

    index_meta: Dict[str, Any] = {
        "release": RELEASE,
        "corpus_id": CORPUS_ID,
        "source_manifest_sha256": sha256_file(MANIFEST_PATH),
        "processed_corpus_sha256": processed_corpus_sha256(),
        "model": encoder.model_name,
        "model_name": encoder.model_name,
        "dim": encoder.dim,
        "total_records": len(records),
        "records": {},
    }

    total_tokens = 0
    total_size = 0
    for i, (record, emb) in enumerate(zip(records, embeddings)):
        record_id = record.get("id") or f"record_{i:05d}"
        metadata = record.get("metadata") or {}
        filename = f"{i:05d}_{record_id.replace('/', '_').replace(' ', '_')[:80]}.npz"
        filepath = EMBEDDINGS_DIR / filename
        np.savez_compressed(str(filepath), tokens=emb)
        num_tokens = emb.shape[0]
        total_tokens += num_tokens
        total_size += filepath.stat().st_size

        index_meta["records"][record_id] = {
            "file": filename,
            "num_tokens": num_tokens,
            "doc_type": metadata.get("doc_type", ""),
            "policy_domain": metadata.get("policy_domain", ""),
            "strategy_domains": metadata.get("strategy_domains", []),
            "id": record_id,
        }

    with INDEX_META.open("w", encoding="utf-8") as f:
        json.dump(index_meta, f, ensure_ascii=False, indent=2)

    print(f"Index written to {COLBERT_INDEX_ROOT}")
    print(f"  Records:  {len(records)}")
    average_tokens = total_tokens / len(records) if records else 0.0
    print(f"  Tokens:   {total_tokens} (avg {average_tokens:.1f}/doc)")
    print(f"  Size:     {total_size / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build ColBERT index for PCF policy knowledge base")
    parser.add_argument("--batch-size", type=int, default=16, help="Encoding batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda)")
    args = parser.parse_args()

    build_colbert_index(batch_size=args.batch_size, device=args.device)
