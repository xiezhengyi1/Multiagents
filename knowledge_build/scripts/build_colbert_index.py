"""Build ColBERT per-token embedding index from existing JSONL corpus.

Usage:
    python knowledge_build/scripts/build_colbert_index.py [--batch-size 32] [--device cpu|cuda]

The index is saved under knowledge_build/data/pcf_policy_r18/colbert_index/ as:
    embeddings/    — one .npz per record (compressed numpy)
    index_meta.json — record metadata map
"""

from __future__ import annotations

import json
import os
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

DATA_ROOT = KNOWLEDGE_BUILD_ROOT / "data" / "pcf_policy_r18"
PROCESSED_ROOT = DATA_ROOT / "processed"
CLAUSE_JSONL = PROCESSED_ROOT / "clauses.jsonl"
SCHEMA_JSONL = PROCESSED_ROOT / "schema.jsonl"
GLOSSARY_JSONL = PROCESSED_ROOT / "glossary.jsonl"
COLBERT_INDEX_ROOT = DATA_ROOT / "colbert_index"
EMBEDDINGS_DIR = COLBERT_INDEX_ROOT / "embeddings"
INDEX_META = COLBERT_INDEX_ROOT / "index_meta.json"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def build_colbert_index(
    batch_size: int = 32,
    device: Optional[str] = None,
) -> None:
    from knowledge_runtime.retrieval.colbert_encoder import ColBERTEncoder

    print(f"Loading corpus from {PROCESSED_ROOT}")
    records = (
        load_jsonl(CLAUSE_JSONL)
        + load_jsonl(SCHEMA_JSONL)
        + load_jsonl(GLOSSARY_JSONL)
    )
    print(f"Total records: {len(records)}")

    encoder = ColBERTEncoder(device=device)
    print(f"Encoder: {encoder.model_name}, device: {encoder._device}, dim: {encoder.dim}")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    texts = [record.get("page_content", "") for record in records]

    started = time.perf_counter()
    embeddings = encoder.encode_documents(texts, batch_size=batch_size, show_progress=True)
    elapsed = time.perf_counter() - started
    print(f"Encoding done in {elapsed:.2f}s ({len(records) / elapsed:.1f} docs/s)")

    index_meta: Dict[str, Any] = {
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
    print(f"  Tokens:   {total_tokens} (avg {total_tokens / len(records):.1f}/doc)")
    print(f"  Size:     {total_size / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build ColBERT index for PCF policy knowledge base")
    parser.add_argument("--batch-size", type=int, default=16, help="Encoding batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda)")
    args = parser.parse_args()

    build_colbert_index(batch_size=args.batch_size, device=args.device)
