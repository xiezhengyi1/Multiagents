"""Offline acceptance contract for the canonical 3GPP Release 19 RAG.

These tests deliberately use only temporary fixture files and in-process imports.
They must not fetch a 3GPP artifact, contact PostgreSQL, or load an embedding model.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

# Importing the build module constructs an SQLAlchemy engine but must never connect
# during this offline suite.  The production code validates that a URL is present.
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://offline:offline@127.0.0.1:5432/offline_test",
)

from database import langchain_pg
from knowledge_build import rag_release
from knowledge_build.scripts import build_pcf_policy_kb as kb
from knowledge_runtime.retrieval import colbert_retriever


EXPECTED_COLLECTIONS = {
    "am_clause": "pcf_am_policy_clauses_r19",
    "am_schema": "pcf_am_policy_schema_r19",
    "sm_clause": "pcf_sm_policy_clauses_r19",
    "sm_schema": "pcf_sm_policy_schema_r19",
    "ursp_clause": "pcf_ursp_clauses_r19",
    "ursp_schema": "pcf_ursp_schema_r19",
    "glossary": "pcf_policy_glossary_r19",
}

# This is intentionally a complete lock rather than a pattern-based check.  It
# pins the exact Release-19 corpus approved for this project: six normative PDFs
# (the five original core specs plus TS 29.571) and four REL-19 OpenAPI files.
EXPECTED_SOURCES = (
    (
        "ts_23503_pdf",
        "23.503",
        "19",
        "19.8.0",
        "19.8.0",
        "https://www.etsi.org/deliver/etsi_ts/123500_123599/123503/19.08.00_60/ts_123503v190800p.pdf",
        "https://www.3gpp.org/ftp/Specs/archive/23_series/23.503/23503-j80.zip",
        "stage2",
        "shared",
        "ts_123503_v19_8_0.pdf",
    ),
    (
        "ts_24526_pdf",
        "24.526",
        "19",
        "19.4.0",
        "19.4.0",
        "https://www.etsi.org/deliver/etsi_ts/124500_124599/124526/19.04.00_60/ts_124526v190400p.pdf",
        "https://www.3gpp.org/ftp/Specs/archive/24_series/24.526/24526-j40.zip",
        "stage3",
        "ursp",
        "ts_124526_v19_4_0.pdf",
    ),
    (
        "ts_29507_pdf",
        "29.507",
        "19",
        "19.7.0",
        "19.7.0",
        "https://www.etsi.org/deliver/etsi_ts/129500_129599/129507/19.07.00_60/ts_129507v190700p.pdf",
        "https://www.3gpp.org/ftp/Specs/archive/29_series/29.507/29507-j70.zip",
        "stage3",
        "am_policy",
        "ts_129507_v19_7_0.pdf",
    ),
    (
        "ts_29512_pdf",
        "29.512",
        "19",
        "19.7.0",
        "19.7.0",
        "https://www.etsi.org/deliver/etsi_ts/129500_129599/129512/19.07.00_60/ts_129512v190700p.pdf",
        "https://www.3gpp.org/ftp/Specs/archive/29_series/29.512/29512-j70.zip",
        "stage3",
        "sm_policy",
        "ts_129512_v19_7_0.pdf",
    ),
    (
        "ts_29525_pdf",
        "29.525",
        "19",
        "19.7.0",
        "19.7.0",
        "https://www.etsi.org/deliver/etsi_ts/129500_129599/129525/19.07.00_60/ts_129525v190700p.pdf",
        "https://www.3gpp.org/ftp/Specs/archive/29_series/29.525/29525-j70.zip",
        "stage3",
        "ursp",
        "ts_129525_v19_7_0.pdf",
    ),
    (
        "ts_29571_pdf",
        "29.571",
        "19",
        "19.7.0",
        "19.7.0",
        "https://www.etsi.org/deliver/etsi_ts/129500_129599/129571/19.07.00_60/ts_129571v190700p.pdf",
        "https://www.3gpp.org/ftp/Specs/archive/29_series/29.571/29571-j70.zip",
        "stage3",
        "shared",
        "ts_129571_v19_7_0.pdf",
    ),
    (
        "ts_29507_yaml",
        "29.507",
        "19",
        "19.7.0",
        "1.4.2",
        "https://forge.3gpp.org/rep/all/5G_APIs/-/raw/REL-19/TS29507_Npcf_AMPolicyControl.yaml",
        "https://www.3gpp.org/ftp/Specs/archive/29_series/29.507/29507-j70.zip",
        "openapi",
        "am_policy",
        "TS29507_Npcf_AMPolicyControl.yaml",
    ),
    (
        "ts_29512_yaml",
        "29.512",
        "19",
        "19.7.0",
        "1.4.2",
        "https://forge.3gpp.org/rep/all/5G_APIs/-/raw/REL-19/TS29512_Npcf_SMPolicyControl.yaml",
        "https://www.3gpp.org/ftp/Specs/archive/29_series/29.512/29512-j70.zip",
        "openapi",
        "sm_policy",
        "TS29512_Npcf_SMPolicyControl.yaml",
    ),
    (
        "ts_29525_yaml",
        "29.525",
        "19",
        "19.7.0",
        "1.4.2",
        "https://forge.3gpp.org/rep/all/5G_APIs/-/raw/REL-19/TS29525_Npcf_UEPolicyControl.yaml",
        "https://www.3gpp.org/ftp/Specs/archive/29_series/29.525/29525-j70.zip",
        "openapi",
        "ursp",
        "TS29525_Npcf_UEPolicyControl.yaml",
    ),
    (
        "ts_29571_yaml",
        "29.571",
        "19",
        "19.7.0",
        "1.6.2",
        "https://forge.3gpp.org/rep/all/5G_APIs/-/raw/REL-19/TS29571_CommonData.yaml",
        "https://www.3gpp.org/ftp/Specs/archive/29_series/29.571/29571-j70.zip",
        "openapi",
        "shared",
        "TS29571_CommonData.yaml",
    ),
)


def _source_contract(source: Any) -> tuple[str, ...]:
    return (
        source.source_id,
        source.spec_id,
        source.release,
        source.spec_version,
        source.version,
        source.source_url,
        source.archive_url,
        source.doc_type,
        source.policy_domain,
        source.local_name,
    )


def _write_fixture_manifest(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for index, source in enumerate(kb.default_sources(), start=1):
        raw_path = tmp_path / source.local_name
        raw_path.write_bytes(f"rel19 fixture {index} {source.source_id}".encode("utf-8"))
        record = asdict(source)
        source_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        record.update(
            {
                "local_path": str(raw_path),
                "sha256": source_sha256,
                "hash": source_sha256,
                "fetched_at": "2026-08-26T00:00:00",
            }
        )
        records.append(record)
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path, records


def _load_fixture_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    manifest_path, records = _write_fixture_manifest(tmp_path)
    monkeypatch.setattr(kb, "RAW_ROOT", tmp_path)
    monkeypatch.setattr(kb, "MANIFEST_PATH", manifest_path)
    return records


def test_release_identity_roots_and_seven_collection_contracts() -> None:
    assert rag_release.RELEASE == "19"
    assert rag_release.CORPUS_ID == "pcf_policy_r19"
    assert rag_release.DATA_ROOT == rag_release.KNOWLEDGE_BUILD_ROOT / "data" / rag_release.CORPUS_ID
    assert rag_release.RAW_ROOT == rag_release.DATA_ROOT / "raw"
    assert rag_release.PROCESSED_ROOT == rag_release.DATA_ROOT / "processed"
    assert rag_release.COLLECTIONS == EXPECTED_COLLECTIONS
    assert len(set(rag_release.COLLECTIONS.values())) == 7
    assert all(value.endswith("_r19") for value in rag_release.COLLECTIONS.values())

    assert kb.RELEASE == rag_release.RELEASE
    assert kb.CORPUS_ID == rag_release.CORPUS_ID
    assert kb.DATA_ROOT == rag_release.DATA_ROOT
    assert kb.RAW_ROOT == rag_release.RAW_ROOT
    assert kb.PROCESSED_ROOT == rag_release.PROCESSED_ROOT

    assert langchain_pg.PCF_AM_POLICY_CLAUSES_COLLECTION == EXPECTED_COLLECTIONS["am_clause"]
    assert langchain_pg.PCF_AM_POLICY_SCHEMA_COLLECTION == EXPECTED_COLLECTIONS["am_schema"]
    assert langchain_pg.PCF_SM_POLICY_CLAUSES_COLLECTION == EXPECTED_COLLECTIONS["sm_clause"]
    assert langchain_pg.PCF_SM_POLICY_SCHEMA_COLLECTION == EXPECTED_COLLECTIONS["sm_schema"]
    assert langchain_pg.PCF_URSP_CLAUSES_COLLECTION == EXPECTED_COLLECTIONS["ursp_clause"]
    assert langchain_pg.PCF_URSP_SCHEMA_COLLECTION == EXPECTED_COLLECTIONS["ursp_schema"]
    assert langchain_pg.PCF_POLICY_GLOSSARY_COLLECTION == EXPECTED_COLLECTIONS["glossary"]


def test_source_configs_are_the_frozen_ten_official_rel19_sources() -> None:
    assert tuple(_source_contract(source) for source in rag_release.SOURCE_CONFIGS) == EXPECTED_SOURCES
    assert tuple(_source_contract(source) for source in kb.default_sources()) == EXPECTED_SOURCES
    assert tuple(kb.default_sources()) == rag_release.SOURCE_CONFIGS

    source_ids = [source.source_id for source in rag_release.SOURCE_CONFIGS]
    assert source_ids == [source[0] for source in EXPECTED_SOURCES]
    assert len(source_ids) == len(set(source_ids)) == 10
    assert sum(source.doc_type == "openapi" for source in rag_release.SOURCE_CONFIGS) == 4
    assert sum(source.doc_type in {"stage2", "stage3"} for source in rag_release.SOURCE_CONFIGS) == 6
    assert all(source.release == "19" for source in rag_release.SOURCE_CONFIGS)


def test_manifest_validation_accepts_complete_rel19_fixture_with_matching_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    records = _load_fixture_manifest(monkeypatch, tmp_path)

    loaded = kb.load_manifest()

    assert [record["source_id"] for record in loaded] == [record["source_id"] for record in records]
    assert all(record["release"] == "19" for record in loaded)
    assert all(len(record["sha256"]) == 64 for record in loaded)
    assert all(record["archive_url"].startswith("https://") for record in loaded)


@pytest.mark.parametrize(
    ("mutation", "error_fragment"),
    [
        (lambda records: records.__setitem__(0, {**records[0], "release": "18"}), "release"),
        (lambda records: records.__setitem__(0, {**records[0], "source_id": "ts_99999_pdf"}), "source"),
        (lambda records: records.__setitem__(0, {**records[0], "sha256": "0" * 64}), "hash"),
        (lambda records: records.__setitem__(0, {**records[0], "hash": ""}), "hash"),
    ],
)
def test_manifest_validation_rejects_wrong_release_source_id_or_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: Any,
    error_fragment: str,
) -> None:
    records = _load_fixture_manifest(monkeypatch, tmp_path)
    mutation(records)
    kb.MANIFEST_PATH.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match=error_fragment):
        kb.load_manifest()


def test_every_record_metadata_carries_release_version_and_archive_provenance() -> None:
    for source in kb.default_sources():
        metadata = kb.make_metadata(
            source=asdict(source),
            doc_type=source.doc_type,
            clause_path="1.2",
            clause_title="Rel-19 acceptance fixture",
            page_start=1,
            page_end=1,
            canonical_title=f"TS {source.spec_id} fixture",
            object_tags=["FixtureObject"],
            citation_anchor=f"{source.spec_id}:1.2",
            strategy_domains=[source.policy_domain] if source.policy_domain != "shared" else [],
        )
        assert metadata["release"] == "19"
        assert metadata["corpus_id"] == "pcf_policy_r19"
        assert metadata["spec_id"] == source.spec_id
        assert metadata["spec_version"] == source.spec_version
        assert metadata["version"] == source.version
        assert metadata["archive_url"] == source.archive_url
        assert metadata["source_url"] == source.source_url


def test_exact_index_is_labeled_as_the_rel19_corpus() -> None:
    records = [
        {
            "id": "ts-29512-fixture",
            "page_content": "Npcf_SMPolicyControl contains SmPolicyDecision and PccRule.",
            "metadata": {
                "canonical_title": "TS 29.512 Npcf_SMPolicyControl",
                "clause_title": "SM policy fixture",
                "object_tags": ["Npcf_SMPolicyControl", "SmPolicyDecision"],
                "normalized_terms": ["Npcf_SMPolicyControl", "SM Policy"],
                "aliases": [],
                "policy_domain": "sm_policy",
                "spec_id": "29.512",
                "citation_anchor": "29.512:1.2",
                "release": "19",
                "spec_version": "19.7.0",
                "version": "19.7.0",
                "archive_url": "https://www.3gpp.org/ftp/Specs/archive/29_series/29.512/29512-j70.zip",
            },
        }
    ]

    index = kb.build_exact_index(records, index_name="rel19_fixture")

    assert index["release"] == "19"
    assert kb.search_exact_index(index, "Npcf_SMPolicyControl", top_k=1)[0]["id"] == "ts-29512-fixture"


def _write_colbert_meta(
    index_root: Path,
    *,
    release: str,
    corpus_id: str,
    source_manifest_sha256: str,
    processed_corpus_sha256: str,
) -> None:
    (index_root / "embeddings").mkdir(parents=True)
    (index_root / "index_meta.json").write_text(
        json.dumps(
            {
                "release": release,
                "corpus_id": corpus_id,
                "source_manifest_sha256": source_manifest_sha256,
                "processed_corpus_sha256": processed_corpus_sha256,
                "model_name": "offline-fixture",
                "dim": 2,
                "total_records": 0,
                "records": {},
            }
        ),
        encoding="utf-8",
    )


def test_colbert_index_meta_requires_matching_release_and_corpus(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text('{"release":"19"}', encoding="utf-8")
    processed_paths = tuple(tmp_path / name for name in ("clauses.jsonl", "schema.jsonl", "glossary.jsonl"))
    for path in processed_paths:
        path.write_text('{"fixture":true}\n', encoding="utf-8")
    monkeypatch.setattr(colbert_retriever, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(colbert_retriever, "PROCESSED_CORPUS_FILES", processed_paths)
    source_manifest_sha256 = colbert_retriever._sha256_file(manifest_path)
    processed_corpus_sha256 = colbert_retriever._processed_corpus_sha256()

    valid_root = tmp_path / "valid"
    _write_colbert_meta(
        valid_root,
        release="19",
        corpus_id="pcf_policy_r19",
        source_manifest_sha256=source_manifest_sha256,
        processed_corpus_sha256=processed_corpus_sha256,
    )
    retriever = colbert_retriever.ColBERTRetriever(index_root=valid_root)
    assert retriever.record_count == 0

    bad_release_root = tmp_path / "bad-release"
    _write_colbert_meta(
        bad_release_root,
        release="18",
        corpus_id="pcf_policy_r19",
        source_manifest_sha256=source_manifest_sha256,
        processed_corpus_sha256=processed_corpus_sha256,
    )
    with pytest.raises(ValueError, match="release"):
        colbert_retriever.ColBERTRetriever(index_root=bad_release_root)

    bad_corpus_root = tmp_path / "bad-corpus"
    _write_colbert_meta(
        bad_corpus_root,
        release="19",
        corpus_id="pcf_policy_r18",
        source_manifest_sha256=source_manifest_sha256,
        processed_corpus_sha256=processed_corpus_sha256,
    )
    with pytest.raises(ValueError, match="corpus"):
        colbert_retriever.ColBERTRetriever(index_root=bad_corpus_root)


def test_rag_production_code_has_no_rel18_docling_or_profile_compatibility_entrypoints() -> None:
    production_files = (
        ROOT / "knowledge_build" / "rag_release.py",
        ROOT / "knowledge_build" / "scripts" / "build_pcf_policy_kb.py",
        ROOT / "knowledge_build" / "scripts" / "build_colbert_index.py",
        ROOT / "database" / "langchain_pg.py",
        ROOT / "src" / "knowledge_runtime" / "retrieval" / "colbert_retriever.py",
        ROOT / "src" / "knowledge_runtime" / "retrieval" / "raw" / "knowledge_tool.py",
        ROOT / "knowledge_build" / "scripts" / "evaluate_pcf_policy_rag.py",
    )
    forbidden_tokens = (
        "r18",
        "rel-18",
        "docling",
        "profile_configs",
        "knowledge_profile",
        "--profile",
        "build_pcf_policy_kb_docling",
    )

    assert not (ROOT / "knowledge_build" / "scripts" / "build_pcf_policy_kb_docling.py").exists()
    offenders = [
        f"{path.relative_to(ROOT)}: {token}"
        for path in production_files
        for token in forbidden_tokens
        if token in path.read_text(encoding="utf-8").lower()
    ]
    assert offenders == []
