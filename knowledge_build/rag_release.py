"""Canonical 3GPP Release 19 corpus configuration for the policy RAG."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RELEASE = "19"
CORPUS_ID = "pcf_policy_r19"
PARSER = "pypdf"

KNOWLEDGE_BUILD_ROOT = Path(__file__).resolve().parent
DATA_ROOT = KNOWLEDGE_BUILD_ROOT / "data" / CORPUS_ID
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
COLBERT_INDEX_ROOT = DATA_ROOT / "colbert_index"

COLLECTIONS = {
    "am_clause": "pcf_am_policy_clauses_r19",
    "am_schema": "pcf_am_policy_schema_r19",
    "sm_clause": "pcf_sm_policy_clauses_r19",
    "sm_schema": "pcf_sm_policy_schema_r19",
    "ursp_clause": "pcf_ursp_clauses_r19",
    "ursp_schema": "pcf_ursp_schema_r19",
    "glossary": "pcf_policy_glossary_r19",
}


@dataclass(frozen=True)
class ReleaseSource:
    source_id: str
    spec_id: str
    title: str
    release: str
    spec_version: str
    version: str
    source_url: str
    archive_url: str
    doc_type: str
    policy_domain: str
    local_name: str


SOURCE_CONFIGS: tuple[ReleaseSource, ...] = (
    ReleaseSource(
        "ts_23503_pdf",
        "23.503",
        "Policy and charging control framework for the 5GS",
        RELEASE,
        "19.8.0",
        "19.8.0",
        "https://www.etsi.org/deliver/etsi_ts/123500_123599/123503/19.08.00_60/ts_123503v190800p.pdf",
        "https://www.3gpp.org/ftp/Specs/archive/23_series/23.503/23503-j80.zip",
        "stage2",
        "shared",
        "ts_123503_v19_8_0.pdf",
    ),
    ReleaseSource(
        "ts_24526_pdf",
        "24.526",
        "UE policies for 5GS",
        RELEASE,
        "19.4.0",
        "19.4.0",
        "https://www.etsi.org/deliver/etsi_ts/124500_124599/124526/19.04.00_60/ts_124526v190400p.pdf",
        "https://www.3gpp.org/ftp/Specs/archive/24_series/24.526/24526-j40.zip",
        "stage3",
        "ursp",
        "ts_124526_v19_4_0.pdf",
    ),
    ReleaseSource(
        "ts_29507_pdf",
        "29.507",
        "Access and Mobility Policy Control Service",
        RELEASE,
        "19.7.0",
        "19.7.0",
        "https://www.etsi.org/deliver/etsi_ts/129500_129599/129507/19.07.00_60/ts_129507v190700p.pdf",
        "https://www.3gpp.org/ftp/Specs/archive/29_series/29.507/29507-j70.zip",
        "stage3",
        "am_policy",
        "ts_129507_v19_7_0.pdf",
    ),
    ReleaseSource(
        "ts_29512_pdf",
        "29.512",
        "Session Management Policy Control Service",
        RELEASE,
        "19.7.0",
        "19.7.0",
        "https://www.etsi.org/deliver/etsi_ts/129500_129599/129512/19.07.00_60/ts_129512v190700p.pdf",
        "https://www.3gpp.org/ftp/Specs/archive/29_series/29.512/29512-j70.zip",
        "stage3",
        "sm_policy",
        "ts_129512_v19_7_0.pdf",
    ),
    ReleaseSource(
        "ts_29525_pdf",
        "29.525",
        "UE Policy Control Service",
        RELEASE,
        "19.7.0",
        "19.7.0",
        "https://www.etsi.org/deliver/etsi_ts/129500_129599/129525/19.07.00_60/ts_129525v190700p.pdf",
        "https://www.3gpp.org/ftp/Specs/archive/29_series/29.525/29525-j70.zip",
        "stage3",
        "ursp",
        "ts_129525_v19_7_0.pdf",
    ),
    ReleaseSource(
        "ts_29571_pdf",
        "29.571",
        "Common Data Types for Service Based Interfaces",
        RELEASE,
        "19.7.0",
        "19.7.0",
        "https://www.etsi.org/deliver/etsi_ts/129500_129599/129571/19.07.00_60/ts_129571v190700p.pdf",
        "https://www.3gpp.org/ftp/Specs/archive/29_series/29.571/29571-j70.zip",
        "stage3",
        "shared",
        "ts_129571_v19_7_0.pdf",
    ),
    ReleaseSource(
        "ts_29507_yaml",
        "29.507",
        "Npcf_AMPolicyControl OpenAPI",
        RELEASE,
        "19.7.0",
        "1.4.2",
        "https://forge.3gpp.org/rep/all/5G_APIs/-/raw/REL-19/TS29507_Npcf_AMPolicyControl.yaml",
        "https://www.3gpp.org/ftp/Specs/archive/29_series/29.507/29507-j70.zip",
        "openapi",
        "am_policy",
        "TS29507_Npcf_AMPolicyControl.yaml",
    ),
    ReleaseSource(
        "ts_29512_yaml",
        "29.512",
        "Npcf_SMPolicyControl OpenAPI",
        RELEASE,
        "19.7.0",
        "1.4.2",
        "https://forge.3gpp.org/rep/all/5G_APIs/-/raw/REL-19/TS29512_Npcf_SMPolicyControl.yaml",
        "https://www.3gpp.org/ftp/Specs/archive/29_series/29.512/29512-j70.zip",
        "openapi",
        "sm_policy",
        "TS29512_Npcf_SMPolicyControl.yaml",
    ),
    ReleaseSource(
        "ts_29525_yaml",
        "29.525",
        "Npcf_UEPolicyControl OpenAPI",
        RELEASE,
        "19.7.0",
        "1.4.2",
        "https://forge.3gpp.org/rep/all/5G_APIs/-/raw/REL-19/TS29525_Npcf_UEPolicyControl.yaml",
        "https://www.3gpp.org/ftp/Specs/archive/29_series/29.525/29525-j70.zip",
        "openapi",
        "ursp",
        "TS29525_Npcf_UEPolicyControl.yaml",
    ),
    ReleaseSource(
        "ts_29571_yaml",
        "29.571",
        "Common Data Types OpenAPI",
        RELEASE,
        "19.7.0",
        "1.6.2",
        "https://forge.3gpp.org/rep/all/5G_APIs/-/raw/REL-19/TS29571_CommonData.yaml",
        "https://www.3gpp.org/ftp/Specs/archive/29_series/29.571/29571-j70.zip",
        "openapi",
        "shared",
        "TS29571_CommonData.yaml",
    ),
)


__all__ = [
    "COLBERT_INDEX_ROOT",
    "COLLECTIONS",
    "CORPUS_ID",
    "DATA_ROOT",
    "PARSER",
    "PROCESSED_ROOT",
    "RAW_ROOT",
    "RELEASE",
    "ReleaseSource",
    "SOURCE_CONFIGS",
]
