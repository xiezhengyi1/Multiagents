from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
import yaml
from pypdf import PdfReader

SCRIPT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BUILD_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = KNOWLEDGE_BUILD_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from database.langchain_pg import (
    PCF_AM_POLICY_CLAUSES_COLLECTION,
    PCF_AM_POLICY_SCHEMA_COLLECTION,
    PCF_POLICY_GLOSSARY_COLLECTION,
    PCF_SM_POLICY_CLAUSES_COLLECTION,
    PCF_SM_POLICY_SCHEMA_COLLECTION,
    PCF_URSP_CLAUSES_COLLECTION,
    PCF_URSP_SCHEMA_COLLECTION,
    build_pgvector_document,
    rebuild_pgvector_collection,
)
from shared.logging import setup_logger

logger = setup_logger(__name__)

DATA_ROOT = KNOWLEDGE_BUILD_ROOT / "data" / "pcf_policy_r18"
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
MANIFEST_PATH = PROCESSED_ROOT / "source_manifest.json"
CLAUSE_JSONL = PROCESSED_ROOT / "clauses.jsonl"
SCHEMA_JSONL = PROCESSED_ROOT / "schema.jsonl"
GLOSSARY_JSONL = PROCESSED_ROOT / "glossary.jsonl"
SPEC_OBJECT_MAP_JSON = PROCESSED_ROOT / "spec_object_map.json"
TERM_ALIAS_MAP_JSON = PROCESSED_ROOT / "term_alias_map.json"
CLAUSE_EXACT_INDEX_JSON = PROCESSED_ROOT / "clause_exact_index.json"
SCHEMA_EXACT_INDEX_JSON = PROCESSED_ROOT / "schema_exact_index.json"
GLOSSARY_EXACT_INDEX_JSON = PROCESSED_ROOT / "glossary_exact_index.json"
RETRIEVAL_EVAL_JSON = PROCESSED_ROOT / "retrieval_eval_queries.json"

MIN_CHUNK_TOKENS = 120
MAX_CHUNK_TOKENS = 280
TARGET_EMBED_CHUNK_LENGTH = 1800
MAX_EMBED_INPUT_LENGTH = 8192

HEADING_RE = re.compile(r"^(?P<num>\d+(?:\.\d+){0,5})\s+(?P<title>.+)$")
TABLE_RE = re.compile(r"^(Table\s+\d+(?:\.\d+)?(?:[-:]\d+)?[^\n]*)$", re.MULTILINE)
TOKEN_RE = re.compile(r"[A-Za-z0-9_.-]+")
CHANGE_REQUEST_RE = re.compile(r"\bC\d-\d+\b", re.IGNORECASE)

MINIMAL_KB_SOURCE_IDS = {
    "ts_23503_pdf",
    "ts_24526_pdf",
    "ts_29507_pdf",
    "ts_29507_yaml",
    "ts_29512_pdf",
    "ts_29512_yaml",
    "ts_29525_pdf",
    "ts_29525_yaml",
    "ts_29571_yaml",
}

AM_POLICY_RELEVANCE_TERMS = (
    "ampolicy",
    "am policy",
    "access and mobility policy",
    "npcf_ampolicycontrol",
    "policyassociation",
    "policy association",
    "policyassociationrequest",
    "policyassociationupdaterequest",
    "amrequestedvaluerep",
    "requesttrigger",
    "allowed nssai",
    "allowedsnssais",
    "target nssai",
    "targetsnssais",
    "partially allowed nssai",
    "pending nssai",
    "rejected snssais",
    "service area restriction",
    "servareares",
    "rfsp",
    "rfspindex",
    "presence reporting area",
    "pra",
    "smf selection",
    "ue slice mbr",
)

SM_POLICY_RELEVANCE_TERMS = (
    "smpolicy",
    "sm policy",
    "npcf_smpolicycontrol",
    "smpolicydecision",
    "smpolicycontextdata",
    "smpolicyupdatecontextdata",
    "pccrule",
    "pcc rule",
    "qosdata",
    "qos dec",
    "sessionrule",
    "policycontrolrequesttrigger",
    "usage monitoring",
    "revalidation",
)

URSP_RELEVANCE_TERMS = (
    "ursp",
    "ue route selection policy",
    "ue policy",
    "npcf_uepolicycontrol",
    "uepolicysection",
    "traffic descriptor",
    "trafficdescriptor",
    "route selection descriptor",
    "routeselectiondescriptor",
    "route selection",
    "os id",
    "os app id",
    "access traffic steering",
    "switching",
    "splitting",
)

CANONICAL_TERM_ALIASES: Dict[str, Dict[str, Any]] = {
    "Npcf_SMPolicyControl": {
        "aliases": ["SmPolicy", "SM Policy", "SM Policy Control", "SmPolicyDecision API"],
        "related_specs": ["29.512", "23.503"],
        "related_objects": ["SmPolicyContextData", "SmPolicyDecision"],
        "policy_domain": "sm_policy",
    },
    "SmPolicyDecision": {
        "aliases": ["SM policy decision", "session policy decision", "smpolicydecision"],
        "related_specs": ["29.512", "23.503"],
        "related_objects": ["SmPolicyDecision", "PccRule", "QosData", "SessionRule"],
        "policy_domain": "sm_policy",
    },
    "PccRule": {
        "aliases": ["PCC rule", "policy control rule", "pccRules"],
        "related_specs": ["23.503", "29.512", "29.213", "29.214"],
        "related_objects": ["PccRule", "QosData"],
        "policy_domain": "sm_policy",
    },
    "QosData": {
        "aliases": ["QoS data", "QoS config", "qosDecs"],
        "related_specs": ["23.503", "29.512", "29.214"],
        "related_objects": ["QosData"],
        "policy_domain": "sm_policy",
    },
    "SessionRule": {
        "aliases": ["Session rule", "session-level rule", "sessRules"],
        "related_specs": ["23.503", "29.512"],
        "related_objects": ["SessionRule"],
        "policy_domain": "sm_policy",
    },
    "Npcf_AMPolicyControl": {
        "aliases": [
            "AM Policy",
            "AM Policy Control",
            "Access and Mobility Policy Control",
            "AM policy API",
        ],
        "related_specs": ["29.507", "23.503", "29.571"],
        "related_objects": [
            "PolicyAssociation",
            "PolicyAssociationRequest",
            "PolicyAssociationUpdateRequest",
            "AmRequestedValueRep",
            "RequestTrigger",
            "ServiceAreaRestriction",
            "RfspIndex",
            "PresenceInfo",
        ],
        "policy_domain": "am_policy",
    },
    "PcfAmPolicyControlPolicyAssociation": {
        "aliases": ["PolicyAssociation", "AM Policy Association", "policy association"],
        "related_specs": ["29.507", "23.503", "29.571"],
        "related_objects": [
            "PolicyAssociation",
            "PolicyAssociationRequest",
            "PolicyAssociationUpdateRequest",
            "AmRequestedValueRep",
        ],
        "policy_domain": "am_policy",
    },
    "PcfAmPolicyControlPolicyAssociationRequest": {
        "aliases": ["PolicyAssociationRequest", "AM Policy Association Request"],
        "related_specs": ["29.507", "29.571"],
        "related_objects": [
            "PolicyAssociationRequest",
            "ServiceAreaRestriction",
            "RfspIndex",
            "SmfSelectionData",
        ],
        "policy_domain": "am_policy",
    },
    "PcfAmPolicyControlRequestTrigger": {
        "aliases": ["RequestTrigger", "AM Policy Request Trigger", "policy trigger"],
        "related_specs": ["29.507"],
        "related_objects": ["RequestTrigger", "AmRequestedValueRep"],
        "policy_domain": "am_policy",
    },
    "Allowed NSSAI": {
        "aliases": ["allowedSnssais", "allowed S-NSSAIs", "allowed S-NSSAI"],
        "related_specs": ["29.507", "23.503", "29.571"],
        "related_objects": ["PolicyAssociationRequest", "AmRequestedValueRep", "Snssai"],
        "policy_domain": "am_policy",
    },
    "Target NSSAI": {
        "aliases": ["targetSnssais", "target S-NSSAIs", "target S-NSSAI"],
        "related_specs": ["29.507", "23.503", "29.571"],
        "related_objects": ["PolicyAssociationRequest", "PolicyAssociationUpdateRequest", "Snssai"],
        "policy_domain": "am_policy",
    },
    "ServiceAreaRestriction": {
        "aliases": ["service area restriction", "servAreaRes", "service-area"],
        "related_specs": ["29.507", "29.571", "23.503"],
        "related_objects": ["ServiceAreaRestriction", "PolicyAssociation", "PolicyAssociationRequest"],
        "policy_domain": "am_policy",
    },
    "RfspIndex": {
        "aliases": ["RFSP", "rfsp", "RFSP index"],
        "related_specs": ["29.507", "29.571", "23.503"],
        "related_objects": ["RfspIndex", "PolicyAssociation", "PolicyAssociationRequest"],
        "policy_domain": "am_policy",
    },
    "AmRequestedValueRep": {
        "aliases": ["current mobility requested values", "AM requested value report"],
        "related_specs": ["29.507", "29.571"],
        "related_objects": ["AmRequestedValueRep", "RequestTrigger", "PresenceInfo"],
        "policy_domain": "am_policy",
    },
    "PresenceInfo": {
        "aliases": ["PRA", "presence reporting area", "praStatuses", "pras"],
        "related_specs": ["29.507", "29.571"],
        "related_objects": ["PresenceInfo", "AmRequestedValueRep", "PolicyAssociationUpdateRequest"],
        "policy_domain": "am_policy",
    },
    "Npcf_UEPolicyControl": {
        "aliases": ["UE Policy Control", "URSP control API", "UE policy API"],
        "related_specs": ["29.525", "23.503"],
        "related_objects": ["UePolicySection", "URSP rule"],
        "policy_domain": "ursp",
    },
    "UE Route Selection Policy": {
        "aliases": ["URSP", "UE route selection policy", "route selection policy", "UE policy route selection"],
        "related_specs": ["24.526", "23.503", "29.525"],
        "related_objects": ["URSP rule", "Traffic descriptor", "Route selection descriptor"],
        "policy_domain": "ursp",
    },
    "Traffic descriptor": {
        "aliases": ["trafficDesc", "five tuple matching", "flow descriptions", "traffic descriptor"],
        "related_specs": ["24.526", "23.503"],
        "related_objects": ["Traffic descriptor"],
        "policy_domain": "ursp",
    },
    "Route selection descriptor": {
        "aliases": ["route selection descriptor", "routeSelDesc", "routeSelParamSets"],
        "related_specs": ["24.526", "23.503"],
        "related_objects": ["Route selection descriptor"],
        "policy_domain": "ursp",
    },
    "Route Selection Validation Criteria": {
        "aliases": ["RSVC", "Route Selection Validation Criteria"],
        "related_specs": ["23.503", "24.526"],
        "related_objects": ["Route Selection Validation Criteria"],
        "policy_domain": "ursp",
    },
    "OS Id": {
        "aliases": ["OS Id", "OS App Id", "osId", "appIds"],
        "related_specs": ["24.526", "24.501"],
        "related_objects": ["Traffic descriptor"],
        "policy_domain": "ursp",
    },
}

OBJECT_POLICY_DOMAIN_MAP: Dict[str, str] = {}
for _canonical_term, _config in CANONICAL_TERM_ALIASES.items():
    OBJECT_POLICY_DOMAIN_MAP[_canonical_term] = _config["policy_domain"]
    for _related_object in _config.get("related_objects") or []:
        OBJECT_POLICY_DOMAIN_MAP.setdefault(_related_object, _config["policy_domain"])


@dataclass(frozen=True)
class CorpusSource:
    source_id: str
    spec_id: str
    title: str
    release: str
    version: str
    source_url: str
    doc_type: str
    policy_domain: str
    local_name: str


def _utcnow() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds")


def default_sources() -> List[CorpusSource]:
    return [
        CorpusSource("ts_23503_pdf", "23.503", "Policy and charging control framework for the 5GS", "18", "18.8.0", "https://www.etsi.org/deliver/etsi_ts/123500_123599/123503/18.08.00_60/ts_123503v180800p.pdf", "stage2", "shared", "ts_123503_v18_8_0.pdf"),
        CorpusSource("ts_29507_pdf", "29.507", "Access and Mobility Policy Control Service", "18", "18.8.0", "https://www.etsi.org/deliver/etsi_ts/129500_129599/129507/18.08.00_60/ts_129507v180800p.pdf", "stage3", "am_policy", "ts_129507_v18_8_0.pdf"),
        CorpusSource("ts_29512_pdf", "29.512", "Session Management Policy Control Service", "18", "18.8.0", "https://www.etsi.org/deliver/etsi_ts/129500_129599/129512/18.08.00_60/ts_129512v180800p.pdf", "stage3", "sm_policy", "ts_129512_v18_8_0.pdf"),
        CorpusSource("ts_29514_pdf", "29.514", "Policy Authorization Service", "18", "18.8.0", "https://www.etsi.org/deliver/etsi_ts/129500_129599/129514/18.08.00_60/ts_129514v180800p.pdf", "stage3", "sm_policy", "ts_129514_v18_8_0.pdf"),
        CorpusSource("ts_24526_pdf", "24.526", "UE policies for 5GS", "18", "18.8.0", "https://www.etsi.org/deliver/etsi_ts/124500_124599/124526/18.08.00_60/ts_124526v180800p.pdf", "stage3", "ursp", "ts_124526_v18_8_0.pdf"),
        CorpusSource("ts_29525_pdf", "29.525", "UE Policy Control Service", "18", "18.6.0", "https://www.etsi.org/deliver/etsi_ts/129500_129599/129525/18.06.00_60/ts_129525v180600p.pdf", "stage3", "ursp", "ts_129525_v18_6_0.pdf"),
        CorpusSource("ts_29519_pdf", "29.519", "UDR policy data definitions", "18", "18.8.0", "https://www.etsi.org/deliver/etsi_ts/129500_129599/129519/18.08.00_60/ts_129519v180800p.pdf", "stage3", "ursp", "ts_129519_v18_8_0.pdf"),
        CorpusSource("ts_24501_pdf", "24.501", "NAS protocol for 5GS", "18", "18.8.0", "https://www.etsi.org/deliver/etsi_ts/124500_124599/124501/18.08.00_60/ts_124501v180800p.pdf", "stage3", "shared", "ts_124501_v18_8_0.pdf"),
        CorpusSource("ts_23501_pdf", "23.501", "System architecture for the 5GS", "18", "18.8.0", "https://www.etsi.org/deliver/etsi_ts/123500_123599/123501/18.08.00_60/ts_123501v180800p.pdf", "stage2", "shared", "ts_123501_v18_8_0.pdf"),
        CorpusSource("ts_29507_yaml", "29.507", "Npcf_AMPolicyControl OpenAPI", "18", "1.0.7", "https://forge.3gpp.org/rep/all/5G_APIs/-/raw/REL-18/TS29507_Npcf_AMPolicyControl.yaml", "openapi", "am_policy", "TS29507_Npcf_AMPolicyControl.yaml"),
        CorpusSource("ts_29512_yaml", "29.512", "Npcf_SMPolicyControl OpenAPI", "18", "1.3.4", "https://forge.3gpp.org/rep/all/5G_APIs/-/raw/REL-18/TS29512_Npcf_SMPolicyControl.yaml", "openapi", "sm_policy", "TS29512_Npcf_SMPolicyControl.yaml"),
        CorpusSource("ts_29525_yaml", "29.525", "Npcf_UEPolicyControl OpenAPI", "18", "1.3.3", "https://forge.3gpp.org/rep/all/5G_APIs/-/raw/REL-18/TS29525_Npcf_UEPolicyControl.yaml", "openapi", "ursp", "TS29525_Npcf_UEPolicyControl.yaml"),
        CorpusSource("ts_29571_yaml", "29.571", "Common Data Types OpenAPI", "18", "1.5.4", "https://forge.3gpp.org/rep/all/5G_APIs/-/raw/REL-18/TS29571_CommonData.yaml", "openapi", "shared", "TS29571_CommonData.yaml"),
    ]


def ensure_directories() -> None:
    for path in (RAW_ROOT, PROCESSED_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def estimate_tokens(text: str) -> int:
    normalized = str(text or "").strip()
    if not normalized:
        return 0
    return max(len(normalized.split()), math.ceil(len(normalized) / 4))


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text or ""))]


def normalize_query_terms(text: str) -> List[str]:
    raw_tokens = tokenize(text)
    expanded = list(raw_tokens)
    lowered = str(text or "").lower()
    for canonical, config in CANONICAL_TERM_ALIASES.items():
        if canonical.lower() in lowered or any(alias.lower() in lowered for alias in config["aliases"]):
            expanded.extend(tokenize(canonical))
            for alias in config["aliases"]:
                expanded.extend(tokenize(alias))
    return sorted(set(expanded))


def sanitize_text(text: str) -> str:
    lines = []
    for raw_line in str(text or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            lines.append("")
            continue
        if re.match(r"^3GPP TS \d{2}\.\d{3}", line):
            continue
        if re.match(r"^(ETSI TS|ETSI TR) \d", line):
            continue
        if re.match(r"^\d+\s*$", line):
            continue
        if "Intellectual Property Rights" in line:
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_sources(*, force: bool = False) -> List[Dict[str, Any]]:
    ensure_directories()
    manifest: List[Dict[str, Any]] = []
    session = requests.Session()
    for source in default_sources():
        target = RAW_ROOT / source.local_name
        if force or not target.exists():
            response = session.get(source.source_url, timeout=60)
            response.raise_for_status()
            target.write_bytes(response.content)
            logger.info("Fetched %s -> %s", source.source_id, target)
        record = asdict(source)
        record.update({"local_path": str(target), "sha256": sha256_file(target), "fetched_at": _utcnow()})
        manifest.append(record)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_manifest() -> List[Dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Source manifest not found: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def infer_object_tags(text: str, *, schema_name: str = "", operation_id: str = "") -> List[str]:
    haystack = " ".join(part for part in [schema_name, operation_id, text] if part).lower()
    tags = []
    candidates = [
        "SmPolicyContextData",
        "SmPolicyDecision",
        "SmPolicyUpdateContextData",
        "SmPolicyDeleteData",
        "PccRule",
        "QosData",
        "SessionRule",
        "PolicyAssociation",
        "PolicyAssociationRequest",
        "PolicyAssociationUpdateRequest",
        "AmRequestedValueRep",
        "RequestTrigger",
        "ServiceAreaRestriction",
        "WirelineServiceAreaRestriction",
        "RfspIndex",
        "PresenceInfo",
        "SmfSelectionData",
        "UeSliceMbr",
        "Allowed NSSAI",
        "Target NSSAI",
        "Partially Allowed NSSAI",
        "Pending NSSAI",
        "Rejected S-NSSAI",
        "TrafficControlData",
        "ChargingData",
        "PolicyControlRequestTrigger",
        "RevalidationTime",
        "UsageMonitoringData",
        "RefQosIndication",
        "UePolicySection",
        "URSP rule",
        "Traffic descriptor",
        "Route selection descriptor",
        "Route Selection Validation Criteria",
        "OS Id",
        "DNN",
        "S-NSSAI",
    ]
    normalized_haystack = haystack.replace(" ", "")
    for candidate in candidates:
        lowered = candidate.lower().replace(" ", "")
        if lowered in normalized_haystack:
            tags.append(candidate)
    if "npcf_smpolicycontrol" in haystack:
        tags.append("Npcf_SMPolicyControl")
    if "npcf_ampolicycontrol" in haystack:
        tags.append("Npcf_AMPolicyControl")
    if "npcf_uepolicycontrol" in haystack:
        tags.append("Npcf_UEPolicyControl")
    return sorted(set(tags))


def is_minimal_source(source: Dict[str, Any]) -> bool:
    return str(source.get("source_id") or "").strip() in MINIMAL_KB_SOURCE_IDS


def _infer_strategy_domains(
    *,
    source_policy_domain: str,
    object_tags: List[str],
    searchable_text: str,
) -> List[str]:
    normalized_domain = str(source_policy_domain or "").strip().lower()
    if normalized_domain in {"sm_policy", "ursp"}:
        return [normalized_domain]

    domains = {
        OBJECT_POLICY_DOMAIN_MAP[tag]
        for tag in object_tags
        if tag in OBJECT_POLICY_DOMAIN_MAP
    }
    lowered = searchable_text.lower()
    if any(term in lowered for term in AM_POLICY_RELEVANCE_TERMS):
        domains.add("am_policy")
    if any(term in lowered for term in SM_POLICY_RELEVANCE_TERMS):
        domains.add("sm_policy")
    if any(term in lowered for term in URSP_RELEVANCE_TERMS):
        domains.add("ursp")
    return sorted(domains)


def _should_keep_record(
    *,
    source: Dict[str, Any],
    object_tags: List[str],
    searchable_text: str,
    strategy_domains: List[str],
) -> bool:
    if not is_minimal_source(source):
        return False
    if not strategy_domains:
        return False
    if object_tags:
        return True

    lowered = searchable_text.lower()
    if "am_policy" in strategy_domains and any(term in lowered for term in AM_POLICY_RELEVANCE_TERMS):
        return True
    if "sm_policy" in strategy_domains and any(term in lowered for term in SM_POLICY_RELEVANCE_TERMS):
        return True
    if "ursp" in strategy_domains and any(term in lowered for term in URSP_RELEVANCE_TERMS):
        return True
    return False


def make_metadata(
    *,
    source: Dict[str, Any],
    doc_type: str,
    clause_path: str,
    clause_title: str,
    page_start: int,
    page_end: int,
    canonical_title: str,
    object_tags: List[str],
    table_id: str = "",
    schema_name: str = "",
    operation_id: str = "",
    citation_anchor: str = "",
    related_specs: Optional[List[str]] = None,
    normalized_terms: Optional[List[str]] = None,
    strategy_domains: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "source_id": source["source_id"],
        "spec_id": source["spec_id"],
        "release": source["release"],
        "version": source["version"],
        "source_url": source["source_url"],
        "doc_type": doc_type,
        "policy_domain": source["policy_domain"],
        "clause_path": clause_path,
        "clause_title": clause_title,
        "page_start": page_start,
        "page_end": page_end,
        "table_id": table_id or None,
        "schema_name": schema_name or None,
        "operation_id": operation_id or None,
        "object_tags": object_tags,
        "canonical_title": canonical_title,
        "normalized_terms": normalized_terms or normalize_query_terms(" ".join([canonical_title, clause_title, " ".join(object_tags)])),
        "related_specs": sorted(set(related_specs or [source["spec_id"]])),
        "citation_anchor": citation_anchor or f"{source['spec_id']}:{clause_path or clause_title}",
        "strategy_domains": sorted(set(strategy_domains or ([source["policy_domain"]] if source["policy_domain"] in {"sm_policy", "ursp"} else []))),
    }


def make_record(*, record_id: str, page_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": record_id,
        "page_content": page_content,
        "metadata": metadata,
    }


def _split_oversized_text(text: str, *, max_length: int = TARGET_EMBED_CHUNK_LENGTH) -> List[str]:
    normalized = str(text or "").strip()
    if not normalized:
        return []
    if len(normalized) <= max_length:
        return [normalized]

    parts = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    if len(parts) <= 1:
        parts = [part.strip() for part in normalized.splitlines() if part.strip()]
    if len(parts) <= 1:
        parts = [part.strip() for part in re.split(r"(?<=[.;:])\s+", normalized) if part.strip()]
    if len(parts) <= 1:
        parts = [normalized[index : index + max_length] for index in range(0, len(normalized), max_length)]

    chunks: List[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}\n\n{part}".strip() if current else part
        if len(candidate) <= max_length:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(part) <= max_length:
            current = part
            continue
        chunks.extend(part[index : index + max_length].strip() for index in range(0, len(part), max_length) if part[index : index + max_length].strip())
    if current:
        chunks.append(current)
    return chunks


def _record_with_safe_chunks(*, record_id: str, page_content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = _split_oversized_text(page_content)
    records: List[Dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunks, start=1):
        suffix = f"-part-{chunk_index}" if len(chunks) > 1 else ""
        records.append(make_record(record_id=f"{record_id}{suffix}", page_content=chunk, metadata=metadata))
    return records


def split_large_section(text: str) -> List[str]:
    cleaned = sanitize_text(text)
    if estimate_tokens(cleaned) <= MAX_CHUNK_TOKENS:
        return [cleaned]
    parts = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for part in parts:
        part_tokens = estimate_tokens(part)
        if current and current_tokens + part_tokens > MAX_CHUNK_TOKENS:
            chunks.append("\n\n".join(current).strip())
            current = [part]
            current_tokens = part_tokens
            continue
        current.append(part)
        current_tokens += part_tokens
    if current:
        chunks.append("\n\n".join(current).strip())
    merged: List[str] = []
    for chunk in chunks:
        if merged and estimate_tokens(chunk) < MIN_CHUNK_TOKENS:
            merged[-1] = f"{merged[-1]}\n\n{chunk}".strip()
        else:
            merged.append(chunk)
    return [chunk for chunk in merged if chunk.strip()]


def split_tables(section_text: str) -> List[Dict[str, str]]:
    matches = list(TABLE_RE.finditer(section_text))
    if not matches:
        return [{"kind": "body", "title": "", "text": section_text}]
    segments: List[Dict[str, str]] = []
    cursor = 0
    for index, match in enumerate(matches):
        start = match.start()
        if start > cursor:
            segments.append({"kind": "body", "title": "", "text": section_text[cursor:start]})
        end = matches[index + 1].start() if index + 1 < len(matches) else len(section_text)
        segments.append({"kind": "table", "title": match.group(1).strip(), "text": section_text[start:end]})
        cursor = end
    return [segment for segment in segments if segment["text"].strip()]


def is_probable_heading(line: str) -> bool:
    normalized = str(line or "").strip()
    match = HEADING_RE.match(normalized)
    if not match:
        return False
    if estimate_tokens(normalized) >= 40:
        return False

    number = match.group("num").strip()
    title = match.group("title").strip()
    number_parts = number.split(".")

    if len(number_parts) == 1:
        try:
            if int(number_parts[0]) > 50:
                return False
        except ValueError:
            return False

    if CHANGE_REQUEST_RE.search(title):
        return False
    if title.lower().startswith(("approved at", "corrections", "editorial", "rapporteur")):
        return False
    if re.search(r"\b\d{5,}\b", title):
        return False
    return True


def extract_pdf_sections(source: Dict[str, Any]) -> List[Dict[str, Any]]:
    reader = PdfReader(source["local_path"])
    section_heading = ""
    section_number = ""
    section_start_page = 1
    buffer: List[str] = []
    records: List[Dict[str, Any]] = []
    record_counter = 0

    def flush(current_page: int) -> None:
        nonlocal buffer, section_heading, section_number, section_start_page, record_counter
        text = sanitize_text("\n".join(buffer))
        if not text or not section_number:
            buffer = []
            return
        segments = split_tables(text)
        for segment_index, segment in enumerate(segments, start=1):
            doc_type = source["doc_type"] if segment["kind"] == "body" else "table"
            canonical_title = f"{source['spec_id']} {section_number} {section_heading}".strip()
            if segment["kind"] == "table":
                canonical_title = f"{canonical_title} {segment['title']}".strip()
            object_tags = infer_object_tags(segment["text"])
            searchable_text = f"{canonical_title}\n{segment['text'][:1200]}"
            strategy_domains = _infer_strategy_domains(
                source_policy_domain=source["policy_domain"],
                object_tags=object_tags,
                searchable_text=searchable_text,
            )
            if not _should_keep_record(
                source=source,
                object_tags=object_tags,
                searchable_text=searchable_text,
                strategy_domains=strategy_domains,
            ):
                continue
            related_specs = [source["spec_id"]]
            if source["policy_domain"] == "shared":
                related_specs.extend(["29.512", "29.525", "23.503", "24.526"])
            metadata = make_metadata(
                source=source,
                doc_type=doc_type,
                clause_path=section_number,
                clause_title=section_heading,
                page_start=section_start_page,
                page_end=current_page,
                canonical_title=canonical_title,
                object_tags=object_tags,
                table_id=segment["title"],
                citation_anchor=f"{source['spec_id']}:{section_number}",
                related_specs=related_specs,
                normalized_terms=normalize_query_terms(f"{canonical_title} {segment['text'][:240]}"),
                strategy_domains=strategy_domains,
            )
            for chunk_index, chunk in enumerate(split_large_section(segment["text"]), start=1):
                record_counter += 1
                record_id = (
                    f"{source['spec_id']}-{section_number}-"
                    f"{'tbl' if segment['kind'] == 'table' else 'clause'}-"
                    f"{segment_index}-{chunk_index}-{record_counter}"
                )
                records.extend(_record_with_safe_chunks(record_id=record_id, page_content=chunk, metadata=metadata))
        buffer = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = sanitize_text(page.extract_text() or "")
        if not page_text:
            continue
        lines = page_text.splitlines()
        page_buffer: List[str] = []
        for line in lines:
            match = HEADING_RE.match(line)
            if match and is_probable_heading(line):
                flush(page_number - 1 if page_buffer or buffer else page_number)
                section_number = match.group("num").strip()
                section_heading = match.group("title").strip()
                section_start_page = page_number
                page_buffer = []
                continue
            page_buffer.append(line)
        if page_buffer:
            buffer.append("\n".join(page_buffer))
    flush(len(reader.pages))
    return records


def build_openapi_operation_chunks(source: Dict[str, Any], document: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    paths = document.get("paths") or {}
    for path_name, operation_map in paths.items():
        if not isinstance(operation_map, dict):
            continue
        for method_name, operation in operation_map.items():
            if method_name.lower() not in {"get", "post", "put", "patch", "delete"} or not isinstance(operation, dict):
                continue
            operation_id = str(operation.get("operationId") or f"{method_name.upper()} {path_name}").strip()
            title = str(operation.get("summary") or operation.get("description") or operation_id).strip()
            payload = {
                "method": method_name.upper(),
                "path": path_name,
                "operationId": operation_id,
                "summary": operation.get("summary"),
                "description": operation.get("description"),
                "parameters": operation.get("parameters"),
                "requestBody": operation.get("requestBody"),
                "responses": operation.get("responses"),
            }
            page_content = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
            object_tags = infer_object_tags(page_content, operation_id=operation_id)
            searchable_text = f"{operation_id}\n{title}\n{page_content[:1200]}"
            strategy_domains = _infer_strategy_domains(
                source_policy_domain=source["policy_domain"],
                object_tags=object_tags,
                searchable_text=searchable_text,
            )
            if not _should_keep_record(
                source=source,
                object_tags=object_tags,
                searchable_text=searchable_text,
                strategy_domains=strategy_domains,
            ):
                continue
            metadata = make_metadata(
                source=source,
                doc_type="openapi",
                clause_path=path_name,
                clause_title=title,
                page_start=1,
                page_end=1,
                canonical_title=f"{source['spec_id']} {operation_id}",
                object_tags=object_tags,
                operation_id=operation_id,
                citation_anchor=f"{source['spec_id']}:{operation_id}",
                strategy_domains=strategy_domains,
            )
            record_id = f"{source['spec_id']}-op-{operation_id}".replace("/", "_")
            records.extend(_record_with_safe_chunks(record_id=record_id, page_content=page_content, metadata=metadata))
    return records


def build_openapi_schema_chunks(source: Dict[str, Any], document: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    schemas = (((document.get("components") or {}).get("schemas")) or {})
    for schema_name, schema in schemas.items():
        if not isinstance(schema, dict):
            continue
        page_content = yaml.safe_dump({"schema": schema_name, "definition": schema}, sort_keys=False, allow_unicode=True)
        object_tags = infer_object_tags(page_content, schema_name=schema_name)
        searchable_text = f"{schema_name}\n{page_content[:1200]}"
        strategy_domains = _infer_strategy_domains(
            source_policy_domain=source["policy_domain"],
            object_tags=object_tags,
            searchable_text=searchable_text,
        )
        if not _should_keep_record(
            source=source,
            object_tags=object_tags,
            searchable_text=searchable_text,
            strategy_domains=strategy_domains,
        ):
            continue
        metadata = make_metadata(
            source=source,
            doc_type="openapi",
            clause_path=schema_name,
            clause_title=schema_name,
            page_start=1,
            page_end=1,
            canonical_title=f"{source['spec_id']} schema {schema_name}",
            object_tags=object_tags or [schema_name],
            schema_name=schema_name,
            citation_anchor=f"{source['spec_id']}:{schema_name}",
            strategy_domains=strategy_domains,
        )
        records.extend(
            _record_with_safe_chunks(
                record_id=f"{source['spec_id']}-schema-{schema_name}",
                page_content=page_content,
                metadata=metadata,
            )
        )
    return records


def extract_openapi_chunks(source: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = yaml.safe_load(Path(source["local_path"]).read_text(encoding="utf-8"))
    return build_openapi_operation_chunks(source, payload) + build_openapi_schema_chunks(source, payload)


def build_glossary_records(clause_records: List[Dict[str, Any]], schema_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    aggregated: Dict[str, Dict[str, Any]] = {}
    for canonical, config in CANONICAL_TERM_ALIASES.items():
        aggregated[canonical] = {
            "aliases": list(config["aliases"]),
            "related_specs": list(config["related_specs"]),
            "related_objects": list(config["related_objects"]),
            "policy_domain": config["policy_domain"],
            "citations": [],
        }

    for record in clause_records + schema_records:
        metadata = record["metadata"]
        for object_tag in metadata.get("object_tags") or []:
            aggregated.setdefault(
                object_tag,
                {
                    "aliases": [object_tag],
                    "related_specs": [metadata["spec_id"]],
                    "related_objects": [object_tag],
                    "policy_domain": metadata["policy_domain"],
                    "citations": [],
                },
            )
            aggregated[object_tag]["related_specs"].append(metadata["spec_id"])
            aggregated[object_tag]["citations"].append(metadata["citation_anchor"])

    records: List[Dict[str, Any]] = []
    for canonical, payload in sorted(aggregated.items()):
        aliases = sorted(set(alias for alias in payload["aliases"] if alias))
        related_specs = sorted(set(payload["related_specs"]))
        related_objects = sorted(set(payload["related_objects"]))
        citations = sorted(set(payload["citations"]))[:8]
        page_content = (
            f"Canonical term: {canonical}\n"
            f"Aliases: {', '.join(aliases)}\n"
            f"Related specs: {', '.join(related_specs)}\n"
            f"Related objects: {', '.join(related_objects)}\n"
            f"Citations: {', '.join(citations)}"
        )
        metadata = {
            "spec_id": "glossary",
            "release": "18",
            "version": "r18",
            "source_url": None,
            "doc_type": "glossary",
            "policy_domain": payload["policy_domain"],
            "clause_path": canonical,
            "clause_title": canonical,
            "page_start": 1,
            "page_end": 1,
            "table_id": None,
            "schema_name": None,
            "operation_id": None,
            "object_tags": related_objects or [canonical],
            "canonical_title": canonical,
            "normalized_terms": normalize_query_terms(" ".join([canonical, *aliases])),
            "related_specs": related_specs,
            "citation_anchor": canonical,
            "aliases": aliases,
            "related_objects": related_objects,
            "strategy_domains": [payload["policy_domain"]],
        }
        records.append(make_record(record_id=f"glossary-{canonical}", page_content=page_content, metadata=metadata))
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def build_exact_index(records: List[Dict[str, Any]], *, index_name: str) -> Dict[str, Any]:
    documents = []
    df: Dict[str, int] = {}
    for record in records:
        metadata = record["metadata"]
        content_tokens = tokenize(record["page_content"])
        title_tokens = tokenize(" ".join([metadata.get("canonical_title") or "", metadata.get("clause_title") or ""]))
        object_tokens = tokenize(" ".join(metadata.get("object_tags") or []))
        normalized_tokens = tokenize(" ".join(metadata.get("normalized_terms") or []))
        alias_tokens = tokenize(" ".join(metadata.get("aliases") or []))

        token_counts: Dict[str, int] = {}
        for token in content_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        for token in title_tokens:
            token_counts[token] = token_counts.get(token, 0) + 5
        for token in object_tokens:
            token_counts[token] = token_counts.get(token, 0) + 4
        for token in normalized_tokens:
            token_counts[token] = token_counts.get(token, 0) + 3
        for token in alias_tokens:
            token_counts[token] = token_counts.get(token, 0) + 2

        for token in set(content_tokens + title_tokens + object_tokens + normalized_tokens + alias_tokens):
            df[token] = df.get(token, 0) + 1
        documents.append(
            {
                "id": record["id"],
                "title": metadata.get("canonical_title"),
                "policy_domain": metadata.get("policy_domain"),
                "spec_id": metadata.get("spec_id"),
                "citation_anchor": metadata.get("citation_anchor"),
                "token_counts": token_counts,
                "title_tokens": sorted(set(title_tokens)),
                "object_tokens": sorted(set(object_tokens)),
            }
        )
    return {
        "index_name": index_name,
        "release": "18",
        "doc_count": len(documents),
        "document_frequency": df,
        "documents": documents,
    }


def search_exact_index(index: Dict[str, Any], query: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
    query_terms = normalize_query_terms(query)
    if not query_terms:
        return []
    scored = []
    doc_count = max(1, int(index.get("doc_count") or 0))
    for document in index.get("documents") or []:
        score = 0.0
        token_counts = document.get("token_counts") or {}
        title_tokens = set(document.get("title_tokens") or [])
        object_tokens = set(document.get("object_tokens") or [])
        for term in query_terms:
            tf = token_counts.get(term, 0)
            if tf <= 0:
                continue
            df = max(1, int((index.get("document_frequency") or {}).get(term, 1)))
            idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            field_boost = 1.0
            if term in title_tokens:
                field_boost += 1.5
            if term in object_tokens:
                field_boost += 1.0
            score += tf * idf * field_boost
        if score > 0:
            scored.append({"id": document["id"], "score": round(score, 4), "title": document["title"], "spec_id": document["spec_id"]})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def build_auxiliary_maps(clause_records: List[Dict[str, Any]], schema_records: List[Dict[str, Any]], glossary_records: List[Dict[str, Any]]) -> None:
    spec_object_map: Dict[str, List[Dict[str, Any]]] = {}
    for record in clause_records + schema_records:
        metadata = record["metadata"]
        for object_tag in metadata.get("object_tags") or []:
            spec_object_map.setdefault(object_tag, []).append(
                {
                    "spec_id": metadata["spec_id"],
                    "doc_type": metadata["doc_type"],
                    "citation_anchor": metadata["citation_anchor"],
                    "canonical_title": metadata["canonical_title"],
                }
            )
    term_alias_map = {
        record["metadata"]["canonical_title"]: {
            "aliases": record["metadata"].get("aliases") or [],
            "related_specs": record["metadata"].get("related_specs") or [],
            "related_objects": record["metadata"].get("related_objects") or [],
        }
        for record in glossary_records
    }
    SPEC_OBJECT_MAP_JSON.write_text(json.dumps(spec_object_map, ensure_ascii=False, indent=2), encoding="utf-8")
    TERM_ALIAS_MAP_JSON.write_text(json.dumps(term_alias_map, ensure_ascii=False, indent=2), encoding="utf-8")


def build_eval_queries() -> List[Dict[str, str]]:
    return [
        {"query": "Which AM policy fields carry allowed NSSAI target NSSAI and RFSP", "target_collection": PCF_AM_POLICY_SCHEMA_COLLECTION},
        {"query": "What request triggers are defined for AM Policy Control", "target_collection": PCF_AM_POLICY_SCHEMA_COLLECTION},
        {"query": "How does Npcf_AMPolicyControl relate to service area restriction and presence reporting area", "target_collection": PCF_POLICY_GLOSSARY_COLLECTION},
        {"query": "What core objects are included in SmPolicyDecision", "target_collection": PCF_SM_POLICY_SCHEMA_COLLECTION},
        {"query": "What is the responsibility boundary between Npcf_SMPolicyControl and Npcf_UEPolicyControl", "target_collection": PCF_POLICY_GLOSSARY_COLLECTION},
        {"query": "What descriptors make up a URSP rule", "target_collection": PCF_URSP_CLAUSES_COLLECTION},
        {"query": "What are the standard sources for Traffic descriptor and Route selection descriptor", "target_collection": PCF_URSP_CLAUSES_COLLECTION},
        {"query": "What is the relationship among PccRule QosData and SessionRule", "target_collection": PCF_SM_POLICY_SCHEMA_COLLECTION},
        {"query": "Under what conditions are OS Id and OS App Id used in URSP", "target_collection": PCF_URSP_CLAUSES_COLLECTION},
        {"query": "Does URSP belong to the SM Policy Control API", "target_collection": PCF_POLICY_GLOSSARY_COLLECTION},
        {"query": "Where does access traffic steering switching and splitting appear in the control chain", "target_collection": PCF_URSP_CLAUSES_COLLECTION},
    ]


def build_processed_corpus() -> Dict[str, int]:
    manifest = [source for source in load_manifest() if is_minimal_source(source)]
    if not manifest:
        raise RuntimeError("No minimal knowledge-base sources found in manifest. Run fetch first.")
    clause_records: List[Dict[str, Any]] = []
    schema_records: List[Dict[str, Any]] = []
    for source in manifest:
        if source["doc_type"] == "openapi":
            schema_records.extend(extract_openapi_chunks(source))
        else:
            clause_records.extend(extract_pdf_sections(source))
    glossary_records = build_glossary_records(clause_records, schema_records)
    write_jsonl(CLAUSE_JSONL, clause_records)
    write_jsonl(SCHEMA_JSONL, schema_records)
    write_jsonl(GLOSSARY_JSONL, glossary_records)
    build_auxiliary_maps(clause_records, schema_records, glossary_records)
    clause_exact = build_exact_index(clause_records, index_name="clause_exact")
    schema_exact = build_exact_index(schema_records, index_name="schema_exact")
    glossary_exact = build_exact_index(glossary_records, index_name="glossary_exact")
    CLAUSE_EXACT_INDEX_JSON.write_text(json.dumps(clause_exact, ensure_ascii=False, indent=2), encoding="utf-8")
    SCHEMA_EXACT_INDEX_JSON.write_text(json.dumps(schema_exact, ensure_ascii=False, indent=2), encoding="utf-8")
    GLOSSARY_EXACT_INDEX_JSON.write_text(json.dumps(glossary_exact, ensure_ascii=False, indent=2), encoding="utf-8")
    RETRIEVAL_EVAL_JSON.write_text(json.dumps(build_eval_queries(), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"clauses": len(clause_records), "schema": len(schema_records), "glossary": len(glossary_records)}


def records_to_documents(records: Iterable[Dict[str, Any]]) -> List[Any]:
    return [build_pgvector_document(page_content=record["page_content"], metadata=record["metadata"]) for record in records]


def collection_scoped_ids(collection_name: str, records: Iterable[Dict[str, Any]]) -> List[str]:
    normalized_collection = str(collection_name or "").strip()
    if not normalized_collection:
        raise ValueError("collection_name is required for collection-scoped ids.")
    scoped_ids: List[str] = []
    for record in records:
        record_id = str(record.get("id") or "").strip()
        if not record_id:
            raise ValueError("record id is required for collection-scoped ids.")
        scoped_ids.append(f"{normalized_collection}:{record_id}")
    return scoped_ids


def normalize_records_for_embedding(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_records: List[Dict[str, Any]] = []
    for record in records:
        normalized_records.extend(
            _record_with_safe_chunks(
                record_id=str(record["id"]),
                page_content=str(record["page_content"]),
                metadata=dict(record["metadata"]),
            )
        )
    return normalized_records


def ingest_processed_corpus() -> Dict[str, int]:
    clause_records = normalize_records_for_embedding(load_jsonl(CLAUSE_JSONL))
    schema_records = normalize_records_for_embedding(load_jsonl(SCHEMA_JSONL))
    glossary_records = normalize_records_for_embedding(load_jsonl(GLOSSARY_JSONL))
    if not clause_records or not schema_records or not glossary_records:
        raise RuntimeError("Processed corpus is incomplete. Run build first.")

    am_clause_records = [record for record in clause_records if "am_policy" in (record["metadata"].get("strategy_domains") or [])]
    sm_clause_records = [record for record in clause_records if "sm_policy" in (record["metadata"].get("strategy_domains") or [])]
    ursp_clause_records = [record for record in clause_records if "ursp" in (record["metadata"].get("strategy_domains") or [])]
    am_schema_records = [record for record in schema_records if "am_policy" in (record["metadata"].get("strategy_domains") or [])]
    sm_schema_records = [record for record in schema_records if "sm_policy" in (record["metadata"].get("strategy_domains") or [])]
    ursp_schema_records = [record for record in schema_records if "ursp" in (record["metadata"].get("strategy_domains") or [])]

    targets = [
        (PCF_AM_POLICY_CLAUSES_COLLECTION, am_clause_records),
        (PCF_AM_POLICY_SCHEMA_COLLECTION, am_schema_records),
        (PCF_SM_POLICY_CLAUSES_COLLECTION, sm_clause_records),
        (PCF_SM_POLICY_SCHEMA_COLLECTION, sm_schema_records),
        (PCF_URSP_CLAUSES_COLLECTION, ursp_clause_records),
        (PCF_URSP_SCHEMA_COLLECTION, ursp_schema_records),
        (PCF_POLICY_GLOSSARY_COLLECTION, glossary_records),
    ]
    stats: Dict[str, int] = {}
    for collection_name, records in targets:
        store = rebuild_pgvector_collection(collection_name=collection_name)
        if records:
            store.add_documents(records_to_documents(records), ids=collection_scoped_ids(collection_name, records))
        stats[collection_name] = len(records)
        logger.info("Ingested %s documents into %s.", len(records), collection_name)
    return stats


def run_pipeline(*, force_fetch: bool = False) -> Dict[str, Any]:
    manifest = fetch_sources(force=force_fetch)
    processed = build_processed_corpus()
    ingested = ingest_processed_corpus()
    return {"sources": len(manifest), "processed": processed, "ingested": ingested}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the PCF SM/AM/UE policy standards knowledge base.")
    parser.add_argument("command", choices=["fetch", "build", "ingest", "all"])
    parser.add_argument("--force-fetch", action="store_true")
    args = parser.parse_args()

    ensure_directories()
    if args.command == "fetch":
        result = fetch_sources(force=args.force_fetch)
    elif args.command == "build":
        result = build_processed_corpus()
    elif args.command == "ingest":
        result = ingest_processed_corpus()
    else:
        result = run_pipeline(force_fetch=args.force_fetch)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
