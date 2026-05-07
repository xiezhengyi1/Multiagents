from __future__ import annotations

import json
import os
import re
import signal
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain.tools import ToolRuntime

from shared.tools import tool_with_reason

from shared.runtime import AgentRuntimeContext
from database.langchain_pg import (
    PCF_AM_POLICY_CLAUSES_COLLECTION,
    PCF_AM_POLICY_SCHEMA_COLLECTION,
    PCF_SM_POLICY_CLAUSES_COLLECTION,
    PCF_SM_POLICY_SCHEMA_COLLECTION,
    PCF_URSP_CLAUSES_COLLECTION,
    PCF_URSP_SCHEMA_COLLECTION,
    get_pgvector_store,
)
from knowledge_build.scripts.build_pcf_policy_kb import (
    CLAUSE_EXACT_INDEX_JSON,
    CLAUSE_JSONL,
    GLOSSARY_EXACT_INDEX_JSON,
    GLOSSARY_JSONL,
    SCHEMA_EXACT_INDEX_JSON,
    SCHEMA_JSONL,
    SPEC_OBJECT_MAP_JSON,
    TERM_ALIAS_MAP_JSON,
    normalize_query_terms,
    search_exact_index,
)
from shared.logging import setup_logger

logger = setup_logger(__name__)


DEFAULT_RETURNED_RESULTS = 10
MAX_RETURNED_RESULTS = 30
MAX_CONTENT_CHARS = 1200
MAX_SUMMARY_CHARS = 240
BGE_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
BGE_RERANK_INPUT_CHARS = 3200
BGE_RERANK_TERM_CHARS = 700
RRF_K = 60.0
RRF_SCORE_SCALE = 1000.0
DEBUG_STAGE_TIMEOUT_SECONDS = max(
    1,
    int(str(os.getenv("KNOWLEDGE_DEBUG_STAGE_TIMEOUT_SECONDS", "30")).strip() or "30"),
)
LEXICAL_CANDIDATE_LIMIT = 24
MAX_RERANK_POOL = 24
CROSS_SPEC_SEED_LIMIT = 4
CROSS_SPEC_OBJECT_LIMIT = 6
CROSS_SPEC_CITATION_LIMIT = 24
VECTOR_SEARCH_K = 8
VECTOR_SEARCH_TRIGGER_THRESHOLD = 8
HF_CACHE_ROOT = Path(str(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")).strip()).expanduser()
HF_HUB_MODELS_ROOT = HF_CACHE_ROOT / "hub"
SEARCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "belong",
    "by",
    "carry",
    "core",
    "defined",
    "describe",
    "descriptors",
    "does",
    "do",
    "explain",
    "fields",
    "for",
    "how",
    "in",
    "included",
    "include",
    "is",
    "it",
    "make",
    "of",
    "or",
    "relation",
    "relationship",
    "relate",
    "responsibility",
    "the",
    "their",
    "these",
    "this",
    "those",
    "to",
    "under",
    "used",
    "using",
    "what",
    "where",
    "which",
}


def _resolve_bge_reranker_model() -> str:
    explicit_path = str(os.getenv("BGE_RERANK_MODEL_PATH", "")).strip()
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        if candidate.exists():
            return str(candidate)

    repo_cache_dir = HF_HUB_MODELS_ROOT / "models--BAAI--bge-reranker-v2-m3" / "snapshots"
    if repo_cache_dir.exists():
        snapshots = sorted(
            (path for path in repo_cache_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return str(snapshots[0])

    return BGE_RERANK_MODEL
GLOSSARY_QUERY_KEYWORDS = {
    "acronym",
    "alias",
    "aliases",
    "basics",
    "called",
    "canonical",
    "expand",
    "expansion",
    "mean",
    "means",
    "term",
    "terminology",
    "what is",
}
GENERIC_CLAUSE_TERMS = {
    "general",
    "overview",
    "security",
    "scope",
    "introduction",
    "references",
    "history",
    "foreword",
}
URSP_SCHEMA_HINT_TERMS = {
    "association",
    "contains",
    "enum",
    "field",
    "fields",
    "parameter",
    "parameters",
    "report",
    "reported",
    "requesttrigger",
    "requestedvalue",
    "requestedvaluerep",
    "response",
    "schema",
}
URSP_CLAUSE_HINT_TERMS = {
    "defined",
    "definition",
    "descriptor",
    "descriptors",
    "make up",
    "rule",
    "rules",
    "structure",
    "where",
}
AM_SCHEMA_HINT_TERMS = {
    "allowed nssai",
    "allowedsnssais",
    "am policy",
    "mobility",
    "policy association",
    "policyassociation",
    "requesttrigger",
    "rfsp",
    "service area",
    "service area restriction",
    "target nssai",
    "targetsnssais",
}
AM_CLAUSE_HINT_TERMS = {
    "clause",
    "described",
    "where",
}
SM_QOS_TABLE_HINT_TERMS = {
    "5qi",
    "delay",
    "jitter",
    "latency",
    "packet delay budget",
    "packet error rate",
    "qos requirements",
    "requirements",
}
AM_POLICY_KEYWORDS = (
    "ampolicy",
    "am policy",
    "npcf_ampolicycontrol",
    "policyassociation",
    "policy association",
    "policyassociationrequest",
    "policyassociationupdaterequest",
    "amrequestedvaluerep",
    "requesttrigger",
    "allowed nssai",
    "target nssai",
    "service area",
    "service area restriction",
    "rfsp",
    "pra",
    "presence reporting area",
    "ue ambr",
)
SM_POLICY_KEYWORDS = (
    "smpolicy",
    "sm policy",
    "npcf_smpolicycontrol",
    "pcc",
    "pccrule",
    "qos",
    "qosdata",
    "sessionrule",
    "trafficcontroldata",
    "chargingdata",
    "usagemonitoring",
    "revalidationtime",
)
URSP_KEYWORDS = (
    "ursp",
    "ue policy",
    "npcf_uepolicycontrol",
    "route selection",
    "traffic descriptor",
    "route selection descriptor",
    "steering",
    "switching",
    "splitting",
    "os id",
    "os app id",
)
SCHEMA_QUERY_KEYWORDS = (
    "schema",
    "field",
    "fields",
    "parameter",
    "parameters",
    "enum",
    "object",
    "objects",
    "contains",
    "include",
    "definition",
    "define",
    "properties",
    "property",
)
OPERATION_QUERY_KEYWORDS = (
    "api",
    "endpoint",
    "operation",
    "method",
    "create",
    "delete",
    "update",
    "read",
    "post",
    "get",
    "put",
    "patch",
)
REQUIRED_KB_PATHS = (
    CLAUSE_JSONL,
    SCHEMA_JSONL,
    GLOSSARY_JSONL,
    CLAUSE_EXACT_INDEX_JSON,
    SCHEMA_EXACT_INDEX_JSON,
    GLOSSARY_EXACT_INDEX_JSON,
    SPEC_OBJECT_MAP_JSON,
    TERM_ALIAS_MAP_JSON,
)
COMPOUND_TERM_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+")


def _log_prefix(runtime: ToolRuntime[AgentRuntimeContext] = None) -> str:
    if runtime is None:
        return "[knowledge_tool]"
    ctx = runtime.context
    return (
        f"[knowledge_tool][agent={ctx.agent_name}]"
        f"[session={ctx.session_id}]"
        f"[snapshot={ctx.snapshot_id}]"
    )


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _normalized(text: str) -> str:
    return str(text or "").strip().lower()


def _shorten(text: str, *, limit: int = MAX_CONTENT_CHARS) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 3].rstrip()}..."


def _normalize_search_term(term: str) -> str:
    normalized = _normalized(term)
    if len(normalized) > 4 and normalized.endswith("ies"):
        return f"{normalized[:-3]}y"
    if len(normalized) > 6 and normalized.endswith("ing"):
        return normalized[:-3]
    if len(normalized) > 5 and normalized.endswith("ed"):
        return normalized[:-2]
    if len(normalized) > 4 and normalized.endswith("s") and not normalized.endswith("ss"):
        return normalized[:-1]
    return normalized


def _expand_compound_tokens(text: str) -> List[str]:
    expanded: List[str] = []
    for raw_token in re.findall(r"[A-Za-z0-9_.-]+", str(text or "")):
        parts = COMPOUND_TERM_RE.findall(raw_token.replace("_", " ").replace("-", " ").replace(".", " "))
        expanded.extend(part.lower() for part in parts if part)
    return expanded


def _search_terms(text: str) -> List[str]:
    normalized = normalize_query_terms(text) + _expand_compound_tokens(text)
    filtered = [
        _normalize_search_term(term)
        for term in normalized
        if term not in SEARCH_STOPWORDS and (len(term) > 2 or term in {"am", "sm", "ue", "os"})
    ]
    deduped = sorted(set(term for term in filtered if term))
    if deduped:
        return deduped
    return sorted(set(_normalize_search_term(term) for term in normalized if term))


def _domain_matches(metadata_domain: Any, requested_domain: str) -> bool:
    normalized_request = _normalized(requested_domain)
    if normalized_request in {"", "all"}:
        return True
    if isinstance(metadata_domain, dict):
        strategy_domains = [_normalized(item) for item in metadata_domain.get("strategy_domains") or [] if _normalized(item)]
        if strategy_domains:
            if normalized_request == "mobility":
                return any(item in {"am_policy", "ursp"} for item in strategy_domains)
            return normalized_request in strategy_domains
        normalized_domain = _normalized(metadata_domain.get("policy_domain") or "")
    else:
        normalized_domain = _normalized(metadata_domain)
    if normalized_domain == "shared":
        return True
    if normalized_request == "mobility":
        return normalized_domain in {"am_policy", "ursp"}
    return normalized_domain == normalized_request


def _spec_priority(spec_id: str, domain: str) -> int:
    normalized_domain = _normalized(domain)
    if normalized_domain == "am_policy":
        order = ["29.507", "23.503", "29.571", "24.501", "23.501", "29.519", "29.525", "24.526", "29.512"]
    elif normalized_domain == "sm_policy":
        order = ["29.512", "23.503", "29.571", "29.514", "29.213", "29.214", "24.501", "23.501", "29.519", "29.525", "24.526"]
    elif normalized_domain == "ursp":
        order = ["24.526", "29.525", "23.503", "29.519", "24.501", "23.501", "29.571", "29.512", "29.514"]
    elif normalized_domain == "mobility":
        order = ["29.507", "24.526", "29.525", "23.503", "29.571", "24.501", "23.501", "29.519"]
    else:
        order = ["23.503", "29.507", "29.512", "24.526", "29.525", "29.571", "29.514", "29.519", "24.501", "23.501"]
    try:
        return max(0, 20 - order.index(spec_id))
    except ValueError:
        return 0


def _has_explicit_schema_intent(query: str) -> bool:
    lowered = _normalized(query)
    return any(keyword in lowered for keyword in SCHEMA_QUERY_KEYWORDS)


@lru_cache(maxsize=8)
def _domain_schema_phrase_index(domain: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    schema_names: set[str] = set()
    object_phrases: set[str] = set()
    for record in _record_catalog().values():
        metadata = record.get("metadata") or {}
        if not _domain_matches(metadata, domain):
            continue
        if _normalized(metadata.get("doc_type") or "") != "openapi":
            continue
        schema_name = str(metadata.get("schema_name") or "").strip()
        if schema_name:
            schema_names.add(schema_name)
        for object_tag in metadata.get("object_tags") or []:
            normalized_tag = str(object_tag or "").strip()
            if normalized_tag:
                object_phrases.add(normalized_tag)
    return tuple(sorted(schema_names)), tuple(sorted(object_phrases))


def _schema_phrase_matches(query: str, domain: str) -> Tuple[set[str], set[str]]:
    query_terms = set(_search_terms(query))
    schema_names, object_phrases = _domain_schema_phrase_index(domain)
    matched_schema_names = {
        phrase for phrase in schema_names
        if len(_normalized(phrase)) >= 3 and set(_search_terms(phrase)).issubset(query_terms)
    }
    matched_object_phrases = {
        phrase for phrase in object_phrases
        if len(_normalized(phrase)) >= 3 and set(_search_terms(phrase)).issubset(query_terms)
    }
    return matched_schema_names, matched_object_phrases


@lru_cache(maxsize=128)
def _expanded_query_terms_cached(query: str, domain: str) -> frozenset[str]:
    return frozenset(_expanded_query_terms(query, domain))


@lru_cache(maxsize=128)
def _schema_phrase_matches_cached(query: str, domain: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    matched_schema_names, matched_object_phrases = _schema_phrase_matches(query, domain)
    return tuple(sorted(matched_schema_names)), tuple(sorted(matched_object_phrases))


def _looks_like_schema_object_query(query: str, domain: str) -> bool:
    if _has_explicit_schema_intent(query):
        return True
    if _prefers_clause_location(query, domain) or _prefers_operation(query) or _prefers_glossary(query):
        return False
    if _looks_like_exact_identifier_query(query):
        return False

    query_terms = _search_terms(query)
    if len(query_terms) <= 3 and _is_short_term_query(query):
        return False

    matched_schema_names, matched_object_phrases = _schema_phrase_matches(query, domain)
    technical_phrase_hits = len(matched_schema_names | matched_object_phrases)
    if technical_phrase_hits >= 2:
        return True

    lowered = _normalized(query)
    if matched_schema_names and any(
        phrase in lowered
        for phrase in (
            "carrier",
            "carries",
            "contains",
            "core objects",
            "information",
            "mapping",
            "reports",
        )
    ):
        return True

    return bool(matched_schema_names) and len(query_terms) >= 5


def _prefers_schema(query: str, domain: str = "all") -> bool:
    return _looks_like_schema_object_query(query, domain)


def _prefers_operation(query: str) -> bool:
    lowered = _normalized(query)
    if any(phrase in lowered for phrase in ("which endpoint", "what endpoint", "which operation", "what operation", "which method", "what method")):
        return True
    if bool(re.search(r"\b(endpoint|operation|method)\b", lowered)):
        return True
    has_http_or_action_verb = bool(
        re.search(r"\b(create|creates|created|delete|deletes|deleted|update|updates|updated|read|reads|post|get|put|patch)\b", lowered)
    )
    if has_http_or_action_verb:
        return True
    return bool(re.search(r"\bapi\b", lowered) and re.search(r"\b(create|creates|created|delete|deletes|deleted|update|updates|updated|read|reads)\b", lowered))


def _prefers_glossary(query: str) -> bool:
    lowered = _normalized(query)
    return any(keyword in lowered for keyword in GLOSSARY_QUERY_KEYWORDS)


def _is_short_term_query(query: str) -> bool:
    terms = _search_terms(query)
    if len(terms) > 6:
        return False
    return not _has_explicit_schema_intent(query) and not _prefers_operation(query)


def _looks_like_exact_identifier_query(query: str) -> bool:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return False
    search_terms = _search_terms(normalized_query)
    has_identifier_shape = (
        ":" in normalized_query
        or "_" in normalized_query
        or bool(re.search(r"[A-Z][a-z]+[A-Z]", normalized_query))
        or bool(re.fullmatch(r"[A-Z0-9][A-Z0-9\s_-]{1,}", normalized_query))
    )
    return has_identifier_shape and len(search_terms) <= 6


def _ursp_prefers_schema(query: str) -> bool:
    lowered = _normalized(query)
    if _prefers_schema(query, "ursp"):
        return True
    return any(term in lowered for term in URSP_SCHEMA_HINT_TERMS)


def _ursp_prefers_clause(query: str) -> bool:
    lowered = _normalized(query)
    return any(term in lowered for term in URSP_CLAUSE_HINT_TERMS)


def _am_prefers_schema(query: str) -> bool:
    lowered = _normalized(query)
    if _prefers_schema(query, "am_policy"):
        return True
    return any(term in lowered for term in AM_SCHEMA_HINT_TERMS)


def _sm_prefers_qos_table(query: str) -> bool:
    lowered = _normalized(query)
    if "5qi" in lowered:
        return True
    if "qos" in lowered and ("requirement" in lowered or "requirements" in lowered):
        return True
    return any(term in lowered for term in SM_QOS_TABLE_HINT_TERMS)


def _am_prefers_clause(query: str) -> bool:
    lowered = _normalized(query)
    return "rfsp" in lowered and any(term in lowered for term in AM_CLAUSE_HINT_TERMS)


def _prefers_clause_location(query: str, domain: str) -> bool:
    lowered = _normalized(query)
    if (
        lowered.startswith("where")
        or " described " in f" {lowered} "
        or " defined " in f" {lowered} "
        or "make up" in lowered
        or "structure" in lowered
    ):
        return True
    if domain == "ursp":
        return _ursp_prefers_clause(query)
    if domain == "am_policy":
        return _am_prefers_clause(query)
    return False


def _infer_policy_domain(query: str, category: Optional[str]) -> str:
    normalized_category = _normalized(category)
    if normalized_category in {"sm_policy", "am_policy", "ursp", "mobility", "shared", "all"}:
        return "all" if normalized_category == "shared" else normalized_category

    haystack = " ".join(part for part in [query, category] if part).lower()
    am_hits = sum(1 for keyword in AM_POLICY_KEYWORDS if keyword in haystack)
    sm_hits = sum(1 for keyword in SM_POLICY_KEYWORDS if keyword in haystack)
    ursp_hits = sum(1 for keyword in URSP_KEYWORDS if keyword in haystack)
    mobility_hits = am_hits + ursp_hits
    if am_hits and not sm_hits and not ursp_hits:
        return "am_policy"
    if sm_hits and not ursp_hits:
        return "sm_policy"
    if ursp_hits and not sm_hits and not am_hits:
        return "ursp"
    if mobility_hits and not sm_hits:
        return "mobility"
    return "all"


def _domain_collection_pair(domain: str) -> Tuple[Optional[str], Optional[str]]:
    if domain == "am_policy":
        return PCF_AM_POLICY_SCHEMA_COLLECTION, PCF_AM_POLICY_CLAUSES_COLLECTION
    if domain == "sm_policy":
        return PCF_SM_POLICY_SCHEMA_COLLECTION, PCF_SM_POLICY_CLAUSES_COLLECTION
    if domain == "ursp":
        return PCF_URSP_SCHEMA_COLLECTION, PCF_URSP_CLAUSES_COLLECTION
    return None, None


def _select_collections(query: str, domain: str) -> List[str]:
    prefers_schema = _prefers_schema(query, domain)
    prefers_clause = _prefers_clause_location(query, domain)
    prefers_operation = _prefers_operation(query)
    schema_collection, clause_collection = _domain_collection_pair(domain)
    if schema_collection and clause_collection:
        if prefers_operation or prefers_schema:
            return [schema_collection]
        if prefers_clause:
            return [clause_collection]
        return [clause_collection, schema_collection]
    if domain == "mobility":
        if prefers_operation or prefers_schema:
            return [
                PCF_AM_POLICY_SCHEMA_COLLECTION,
                PCF_URSP_SCHEMA_COLLECTION,
            ]
        if prefers_clause:
            return [
                PCF_AM_POLICY_CLAUSES_COLLECTION,
                PCF_URSP_CLAUSES_COLLECTION,
            ]
        return [
            PCF_AM_POLICY_CLAUSES_COLLECTION,
            PCF_URSP_CLAUSES_COLLECTION,
            PCF_AM_POLICY_SCHEMA_COLLECTION,
            PCF_URSP_SCHEMA_COLLECTION,
        ]
    if prefers_operation or prefers_schema:
        return [
            PCF_AM_POLICY_SCHEMA_COLLECTION,
            PCF_SM_POLICY_SCHEMA_COLLECTION,
            PCF_URSP_SCHEMA_COLLECTION,
        ]
    if prefers_clause:
        return [
            PCF_AM_POLICY_CLAUSES_COLLECTION,
            PCF_SM_POLICY_CLAUSES_COLLECTION,
            PCF_URSP_CLAUSES_COLLECTION,
        ]
    return [
        PCF_AM_POLICY_CLAUSES_COLLECTION,
        PCF_SM_POLICY_CLAUSES_COLLECTION,
        PCF_URSP_CLAUSES_COLLECTION,
        PCF_AM_POLICY_SCHEMA_COLLECTION,
        PCF_SM_POLICY_SCHEMA_COLLECTION,
        PCF_URSP_SCHEMA_COLLECTION,
    ]


@lru_cache(maxsize=1)
def _ensure_processed_corpus() -> Tuple[Path, ...]:
    missing = [path for path in REQUIRED_KB_PATHS if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "PCF policy knowledge base is not built. Missing files: "
            + ", ".join(str(path) for path in missing)
            + ". Run `python knowledge_build/scripts/build_pcf_policy_kb.py build` first."
        )
    return tuple(Path(path) for path in REQUIRED_KB_PATHS)


@lru_cache(maxsize=1)
def _record_catalog() -> Dict[str, Dict[str, Any]]:
    _ensure_processed_corpus()
    records = _load_jsonl(CLAUSE_JSONL) + _load_jsonl(SCHEMA_JSONL) + _load_jsonl(GLOSSARY_JSONL)
    return {record["id"]: record for record in records}


@lru_cache(maxsize=1)
def _records_by_citation() -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in _record_catalog().values():
        citation = str((record.get("metadata") or {}).get("citation_anchor") or "").strip()
        if citation:
            grouped.setdefault(citation, []).append(record)
    return grouped


@lru_cache(maxsize=1)
def _clause_exact_index() -> Dict[str, Any]:
    _ensure_processed_corpus()
    return _load_json(CLAUSE_EXACT_INDEX_JSON)


@lru_cache(maxsize=1)
def _schema_exact_index() -> Dict[str, Any]:
    _ensure_processed_corpus()
    return _load_json(SCHEMA_EXACT_INDEX_JSON)


@lru_cache(maxsize=1)
def _glossary_exact_index() -> Dict[str, Any]:
    _ensure_processed_corpus()
    return _load_json(GLOSSARY_EXACT_INDEX_JSON)


@lru_cache(maxsize=1)
def _spec_object_map() -> Dict[str, List[Dict[str, Any]]]:
    _ensure_processed_corpus()
    return _load_json(SPEC_OBJECT_MAP_JSON)


@lru_cache(maxsize=1)
def _term_alias_map() -> Dict[str, Dict[str, Any]]:
    _ensure_processed_corpus()
    return _load_json(TERM_ALIAS_MAP_JSON)


def _candidate_key(metadata: Dict[str, Any], fallback_id: str = "") -> str:
    for key in ("citation_anchor", "schema_name", "operation_id", "canonical_title"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return fallback_id


def _sanitize_limit(limit: Optional[int]) -> int:
    if limit is None:
        return DEFAULT_RETURNED_RESULTS
    try:
        parsed = int(limit)
    except (TypeError, ValueError):
        return DEFAULT_RETURNED_RESULTS
    return max(1, min(parsed, MAX_RETURNED_RESULTS))


def _doc_type_priority(doc_type: str, query: str, domain: str) -> float:
    normalized_doc_type = _normalized(doc_type)
    prefers_schema = _prefers_schema(query, domain)
    prefers_clause = _prefers_clause_location(query, domain)
    prefers_operation = _prefers_operation(query)
    prefers_glossary = _prefers_glossary(query)
    if prefers_operation:
        if normalized_doc_type == "openapi":
            return 320.0
        if normalized_doc_type == "glossary":
            return 10.0
        if normalized_doc_type in {"table", "stage2", "stage3"}:
            return -40.0
        return 0.0

    if prefers_schema:
        if normalized_doc_type == "openapi":
            return 260.0
        if normalized_doc_type == "glossary":
            return 35.0
        if normalized_doc_type == "table":
            return 70.0
        if normalized_doc_type in {"stage2", "stage3"}:
            return 35.0
        return 0.0

    if prefers_clause:
        if normalized_doc_type in {"table", "stage2", "stage3"}:
            return 260.0
        if normalized_doc_type == "glossary":
            return 20.0
        if normalized_doc_type == "openapi":
            return -60.0
        return 0.0

    if prefers_glossary:
        if normalized_doc_type == "glossary":
            return 420.0
        if normalized_doc_type == "openapi":
            return 40.0
        if normalized_doc_type in {"stage2", "stage3", "table"}:
            return 20.0
        return 0.0

    if normalized_doc_type == "glossary":
        return 220.0
    if normalized_doc_type == "table":
        return 170.0
    if normalized_doc_type in {"stage2", "stage3"}:
        return 140.0
    if normalized_doc_type == "openapi":
        score = 110.0 if domain == "sm_policy" else 90.0
    else:
        score = 0.0

    if domain == "ursp":
        if _ursp_prefers_schema(query):
            if normalized_doc_type == "openapi":
                score += 120.0
            elif normalized_doc_type in {"stage2", "stage3"}:
                score -= 45.0
        elif _ursp_prefers_clause(query):
            if normalized_doc_type in {"stage2", "stage3"}:
                score += 110.0
            elif normalized_doc_type == "openapi":
                score -= 70.0
            elif normalized_doc_type == "glossary":
                score -= 30.0
    elif domain == "am_policy":
        if _am_prefers_clause(query):
            if normalized_doc_type in {"stage2", "stage3", "table"}:
                score += 150.0
            elif normalized_doc_type == "openapi":
                score -= 80.0
        elif _am_prefers_schema(query):
            if normalized_doc_type == "openapi":
                score += 135.0
            elif normalized_doc_type == "glossary":
                score += 40.0
            elif normalized_doc_type in {"stage2", "stage3"}:
                score -= 55.0
    elif domain == "sm_policy":
        if _sm_prefers_qos_table(query):
            if normalized_doc_type == "table":
                score += 170.0
            elif normalized_doc_type == "openapi":
                score += 55.0
            elif normalized_doc_type in {"stage2", "stage3"}:
                score -= 50.0
    return score


def _expanded_query_terms(query: str, domain: str) -> set[str]:
    base_terms = set(_search_terms(query))
    query_text = _normalized(query)
    alias_map = _term_alias_map()
    expanded = set(base_terms)
    short_term_query = _is_short_term_query(query)
    for canonical_title, payload in alias_map.items():
        phrases = [str(canonical_title)]
        phrases.extend(str(item) for item in payload.get("aliases") or [])
        matched = False
        for phrase in phrases:
            normalized_phrase = _normalized(phrase)
            if len(normalized_phrase) < 3 or normalized_phrase not in query_text:
                continue
            expanded.update(_search_terms(phrase))
            matched = True
        if not matched:
            continue
        expanded.update(_search_terms(canonical_title))
        if not short_term_query:
            expanded.update(
                token
                for item in payload.get("related_objects") or []
                for token in _search_terms(str(item))
            )
        if domain in {"am_policy", "mobility"}:
            if not short_term_query:
                expanded.update(
                    token
                    for item in payload.get("related_specs") or []
                    for token in _search_terms(str(item))
                )
    if domain == "sm_policy" and _sm_prefers_qos_table(query):
        expanded.update(
            {
                "5qi",
                "packet",
                "delay",
                "budget",
                "error",
                "rate",
                "qos",
                "qosdata",
                "qoscharacteristics",
                "latency",
                "jitter",
                "requirement",
            }
        )
    return expanded


def _glossary_phrase_bonus(metadata: Dict[str, Any], query: str) -> float:
    query_terms = set(_search_terms(query))
    query_text = _normalized(query)
    phrases: List[Tuple[str, float]] = []
    canonical_title = str(metadata.get("canonical_title") or "").strip()
    if canonical_title:
        phrases.append((canonical_title, 220.0))
    for alias in metadata.get("aliases") or []:
        phrases.append((str(alias), 260.0))
    for tag in metadata.get("object_tags") or []:
        phrases.append((str(tag), 180.0))

    best = 0.0
    for phrase, base in phrases:
        normalized_phrase = _normalized(phrase)
        if len(normalized_phrase) < 3 or normalized_phrase not in query_text:
            continue
        phrase_terms = set(_search_terms(phrase))
        overlap = len(query_terms.intersection(phrase_terms))
        score = base + 42.0 * overlap
        if normalized_phrase == query_text:
            score += 160.0
        best = max(best, score)
    return best


def _glossary_strong_candidates(query: str, domain: str) -> List[Dict[str, Any]]:
    if _prefers_schema(query, domain) or _prefers_clause_location(query, domain) or _prefers_operation(query):
        return []
    candidates: List[Dict[str, Any]] = []
    for record in _record_catalog().values():
        metadata = record.get("metadata") or {}
        if _normalized(metadata.get("doc_type") or "") != "glossary":
            continue
        if not _domain_matches(metadata, domain):
            continue
        phrase_bonus = _glossary_phrase_bonus(metadata, query)
        if phrase_bonus <= 0:
            continue
        score = 1500.0 + phrase_bonus + _metadata_query_bonus(metadata, query, domain)
        candidates.append(_record_to_candidate(record, score=score, retrieval="glossary_phrase"))
    return _dedupe_candidates(candidates, limit=6)


def _metadata_term_sets(metadata: Dict[str, Any]) -> Tuple[set[str], set[str], set[str]]:
    title_parts = [
        str(metadata.get("canonical_title") or "").strip(),
        str(metadata.get("clause_title") or "").strip(),
        str(metadata.get("schema_name") or "").strip(),
        str(metadata.get("operation_id") or "").strip(),
        str(metadata.get("table_id") or "").strip(),
    ]
    title_terms = set(_search_terms(" ".join(part for part in title_parts if part)))
    object_terms = {
        token
        for item in metadata.get("object_tags") or []
        for token in _search_terms(str(item))
    }
    normalized_terms = set(
        term
        for term in metadata.get("normalized_terms") or []
        if term not in SEARCH_STOPWORDS and (len(term) > 2 or term in {"am", "sm", "ue", "os"})
    )
    return title_terms, object_terms, normalized_terms


def _is_generic_clause_candidate(metadata: Dict[str, Any]) -> bool:
    normalized_doc_type = _normalized(metadata.get("doc_type") or "")
    if normalized_doc_type not in {"stage2", "stage3", "table"}:
        return False
    title = " ".join(
        part
        for part in [
            str(metadata.get("clause_title") or "").strip(),
            str(metadata.get("canonical_title") or "").strip(),
        ]
        if part
    ).lower()
    return any(term in title for term in GENERIC_CLAUSE_TERMS)


def _identifier_match_bonus(identifier: Any, query_terms: set[str]) -> float:
    identifier_terms = set(_search_terms(str(identifier or "")))
    if not identifier_terms:
        return 0.0
    overlap = query_terms.intersection(identifier_terms)
    if not overlap:
        return 0.0
    precision = len(overlap) / len(identifier_terms)
    return 60.0 * len(overlap) + 140.0 * precision


def _should_skip_generic_clause_candidate(metadata: Dict[str, Any], query: str) -> bool:
    if not _is_generic_clause_candidate(metadata):
        return False
    query_terms = set(_search_terms(query))
    title_terms, object_terms, _normalized_terms = _metadata_term_sets(metadata)
    return not bool(query_terms.intersection(title_terms | object_terms))


def _metadata_query_bonus(metadata: Dict[str, Any], query: str, domain: str) -> float:
    score = _spec_priority(str(metadata.get("spec_id") or ""), domain)
    score += _doc_type_priority(str(metadata.get("doc_type") or ""), query, domain)
    normalized_doc_type = _normalized(metadata.get("doc_type") or "")
    query_terms = set(_expanded_query_terms_cached(query, domain))
    matched_schema_names, _matched_object_phrases = _schema_phrase_matches_cached(query, domain)
    matched_schema_names_normalized = {_normalized(name) for name in matched_schema_names}
    title_terms, object_terms, normalized_terms = _metadata_term_sets(metadata)
    score += 24.0 * len(query_terms.intersection(title_terms))
    score += 14.0 * len(query_terms.intersection(object_terms))
    score += 6.0 * len(query_terms.intersection(normalized_terms))
    if _prefers_schema(query, domain):
        score += 120.0 * len(query_terms.intersection(title_terms))
        score += 80.0 * len(query_terms.intersection(object_terms))
        if normalized_doc_type == "glossary":
            score -= 280.0
    if _prefers_clause_location(query, domain):
        if normalized_doc_type == "openapi":
            score -= 240.0
        elif normalized_doc_type == "glossary":
            score -= 90.0
    if _is_generic_clause_candidate(metadata):
        score -= 180.0
        if query_terms.intersection(title_terms | object_terms):
            score += 120.0
    score += _identifier_match_bonus(metadata.get("schema_name"), query_terms)
    if _prefers_schema(query, domain) and metadata.get("schema_name"):
        normalized_schema_name = _normalized(metadata.get("schema_name") or "")
        if matched_schema_names_normalized:
            score += 240.0 if normalized_schema_name in matched_schema_names_normalized else -180.0
        schema_terms = set(_search_terms(str(metadata.get("schema_name") or "")))
        overlap = len(query_terms.intersection(schema_terms))
        if overlap:
            score += 220.0 + 80.0 * overlap
    if metadata.get("operation_id"):
        score += _identifier_match_bonus(metadata.get("operation_id"), query_terms)
        if _prefers_operation(query):
            score += 140.0
            operation_id = _normalized(metadata.get("operation_id") or "")
            query_text = _normalized(query)
            for verb in ("create", "update", "delete", "get", "read"):
                if verb in query_text:
                    score += 180.0 if verb in operation_id else -120.0
        else:
            score -= 680.0
    elif _prefers_operation(query) and _normalized(metadata.get("doc_type") or "") == "openapi":
        score -= 140.0
    return score


def _sm_qos_requirement_candidates(query: str, domain: str) -> List[Dict[str, Any]]:
    if domain != "sm_policy" or not _sm_prefers_qos_table(query):
        return []
    candidates: List[Dict[str, Any]] = []
    for record in _record_catalog().values():
        metadata = record.get("metadata") or {}
        if not _domain_matches(metadata, domain):
            continue
        doc_type = _normalized(metadata.get("doc_type") or "")
        object_tags = {str(item) for item in metadata.get("object_tags") or []}
        title = _normalized(metadata.get("canonical_title") or "")
        if not ({"QosData", "QosCharacteristics", "5Qi"} & object_tags or "qosdata" in title or "5qi" in title):
            continue
        if doc_type not in {"table", "openapi", "glossary"}:
            continue
        base = 2480.0 if doc_type == "table" else 1700.0 if doc_type == "openapi" else 1260.0
        page_text = _normalized(record.get("page_content") or "")
        if "5.6.2.8" in title or "definition of type qosdata" in title or "packetdelaybudget" in page_text:
            base += 820.0
        candidates.append(_record_to_candidate(record, score=base + _metadata_query_bonus(metadata, query, domain), retrieval="sm_qos_requirements"))
    return _dedupe_candidates(candidates, limit=6)


def _schema_object_graph_candidates(query: str, domain: str) -> List[Dict[str, Any]]:
    if not _prefers_schema(query, domain):
        return []

    query_terms = _expanded_query_terms(query, domain)
    matched_schema_names, matched_object_phrases = _schema_phrase_matches(query, domain)
    matched_schema_names_normalized = {_normalized(name) for name in matched_schema_names}
    matched_object_phrases_normalized = {_normalized(name) for name in matched_object_phrases}
    candidates: List[Dict[str, Any]] = []
    for record in _record_catalog().values():
        metadata = record.get("metadata") or {}
        if not _domain_matches(metadata, domain):
            continue
        if _normalized(metadata.get("doc_type") or "") != "openapi":
            continue
        title_terms, object_terms, _ = _metadata_term_sets(metadata)
        title_overlap = len(query_terms.intersection(title_terms))
        object_overlap = len(query_terms.intersection(object_terms))
        schema_name_terms = set(_search_terms(str(metadata.get("schema_name") or "")))
        schema_overlap = len(query_terms.intersection(schema_name_terms))
        if title_overlap == 0 and object_overlap == 0 and schema_overlap == 0:
            continue
        score = 2440.0 + 180.0 * schema_overlap + 120.0 * title_overlap + 90.0 * object_overlap
        normalized_schema_name = _normalized(metadata.get("schema_name") or "")
        if matched_schema_names_normalized:
            score += 420.0 if normalized_schema_name in matched_schema_names_normalized else -180.0
        object_phrase_overlap = sum(
            1
            for object_tag in metadata.get("object_tags") or []
            if _normalized(object_tag) in matched_object_phrases_normalized
        )
        score += 170.0 * object_phrase_overlap
        score += _metadata_query_bonus(metadata, query, domain)
        candidates.append(_record_to_candidate(record, score=score, retrieval="schema_object_graph"))
    return _dedupe_candidates(candidates, limit=8)


def _schema_definition_candidates(query: str, domain: str) -> List[Dict[str, Any]]:
    if not _prefers_schema(query, domain):
        return []

    query_terms = _expanded_query_terms(query, domain)
    query_text = _normalized(query)
    matched_schema_names, matched_object_phrases = _schema_phrase_matches(query, domain)
    matched_schema_names_normalized = {_normalized(name) for name in matched_schema_names}
    matched_object_phrases_normalized = {_normalized(name) for name in matched_object_phrases}
    candidates: List[Dict[str, Any]] = []
    for record in _record_catalog().values():
        metadata = record.get("metadata") or {}
        if not _domain_matches(metadata, domain):
            continue
        if _normalized(metadata.get("doc_type") or "") != "openapi":
            continue
        schema_name = str(metadata.get("schema_name") or "").strip()
        if not schema_name:
            continue
        title_terms, object_terms, normalized_terms = _metadata_term_sets(metadata)
        content_terms = set(_search_terms(str(record.get("page_content") or "")))
        schema_terms = set(_search_terms(schema_name))
        title_overlap = len(query_terms.intersection(title_terms))
        object_overlap = len(query_terms.intersection(object_terms))
        content_overlap = len(query_terms.intersection(content_terms | normalized_terms))
        schema_overlap = len(query_terms.intersection(schema_terms))
        if title_overlap == 0 and object_overlap == 0 and content_overlap == 0 and schema_overlap == 0:
            continue
        score = 2320.0 + 220.0 * schema_overlap + 105.0 * title_overlap + 90.0 * object_overlap + 32.0 * content_overlap
        normalized_schema_name = _normalized(schema_name)
        if normalized_schema_name in query_text:
            score += 380.0
        if matched_schema_names_normalized:
            score += 460.0 if normalized_schema_name in matched_schema_names_normalized else -180.0
        object_phrase_overlap = sum(
            1
            for object_tag in metadata.get("object_tags") or []
            if _normalized(object_tag) in matched_object_phrases_normalized
        )
        score += 140.0 * object_phrase_overlap
        score += _metadata_query_bonus(metadata, query, domain)
        candidates.append(_record_to_candidate(record, score=score, retrieval="schema_definition"))
    return _dedupe_candidates(candidates, limit=8)


def _am_rfsp_clause_candidates(query: str, domain: str) -> List[Dict[str, Any]]:
    if domain != "am_policy" or not _am_prefers_clause(query):
        return []
    candidates: List[Dict[str, Any]] = []
    for record in _record_catalog().values():
        metadata = record.get("metadata") or {}
        if not _domain_matches(metadata, domain):
            continue
        doc_type = _normalized(metadata.get("doc_type") or "")
        if doc_type not in {"stage2", "stage3", "table"}:
            continue
        object_tags = {str(item) for item in metadata.get("object_tags") or []}
        title = _normalized(metadata.get("canonical_title") or "")
        page_text = _normalized(record.get("page_content") or "")
        if "RfspIndex" not in object_tags and "rfsp" not in title and "rfsp index" not in page_text:
            continue
        base = 2520.0
        if "4.2.2.3.2" in title or "rfsp index" in title or "rfsp index" in page_text:
            base += 760.0
        candidates.append(_record_to_candidate(record, score=base + _metadata_query_bonus(metadata, query, domain), retrieval="am_rfsp_clause"))
    return _dedupe_candidates(candidates, limit=6)


def _operation_intent_candidates(query: str, domain: str) -> List[Dict[str, Any]]:
    if not _prefers_operation(query):
        return []
    query_text = _normalized(query)
    query_terms = _expanded_query_terms(query, domain)
    requested_verbs = [verb for verb in ("create", "update", "delete", "get", "read") if verb in query_text]
    candidates: List[Dict[str, Any]] = []
    for record in _record_catalog().values():
        metadata = record.get("metadata") or {}
        if not _domain_matches(metadata, domain):
            continue
        operation_id = str(metadata.get("operation_id") or "").strip()
        if not operation_id:
            continue
        operation_terms = set(_search_terms(operation_id))
        overlap = len(query_terms.intersection(operation_terms))
        if not overlap and not any(verb in _normalized(operation_id) for verb in requested_verbs):
            continue
        score = 2300.0 + 120.0 * overlap + _metadata_query_bonus(metadata, query, domain)
        normalized_operation = _normalized(operation_id)
        for verb in requested_verbs:
            score += 260.0 if verb in normalized_operation else -140.0
        candidates.append(_record_to_candidate(record, score=score, retrieval="operation_intent"))
    return _dedupe_candidates(candidates, limit=6)


def _record_to_candidate(record: Dict[str, Any], *, score: float, retrieval: str) -> Dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    return {
        "id": record.get("id"),
        "page_content": str(record.get("page_content") or ""),
        "metadata": metadata,
        "score": score,
        "retrieval": retrieval,
    }


def _doc_to_candidate(doc: Any, *, score: float, retrieval: str) -> Dict[str, Any]:
    metadata = doc.metadata if isinstance(getattr(doc, "metadata", None), dict) else {}
    return {
        "id": None,
        "page_content": str(getattr(doc, "page_content", "") or ""),
        "metadata": metadata,
        "score": score,
        "retrieval": retrieval,
    }


@lru_cache(maxsize=8)
def _bge_reranker(top_n: int) -> Any:
    try:
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "BGE reranking requires LangChain's HuggingFace cross-encoder support. "
            "Install `sentence-transformers` and ensure `langchain_community.cross_encoders.HuggingFaceCrossEncoder` is available."
        ) from exc

    model = HuggingFaceCrossEncoder(model_name=_resolve_bge_reranker_model())
    return CrossEncoderReranker(model=model, top_n=top_n)


def warmup_knowledge_tool_models() -> Dict[str, Any]:
    warmed: Dict[str, Any] = {
        "corpus": [],
        "vectorstores": [],
        "rerankers": [],
    }

    warmed["corpus"] = [str(path) for path in _ensure_processed_corpus()]

    collection_names = [
        PCF_AM_POLICY_CLAUSES_COLLECTION,
        PCF_AM_POLICY_SCHEMA_COLLECTION,
        PCF_SM_POLICY_CLAUSES_COLLECTION,
        PCF_SM_POLICY_SCHEMA_COLLECTION,
        PCF_URSP_CLAUSES_COLLECTION,
        PCF_URSP_SCHEMA_COLLECTION,
    ]
    for collection_name in collection_names:
        get_pgvector_store(collection_name=collection_name)
        warmed["vectorstores"].append(collection_name)

    _bge_reranker(1)
    warmed["rerankers"].append(BGE_RERANK_MODEL)
    return warmed


def _limited_terms(values: Iterable[Any], *, limit: int = BGE_RERANK_TERM_CHARS) -> str:
    seen: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.append(text)
        joined = ", ".join(seen)
        if len(joined) >= limit:
            return joined[:limit].rstrip(" ,")
    return ", ".join(seen)


def _bge_rerank_query(query: str, domain: str) -> str:
    expanded_terms = _limited_terms(sorted(_expanded_query_terms(query, domain)))
    if not expanded_terms:
        return query
    return f"Query: {query}\nExpanded technical terms: {expanded_terms}"


def _candidate_rerank_text(candidate: Dict[str, Any]) -> str:
    metadata = candidate["metadata"]
    result_key = _candidate_key(metadata, candidate.get("id") or "")
    aliases = _limited_terms(metadata.get("aliases") or [])
    objects = _limited_terms(metadata.get("object_tags") or [])
    normalized_terms = _limited_terms(metadata.get("normalized_terms") or [])
    parts = [
        f"Candidate ResultKey: {result_key}",
        f"Candidate title: {metadata.get('canonical_title', '')}",
        f"Candidate knowledge type: {metadata.get('doc_type', '')}",
        f"Candidate policy domain: {metadata.get('policy_domain', '')}",
        f"Candidate schema name: {metadata.get('schema_name', '')}",
        f"Candidate operation id: {metadata.get('operation_id', '')}",
        f"Candidate clause path: {metadata.get('clause_path', '')}",
        f"Candidate table id: {metadata.get('table_id', '')}",
        f"Candidate aliases: {aliases}",
        f"Candidate object tags: {objects}",
        f"Candidate normalized technical terms: {normalized_terms}",
        f"Content: {candidate.get('page_content', '')}",
    ]
    return _shorten(" ".join(part for part in parts if part), limit=BGE_RERANK_INPUT_CHARS)


def _bge_rerank_candidates(query: str, domain: str, candidates: Iterable[Dict[str, Any]], *, limit: int) -> List[Dict[str, Any]]:
    candidate_list = list(candidates)
    if not candidate_list:
        return []
    rerank_query = _bge_rerank_query(query, domain)
    compressor = _bge_reranker(limit)
    documents = [
        Document(
            page_content=_candidate_rerank_text(candidate),
            metadata={"candidate_index": index},
        )
        for index, candidate in enumerate(candidate_list)
    ]
    reranked_documents = list(compressor.compress_documents(documents, query=rerank_query))
    if not reranked_documents:
        raise RuntimeError("BGE reranker returned no documents for a non-empty candidate set.")

    ordered: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for document in reranked_documents[:limit]:
        index = document.metadata.get("candidate_index")
        if not isinstance(index, int):
            raise RuntimeError("BGE reranker did not preserve candidate_index metadata.")
        if index in seen:
            continue
        seen.add(index)
        ordered.append(candidate_list[index])
    return ordered


def _dedupe_candidates(candidates: Iterable[Dict[str, Any]], *, limit: int = DEFAULT_RETURNED_RESULTS) -> List[Dict[str, Any]]:
    return _aggregate_parent_candidates(candidates, limit=limit)


def _retrieval_family(retrieval: str) -> str:
    normalized = str(retrieval or "").strip()
    return normalized.split(":", 1)[0] if normalized else "unknown"


def _aggregate_parent_candidates(
    candidates: Iterable[Dict[str, Any]],
    *,
    limit: int = DEFAULT_RETURNED_RESULTS,
) -> List[Dict[str, Any]]:
    candidate_list = list(candidates)
    if not candidate_list:
        return []

    family_rankings: Dict[str, Dict[str, int]] = defaultdict(dict)
    family_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in candidate_list:
        parent_key = _candidate_key(candidate["metadata"], candidate.get("id") or "")
        if not parent_key:
            continue
        family = _retrieval_family(candidate.get("retrieval") or "")
        family_buckets[family].append(candidate)

    for family, bucket in family_buckets.items():
        ranked_bucket = sorted(bucket, key=lambda item: item["score"], reverse=True)
        rank_counter = 1
        seen_parent_keys: set[str] = set()
        for candidate in ranked_bucket:
            parent_key = _candidate_key(candidate["metadata"], candidate.get("id") or "")
            if not parent_key or parent_key in seen_parent_keys:
                continue
            seen_parent_keys.add(parent_key)
            family_rankings[family][parent_key] = rank_counter
            rank_counter += 1

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in candidate_list:
        metadata = candidate["metadata"]
        parent_key = _candidate_key(metadata, candidate.get("id") or "")
        if not parent_key:
            continue
        grouped[parent_key].append(candidate)

    aggregated: List[Dict[str, Any]] = []
    for parent_key, group in grouped.items():
        ranked_group = sorted(group, key=lambda item: item["score"], reverse=True)
        representative = dict(ranked_group[0])
        representative_metadata = dict(representative["metadata"])
        support_count = len(ranked_group)
        retrieval_families = sorted({_retrieval_family(item.get("retrieval") or "") for item in ranked_group})
        rrf_score = 0.0
        for family in retrieval_families:
            rank = family_rankings.get(family, {}).get(parent_key)
            if rank is not None:
                rrf_score += 1.0 / (RRF_K + float(rank))
        representative["score"] = round(float(representative["score"]) + rrf_score * RRF_SCORE_SCALE, 4)
        representative_retrieval = str(representative.get("retrieval") or "")
        if support_count > 1 or len(retrieval_families) > 1:
            representative["retrieval"] = (
                representative_retrieval
                if representative_retrieval.startswith("parent_aggregate:")
                else f"parent_aggregate:{representative_retrieval}"
            )
        else:
            representative["retrieval"] = representative_retrieval
        representative["metadata"] = {
            **representative_metadata,
            "parent_result_key": parent_key,
            "support_count": support_count,
            "supporting_retrievals": retrieval_families,
            "rrf_score": round(rrf_score, 6),
        }
        aggregated.append(representative)

    return sorted(aggregated, key=lambda item: item["score"], reverse=True)[:limit]


def _direct_record_matches(key: str, domain: str) -> List[Dict[str, Any]]:
    normalized_key = _normalized(key)
    catalog = _record_catalog()
    direct_hits: List[Dict[str, Any]] = []
    for record in catalog.values():
        metadata = record.get("metadata") or {}
        if not _domain_matches(metadata, domain):
            continue
        search_fields = [
            metadata.get("canonical_title"),
            metadata.get("schema_name"),
            metadata.get("operation_id"),
            metadata.get("citation_anchor"),
            metadata.get("clause_path"),
            metadata.get("clause_title"),
        ]
        search_fields.extend(metadata.get("aliases") or [])
        search_fields.extend(metadata.get("object_tags") or [])
        score = 0.0
        if _normalized(metadata.get("schema_name") or "") == normalized_key:
            score = 2300.0
        elif _normalized(metadata.get("operation_id") or "") == normalized_key:
            score = 2200.0
        elif _normalized(metadata.get("canonical_title") or "") == normalized_key:
            score = 2150.0
        elif any(_normalized(alias) == normalized_key for alias in metadata.get("aliases") or []):
            score = 2050.0
        elif any(_normalized(tag) == normalized_key for tag in metadata.get("object_tags") or []):
            score = 1950.0
        elif any(_normalized(field) == normalized_key for field in search_fields if field):
            score = 1800.0
        if score > 0:
            score += _metadata_query_bonus(metadata, key, domain)
            direct_hits.append(_record_to_candidate(record, score=score, retrieval="exact_identifier"))
    return _dedupe_candidates(direct_hits, limit=3)


def _exact_search_candidates(query: str, domain: str) -> List[Dict[str, Any]]:
    catalog = _record_catalog()
    candidates: List[Dict[str, Any]] = []
    prefers_schema = _prefers_schema(query, domain)
    for hit in search_exact_index(_clause_exact_index(), query, top_k=12):
        record = catalog.get(hit["id"])
        if record is None:
            continue
        metadata = record.get("metadata") or {}
        if not _domain_matches(metadata, domain):
            continue
        if _should_skip_generic_clause_candidate(metadata, query):
            continue
        clause_base = 1120.0 if prefers_schema else 1260.0
        score = clause_base + float(hit["score"]) + _metadata_query_bonus(metadata, query, domain)
        candidates.append(_record_to_candidate(record, score=score, retrieval="exact_clause"))
    for hit in search_exact_index(_schema_exact_index(), query, top_k=12):
        record = catalog.get(hit["id"])
        if record is None:
            continue
        metadata = record.get("metadata") or {}
        if not _domain_matches(metadata, domain):
            continue
        schema_base = 1380.0 if prefers_schema else 1200.0
        score = schema_base + float(hit["score"]) + _metadata_query_bonus(metadata, query, domain)
        candidates.append(_record_to_candidate(record, score=score, retrieval="exact_schema"))
    for hit in search_exact_index(_glossary_exact_index(), query, top_k=12):
        record = catalog.get(hit["id"])
        if record is None:
            continue
        metadata = record.get("metadata") or {}
        if not _domain_matches(metadata, domain):
            continue
        score = 1000.0 + float(hit["score"]) + _metadata_query_bonus(metadata, query, domain)
        candidates.append(_record_to_candidate(record, score=score, retrieval="exact_glossary"))
    return _dedupe_candidates(candidates, limit=6)


def _top_cross_spec_seeds(seed_candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(
        list(seed_candidates),
        key=lambda item: float(item.get("score") or 0.0),
        reverse=True,
    )
    return ranked[:CROSS_SPEC_SEED_LIMIT]


def _rank_cross_spec_objects(
    seed_candidates: Iterable[Dict[str, Any]],
    query: str,
    alias_map: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, float]]:
    query_terms = set(_search_terms(query))
    object_scores: Dict[str, float] = defaultdict(float)

    for candidate in _top_cross_spec_seeds(seed_candidates):
        metadata = candidate["metadata"]
        seed_score = float(candidate.get("score") or 0.0)
        canonical_title = str(metadata.get("canonical_title") or "").strip()
        candidate_objects = set(str(item).strip() for item in (metadata.get("object_tags") or []) if str(item).strip())
        if canonical_title and canonical_title in alias_map:
            candidate_objects.update(
                str(item).strip()
                for item in (alias_map[canonical_title].get("related_objects") or [])
                if str(item).strip()
            )

        for object_name in candidate_objects:
            overlap = len(query_terms.intersection(_search_terms(object_name)))
            object_scores[object_name] += seed_score + overlap * 80.0

    return sorted(
        object_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )[:CROSS_SPEC_OBJECT_LIMIT]


def _rank_cross_spec_citations(
    ranked_objects: Iterable[Tuple[str, float]],
    spec_map: Dict[str, List[Dict[str, Any]]],
) -> List[Tuple[str, str, float]]:
    citation_scores: Dict[str, Tuple[str, float]] = {}

    for object_name, object_score in ranked_objects:
        seen_citations: set[str] = set()
        for mapping in spec_map.get(object_name) or []:
            citation = str(mapping.get("citation_anchor") or "").strip()
            if not citation or citation in seen_citations:
                continue
            seen_citations.add(citation)
            current = citation_scores.get(citation)
            if current is None or object_score > current[1]:
                citation_scores[citation] = (object_name, object_score)

    ranked_citations = sorted(
        citation_scores.items(),
        key=lambda item: (-item[1][1], item[0]),
    )[:CROSS_SPEC_CITATION_LIMIT]
    return [(citation, object_name, object_score) for citation, (object_name, object_score) in ranked_citations]


def _expand_cross_spec_candidates(
    seed_candidates: Iterable[Dict[str, Any]],
    domain: str,
    query: str,
    emit: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    started_at = time.perf_counter()
    alias_map = _term_alias_map()
    selected_seeds = _top_cross_spec_seeds(seed_candidates)
    if emit is not None:
        emit(f"stage=expand_cross_spec_candidates detail=selected_seeds count={len(selected_seeds)}")

    ranked_objects = _rank_cross_spec_objects(selected_seeds, query, alias_map)
    if emit is not None:
        emit(f"stage=expand_cross_spec_candidates detail=selected_objects count={len(ranked_objects)}")
    if not ranked_objects:
        # logger.info("knowledge_tool cross-spec expand skipped: no ranked_objects")
        return []

    spec_map = _spec_object_map()
    by_citation = _records_by_citation()
    ranked_citations = _rank_cross_spec_citations(ranked_objects, spec_map)
    if emit is not None:
        emit(f"stage=expand_cross_spec_candidates detail=selected_citations count={len(ranked_citations)}")

    expanded: List[Dict[str, Any]] = []
    for citation, object_name, object_score in ranked_citations:
        for record in by_citation.get(citation) or []:
            metadata = record.get("metadata") or {}
            if not _domain_matches(metadata, domain):
                continue
            score = 900.0 + min(object_score, 4000.0) * 0.01 + _metadata_query_bonus(metadata, query, domain)
            expanded.append(_record_to_candidate(record, score=score, retrieval=f"cross_spec:{object_name}"))

    deduped = _dedupe_candidates(expanded, limit=4)
    # logger.info(
    #     "knowledge_tool cross-spec expand complete: selected_seeds=%s selected_objects=%s selected_citations=%s expanded=%s deduped=%s duration_ms=%.2f",
    #     len(selected_seeds),
    #     len(ranked_objects),
    #     len(ranked_citations),
    #     len(expanded),
    #     len(deduped),
    #     (time.perf_counter() - started_at) * 1000.0,
    # )
    return deduped


def _vector_search_candidates(query: str, domain: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    query_terms = set(_search_terms(query))
    candidates: List[Dict[str, Any]] = []
    errors: List[str] = []
    collections = _select_collections(query, domain)
    for collection_name in collections:
        try:
            store = get_pgvector_store(collection_name=collection_name)
            docs = store.similarity_search_with_score(query, k=20)
        except Exception as exc:
            errors.append(f"{collection_name}: {exc}")
            continue
        for rank, (doc, raw_score) in enumerate(docs):
            metadata = doc.metadata if isinstance(getattr(doc, "metadata", None), dict) else {}
            if not _domain_matches(metadata, domain):
                continue
            if _is_generic_clause_candidate(metadata):
                continue
            score = 300.0 - rank * 10.0 + _metadata_query_bonus(metadata, query, domain)
            if raw_score is not None:
                try:
                    score -= float(raw_score) * 15.0
                except (TypeError, ValueError):
                    pass
            title_terms, object_terms, normalized_terms = _metadata_term_sets(metadata)
            score += 10.0 * len(query_terms.intersection(title_terms))
            score += 8.0 * len(query_terms.intersection(object_terms))
            score += 4.0 * len(query_terms.intersection(normalized_terms))
            candidates.append(_doc_to_candidate(doc, score=score, retrieval=f"vector:{collection_name}"))
    return _aggregate_parent_candidates(candidates, limit=20), errors


def _resolve_explicit_key_candidates(key: str, domain: str) -> List[Dict[str, Any]]:
    normalized_key = _normalized(key)
    if not normalized_key:
        return []

    matches: List[Dict[str, Any]] = []
    for record in _record_catalog().values():
        metadata = record.get("metadata") or {}
        if not _domain_matches(metadata, domain):
            continue
        candidate_key = _candidate_key(metadata, record.get("id") or "")
        if _normalized(candidate_key) != normalized_key:
            continue
        score = 2600.0 + _metadata_query_bonus(metadata, key, domain)
        matches.append(_record_to_candidate(record, score=score, retrieval="selected_key"))
    return _dedupe_candidates(matches, limit=6)


def _result_summary(candidate: Dict[str, Any]) -> str:
    metadata = candidate["metadata"]
    title = str(
        metadata.get("canonical_title")
        or metadata.get("schema_name")
        or metadata.get("operation_id")
        or candidate.get("id")
        or ""
    ).strip()
    snippet = _shorten(candidate["page_content"], limit=MAX_SUMMARY_CHARS)
    return f"{title} | {snippet}"


def _render_result(candidate: Dict[str, Any]) -> str:
    metadata = candidate["metadata"]
    result_key = _candidate_key(metadata, candidate.get("id") or "")
    locator_parts = []
    if metadata.get("schema_name"):
        locator_parts.append(f"schema={metadata['schema_name']}")
    if metadata.get("operation_id"):
        locator_parts.append(f"operation={metadata['operation_id']}")
    if metadata.get("clause_path"):
        locator_parts.append(f"clause={metadata['clause_path']}")
    if metadata.get("table_id"):
        locator_parts.append(f"table={metadata['table_id']}")
    locator = ", ".join(locator_parts) or "locator=unknown"

    lines = [
        f"ResultKey: {result_key}",
        f"Title: {metadata.get('canonical_title', '')}",
        f"Retrieval: {candidate['retrieval']}",
        f"Spec: {metadata.get('spec_id', '')} Release {metadata.get('release', '')} Version {metadata.get('version', '')}",
        f"Type: {metadata.get('doc_type', '')} Domain: {metadata.get('policy_domain', '')}",
        f"Locator: {locator}",
        f"Objects: {', '.join(metadata.get('object_tags') or []) or 'none'}",
        f"Citation: {metadata.get('citation_anchor', '')}",
    ]
    if metadata.get("source_url"):
        lines.append(f"Source: {metadata['source_url']}")
    lines.append(f"Summary: {_result_summary(candidate)}")
    lines.append(f"Content: {_shorten(candidate['page_content'])}")
    return "\n".join(lines)


def _assemble_response(query: str, domain: str, *, limit: int = DEFAULT_RETURNED_RESULTS) -> Tuple[List[Dict[str, Any]], List[str]]:
    direct = _direct_record_matches(query, domain)
    glossary = _glossary_strong_candidates(query, domain)
    domain_special = [
        *_schema_object_graph_candidates(query, domain),
        *_schema_definition_candidates(query, domain),
        *_sm_qos_requirement_candidates(query, domain),
        *_am_rfsp_clause_candidates(query, domain),
        *_operation_intent_candidates(query, domain),
    ]
    exact = _exact_search_candidates(query, domain)
    rerank_pool_limit = max(limit * 4, MAX_RERANK_POOL)
    seeds = _dedupe_candidates([*direct, *glossary, *domain_special, *exact], limit=rerank_pool_limit)
    if direct and _looks_like_exact_identifier_query(query):
        return _bge_rerank_candidates(query, domain, seeds, limit=limit), []
    cross_spec = _expand_cross_spec_candidates(seeds, domain, query)
    vector, errors = _vector_search_candidates(query, domain)
    combined = _dedupe_candidates([*seeds, *cross_spec, *vector], limit=rerank_pool_limit)
    return _bge_rerank_candidates(query, domain, combined, limit=limit), errors


def _format_results(
    query: str,
    candidates: List[Dict[str, Any]],
    errors: List[str],
    *,
    include_selection_guide: bool = False,
) -> str:
    if not candidates:
        if errors:
            return (
                f"No relevant knowledge found for '{query}'. "
                f"Dense retrieval failed: {' | '.join(errors)}"
            )
        return f"No relevant knowledge found for '{query}'."

    blocks = [_render_result(candidate) for candidate in candidates]
    if include_selection_guide:
        selection_lines = [
            f"CandidateKeys for query '{query}':",
            *[
                f"{index}. {_candidate_key(candidate['metadata'], candidate.get('id') or '')} | {_result_summary(candidate)}"
                for index, candidate in enumerate(candidates, start=1)
            ],
            "Use get_knowledge_by_key with an exact ResultKey from the list above for the second-stage lookup.",
        ]
        blocks.insert(0, "\n".join(selection_lines))
    if errors:
        blocks.append("Warnings: Dense retrieval failed for collections: " + " | ".join(errors))
    return "\n\n---\n\n".join(blocks)


def debug_search_semantic_knowledge_pipeline(
    query: str,
    category: Optional[str] = None,
    limit: int = DEFAULT_RETURNED_RESULTS,
    *,
    verbose: bool = True,
    stage_timeout_seconds: int = DEBUG_STAGE_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """
    调试 search_semantic_knowledge 的内部链路。

    返回每个阶段的耗时、候选数、错误信息和最终结果预览，用于定位是
    exact/glossary、PGVector embedding 检索，还是 BGE rerank 出现阻塞。
    """
    normalized_query = str(query or "").strip()
    if not normalized_query:
        raise ValueError("query must not be empty")

    sanitized_limit = _sanitize_limit(limit)
    timings_ms: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    debug_errors: List[str] = []
    total_start = time.perf_counter()

    def _emit(message: str) -> None:
        if verbose:
            print(f"[knowledge-debug] {message}", flush=True)

    class _StageTimeoutError(TimeoutError):
        pass

    def _run_with_timeout(stage_name: str, func):
        if stage_timeout_seconds <= 0:
            return func()
        previous_handler = signal.getsignal(signal.SIGALRM)

        def _handle_timeout(_signum, _frame):
            raise _StageTimeoutError(f"{stage_name} timed out after {stage_timeout_seconds}s")

        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(stage_timeout_seconds)
        try:
            return func()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)

    stage_start = time.perf_counter()
    _emit("stage=ensure_processed_corpus start")
    _ensure_processed_corpus()
    timings_ms["ensure_processed_corpus"] = round((time.perf_counter() - stage_start) * 1000.0, 2)
    _emit(f"stage=ensure_processed_corpus done duration_ms={timings_ms['ensure_processed_corpus']}")

    stage_start = time.perf_counter()
    _emit("stage=infer_policy_domain start")
    domain = _infer_policy_domain(normalized_query, category)
    timings_ms["infer_policy_domain"] = round((time.perf_counter() - stage_start) * 1000.0, 2)
    _emit(f"stage=infer_policy_domain done duration_ms={timings_ms['infer_policy_domain']} domain={domain}")

    stage_start = time.perf_counter()
    _emit("stage=direct_record_matches start")
    direct = _direct_record_matches(normalized_query, domain)
    counts["direct"] = len(direct)
    timings_ms["direct_record_matches"] = round((time.perf_counter() - stage_start) * 1000.0, 2)
    _emit(f"stage=direct_record_matches done duration_ms={timings_ms['direct_record_matches']} count={counts['direct']}")

    stage_start = time.perf_counter()
    _emit("stage=glossary_strong_candidates start")
    glossary = _glossary_strong_candidates(normalized_query, domain)
    counts["glossary"] = len(glossary)
    timings_ms["glossary_strong_candidates"] = round((time.perf_counter() - stage_start) * 1000.0, 2)
    _emit(f"stage=glossary_strong_candidates done duration_ms={timings_ms['glossary_strong_candidates']} count={counts['glossary']}")

    stage_start = time.perf_counter()
    _emit("stage=domain_special_candidates start")
    domain_special = [
        *_schema_object_graph_candidates(normalized_query, domain),
        *_schema_definition_candidates(normalized_query, domain),
        *_sm_qos_requirement_candidates(normalized_query, domain),
        *_am_rfsp_clause_candidates(normalized_query, domain),
        *_operation_intent_candidates(normalized_query, domain),
    ]
    counts["domain_special"] = len(domain_special)
    timings_ms["domain_special_candidates"] = round((time.perf_counter() - stage_start) * 1000.0, 2)
    _emit(f"stage=domain_special_candidates done duration_ms={timings_ms['domain_special_candidates']} count={counts['domain_special']}")

    stage_start = time.perf_counter()
    _emit("stage=exact_search_candidates start")
    exact = _exact_search_candidates(normalized_query, domain)
    counts["exact"] = len(exact)
    timings_ms["exact_search_candidates"] = round((time.perf_counter() - stage_start) * 1000.0, 2)
    _emit(f"stage=exact_search_candidates done duration_ms={timings_ms['exact_search_candidates']} count={counts['exact']}")

    rerank_pool_limit = max(sanitized_limit * 4, 20)
    stage_start = time.perf_counter()
    _emit("stage=seed_dedupe start")
    seeds = _dedupe_candidates([*direct, *glossary, *domain_special, *exact], limit=rerank_pool_limit)
    counts["seeds"] = len(seeds)
    timings_ms["seed_dedupe"] = round((time.perf_counter() - stage_start) * 1000.0, 2)
    _emit(f"stage=seed_dedupe done duration_ms={timings_ms['seed_dedupe']} count={counts['seeds']}")

    stage_start = time.perf_counter()
    _emit("stage=expand_cross_spec_candidates start")
    cross_spec = _run_with_timeout(
        "expand_cross_spec_candidates",
        lambda: _expand_cross_spec_candidates(seeds, domain, normalized_query, emit=_emit),
    )
    counts["cross_spec"] = len(cross_spec)
    timings_ms["expand_cross_spec_candidates"] = round((time.perf_counter() - stage_start) * 1000.0, 2)
    _emit(f"stage=expand_cross_spec_candidates done duration_ms={timings_ms['expand_cross_spec_candidates']} count={counts['cross_spec']}")

    vector: List[Dict[str, Any]] = []
    vector_errors: List[str] = []
    stage_start = time.perf_counter()
    _emit("stage=vector_search_candidates start")
    try:
        vector, vector_errors = _run_with_timeout(
            "vector_search_candidates",
            lambda: _vector_search_candidates(normalized_query, domain),
        )
    except Exception as exc:
        vector_errors = [f"vector_search_exception: {exc}"]
    counts["vector"] = len(vector)
    debug_errors.extend(vector_errors)
    timings_ms["vector_search_candidates"] = round((time.perf_counter() - stage_start) * 1000.0, 2)
    _emit(
        f"stage=vector_search_candidates done duration_ms={timings_ms['vector_search_candidates']} count={counts['vector']} errors={len(vector_errors)}"
    )

    stage_start = time.perf_counter()
    _emit("stage=combined_dedupe start")
    combined = _dedupe_candidates([*seeds, *cross_spec, *vector], limit=rerank_pool_limit)
    counts["combined"] = len(combined)
    timings_ms["combined_dedupe"] = round((time.perf_counter() - stage_start) * 1000.0, 2)
    _emit(f"stage=combined_dedupe done duration_ms={timings_ms['combined_dedupe']} count={counts['combined']}")

    reranked: List[Dict[str, Any]] = []
    rerank_error = ""
    stage_start = time.perf_counter()
    _emit("stage=bge_rerank_candidates start")
    try:
        reranked = _run_with_timeout(
            "bge_rerank_candidates",
            lambda: _bge_rerank_candidates(normalized_query, domain, combined, limit=sanitized_limit),
        )
    except Exception as exc:
        rerank_error = str(exc)
        debug_errors.append(f"rerank_exception: {exc}")
    counts["reranked"] = len(reranked)
    timings_ms["bge_rerank_candidates"] = round((time.perf_counter() - stage_start) * 1000.0, 2)
    _emit(
        f"stage=bge_rerank_candidates done duration_ms={timings_ms['bge_rerank_candidates']} count={counts['reranked']} error={'yes' if rerank_error else 'no'}"
    )

    stage_start = time.perf_counter()
    _emit("stage=format_results start")
    formatted_result = _format_results(
        normalized_query,
        reranked,
        debug_errors,
        include_selection_guide=True,
    )
    timings_ms["format_results"] = round((time.perf_counter() - stage_start) * 1000.0, 2)
    timings_ms["total"] = round((time.perf_counter() - total_start) * 1000.0, 2)
    _emit(f"stage=format_results done duration_ms={timings_ms['format_results']}")
    _emit(f"stage=total done duration_ms={timings_ms['total']}")

    def _candidate_keys(items: List[Dict[str, Any]], *, top_k: int = 5) -> List[str]:
        keys: List[str] = []
        for candidate in items[:top_k]:
            keys.append(_candidate_key(candidate["metadata"], candidate.get("id") or ""))
        return keys

    return {
        "query": normalized_query,
        "category": category,
        "domain": domain,
        "limit": sanitized_limit,
        "counts": counts,
        "timings_ms": timings_ms,
        "errors": debug_errors,
        "rerank_error": rerank_error,
        "seed_keys": _candidate_keys(seeds),
        "combined_keys": _candidate_keys(combined),
        "reranked_keys": _candidate_keys(reranked),
        "result_preview": formatted_result[:2000],
    }


@tool_with_reason
def search_semantic_knowledge(
    query: str,
    category: Optional[str] = None,
    limit: int = DEFAULT_RETURNED_RESULTS,
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """
    Search the Release-18 PCF policy standards knowledge base with exact and dense retrieval.
    The optional `category` is treated as a domain hint (`sm_policy` or `ursp`), not the old business category field.
    """
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return "No relevant knowledge found for an empty query."

    try:
        _ensure_processed_corpus()
        domain = _infer_policy_domain(normalized_query, category)
        candidates, errors = _assemble_response(
            normalized_query,
            domain,
            limit=_sanitize_limit(limit),
        )
        return _format_results(
            normalized_query,
            candidates,
            errors,
            include_selection_guide=True,
        )
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"{_log_prefix(runtime)} Knowledge search failed for '{normalized_query}': {exc}") from exc


@tool_with_reason
def get_knowledge_by_key(
    key: str,
    category: Optional[str] = None,
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """
    Retrieve standards knowledge by exact ResultKey, canonical term, alias, schema name, operationId, or citation anchor.
    """
    normalized_key = str(key or "").strip()
    if not normalized_key:
        return "Knowledge item not found for key: "

    try:
        _ensure_processed_corpus()
        domain = _infer_policy_domain(normalized_key, category)
        selected = _resolve_explicit_key_candidates(normalized_key, domain)
        if selected:
            cross_spec = _expand_cross_spec_candidates(selected, domain, normalized_key)
            return _format_results(normalized_key, _dedupe_candidates([*selected, *cross_spec], limit=6), [])

        direct = _direct_record_matches(normalized_key, domain)
        if direct:
            cross_spec = _expand_cross_spec_candidates(direct, domain, normalized_key)
            return _format_results(normalized_key, _dedupe_candidates([*direct, *cross_spec], limit=6), [])

        exact = _exact_search_candidates(normalized_key, domain)
        if not exact:
            return f"Knowledge item not found for key: {normalized_key}"
        cross_spec = _expand_cross_spec_candidates(exact, domain, normalized_key)
        return _format_results(normalized_key, _dedupe_candidates([*exact, *cross_spec], limit=6), [])
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"{_log_prefix(runtime)} Key lookup failed for '{normalized_key}': {exc}") from exc


__all__ = [
    "debug_search_semantic_knowledge_pipeline",
    "get_knowledge_by_key",
    "search_semantic_knowledge",
]



