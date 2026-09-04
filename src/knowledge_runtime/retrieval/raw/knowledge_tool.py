from __future__ import annotations

import json
import math
import os
import re
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from langchain_core.documents import Document

from database.langchain_pg import (
    get_pgvector_store,
)
from knowledge_build.rag_release import (
    COLLECTIONS,
    CORPUS_ID,
    PROCESSED_ROOT,
    RELEASE,
    SOURCE_CONFIGS,
)
from knowledge_runtime.contracts import (
    KnowledgeCitation,
    KnowledgeDomain,
    KnowledgeHit,
    KnowledgeQuery,
    KnowledgeResponse,
)
from knowledge_runtime.retrieval.colbert_encoder import get_colbert_encoder
from knowledge_runtime.retrieval.colbert_retriever import get_colbert_retriever

PCF_AM_POLICY_CLAUSES_COLLECTION = COLLECTIONS["am_clause"]
PCF_AM_POLICY_SCHEMA_COLLECTION = COLLECTIONS["am_schema"]
PCF_SM_POLICY_CLAUSES_COLLECTION = COLLECTIONS["sm_clause"]
PCF_SM_POLICY_SCHEMA_COLLECTION = COLLECTIONS["sm_schema"]
PCF_URSP_CLAUSES_COLLECTION = COLLECTIONS["ursp_clause"]
PCF_URSP_SCHEMA_COLLECTION = COLLECTIONS["ursp_schema"]

MANIFEST_PATH = PROCESSED_ROOT / "source_manifest.json"
CLAUSE_JSONL = PROCESSED_ROOT / "clauses.jsonl"
SCHEMA_JSONL = PROCESSED_ROOT / "schema.jsonl"
GLOSSARY_JSONL = PROCESSED_ROOT / "glossary.jsonl"
SPEC_OBJECT_MAP_JSON = PROCESSED_ROOT / "spec_object_map.json"
TERM_ALIAS_MAP_JSON = PROCESSED_ROOT / "term_alias_map.json"
CLAUSE_EXACT_INDEX_JSON = PROCESSED_ROOT / "clause_exact_index.json"
SCHEMA_EXACT_INDEX_JSON = PROCESSED_ROOT / "schema_exact_index.json"
GLOSSARY_EXACT_INDEX_JSON = PROCESSED_ROOT / "glossary_exact_index.json"


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9_.-]+", text)]


def normalize_query_terms(text: str) -> list[str]:
    lowered = text.lower()
    expanded = _tokenize(text)
    for canonical, config in _term_alias_map().items():
        aliases = [str(item) for item in config.get("aliases", [])]
        if canonical.lower() in lowered or any(
            alias.lower() in lowered for alias in aliases
        ):
            expanded.extend(_tokenize(canonical))
            for alias in aliases:
                expanded.extend(_tokenize(alias))
    return sorted(set(expanded))


def search_exact_index(
    index: dict[str, Any], query: str, *, top_k: int = 5
) -> list[dict[str, Any]]:
    query_terms = normalize_query_terms(query)
    scored: list[dict[str, Any]] = []
    doc_count = max(1, int(index["doc_count"]))
    document_frequency = index["document_frequency"]
    for document in index["documents"]:
        score = 0.0
        token_counts = document["token_counts"]
        title_tokens = set(document["title_tokens"])
        object_tokens = set(document["object_tokens"])
        for term in query_terms:
            tf = token_counts.get(term, 0)
            if tf == 0:
                continue
            df = max(1, int(document_frequency.get(term, 1)))
            idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            field_boost = 1.0
            if term in title_tokens:
                field_boost += 1.5
            if term in object_tokens:
                field_boost += 1.0
            score += tf * idf * field_boost
        if score > 0:
            scored.append(
                {
                    "id": document["id"],
                    "score": round(score, 4),
                    "title": document["title"],
                    "spec_id": document["spec_id"],
                }
            )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


DEFAULT_RETURNED_RESULTS = 10
DenseRetrieval = Literal["colbert", "pgvector_bge"]
MAX_CONTENT_CHARS = 1200
MAX_SUMMARY_CHARS = 240
BGE_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
BGE_RERANK_INPUT_CHARS = 3200
BGE_RERANK_TERM_CHARS = 700
RRF_K = 60.0
RRF_SCORE_SCALE = 1000.0
LEXICAL_CANDIDATE_LIMIT = 24
MAX_RERANK_POOL = 24
CROSS_SPEC_SEED_LIMIT = 4
CROSS_SPEC_OBJECT_LIMIT = 6
CROSS_SPEC_CITATION_LIMIT = 24
VECTOR_SEARCH_K = 8
VECTOR_SEARCH_TRIGGER_THRESHOLD = 8
HF_CACHE_ROOT = Path(
    str(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")).strip()
).expanduser()
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

    repo_cache_dir = (
        HF_HUB_MODELS_ROOT / "models--BAAI--bge-reranker-v2-m3" / "snapshots"
    )
    if repo_cache_dir.exists():
        snapshots = sorted(
            (path for path in repo_cache_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return str(snapshots[0])

    return BGE_RERANK_MODEL


def _bge_reranker_model_kwargs() -> dict[str, Any]:
    """Return optional CrossEncoder settings configured for this process.

    The default remains the library-selected device.  Setting
    ``BGE_RERANK_DEVICE=cpu`` keeps the cross-encoder off the GPU while the
    LLM server can continue using it.
    """
    device = str(os.getenv("BGE_RERANK_DEVICE", "")).strip()
    return {"device": device} if device else {}


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
    MANIFEST_PATH,
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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
    if (
        len(normalized) > 4
        and normalized.endswith("s")
        and not normalized.endswith("ss")
    ):
        return normalized[:-1]
    return normalized


def _expand_compound_tokens(text: str) -> list[str]:
    expanded: list[str] = []
    for raw_token in re.findall(r"[A-Za-z0-9_.-]+", str(text or "")):
        parts = COMPOUND_TERM_RE.findall(
            raw_token.replace("_", " ").replace("-", " ").replace(".", " ")
        )
        expanded.extend(part.lower() for part in parts if part)
    return expanded


def _search_terms(text: str) -> list[str]:
    normalized = normalize_query_terms(text) + _expand_compound_tokens(text)
    filtered = [
        _normalize_search_term(term)
        for term in normalized
        if term not in SEARCH_STOPWORDS
        and (len(term) > 2 or term in {"am", "sm", "ue", "os"})
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
        strategy_domains = [
            _normalized(item)
            for item in metadata_domain.get("strategy_domains") or []
            if _normalized(item)
        ]
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


def _domain_match_label(metadata: dict[str, Any], requested_domain: str) -> str:
    normalized_request = _normalized(requested_domain)
    if normalized_request in {"", "all"}:
        return "all"
    if _domain_matches(metadata, requested_domain):
        strategy_domains = [
            _normalized(item)
            for item in (metadata.get("strategy_domains") or [])
            if _normalized(item)
        ]
        if strategy_domains and normalized_request in strategy_domains:
            return "exact"
        normalized_domain = _normalized(metadata.get("policy_domain") or "")
        if normalized_domain == "shared":
            return "shared"
        if normalized_request == "mobility" and normalized_domain in {
            "am_policy",
            "ursp",
        }:
            return "exact"
        return "exact" if normalized_domain == normalized_request else "shared"
    return "soft_mismatch"


def _candidate_domain_allowed(
    metadata: dict[str, Any], requested_domain: str, query: str
) -> bool:
    del query
    return _domain_matches(metadata, requested_domain)


def _spec_priority(spec_id: str, domain: str) -> int:
    normalized_domain = _normalized(domain)
    if normalized_domain == "am_policy":
        order = [
            "29.507",
            "23.503",
            "29.571",
            "24.501",
            "23.501",
            "29.519",
            "29.525",
            "24.526",
            "29.512",
        ]
    elif normalized_domain == "sm_policy":
        order = [
            "29.512",
            "23.503",
            "29.571",
            "29.514",
            "29.213",
            "29.214",
            "24.501",
            "23.501",
            "29.519",
            "29.525",
            "24.526",
        ]
    elif normalized_domain == "ursp":
        order = [
            "24.526",
            "29.525",
            "23.503",
            "29.519",
            "24.501",
            "23.501",
            "29.571",
            "29.512",
            "29.514",
        ]
    elif normalized_domain == "mobility":
        order = [
            "29.507",
            "24.526",
            "29.525",
            "23.503",
            "29.571",
            "24.501",
            "23.501",
            "29.519",
        ]
    else:
        order = [
            "23.503",
            "29.507",
            "29.512",
            "24.526",
            "29.525",
            "29.571",
            "29.514",
            "29.519",
            "24.501",
            "23.501",
        ]
    try:
        return max(0, 20 - order.index(spec_id))
    except ValueError:
        return 0


def _has_explicit_schema_intent(query: str) -> bool:
    lowered = _normalized(query)
    return any(keyword in lowered for keyword in SCHEMA_QUERY_KEYWORDS)


@lru_cache(maxsize=8)
def _domain_schema_phrase_index(domain: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
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


def _schema_phrase_matches(query: str, domain: str) -> tuple[set[str], set[str]]:
    query_terms = set(_search_terms(query))
    schema_names, object_phrases = _domain_schema_phrase_index(domain)
    matched_schema_names = {
        phrase
        for phrase in schema_names
        if len(_normalized(phrase)) >= 3
        and set(_search_terms(phrase)).issubset(query_terms)
    }
    matched_object_phrases = {
        phrase
        for phrase in object_phrases
        if len(_normalized(phrase)) >= 3
        and set(_search_terms(phrase)).issubset(query_terms)
    }
    return matched_schema_names, matched_object_phrases


@lru_cache(maxsize=128)
def _expanded_query_terms_cached(query: str, domain: str) -> frozenset[str]:
    return frozenset(_expanded_query_terms(query, domain))


@lru_cache(maxsize=128)
def _schema_phrase_matches_cached(
    query: str, domain: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    matched_schema_names, matched_object_phrases = _schema_phrase_matches(query, domain)
    return tuple(sorted(matched_schema_names)), tuple(sorted(matched_object_phrases))


def _looks_like_schema_object_query(query: str, domain: str) -> bool:
    if _has_explicit_schema_intent(query):
        return True
    if (
        _prefers_clause_location(query, domain)
        or _prefers_operation(query)
        or _prefers_glossary(query)
    ):
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
    if any(
        phrase in lowered
        for phrase in (
            "which endpoint",
            "what endpoint",
            "which operation",
            "what operation",
            "which method",
            "what method",
        )
    ):
        return True
    if bool(re.search(r"\b(endpoint|operation|method)\b", lowered)):
        return True
    has_http_or_action_verb = bool(
        re.search(
            r"\b(create|creates|created|delete|deletes|deleted|update|updates|updated|read|reads|post|get|put|patch)\b",
            lowered,
        )
    )
    if has_http_or_action_verb:
        return True
    return bool(
        re.search(r"\bapi\b", lowered)
        and re.search(
            r"\b(create|creates|created|delete|deletes|deleted|update|updates|updated|read|reads)\b",
            lowered,
        )
    )


def _prefers_glossary(query: str) -> bool:
    lowered = _normalized(query)
    return any(keyword in lowered for keyword in GLOSSARY_QUERY_KEYWORDS)


def _query_intent(query: str, domain: str) -> str:
    if _looks_like_exact_identifier_query(query):
        return "exact_identifier"
    if _prefers_operation(query):
        return "operation_lookup"
    if _prefers_clause_location(query, domain):
        return "clause_location"
    if _is_short_term_query(query):
        return "glossary_term"
    if _prefers_glossary(query):
        return "glossary_term"
    if _prefers_schema(query, domain):
        return "schema_relation"
    return "generic_fact"


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


def _infer_policy_domain(query: str, category: str | None) -> str:
    normalized_category = _normalized(category)
    if normalized_category in {
        "sm_policy",
        "am_policy",
        "ursp",
        "mobility",
        "cross_domain",
        "shared",
        "all",
    }:
        return (
            "all"
            if normalized_category in {"cross_domain", "shared"}
            else normalized_category
        )

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


def _domain_collection_pair(domain: str) -> tuple[str | None, str | None]:
    if domain == "am_policy":
        return PCF_AM_POLICY_SCHEMA_COLLECTION, PCF_AM_POLICY_CLAUSES_COLLECTION
    if domain == "sm_policy":
        return PCF_SM_POLICY_SCHEMA_COLLECTION, PCF_SM_POLICY_CLAUSES_COLLECTION
    if domain == "ursp":
        return PCF_URSP_SCHEMA_COLLECTION, PCF_URSP_CLAUSES_COLLECTION
    return None, None


def _select_collections(query: str, domain: str) -> list[str]:
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
def _ensure_processed_corpus() -> tuple[Path, ...]:
    missing = [path for path in REQUIRED_KB_PATHS if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            f"Release-{RELEASE} PCF policy corpus {CORPUS_ID} is not built. Missing files: "
            + ", ".join(str(path) for path in missing)
            + ". Run `python knowledge_build/scripts/build_pcf_policy_kb.py build` first."
        )

    manifest = _load_json(MANIFEST_PATH)
    if not isinstance(manifest, list):
        raise RuntimeError(
            f"Release-{RELEASE} PCF policy corpus {CORPUS_ID} has an invalid source manifest: "
            f"{MANIFEST_PATH} must contain a JSON list."
        )

    expected_source_ids = [source.source_id for source in SOURCE_CONFIGS]
    if len(expected_source_ids) != len(set(expected_source_ids)):
        raise RuntimeError(
            f"Release-{RELEASE} PCF policy corpus {CORPUS_ID} has duplicate canonical source IDs."
        )
    expected_sources = {source.source_id: source for source in SOURCE_CONFIGS}

    manifest_source_ids: list[str] = []
    invalid_sources: list[str] = []
    for source in manifest:
        if not isinstance(source, dict):
            invalid_sources.append("non-object entry")
            continue
        source_id = str(source.get("source_id") or "").strip()
        if not source_id:
            invalid_sources.append("missing source_id")
            continue
        manifest_source_ids.append(source_id)
        source_release = str(source.get("release") or "").strip()
        if source_release != str(RELEASE):
            invalid_sources.append(
                f"{source_id}: release={source_release or '<missing>'} expected={RELEASE}"
            )
        expected_source = expected_sources.get(source_id)
        if expected_source is None:
            continue
        for field_name in ("spec_version", "version", "source_url", "archive_url"):
            actual_value = str(source.get(field_name) or "").strip()
            expected_value = str(getattr(expected_source, field_name)).strip()
            if actual_value != expected_value:
                invalid_sources.append(
                    f"{source_id}: {field_name}={actual_value or '<missing>'} "
                    f"expected={expected_value}"
                )

    expected_source_id_set = set(expected_source_ids)
    manifest_source_id_set = set(manifest_source_ids)
    duplicate_source_ids = sorted(
        source_id
        for source_id in manifest_source_id_set
        if manifest_source_ids.count(source_id) > 1
    )
    if (
        invalid_sources
        or duplicate_source_ids
        or manifest_source_id_set != expected_source_id_set
        or len(manifest_source_ids) != len(expected_source_ids)
    ):
        details: list[str] = []
        if invalid_sources:
            details.append("invalid=" + ", ".join(invalid_sources))
        if duplicate_source_ids:
            details.append("duplicate=" + ", ".join(duplicate_source_ids))
        missing_source_ids = sorted(expected_source_id_set - manifest_source_id_set)
        if missing_source_ids:
            details.append("missing=" + ", ".join(missing_source_ids))
        unexpected_source_ids = sorted(manifest_source_id_set - expected_source_id_set)
        if unexpected_source_ids:
            details.append("unexpected=" + ", ".join(unexpected_source_ids))
        raise RuntimeError(
            f"Release-{RELEASE} PCF policy corpus {CORPUS_ID} source manifest does not match "
            f"the canonical Release-{RELEASE} source set ({'; '.join(details)})."
        )

    return tuple(Path(path) for path in REQUIRED_KB_PATHS)


@lru_cache(maxsize=1)
def _record_catalog() -> dict[str, dict[str, Any]]:
    _ensure_processed_corpus()
    records = (
        _load_jsonl(CLAUSE_JSONL)
        + _load_jsonl(SCHEMA_JSONL)
        + _load_jsonl(GLOSSARY_JSONL)
    )
    return {record["id"]: record for record in records}


@lru_cache(maxsize=1)
def _records_by_citation() -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in _record_catalog().values():
        citation = str(
            (record.get("metadata") or {}).get("citation_anchor") or ""
        ).strip()
        if citation:
            grouped.setdefault(citation, []).append(record)
    return grouped


@lru_cache(maxsize=1)
def _clause_exact_index() -> dict[str, Any]:
    _ensure_processed_corpus()
    return _load_json(CLAUSE_EXACT_INDEX_JSON)


@lru_cache(maxsize=1)
def _schema_exact_index() -> dict[str, Any]:
    _ensure_processed_corpus()
    return _load_json(SCHEMA_EXACT_INDEX_JSON)


@lru_cache(maxsize=1)
def _glossary_exact_index() -> dict[str, Any]:
    _ensure_processed_corpus()
    return _load_json(GLOSSARY_EXACT_INDEX_JSON)


@lru_cache(maxsize=1)
def _spec_object_map() -> dict[str, list[dict[str, Any]]]:
    _ensure_processed_corpus()
    return _load_json(SPEC_OBJECT_MAP_JSON)


@lru_cache(maxsize=1)
def _term_alias_map() -> dict[str, dict[str, Any]]:
    _ensure_processed_corpus()
    return _load_json(TERM_ALIAS_MAP_JSON)


def _candidate_key(metadata: dict[str, Any], fallback_id: str = "") -> str:
    for key in ("citation_anchor", "schema_name", "operation_id", "canonical_title"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return fallback_id


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


def _glossary_phrase_bonus(metadata: dict[str, Any], query: str) -> float:
    query_terms = set(_search_terms(query))
    query_text = _normalized(query)
    phrases: list[tuple[str, float]] = []
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


def _glossary_strong_candidates(query: str, domain: str) -> list[dict[str, Any]]:
    if (
        _prefers_schema(query, domain)
        or _prefers_clause_location(query, domain)
        or _prefers_operation(query)
    ):
        return []
    candidates: list[dict[str, Any]] = []
    for record in _record_catalog().values():
        metadata = record.get("metadata") or {}
        if _normalized(metadata.get("doc_type") or "") != "glossary":
            continue
        if not _domain_matches(metadata, domain):
            continue
        phrase_bonus = _glossary_phrase_bonus(metadata, query)
        if phrase_bonus <= 0:
            continue
        score = 420.0 + phrase_bonus + _metadata_query_bonus(metadata, query, domain)
        candidates.append(
            _record_to_candidate(
                record,
                score=score,
                retrieval="glossary_phrase",
                features=_candidate_features(
                    metadata,
                    query=query,
                    domain=domain,
                    match_type="glossary_phrase",
                    lexical_score=phrase_bonus,
                ),
            )
        )
    return _dedupe_candidates(candidates, limit=6)


def _metadata_term_sets(
    metadata: dict[str, Any],
) -> tuple[set[str], set[str], set[str]]:
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
        if term not in SEARCH_STOPWORDS
        and (len(term) > 2 or term in {"am", "sm", "ue", "os"})
    )
    return title_terms, object_terms, normalized_terms


def _doc_type_matches_query(metadata: dict[str, Any], query: str, domain: str) -> bool:
    return _doc_type_priority(str(metadata.get("doc_type") or ""), query, domain) > 0


def _candidate_features(
    metadata: dict[str, Any],
    *,
    query: str,
    domain: str,
    match_type: str,
    lexical_score: float = 0.0,
    vector_rank: int | None = None,
    cross_spec_source: str | None = None,
    is_exact_identifier: bool = False,
) -> dict[str, Any]:
    query_terms = set(_expanded_query_terms_cached(query, domain))
    _title_terms, object_terms, _normalized_terms = _metadata_term_sets(metadata)
    object_overlap_count = len(query_terms.intersection(object_terms))
    return {
        "match_type": match_type,
        "lexical_score": round(float(lexical_score), 4),
        "vector_rank": vector_rank,
        "query_intent": _query_intent(query, domain),
        "domain_match": _domain_match_label(metadata, domain),
        "doc_type_match": _doc_type_matches_query(metadata, query, domain),
        "object_overlap_count": object_overlap_count,
        "cross_spec_source": cross_spec_source,
        "is_exact_identifier": bool(is_exact_identifier),
    }


def _is_generic_clause_candidate(metadata: dict[str, Any]) -> bool:
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


def _should_skip_generic_clause_candidate(metadata: dict[str, Any], query: str) -> bool:
    if not _is_generic_clause_candidate(metadata):
        return False
    query_terms = set(_search_terms(query))
    title_terms, object_terms, _normalized_terms = _metadata_term_sets(metadata)
    return not bool(query_terms.intersection(title_terms | object_terms))


def _metadata_query_bonus(metadata: dict[str, Any], query: str, domain: str) -> float:
    domain_match = _domain_match_label(metadata, domain)
    score = _spec_priority(str(metadata.get("spec_id") or ""), domain)
    if domain_match == "soft_mismatch":
        score -= 260.0
    elif domain_match == "shared":
        score += 25.0
    score += _doc_type_priority(str(metadata.get("doc_type") or ""), query, domain)
    normalized_doc_type = _normalized(metadata.get("doc_type") or "")
    query_terms = set(_expanded_query_terms_cached(query, domain))
    matched_schema_names, _matched_object_phrases = _schema_phrase_matches_cached(
        query, domain
    )
    matched_schema_names_normalized = {
        _normalized(name) for name in matched_schema_names
    }
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
            score += (
                240.0
                if normalized_schema_name in matched_schema_names_normalized
                else -180.0
            )
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
    elif (
        _prefers_operation(query)
        and _normalized(metadata.get("doc_type") or "") == "openapi"
    ):
        score -= 140.0
    return score


def _sm_qos_requirement_candidates(query: str, domain: str) -> list[dict[str, Any]]:
    if domain != "sm_policy" or not _sm_prefers_qos_table(query):
        return []
    candidates: list[dict[str, Any]] = []
    for record in _record_catalog().values():
        metadata = record.get("metadata") or {}
        if not _domain_matches(metadata, domain):
            continue
        doc_type = _normalized(metadata.get("doc_type") or "")
        object_tags = {str(item) for item in metadata.get("object_tags") or []}
        title = _normalized(metadata.get("canonical_title") or "")
        if not (
            {"QosData", "QosCharacteristics", "5Qi"} & object_tags
            or "qosdata" in title
            or "5qi" in title
        ):
            continue
        if doc_type not in {"table", "openapi", "glossary"}:
            continue
        base = (
            320.0 if doc_type == "table" else 240.0 if doc_type == "openapi" else 180.0
        )
        page_text = _normalized(record.get("page_content") or "")
        if (
            "5.6.2.8" in title
            or "definition of type qosdata" in title
            or "packetdelaybudget" in page_text
        ):
            base += 180.0
        candidates.append(
            _record_to_candidate(
                record,
                score=base + _metadata_query_bonus(metadata, query, domain),
                retrieval="sm_qos_requirements",
                features=_candidate_features(
                    metadata,
                    query=query,
                    domain=domain,
                    match_type="sm_qos_requirements",
                ),
            )
        )
    return _dedupe_candidates(candidates, limit=6)


def _schema_object_graph_candidates(query: str, domain: str) -> list[dict[str, Any]]:
    if not _prefers_schema(query, domain):
        return []

    query_terms = _expanded_query_terms(query, domain)
    matched_schema_names, matched_object_phrases = _schema_phrase_matches(query, domain)
    matched_schema_names_normalized = {
        _normalized(name) for name in matched_schema_names
    }
    matched_object_phrases_normalized = {
        _normalized(name) for name in matched_object_phrases
    }
    candidates: list[dict[str, Any]] = []
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
        score = (
            280.0 + 90.0 * schema_overlap + 60.0 * title_overlap + 45.0 * object_overlap
        )
        normalized_schema_name = _normalized(metadata.get("schema_name") or "")
        if matched_schema_names_normalized:
            score += (
                110.0
                if normalized_schema_name in matched_schema_names_normalized
                else -60.0
            )
        object_phrase_overlap = sum(
            1
            for object_tag in metadata.get("object_tags") or []
            if _normalized(object_tag) in matched_object_phrases_normalized
        )
        score += 70.0 * object_phrase_overlap
        score += _metadata_query_bonus(metadata, query, domain)
        candidates.append(
            _record_to_candidate(
                record,
                score=score,
                retrieval="schema_object_graph",
                features=_candidate_features(
                    metadata,
                    query=query,
                    domain=domain,
                    match_type="schema_object_graph",
                ),
            )
        )
    return _dedupe_candidates(candidates, limit=8)


def _schema_definition_candidates(query: str, domain: str) -> list[dict[str, Any]]:
    if not _prefers_schema(query, domain):
        return []

    query_terms = _expanded_query_terms(query, domain)
    query_text = _normalized(query)
    matched_schema_names, matched_object_phrases = _schema_phrase_matches(query, domain)
    matched_schema_names_normalized = {
        _normalized(name) for name in matched_schema_names
    }
    matched_object_phrases_normalized = {
        _normalized(name) for name in matched_object_phrases
    }
    candidates: list[dict[str, Any]] = []
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
        content_overlap = len(
            query_terms.intersection(content_terms | normalized_terms)
        )
        schema_overlap = len(query_terms.intersection(schema_terms))
        if (
            title_overlap == 0
            and object_overlap == 0
            and content_overlap == 0
            and schema_overlap == 0
        ):
            continue
        score = (
            260.0
            + 110.0 * schema_overlap
            + 52.0 * title_overlap
            + 45.0 * object_overlap
            + 18.0 * content_overlap
        )
        normalized_schema_name = _normalized(schema_name)
        if normalized_schema_name in query_text:
            score += 140.0
        if matched_schema_names_normalized:
            score += (
                120.0
                if normalized_schema_name in matched_schema_names_normalized
                else -60.0
            )
        object_phrase_overlap = sum(
            1
            for object_tag in metadata.get("object_tags") or []
            if _normalized(object_tag) in matched_object_phrases_normalized
        )
        score += 65.0 * object_phrase_overlap
        score += _metadata_query_bonus(metadata, query, domain)
        candidates.append(
            _record_to_candidate(
                record,
                score=score,
                retrieval="schema_definition",
                features=_candidate_features(
                    metadata,
                    query=query,
                    domain=domain,
                    match_type="schema_definition",
                ),
            )
        )
    return _dedupe_candidates(candidates, limit=8)


def _am_rfsp_clause_candidates(query: str, domain: str) -> list[dict[str, Any]]:
    if domain != "am_policy" or not _am_prefers_clause(query):
        return []
    candidates: list[dict[str, Any]] = []
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
        if (
            "RfspIndex" not in object_tags
            and "rfsp" not in title
            and "rfsp index" not in page_text
        ):
            continue
        base = 320.0
        if "4.2.2.3.2" in title or "rfsp index" in title or "rfsp index" in page_text:
            base += 140.0
        candidates.append(
            _record_to_candidate(
                record,
                score=base + _metadata_query_bonus(metadata, query, domain),
                retrieval="am_rfsp_clause",
                features=_candidate_features(
                    metadata,
                    query=query,
                    domain=domain,
                    match_type="am_rfsp_clause",
                ),
            )
        )
    return _dedupe_candidates(candidates, limit=6)


def _operation_intent_candidates(query: str, domain: str) -> list[dict[str, Any]]:
    if not _prefers_operation(query):
        return []
    query_text = _normalized(query)
    query_terms = _expanded_query_terms(query, domain)
    requested_verbs = [
        verb
        for verb in ("create", "update", "delete", "get", "read")
        if verb in query_text
    ]
    candidates: list[dict[str, Any]] = []
    for record in _record_catalog().values():
        metadata = record.get("metadata") or {}
        if not _domain_matches(metadata, domain):
            continue
        operation_id = str(metadata.get("operation_id") or "").strip()
        if not operation_id:
            continue
        operation_terms = set(_search_terms(operation_id))
        overlap = len(query_terms.intersection(operation_terms))
        if not overlap and not any(
            verb in _normalized(operation_id) for verb in requested_verbs
        ):
            continue
        score = 280.0 + 70.0 * overlap + _metadata_query_bonus(metadata, query, domain)
        normalized_operation = _normalized(operation_id)
        for verb in requested_verbs:
            score += 90.0 if verb in normalized_operation else -50.0
        candidates.append(
            _record_to_candidate(
                record,
                score=score,
                retrieval="operation_intent",
                features=_candidate_features(
                    metadata,
                    query=query,
                    domain=domain,
                    match_type="operation_intent",
                ),
            )
        )
    return _dedupe_candidates(candidates, limit=6)


def _record_to_candidate(
    record: dict[str, Any],
    *,
    score: float,
    retrieval: str,
    features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    return {
        "id": record.get("id"),
        "page_content": str(record.get("page_content") or ""),
        "metadata": metadata,
        "score": score,
        "retrieval": retrieval,
        "features": dict(features or {}),
    }


def _doc_to_candidate(
    doc: Any,
    *,
    score: float,
    retrieval: str,
    features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = doc.metadata if isinstance(getattr(doc, "metadata", None), dict) else {}
    return {
        "id": None,
        "page_content": str(getattr(doc, "page_content", "") or ""),
        "metadata": metadata,
        "score": score,
        "retrieval": retrieval,
        "features": dict(features or {}),
    }


@lru_cache(maxsize=1)
def _bge_cross_encoder() -> Any:
    try:
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "BGE reranking requires LangChain's HuggingFace cross-encoder support. "
            "Install `sentence-transformers` and ensure `langchain_community.cross_encoders.HuggingFaceCrossEncoder` is available."
        ) from exc

    return HuggingFaceCrossEncoder(
        model_name=_resolve_bge_reranker_model(),
        model_kwargs=_bge_reranker_model_kwargs(),
    )


def _bge_reranker(top_n: int) -> Any:
    """Create a cheap top-N wrapper around the process-wide cached model.

    ``top_n`` differs between startup warmup and real queries.  It must not be
    part of the model cache key, otherwise each distinct result limit loads a
    second full cross-encoder copy.
    """
    try:
        from langchain_classic.retrievers.document_compressors import (
            CrossEncoderReranker,
        )
    except ImportError as exc:
        raise RuntimeError(
            "BGE reranking requires LangChain's HuggingFace cross-encoder support. "
            "Install `sentence-transformers` and ensure `langchain_community.cross_encoders.HuggingFaceCrossEncoder` is available."
        ) from exc

    return CrossEncoderReranker(model=_bge_cross_encoder(), top_n=top_n)


def warmup_knowledge_tool_models() -> dict[str, Any]:
    warmed: dict[str, Any] = {
        "corpus": [],
        "vectorstores": [],
        "rerankers": [],
        "colbert": None,
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

    if _colbert_available():
        encoder = get_colbert_encoder()
        retriever = get_colbert_retriever()
        retriever.preload()
        warmed["colbert"] = {
            "model": encoder.model_name,
            "dim": encoder.dim,
            "records": retriever.record_count,
        }

    return warmed


def _limited_terms(values: Iterable[Any], *, limit: int = BGE_RERANK_TERM_CHARS) -> str:
    seen: list[str] = []
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
    intent = _query_intent(query, domain)
    if not expanded_terms:
        return f"Query: {query}\nQuery intent: {intent}"
    return f"Query: {query}\nQuery intent: {intent}\nExpanded technical terms: {expanded_terms}"


def _candidate_rerank_text(candidate: dict[str, Any]) -> str:
    metadata = candidate["metadata"]
    features = candidate.get("features") or {}
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
        f"Candidate query intent: {features.get('query_intent', '')}",
        f"Candidate match type: {features.get('match_type', '')}",
        f"Candidate domain match: {features.get('domain_match', '')}",
        f"Candidate lexical score: {features.get('lexical_score', 0.0)}",
        f"Candidate vector rank: {features.get('vector_rank', '')}",
        f"Candidate exact identifier match: {features.get('is_exact_identifier', False)}",
        f"Candidate object overlap count: {features.get('object_overlap_count', 0)}",
        f"Content: {candidate.get('page_content', '')}",
    ]
    return _shorten(
        " ".join(part for part in parts if part), limit=BGE_RERANK_INPUT_CHARS
    )


def _bge_rerank_candidates(
    query: str, domain: str, candidates: Iterable[dict[str, Any]], *, limit: int
) -> list[dict[str, Any]]:
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
    reranked_documents = list(
        compressor.compress_documents(documents, query=rerank_query)
    )
    if not reranked_documents:
        raise RuntimeError(
            "BGE reranker returned no documents for a non-empty candidate set."
        )

    ordered: list[dict[str, Any]] = []
    seen: set[int] = set()
    for document in reranked_documents[:limit]:
        index = document.metadata.get("candidate_index")
        if not isinstance(index, int):
            raise RuntimeError(
                "BGE reranker did not preserve candidate_index metadata."
            )
        if index in seen:
            continue
        seen.add(index)
        ordered.append(candidate_list[index])
    return ordered


def _dedupe_candidates(
    candidates: Iterable[dict[str, Any]], *, limit: int = DEFAULT_RETURNED_RESULTS
) -> list[dict[str, Any]]:
    return _aggregate_parent_candidates(candidates, limit=limit)


def _retrieval_family(retrieval: str) -> str:
    normalized = str(retrieval or "").strip()
    return normalized.split(":", 1)[0] if normalized else "unknown"


def _aggregate_parent_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    limit: int = DEFAULT_RETURNED_RESULTS,
) -> list[dict[str, Any]]:
    candidate_list = list(candidates)
    if not candidate_list:
        return []

    family_rankings: dict[str, dict[str, int]] = defaultdict(dict)
    family_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
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
            parent_key = _candidate_key(
                candidate["metadata"], candidate.get("id") or ""
            )
            if not parent_key or parent_key in seen_parent_keys:
                continue
            seen_parent_keys.add(parent_key)
            family_rankings[family][parent_key] = rank_counter
            rank_counter += 1

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidate_list:
        metadata = candidate["metadata"]
        parent_key = _candidate_key(metadata, candidate.get("id") or "")
        if not parent_key:
            continue
        grouped[parent_key].append(candidate)

    aggregated: list[dict[str, Any]] = []
    for parent_key, group in grouped.items():
        ranked_group = sorted(group, key=lambda item: item["score"], reverse=True)
        representative = dict(ranked_group[0])
        representative_metadata = dict(representative["metadata"])
        representative_features = dict(representative.get("features") or {})
        support_count = len(ranked_group)
        retrieval_families = sorted(
            {_retrieval_family(item.get("retrieval") or "") for item in ranked_group}
        )
        rrf_score = 0.0
        for family in retrieval_families:
            rank = family_rankings.get(family, {}).get(parent_key)
            if rank is not None:
                rrf_score += 1.0 / (RRF_K + float(rank))
        representative["score"] = round(
            rrf_score * RRF_SCORE_SCALE + float(ranked_group[0]["score"]) * 0.01, 4
        )
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
        representative_features["support_count"] = support_count
        representative_features["supporting_retrievals"] = retrieval_families
        representative["features"] = representative_features
        aggregated.append(representative)

    return sorted(aggregated, key=lambda item: item["score"], reverse=True)[:limit]


def _direct_record_matches(key: str, domain: str) -> list[dict[str, Any]]:
    normalized_key = _normalized(key)
    catalog = _record_catalog()
    direct_hits: list[dict[str, Any]] = []
    for record in catalog.values():
        metadata = record.get("metadata") or {}
        if not _candidate_domain_allowed(metadata, domain, key):
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
        elif any(
            _normalized(alias) == normalized_key
            for alias in metadata.get("aliases") or []
        ):
            score = 2050.0
        elif any(
            _normalized(tag) == normalized_key
            for tag in metadata.get("object_tags") or []
        ):
            score = 1950.0
        elif any(
            _normalized(field) == normalized_key for field in search_fields if field
        ):
            score = 1800.0
        if score > 0:
            score += _metadata_query_bonus(metadata, key, domain)
            direct_hits.append(
                _record_to_candidate(
                    record,
                    score=score,
                    retrieval="exact_identifier",
                    features=_candidate_features(
                        metadata,
                        query=key,
                        domain=domain,
                        match_type="exact_identifier",
                        lexical_score=score,
                        is_exact_identifier=True,
                    ),
                )
            )
    return _dedupe_candidates(direct_hits, limit=3)


def _exact_search_candidates(query: str, domain: str) -> list[dict[str, Any]]:
    catalog = _record_catalog()
    candidates: list[dict[str, Any]] = []
    prefers_schema = _prefers_schema(query, domain)
    for hit in search_exact_index(_clause_exact_index(), query, top_k=12):
        record = catalog.get(hit["id"])
        if record is None:
            continue
        metadata = record.get("metadata") or {}
        if not _candidate_domain_allowed(metadata, domain, query):
            continue
        if _should_skip_generic_clause_candidate(metadata, query):
            continue
        clause_base = 120.0 if prefers_schema else 180.0
        score = (
            clause_base
            + float(hit["score"])
            + _metadata_query_bonus(metadata, query, domain)
        )
        candidates.append(
            _record_to_candidate(
                record,
                score=score,
                retrieval="exact_clause",
                features=_candidate_features(
                    metadata,
                    query=query,
                    domain=domain,
                    match_type="exact_clause",
                    lexical_score=float(hit["score"]),
                ),
            )
        )
    for hit in search_exact_index(_schema_exact_index(), query, top_k=12):
        record = catalog.get(hit["id"])
        if record is None:
            continue
        metadata = record.get("metadata") or {}
        if not _candidate_domain_allowed(metadata, domain, query):
            continue
        schema_base = 220.0 if prefers_schema else 150.0
        score = (
            schema_base
            + float(hit["score"])
            + _metadata_query_bonus(metadata, query, domain)
        )
        candidates.append(
            _record_to_candidate(
                record,
                score=score,
                retrieval="exact_schema",
                features=_candidate_features(
                    metadata,
                    query=query,
                    domain=domain,
                    match_type="exact_schema",
                    lexical_score=float(hit["score"]),
                ),
            )
        )
    for hit in search_exact_index(_glossary_exact_index(), query, top_k=12):
        record = catalog.get(hit["id"])
        if record is None:
            continue
        metadata = record.get("metadata") or {}
        if not _candidate_domain_allowed(metadata, domain, query):
            continue
        score = (
            140.0 + float(hit["score"]) + _metadata_query_bonus(metadata, query, domain)
        )
        candidates.append(
            _record_to_candidate(
                record,
                score=score,
                retrieval="exact_glossary",
                features=_candidate_features(
                    metadata,
                    query=query,
                    domain=domain,
                    match_type="exact_glossary",
                    lexical_score=float(hit["score"]),
                ),
            )
        )
    return _dedupe_candidates(candidates, limit=6)


def _top_cross_spec_seeds(
    seed_candidates: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked = sorted(
        list(seed_candidates),
        key=lambda item: float(item.get("score") or 0.0),
        reverse=True,
    )
    return ranked[:CROSS_SPEC_SEED_LIMIT]


def _rank_cross_spec_objects(
    seed_candidates: Iterable[dict[str, Any]],
    query: str,
    alias_map: dict[str, dict[str, Any]],
) -> list[tuple[str, float]]:
    query_terms = set(_search_terms(query))
    object_scores: dict[str, float] = defaultdict(float)

    for candidate in _top_cross_spec_seeds(seed_candidates):
        metadata = candidate["metadata"]
        seed_score = float(candidate.get("score") or 0.0)
        canonical_title = str(metadata.get("canonical_title") or "").strip()
        candidate_objects = set(
            str(item).strip()
            for item in (metadata.get("object_tags") or [])
            if str(item).strip()
        )
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
    ranked_objects: Iterable[tuple[str, float]],
    spec_map: dict[str, list[dict[str, Any]]],
) -> list[tuple[str, str, float]]:
    citation_scores: dict[str, tuple[str, float]] = {}

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
    return [
        (citation, object_name, object_score)
        for citation, (object_name, object_score) in ranked_citations
    ]


def _expand_cross_spec_candidates(
    seed_candidates: Iterable[dict[str, Any]],
    domain: str,
    query: str,
    emit: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    alias_map = _term_alias_map()
    selected_seeds = _top_cross_spec_seeds(seed_candidates)
    if emit is not None:
        emit(
            f"stage=expand_cross_spec_candidates detail=selected_seeds count={len(selected_seeds)}"
        )

    ranked_objects = _rank_cross_spec_objects(selected_seeds, query, alias_map)
    if emit is not None:
        emit(
            f"stage=expand_cross_spec_candidates detail=selected_objects count={len(ranked_objects)}"
        )
    if not ranked_objects:
        # logger.info("knowledge_tool cross-spec expand skipped: no ranked_objects")
        return []

    spec_map = _spec_object_map()
    by_citation = _records_by_citation()
    ranked_citations = _rank_cross_spec_citations(ranked_objects, spec_map)
    if emit is not None:
        emit(
            f"stage=expand_cross_spec_candidates detail=selected_citations count={len(ranked_citations)}"
        )

    expanded: list[dict[str, Any]] = []
    for citation, object_name, object_score in ranked_citations:
        for record in by_citation.get(citation) or []:
            metadata = record.get("metadata") or {}
            if not _candidate_domain_allowed(metadata, domain, query):
                continue
            score = (
                110.0
                + min(object_score, 4000.0) * 0.01
                + _metadata_query_bonus(metadata, query, domain)
            )
            expanded.append(
                _record_to_candidate(
                    record,
                    score=score,
                    retrieval=f"cross_spec:{object_name}",
                    features=_candidate_features(
                        metadata,
                        query=query,
                        domain=domain,
                        match_type="cross_spec",
                        cross_spec_source=object_name,
                    ),
                )
            )

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


def _vector_search_candidates(query: str, domain: str) -> list[dict[str, Any]]:
    query_terms = set(_search_terms(query))
    candidates: list[dict[str, Any]] = []
    collections = _select_collections(query, domain)
    for collection_name in collections:
        store = get_pgvector_store(collection_name=collection_name)
        docs = store.similarity_search_with_score(query, k=20)
        for rank, (doc, raw_score) in enumerate(docs):
            metadata = (
                doc.metadata if isinstance(getattr(doc, "metadata", None), dict) else {}
            )
            if not _candidate_domain_allowed(metadata, domain, query):
                continue
            if _is_generic_clause_candidate(metadata):
                continue
            score = 300.0 - rank * 10.0 + _metadata_query_bonus(metadata, query, domain)
            if raw_score is not None:
                score -= float(raw_score) * 15.0
            title_terms, object_terms, normalized_terms = _metadata_term_sets(metadata)
            score += 10.0 * len(query_terms.intersection(title_terms))
            score += 8.0 * len(query_terms.intersection(object_terms))
            score += 4.0 * len(query_terms.intersection(normalized_terms))
            candidates.append(
                _doc_to_candidate(
                    doc,
                    score=score,
                    retrieval=f"vector:{collection_name}",
                    features=_candidate_features(
                        metadata,
                        query=query,
                        domain=domain,
                        match_type="vector",
                        vector_rank=rank + 1,
                    ),
                )
            )
    return _aggregate_parent_candidates(candidates, limit=20)


def _colbert_search_candidates(
    query: str,
    domain: str,
    *,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Retrieve candidates via ColBERT late interaction MaxSim search."""
    retriever = get_colbert_retriever()
    if retriever is None:
        raise FileNotFoundError(
            "ColBERT index not available; run build_colbert_index.py"
        )

    encoder = get_colbert_encoder()
    query_emb = encoder.encode_query(query)
    if query_emb.shape[0] == 0:
        raise ValueError("ColBERT encoder returned an empty query embedding")

    def _domain_filter(record_meta: dict[str, Any]) -> bool:
        return _domain_matches(record_meta, domain)

    ranked = retriever.retrieve(query_emb, top_k=top_k, domain_filter=_domain_filter)

    catalog = _record_catalog()
    candidates: list[dict[str, Any]] = []
    for record_id, maxsim_score, _doc_emb in ranked:
        record = catalog.get(record_id)
        if record is None:
            raise KeyError(f"ColBERT index references unknown record {record_id}")
        metadata = record.get("metadata") or {}
        candidates.append(
            _record_to_candidate(
                record,
                score=maxsim_score * 100.0,
                retrieval="colbert_maxsim",
                features=_candidate_features(
                    metadata,
                    query=query,
                    domain=domain,
                    match_type="colbert_maxsim",
                    lexical_score=maxsim_score,
                ),
            )
        )
    return candidates


def _colbert_rerank_candidates(
    query: str,
    candidates: Iterable[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Re-rank candidates using ColBERT MaxSim scores."""
    retriever = get_colbert_retriever()
    if retriever is None:
        raise FileNotFoundError(
            "ColBERT index not available; run build_colbert_index.py"
        )

    encoder = get_colbert_encoder()
    query_emb = encoder.encode_query(query)
    if query_emb.shape[0] == 0:
        raise ValueError("ColBERT encoder returned an empty query embedding")

    candidate_list = list(candidates)
    if not candidate_list:
        return []

    record_ids = [
        candidate.get("id") or _candidate_key(candidate.get("metadata", {}))
        for candidate in candidate_list
    ]

    scored = retriever.rerank(query_emb, record_ids, top_k=len(record_ids))

    score_map: dict[str, float] = {rid: s for rid, s in scored}
    unscored = sorted(set(record_ids) - set(score_map))
    if unscored:
        raise KeyError(
            "ColBERT index is missing rerank candidates: " + ", ".join(unscored)
        )
    scored_candidates = []
    for candidate in candidate_list:
        rid = candidate.get("id") or _candidate_key(candidate.get("metadata", {}))
        candidate = dict(candidate)
        candidate["score"] = score_map[rid] * 100.0
        candidate["retrieval"] = f"colbert_rerank:{candidate.get('retrieval', '')}"
        scored_candidates.append(candidate)

    scored_candidates.sort(key=lambda c: c["score"], reverse=True)
    return scored_candidates[:limit]


def _colbert_available() -> bool:
    return get_colbert_retriever() is not None


def _resolve_explicit_key_candidates(key: str, domain: str) -> list[dict[str, Any]]:
    normalized_key = _normalized(key)
    if not normalized_key:
        return []

    matches: list[dict[str, Any]] = []
    for record in _record_catalog().values():
        metadata = record.get("metadata") or {}
        if not _candidate_domain_allowed(metadata, domain, key):
            continue
        candidate_key = _candidate_key(metadata, record.get("id") or "")
        if _normalized(candidate_key) != normalized_key:
            continue
        score = 2600.0 + _metadata_query_bonus(metadata, key, domain)
        matches.append(
            _record_to_candidate(
                record,
                score=score,
                retrieval="selected_key",
                features=_candidate_features(
                    metadata,
                    query=key,
                    domain=domain,
                    match_type="selected_key",
                    lexical_score=score,
                    is_exact_identifier=True,
                ),
            )
        )
    return _dedupe_candidates(matches, limit=6)


@dataclass(frozen=True)
class RetrievalStage:
    name: str
    duration_ms: float
    candidate_count: int | None = None
    errors: tuple[str, ...] = ()
    skipped: bool = False


RetrievalObserver = Callable[[RetrievalStage], None]


def _emit_retrieval_stage(
    observer: RetrievalObserver | None,
    stage: str,
    duration_ms: float,
    candidate_count: int | None = None,
    errors: Iterable[str] = (),
    skipped: bool = False,
) -> None:
    if observer is None:
        return
    observer(
        RetrievalStage(
            stage, round(duration_ms, 2), candidate_count, tuple(errors), skipped
        )
    )


def _observe_retrieval_stage(
    observer: RetrievalObserver | None,
    stage: str,
    operation: Callable[[], Any],
    *,
    candidates: bool = False,
) -> Any:
    started = time.perf_counter()
    try:
        result = operation()
    except Exception as exc:
        _emit_retrieval_stage(
            observer,
            stage,
            (time.perf_counter() - started) * 1000.0,
            errors=(str(exc),),
        )
        raise
    _emit_retrieval_stage(
        observer,
        stage,
        (time.perf_counter() - started) * 1000.0,
        candidate_count=len(result) if candidates else None,
    )
    return result


def _observe_retrieval_skip(
    observer: RetrievalObserver | None,
    stage: str,
    *,
    candidate_count: int = 0,
) -> None:
    _emit_retrieval_stage(observer, stage, 0.0, candidate_count, skipped=True)


def _run_retrieval_pipeline(
    query: str,
    domain: str,
    *,
    dense_retrieval: DenseRetrieval,
    limit: int = DEFAULT_RETURNED_RESULTS,
    observer: RetrievalObserver | None = None,
) -> list[dict[str, Any]]:
    def stage(name: str, operation: Callable[[], Any], **options: Any) -> Any:
        return _observe_retrieval_stage(observer, name, operation, **options)

    intent = stage("query_intent", lambda: _query_intent(query, domain))
    direct = stage(
        "direct_record_matches",
        lambda: _direct_record_matches(query, domain),
        candidates=True,
    )
    glossary = stage(
        "glossary_strong_candidates",
        lambda: _glossary_strong_candidates(query, domain),
        candidates=True,
    )
    domain_special = stage(
        "domain_special_candidates",
        lambda: [
            *_schema_object_graph_candidates(query, domain),
            *_schema_definition_candidates(query, domain),
            *_sm_qos_requirement_candidates(query, domain),
            *_am_rfsp_clause_candidates(query, domain),
            *_operation_intent_candidates(query, domain),
        ],
        candidates=True,
    )
    exact = stage(
        "exact_search_candidates",
        lambda: _exact_search_candidates(query, domain),
        candidates=True,
    )
    rerank_pool_limit = max(limit * 4, MAX_RERANK_POOL)
    use_colbert = dense_retrieval == "colbert"

    if intent == "exact_identifier":
        seed_inputs = [*direct, *glossary, *exact]
    elif intent == "glossary_term":
        seed_inputs = [*glossary, *direct, *exact]
    elif intent in {"operation_lookup", "clause_location"}:
        seed_inputs = [*domain_special, *direct, *exact]
    elif intent == "schema_relation":
        seed_inputs = [*domain_special, *direct, *exact, *glossary]
    else:
        seed_inputs = [*direct, *glossary, *domain_special, *exact]
    seeds = stage(
        "seed_dedupe",
        lambda: _dedupe_candidates(seed_inputs, limit=rerank_pool_limit),
        candidates=True,
    )

    cross_spec: list[dict[str, Any]] = []
    if intent == "schema_relation":
        cross_spec = stage(
            "expand_cross_spec_candidates",
            lambda: _expand_cross_spec_candidates(seeds, domain, query),
            candidates=True,
        )
    else:
        _observe_retrieval_skip(observer, "expand_cross_spec_candidates")

    dense: list[dict[str, Any]] = []
    dense_stage = (
        "colbert_search_candidates" if use_colbert else "vector_search_candidates"
    )
    if intent != "exact_identifier" or len(seeds) < limit:
        dense = stage(
            dense_stage,
            lambda: (
                _colbert_search_candidates(query, domain, top_k=20)
                if use_colbert
                else _vector_search_candidates(query, domain)
            ),
            candidates=True,
        )
    else:
        _observe_retrieval_skip(observer, dense_stage)

    combined = stage(
        "combined_dedupe",
        lambda: _dedupe_candidates(
            [*seeds, *cross_spec, *dense], limit=rerank_pool_limit
        ),
        candidates=True,
    )
    rerank_stage = (
        "colbert_rerank_candidates" if use_colbert else "bge_rerank_candidates"
    )
    reranked = stage(
        rerank_stage,
        lambda: (
            _colbert_rerank_candidates(query, combined, limit=limit)
            if use_colbert
            else _bge_rerank_candidates(query, domain, combined, limit=limit)
        ),
        candidates=True,
    )
    return reranked


def _knowledge_domain(domain: str) -> str:
    normalized = str(domain).strip().lower()
    if normalized == "mobility":
        return "am_policy"
    if normalized == "all":
        return "cross_domain"
    if normalized not in {"sm_policy", "am_policy", "ursp", "cross_domain"}:
        raise ValueError(f"unsupported knowledge domain: {domain}")
    return normalized


def _knowledge_response(
    *,
    query: KnowledgeQuery,
    candidates: list[dict[str, Any]],
) -> KnowledgeResponse:
    hits = []
    for candidate in candidates:
        metadata = dict(candidate["metadata"])
        result_key = _candidate_key(metadata, candidate.get("id") or "")
        hits.append(
            KnowledgeHit(
                citation=KnowledgeCitation(
                    result_key=result_key,
                    spec_id=str(metadata["spec_id"]),
                    release=str(metadata["release"]),
                    document_version=str(metadata["version"]),
                    anchor=str(metadata["citation_anchor"]),
                    source_url=str(metadata["source_url"])
                    if metadata.get("source_url")
                    else None,
                ),
                title=str(
                    metadata.get("canonical_title")
                    or metadata.get("schema_name")
                    or metadata.get("operation_id")
                    or result_key
                ),
                document_type=str(metadata["doc_type"]),
                policy_domain=_knowledge_domain(str(metadata["policy_domain"])),
                retrieval_method=str(candidate["retrieval"]),
                score=float(candidate["score"]),
                content=str(candidate["page_content"]),
                object_tags=[str(item) for item in metadata.get("object_tags") or []],
            )
        )
    return KnowledgeResponse(query=query, hits=hits)


def run_semantic_knowledge_pipeline(
    query: str,
    category: str | None = None,
    limit: int = DEFAULT_RETURNED_RESULTS,
    *,
    dense_retrieval: DenseRetrieval,
    observer: RetrievalObserver | None = None,
) -> list[dict[str, Any]]:
    """Run the canonical retrieval path, optionally emitting stage diagnostics."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        raise ValueError("query must not be empty")

    total_start = time.perf_counter()
    try:
        _observe_retrieval_stage(
            observer,
            "ensure_processed_corpus",
            _ensure_processed_corpus,
        )
        domain = _observe_retrieval_stage(
            observer,
            "infer_policy_domain",
            lambda: _infer_policy_domain(normalized_query, category),
        )
        return _run_retrieval_pipeline(
            normalized_query,
            domain,
            dense_retrieval=dense_retrieval,
            limit=limit,
            observer=observer,
        )
    finally:
        _emit_retrieval_stage(
            observer,
            "total",
            duration_ms=(time.perf_counter() - total_start) * 1000.0,
        )


class HybridKnowledgeBackend:
    """Concrete local/PGVector backend injected into the standard RAG tools."""

    def __init__(self, *, dense_retrieval: DenseRetrieval) -> None:
        self.dense_retrieval = dense_retrieval

    def search(
        self,
        *,
        query: str,
        category: KnowledgeDomain | None,
        limit: int,
    ) -> KnowledgeResponse:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        inferred_domain = _infer_policy_domain(normalized_query, category)
        knowledge_query = KnowledgeQuery(
            query=normalized_query,
            domain=_knowledge_domain(inferred_domain),
            limit=limit,
        )
        candidates = run_semantic_knowledge_pipeline(
            normalized_query,
            category=category,
            dense_retrieval=self.dense_retrieval,
            limit=knowledge_query.limit,
        )
        return _knowledge_response(query=knowledge_query, candidates=candidates)

    def get_by_key(
        self,
        *,
        key: str,
        category: KnowledgeDomain | None,
    ) -> KnowledgeResponse:
        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError("key must not be empty")
        _ensure_processed_corpus()
        domain = _infer_policy_domain(normalized_key, category)
        knowledge_query = KnowledgeQuery(
            query=normalized_key,
            domain=_knowledge_domain(domain),
            limit=6,
        )
        selected = _resolve_explicit_key_candidates(normalized_key, domain)
        candidates: list[dict[str, Any]] = []
        if selected:
            cross_spec = _expand_cross_spec_candidates(selected, domain, normalized_key)
            candidates = _dedupe_candidates([*selected, *cross_spec], limit=6)
        return _knowledge_response(query=knowledge_query, candidates=candidates)


__all__ = [
    "DenseRetrieval",
    "HybridKnowledgeBackend",
    "RetrievalStage",
    "run_semantic_knowledge_pipeline",
]
