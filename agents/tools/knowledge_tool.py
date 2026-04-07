from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain.tools import ToolRuntime, tool

from agent_runtime import AgentRuntimeContext
from database.langchain_pg import (
    PCF_POLICY_GLOSSARY_COLLECTION,
    PCF_SM_POLICY_CLAUSES_COLLECTION,
    PCF_SM_POLICY_SCHEMA_COLLECTION,
    PCF_URSP_CLAUSES_COLLECTION,
    PCF_URSP_SCHEMA_COLLECTION,
    get_pgvector_store,
)
from knowledge_scripts.build_pcf_policy_kb import (
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


DEFAULT_RETURNED_RESULTS = 10
MAX_RETURNED_RESULTS = 30
MAX_CONTENT_CHARS = 1200
MAX_SUMMARY_CHARS = 240
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
    "request",
    "response",
    "api",
    "object",
    "objects",
    "contains",
    "include",
    "definition",
    "define",
    "allowed",
    "value",
    "values",
)
REQUIRED_KB_PATHS = (
    CLAUSE_JSONL,
    SCHEMA_JSONL,
    GLOSSARY_JSONL,
    SCHEMA_EXACT_INDEX_JSON,
    GLOSSARY_EXACT_INDEX_JSON,
    SPEC_OBJECT_MAP_JSON,
    TERM_ALIAS_MAP_JSON,
)


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


def _domain_matches(metadata_domain: str, requested_domain: str) -> bool:
    normalized_domain = _normalized(metadata_domain)
    normalized_request = _normalized(requested_domain)
    if normalized_request in {"", "all"}:
        return True
    if normalized_domain == "shared":
        return True
    return normalized_domain == normalized_request


def _spec_priority(spec_id: str, domain: str) -> int:
    normalized_domain = _normalized(domain)
    if normalized_domain == "sm_policy":
        order = ["29.512", "23.503", "29.571", "29.514", "29.213", "29.214", "24.501", "23.501", "29.519", "29.525", "24.526"]
    elif normalized_domain == "ursp":
        order = ["24.526", "29.525", "23.503", "29.519", "24.501", "23.501", "29.571", "29.512", "29.514"]
    else:
        order = ["23.503", "29.512", "24.526", "29.525", "29.571", "29.514", "29.519", "24.501", "23.501"]
    try:
        return max(0, 20 - order.index(spec_id))
    except ValueError:
        return 0


def _prefers_schema(query: str) -> bool:
    lowered = _normalized(query)
    return any(keyword in lowered for keyword in SCHEMA_QUERY_KEYWORDS)


def _infer_policy_domain(query: str, category: Optional[str]) -> str:
    normalized_category = _normalized(category)
    if normalized_category in {"sm_policy", "ursp", "shared", "all"}:
        return "all" if normalized_category == "shared" else normalized_category

    haystack = " ".join(part for part in [query, category] if part).lower()
    sm_hits = sum(1 for keyword in SM_POLICY_KEYWORDS if keyword in haystack)
    ursp_hits = sum(1 for keyword in URSP_KEYWORDS if keyword in haystack)
    if sm_hits and not ursp_hits:
        return "sm_policy"
    if ursp_hits and not sm_hits:
        return "ursp"
    return "all"


def _select_collections(query: str, domain: str) -> List[str]:
    prefers_schema = _prefers_schema(query)
    if domain == "sm_policy":
        return [PCF_SM_POLICY_SCHEMA_COLLECTION, PCF_SM_POLICY_CLAUSES_COLLECTION] if prefers_schema else [PCF_SM_POLICY_CLAUSES_COLLECTION, PCF_SM_POLICY_SCHEMA_COLLECTION]
    if domain == "ursp":
        return [PCF_URSP_SCHEMA_COLLECTION, PCF_URSP_CLAUSES_COLLECTION] if prefers_schema else [PCF_URSP_CLAUSES_COLLECTION, PCF_URSP_SCHEMA_COLLECTION]
    if prefers_schema:
        return [
            PCF_SM_POLICY_SCHEMA_COLLECTION,
            PCF_URSP_SCHEMA_COLLECTION,
            PCF_SM_POLICY_CLAUSES_COLLECTION,
            PCF_URSP_CLAUSES_COLLECTION,
            PCF_POLICY_GLOSSARY_COLLECTION,
        ]
    return [
        PCF_SM_POLICY_CLAUSES_COLLECTION,
        PCF_URSP_CLAUSES_COLLECTION,
        PCF_SM_POLICY_SCHEMA_COLLECTION,
        PCF_URSP_SCHEMA_COLLECTION,
        PCF_POLICY_GLOSSARY_COLLECTION,
    ]


@lru_cache(maxsize=1)
def _ensure_processed_corpus() -> Tuple[Path, ...]:
    missing = [path for path in REQUIRED_KB_PATHS if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "PCF policy knowledge base is not built. Missing files: "
            + ", ".join(str(path) for path in missing)
            + ". Run `python knowledge_scripts/build_pcf_policy_kb.py build` first."
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
    prefers_schema = _prefers_schema(query)
    if prefers_schema:
        if normalized_doc_type == "openapi":
            return 260.0
        if normalized_doc_type == "glossary":
            return 180.0
        if normalized_doc_type == "table":
            return 120.0
        if normalized_doc_type in {"stage2", "stage3"}:
            return 100.0
        return 0.0

    if normalized_doc_type == "glossary":
        return 220.0
    if normalized_doc_type == "table":
        return 170.0
    if normalized_doc_type in {"stage2", "stage3"}:
        return 140.0
    if normalized_doc_type == "openapi":
        return 110.0 if domain == "sm_policy" else 90.0
    return 0.0


def _metadata_query_bonus(metadata: Dict[str, Any], query: str, domain: str) -> float:
    score = _spec_priority(str(metadata.get("spec_id") or ""), domain)
    score += _doc_type_priority(str(metadata.get("doc_type") or ""), query, domain)
    query_terms = set(normalize_query_terms(query))
    object_terms = {_normalized(item) for item in metadata.get("object_tags") or []}
    score += 12.0 * len(query_terms.intersection(object_terms))
    return score


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


def _dedupe_candidates(candidates: Iterable[Dict[str, Any]], *, limit: int = DEFAULT_RETURNED_RESULTS) -> List[Dict[str, Any]]:
    chosen: Dict[str, Dict[str, Any]] = {}
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        metadata = candidate["metadata"]
        dedupe_key = _candidate_key(metadata, candidate.get("id") or "")
        current = chosen.get(dedupe_key)
        if current is None or candidate["score"] > current["score"]:
            chosen[dedupe_key] = candidate
    return list(sorted(chosen.values(), key=lambda item: item["score"], reverse=True))[:limit]


def _direct_record_matches(key: str, domain: str) -> List[Dict[str, Any]]:
    normalized_key = _normalized(key)
    catalog = _record_catalog()
    direct_hits: List[Dict[str, Any]] = []
    for record in catalog.values():
        metadata = record.get("metadata") or {}
        if not _domain_matches(str(metadata.get("policy_domain") or ""), domain):
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
    for hit in search_exact_index(_schema_exact_index(), query, top_k=12):
        record = catalog.get(hit["id"])
        if record is None:
            continue
        metadata = record.get("metadata") or {}
        if not _domain_matches(str(metadata.get("policy_domain") or ""), domain):
            continue
        score = 1200.0 + float(hit["score"]) + _metadata_query_bonus(metadata, query, domain)
        candidates.append(_record_to_candidate(record, score=score, retrieval="exact_schema"))
    for hit in search_exact_index(_glossary_exact_index(), query, top_k=12):
        record = catalog.get(hit["id"])
        if record is None:
            continue
        metadata = record.get("metadata") or {}
        if not _domain_matches(str(metadata.get("policy_domain") or ""), domain):
            continue
        score = 1000.0 + float(hit["score"]) + _metadata_query_bonus(metadata, query, domain)
        candidates.append(_record_to_candidate(record, score=score, retrieval="exact_glossary"))
    return _dedupe_candidates(candidates, limit=6)


def _expand_cross_spec_candidates(seed_candidates: Iterable[Dict[str, Any]], domain: str, query: str) -> List[Dict[str, Any]]:
    object_names: List[str] = []
    alias_map = _term_alias_map()
    for candidate in seed_candidates:
        metadata = candidate["metadata"]
        canonical_title = str(metadata.get("canonical_title") or "").strip()
        object_names.extend(metadata.get("object_tags") or [])
        if canonical_title and canonical_title in alias_map:
            object_names.extend(alias_map[canonical_title].get("related_objects") or [])

    seen_objects = sorted(set(name for name in object_names if name))
    if not seen_objects:
        return []

    spec_map = _spec_object_map()
    by_citation = _records_by_citation()
    expanded: List[Dict[str, Any]] = []
    for object_name in seen_objects:
        for mapping in spec_map.get(object_name) or []:
            citation = str(mapping.get("citation_anchor") or "").strip()
            if not citation:
                continue
            for record in by_citation.get(citation) or []:
                metadata = record.get("metadata") or {}
                if not _domain_matches(str(metadata.get("policy_domain") or ""), domain):
                    continue
                score = 900.0 + _metadata_query_bonus(metadata, query, domain)
                expanded.append(_record_to_candidate(record, score=score, retrieval=f"cross_spec:{object_name}"))
    return _dedupe_candidates(expanded, limit=4)


def _vector_search_candidates(query: str, domain: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    query_terms = set(normalize_query_terms(query))
    candidates: List[Dict[str, Any]] = []
    errors: List[str] = []
    collections = _select_collections(query, domain)
    for collection_name in collections:
        try:
            store = get_pgvector_store(collection_name=collection_name)
            docs = store.similarity_search(query, k=3)
        except Exception as exc:
            errors.append(f"{collection_name}: {exc}")
            continue
        for rank, doc in enumerate(docs):
            metadata = doc.metadata if isinstance(getattr(doc, "metadata", None), dict) else {}
            if not _domain_matches(str(metadata.get("policy_domain") or ""), domain):
                continue
            score = 300.0 - rank * 10.0 + _metadata_query_bonus(metadata, query, domain)
            object_terms = {_normalized(item) for item in metadata.get("object_tags") or []}
            score += 8.0 * len(query_terms.intersection(object_terms))
            candidates.append(_doc_to_candidate(doc, score=score, retrieval=f"vector:{collection_name}"))
    return _dedupe_candidates(candidates, limit=6), errors


def _resolve_explicit_key_candidates(key: str, domain: str) -> List[Dict[str, Any]]:
    normalized_key = _normalized(key)
    if not normalized_key:
        return []

    matches: List[Dict[str, Any]] = []
    for record in _record_catalog().values():
        metadata = record.get("metadata") or {}
        if not _domain_matches(str(metadata.get("policy_domain") or ""), domain):
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
    exact = _exact_search_candidates(query, domain)
    seeds = _dedupe_candidates([*direct, *exact], limit=6)
    cross_spec = _expand_cross_spec_candidates(seeds, domain, query)
    vector, errors = _vector_search_candidates(query, domain)
    combined = _dedupe_candidates([*seeds, *cross_spec, *vector], limit=limit)
    return combined, errors


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


@tool
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


@tool
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


__all__ = ["get_knowledge_by_key", "search_semantic_knowledge"]
