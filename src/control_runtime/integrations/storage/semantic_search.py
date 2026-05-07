from __future__ import annotations

import copy
import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List

from database.models import UeContextRecord
from shared.logging import setup_logger

from .flow_catalog import _normalize_catalog_flow
from .session_store import _get_latest_graph_app_data, session_scope

logger = setup_logger(__name__)

def _normalize_semantic_lookup_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(value or "").strip().lower()).strip()


def _compact_semantic_lookup_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").strip().lower())


def _token_overlap_score(query: str, target: str) -> float:
    query_tokens = {token for token in _normalize_semantic_lookup_text(query).split() if token}
    target_tokens = {token for token in _normalize_semantic_lookup_text(target).split() if token}
    if not query_tokens or not target_tokens:
        return 0.0
    return len(query_tokens & target_tokens) / len(query_tokens)


def _semantic_name_score(query: Any, target: Any) -> float:
    normalized_query = _normalize_semantic_lookup_text(query)
    normalized_target = _normalize_semantic_lookup_text(target)
    if not normalized_query or not normalized_target:
        return 0.0

    compact_query = _compact_semantic_lookup_text(query)
    compact_target = _compact_semantic_lookup_text(target)
    if compact_query and compact_query == compact_target:
        return 1.0

    scores = [
        SequenceMatcher(None, normalized_query, normalized_target).ratio(),
        _token_overlap_score(normalized_query, normalized_target),
    ]
    if compact_query and compact_target and (compact_query in compact_target or compact_target in compact_query):
        scores.append(0.95)
    return max(scores)


def search_flow_targets_by_semantic(
    *,
    supi: str = "",
    app_name: str = "",
    flow_name: str = "",
    snapshot_id: str = "",
    limit: int = 5,
    min_score: float = 0.35,
) -> Dict[str, Any]:
    normalized_supi = str(supi or "").strip()
    normalized_app_name = str(app_name or "").strip()
    normalized_flow_name = str(flow_name or "").strip()
    if not normalized_app_name and not normalized_flow_name:
        return {
            "query": {"supi": normalized_supi, "app_name": normalized_app_name, "flow_name": normalized_flow_name},
            "candidate_count": 0,
            "candidates": [],
        }

    app_data = _get_latest_graph_app_data(snapshot_id=snapshot_id)
    candidates: List[Dict[str, Any]] = []

    for app in app_data:
        if not isinstance(app, dict):
            continue
        if normalized_supi and str(app.get("supi") or "").strip() != normalized_supi:
            continue
        flows = app.get("flows") or []
        app_score = _semantic_name_score(normalized_app_name, app.get("name")) if normalized_app_name else 0.0
        for flow in flows:
            if not isinstance(flow, dict):
                continue
            flow_score = _semantic_name_score(normalized_flow_name, flow.get("name")) if normalized_flow_name else 0.0
            if normalized_app_name and normalized_flow_name:
                combined_component_threshold = max(min_score, 0.5)
                if app_score < combined_component_threshold or flow_score < combined_component_threshold:
                    continue
                overall_score = (app_score * 0.45) + (flow_score * 0.55)
            elif normalized_app_name:
                overall_score = app_score
            else:
                overall_score = flow_score
            if overall_score < min_score:
                continue

            candidate = _normalize_catalog_flow(app, flow)
            candidate.update(
                {
                    "match_score": round(overall_score, 4),
                    "app_name_score": round(app_score, 4),
                    "flow_name_score": round(flow_score, 4),
                }
            )
            candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            float(item.get("match_score") or 0.0),
            float(item.get("flow_name_score") or 0.0),
            float(item.get("app_name_score") or 0.0),
            str(item.get("supi") or ""),
            str(item.get("app_id") or ""),
            str(item.get("flow_id") or ""),
        ),
        reverse=True,
    )

    bounded_limit = max(1, int(limit or 5))
    return {
        "query": {"supi": normalized_supi, "app_name": normalized_app_name, "flow_name": normalized_flow_name},
        "candidate_count": len(candidates),
        "candidates": candidates[:bounded_limit],
    }


def _flatten_semantic_tokens(value: Any) -> List[str]:
    tokens: List[str] = []
    if isinstance(value, dict):
        sst = str(value.get("sst") or "").strip()
        sd = str(value.get("sd") or "").strip()
        if sst or sd:
            combined = "-".join(part for part in (sst, sd) if part)
            normalized = _normalize_semantic_lookup_text(combined)
            if normalized:
                tokens.append(normalized)
        for item in value.values():
            tokens.extend(_flatten_semantic_tokens(item))
        try:
            serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            serialized = str(value)
        normalized = _normalize_semantic_lookup_text(serialized)
        if normalized:
            tokens.append(normalized)
    elif isinstance(value, list):
        for item in value:
            tokens.extend(_flatten_semantic_tokens(item))
    else:
        normalized = _normalize_semantic_lookup_text(value)
        if normalized:
            tokens.append(normalized)
    return list(dict.fromkeys(token for token in tokens if token))


def _match_semantic_query(query: str, tokens: List[str], *, min_score: float = 0.45) -> float:
    normalized_query = _normalize_semantic_lookup_text(query)
    if not normalized_query or not tokens:
        return 0.0

    compact_query = _compact_semantic_lookup_text(query)
    best_score = 0.0
    for token in tokens:
        if not token:
            continue
        score = _semantic_name_score(normalized_query, token)
        compact_token = _compact_semantic_lookup_text(token)
        if compact_query and compact_token and (compact_query in compact_token or compact_token in compact_query):
            score = max(score, 0.95)
        best_score = max(best_score, score)

    return best_score if best_score >= min_score else 0.0


def search_am_policy_targets_by_context(
    *,
    supi: str = "",
    association_id: str = "",
    allowed_snssai: str = "",
    target_snssai: str = "",
    service_area: str = "",
    rfsp: str = "",
    access_type: str = "",
    limit: int = 5,
) -> Dict[str, Any]:
    normalized_supi = str(supi or "").strip()
    normalized_association_id = str(association_id or "").strip()
    normalized_allowed = str(allowed_snssai or "").strip()
    normalized_target = str(target_snssai or "").strip()
    normalized_service_area = str(service_area or "").strip()
    normalized_rfsp = str(rfsp or "").strip()
    normalized_access_type = str(access_type or "").strip()

    query_payload = {
        "supi": normalized_supi,
        "association_id": normalized_association_id,
        "allowed_snssai": normalized_allowed,
        "target_snssai": normalized_target,
        "service_area": normalized_service_area,
        "rfsp": normalized_rfsp,
        "access_type": normalized_access_type,
    }
    if not any(query_payload.values()):
        return {
            "query": query_payload,
            "candidate_count": 0,
            "candidates": [],
        }

    bounded_limit = max(1, int(limit or 5))
    fetch_limit = max(20, bounded_limit * 10)

    try:
        with session_scope() as session:
            query = session.query(UeContextRecord)
            if normalized_supi:
                query = query.filter(UeContextRecord.supi == normalized_supi)
            rows = query.order_by(UeContextRecord.updated_at.desc()).limit(fetch_limit).all()
            row_payloads = [
                {
                    "supi": str(row.supi or "").strip(),
                    "am_policy_context": copy.deepcopy(row.am_policy_context or {}),
                    "access_mobility_context": copy.deepcopy(row.access_mobility_context or {}),
                    "mobility_summary": copy.deepcopy(row.mobility_summary or {}),
                }
                for row in rows
            ]
    except Exception as exc:
        logger.error(f"Failed to search AM policy targets: {exc}")
        return {
            "query": query_payload,
            "candidate_count": 0,
            "candidates": [],
        }

    candidates: List[Dict[str, Any]] = []
    for row in row_payloads:
        am_policy_context = row.get("am_policy_context") or {}
        access_mobility_context = row.get("access_mobility_context") or {}
        mobility_summary = row.get("mobility_summary") or {}

        association_map = am_policy_context.get("associations") or {}
        association_ids = [str(key or "").strip() for key in association_map.keys() if str(key or "").strip()]
        current_association_id = str(mobility_summary.get("currentAssociationId") or "").strip()
        if current_association_id and current_association_id not in association_ids:
            association_ids.append(current_association_id)

        scores: List[float] = []
        match_reasons: List[str] = []

        if normalized_supi:
            scores.append(1.0)
            match_reasons.append("supi")

        if normalized_association_id:
            score = _match_semantic_query(normalized_association_id, association_ids, min_score=0.6)
            if score <= 0.0:
                continue
            scores.append(score)
            match_reasons.append("association_id")

        if normalized_allowed:
            score = _match_semantic_query(
                normalized_allowed,
                _flatten_semantic_tokens(am_policy_context.get("allowedSnssais") or []),
            )
            if score <= 0.0:
                continue
            scores.append(score)
            match_reasons.append("allowed_snssai")

        if normalized_target:
            score = _match_semantic_query(
                normalized_target,
                _flatten_semantic_tokens(am_policy_context.get("targetSnssais") or []),
            )
            if score <= 0.0:
                continue
            scores.append(score)
            match_reasons.append("target_snssai")

        if normalized_service_area:
            service_area_payload = (
                am_policy_context.get("servAreaRes")
                or am_policy_context.get("wlServAreaRes")
                or mobility_summary.get("currentServAreaRes")
                or {}
            )
            score = _match_semantic_query(
                normalized_service_area,
                _flatten_semantic_tokens(service_area_payload),
            )
            if score <= 0.0:
                continue
            scores.append(score)
            match_reasons.append("service_area")

        if normalized_rfsp:
            candidate_rfsp = am_policy_context.get("rfsp")
            if candidate_rfsp is None:
                candidate_rfsp = mobility_summary.get("currentRfsp")
            score = _match_semantic_query(normalized_rfsp, _flatten_semantic_tokens(candidate_rfsp), min_score=0.9)
            if score <= 0.0:
                continue
            scores.append(score)
            match_reasons.append("rfsp")

        if normalized_access_type:
            score = _match_semantic_query(
                normalized_access_type,
                _flatten_semantic_tokens(access_mobility_context.get("accessType")),
                min_score=0.8,
            )
            if score <= 0.0:
                continue
            scores.append(score)
            match_reasons.append("access_type")

        if not scores:
            continue

        candidates.append(
            {
                "supi": row.get("supi") or "",
                "association_ids": association_ids,
                "allowed_snssais": am_policy_context.get("allowedSnssais") or [],
                "target_snssais": am_policy_context.get("targetSnssais") or [],
                "mapping_snssais": am_policy_context.get("mappingSnssais") or [],
                "rfsp": am_policy_context.get("rfsp") if am_policy_context.get("rfsp") is not None else mobility_summary.get("currentRfsp"),
                "access_type": access_mobility_context.get("accessType"),
                "service_area_restriction": (
                    am_policy_context.get("servAreaRes")
                    or am_policy_context.get("wlServAreaRes")
                    or mobility_summary.get("currentServAreaRes")
                    or {}
                ),
                "current_association_id": current_association_id or None,
                "match_score": round(sum(scores) / len(scores), 4),
                "match_reasons": match_reasons,
            }
        )

    candidates.sort(
        key=lambda item: (
            float(item.get("match_score") or 0.0),
            str(item.get("supi") or ""),
        ),
        reverse=True,
    )

    return {
        "query": query_payload,
        "candidate_count": len(candidates),
        "candidates": candidates[:bounded_limit],
    }

