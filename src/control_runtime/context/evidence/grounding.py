from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ...domain.policy_plan import FlowSelector
from ...agents.grounding.common import normalize_domain_evidence, normalize_requested_domains, uses_am_grounding, uses_sm_grounding
from ...agents.grounding.contracts import ExplicitFlowTarget, FlowCandidateEvidence, IntentEvidence


class IntentEvidenceBuilder:
    def build_intent_evidence(
        self,
        *,
        user_input: str,
        supi: str,
        main_directives: Dict[str, Any],
        catalog_payload: Dict[str, Any],
        semantic_candidates: List[Dict[str, Any]],
        am_context_payload: Optional[Dict[str, Any]] = None,
        am_policy_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> IntentEvidence:
        flow_match = re.search(r"(?i)\b(flow-\d+)\b", user_input)
        app_match = re.search(r"(?i)\b(app-\d+)\b", user_input)
        requested_domains = normalize_requested_domains(main_directives.get("requested_domains"))
        resolved_catalog_payload = dict(catalog_payload or {}) if uses_sm_grounding(requested_domains) else {}
        resolved_semantic_candidates = list(semantic_candidates or []) if uses_sm_grounding(requested_domains) else []
        resolved_am_context = dict(am_context_payload or {}) if uses_am_grounding(requested_domains) else {}
        resolved_am_policy_candidates = list(am_policy_candidates or []) if uses_am_grounding(requested_domains) else []
        upstream_candidate_flows = [
            dict(item)
            for item in (main_directives.get("candidate_flows") or [])
            if isinstance(item, dict)
        ]

        app_catalog = resolved_catalog_payload.get("app_catalog") or []
        flow_catalog = resolved_catalog_payload.get("flow_catalog") or []
        explicit_flow_id = flow_match.group(1) if flow_match else ""
        explicit_app_id = app_match.group(1) if app_match else ""

        exact_app = self._match_exact_app_from_input(
            user_input=user_input,
            app_catalog=app_catalog,
            explicit_app_id=explicit_app_id,
        )
        explicit_app_name = str(exact_app.get("app_name") or "").strip() if exact_app is not None else ""
        exact_flow = self._match_exact_flow_from_input(
            user_input=user_input,
            flow_catalog=flow_catalog,
            explicit_flow_id=explicit_flow_id,
        )
        explicit_flow_targets = self._extract_explicit_flow_targets(
            user_input=user_input,
            flow_catalog=flow_catalog,
            exact_flow=exact_flow,
        )
        explicit_flow_name = explicit_flow_targets[0].flow_name if explicit_flow_targets else ""
        strict_explicit_flow_request = bool(explicit_flow_id or explicit_flow_targets)

        candidate_flows: List[FlowCandidateEvidence] = []
        candidate_apps: List[Dict[str, Any]] = []

        if exact_app is not None:
            candidate_apps.append(
                {
                    "app_id": str(exact_app.get("app_id") or "").strip(),
                    "app_name": str(exact_app.get("app_name") or "").strip(),
                }
            )

        if explicit_flow_id:
            for item in flow_catalog:
                if str(item.get("flow_id") or "").strip().lower() == explicit_flow_id.lower():
                    candidate_flows.append(self._flow_candidate_from_catalog(item, score=1.0))
            for item in upstream_candidate_flows:
                if str(item.get("flow_id") or "").strip().lower() == explicit_flow_id.lower():
                    candidate_flows.append(self._flow_candidate_from_catalog(item, score=1.0))
        else:
            explicit_target_names = {
                str(item.flow_name or "").strip()
                for item in explicit_flow_targets
                if str(item.flow_name or "").strip()
            }
            if exact_flow is not None and not explicit_target_names:
                candidate_flows.append(self._flow_candidate_from_catalog(exact_flow, score=1.0))
            for item in flow_catalog:
                flow_name = str(item.get("flow_name") or "").strip()
                if flow_name and flow_name in explicit_target_names:
                    candidate_flows.append(self._flow_candidate_from_catalog(item, score=1.0))
            for item in upstream_candidate_flows:
                flow_name = str(item.get("flow_name") or "").strip()
                if flow_name and flow_name in explicit_target_names:
                    candidate_flows.append(self._flow_candidate_from_catalog(item, score=1.0))

        if resolved_semantic_candidates and not strict_explicit_flow_request:
            candidate_flows.extend(self._flow_candidates_from_semantic_matches(resolved_semantic_candidates))

        candidate_flows = self._deduplicate_flow_candidates(candidate_flows)
        candidate_apps = self._deduplicate_candidate_apps(
            candidate_apps=candidate_apps,
            candidate_flows=candidate_flows,
            app_catalog=app_catalog,
            semantic_candidates=resolved_semantic_candidates,
        )

        return IntentEvidence(
            user_input=user_input,
            supi=str(supi or "").strip(),
            requested_domains=requested_domains,
            retry_scope=str(main_directives.get("retry_scope") or "").strip(),
            main_requested_domains=requested_domains,
            explicit_app_id=explicit_app_id,
            explicit_app_name=explicit_app_name,
            explicit_flow_id=explicit_flow_id,
            explicit_flow_name=explicit_flow_name,
            explicit_flow_targets=explicit_flow_targets,
            candidate_flows=candidate_flows,
            candidate_apps=candidate_apps,
            domain_evidence=normalize_domain_evidence(main_directives.get("domain_evidence")),
            am_context_summary=self._summarize_am_context(resolved_am_context),
            am_policy_candidates=self._normalize_am_policy_candidates(resolved_am_policy_candidates),
            catalog_payload=resolved_catalog_payload,
            semantic_candidates=resolved_semantic_candidates,
            am_context_payload=resolved_am_context,
        )

    @classmethod
    def _flow_candidates_from_semantic_matches(cls, candidates: List[Dict[str, Any]]) -> List[FlowCandidateEvidence]:
        results: List[FlowCandidateEvidence] = []
        for item in candidates or []:
            if not isinstance(item, dict):
                continue
            score = item.get("match_score")
            try:
                normalized_score = float(score)
            except (TypeError, ValueError):
                normalized_score = 0.5
            results.append(cls._flow_candidate_from_catalog(item, score=normalized_score))
        return results

    @staticmethod
    def _deduplicate_flow_candidates(candidates: List[FlowCandidateEvidence]) -> List[FlowCandidateEvidence]:
        deduped: List[FlowCandidateEvidence] = []
        seen: set[Tuple[str, str, str]] = set()
        for candidate in candidates:
            key = (
                str(candidate.supi or "").strip(),
                str(candidate.app_id or "").strip(),
                str(candidate.flow_id or "").strip(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    @staticmethod
    def _deduplicate_candidate_apps(
        *,
        candidate_apps: List[Dict[str, Any]],
        candidate_flows: List[FlowCandidateEvidence],
        app_catalog: List[Dict[str, Any]],
        semantic_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        resolved_apps = list(candidate_apps)
        if not resolved_apps and len({item.app_id for item in candidate_flows if item.app_id}) == 1:
            app_id = next(iter({item.app_id for item in candidate_flows if item.app_id}))
            app_match_row = next((item for item in app_catalog if str(item.get("app_id") or "").strip() == app_id), None)
            if app_match_row is not None:
                resolved_apps.append(
                    {
                        "app_id": app_id,
                        "app_name": str(app_match_row.get("app_name") or "").strip(),
                    }
                )

        if not resolved_apps:
            by_app: Dict[str, Dict[str, Any]] = {}
            for item in semantic_candidates or []:
                if not isinstance(item, dict):
                    continue
                app_id = str(item.get("app_id") or "").strip()
                if not app_id or app_id in by_app:
                    continue
                by_app[app_id] = {
                    "app_id": app_id,
                    "app_name": str(item.get("app_name") or "").strip(),
                }
            resolved_apps.extend(by_app.values())

        deduped: List[Dict[str, Any]] = []
        seen_app_ids: set[str] = set()
        for item in resolved_apps:
            app_id = str(item.get("app_id") or "").strip()
            if not app_id or app_id in seen_app_ids:
                continue
            seen_app_ids.add(app_id)
            deduped.append(
                {
                    "app_id": app_id,
                    "app_name": str(item.get("app_name") or "").strip(),
                }
            )
        return deduped

    @staticmethod
    def _flow_candidate_from_catalog(flow: Dict[str, Any], *, score: float) -> FlowCandidateEvidence:
        service = flow.get("service") if isinstance(flow.get("service"), dict) else {}
        return FlowCandidateEvidence(
            supi=str(flow.get("supi") or "").strip(),
            app_id=str(flow.get("app_id") or "").strip(),
            app_name=str(flow.get("app_name") or "").strip() or None,
            flow_id=str(flow.get("flow_id") or "").strip(),
            flow_name=str(flow.get("flow_name") or "").strip(),
            service_type=str(service.get("service_type") or "").strip() or None,
            service_type_id=service.get("service_type_id"),
            score=score,
        )

    @staticmethod
    def _build_flow_selector_from_catalog(flow: Dict[str, Any]) -> FlowSelector:
        service = flow.get("service") if isinstance(flow.get("service"), dict) else {}
        sla = flow.get("sla") if isinstance(flow.get("sla"), dict) else {}
        allocation = flow.get("allocation") if isinstance(flow.get("allocation"), dict) else {}
        traffic = flow.get("traffic") if isinstance(flow.get("traffic"), dict) else {}
        five_tuple = traffic.get("five_tuple")
        return FlowSelector(
            supi=str(flow.get("supi") or "").strip(),
            app_id=str(flow.get("app_id") or "").strip(),
            app_name=str(flow.get("app_name") or "").strip() or None,
            flow_id=str(flow.get("flow_id") or "").strip() or None,
            target_type="flow",
            name=str(flow.get("flow_name") or flow.get("flow_id") or "").strip(),
            service_type=str(service.get("service_type") or "").strip() or None,
            service_type_id=service.get("service_type_id"),
            bw_ul=sla.get("bandwidth_ul"),
            bw_dl=sla.get("bandwidth_dl"),
            gbr_ul=sla.get("guaranteed_bandwidth_ul"),
            gbr_dl=sla.get("guaranteed_bandwidth_dl"),
            lat=sla.get("latency"),
            loss_req=sla.get("loss_rate"),
            jitter_req=sla.get("jitter"),
            priority=sla.get("priority"),
            description=str(flow.get("flow_name") or "").strip() or None,
            five_tuple=list(five_tuple) if isinstance(five_tuple, (list, tuple)) else None,
            current_slice_snssai=str(allocation.get("current_slice_snssai") or "").strip() or None,
            current_bw_ul=allocation.get("allocated_bandwidth_ul"),
            current_bw_dl=allocation.get("allocated_bandwidth_dl"),
            resolution_status="resolved",
        )

    @staticmethod
    def _match_exact_app_from_input(
        *,
        user_input: str,
        app_catalog: List[Dict[str, Any]],
        explicit_app_id: str,
    ) -> Optional[Dict[str, Any]]:
        normalized_input = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(user_input or "").lower())
        if explicit_app_id:
            return next(
                (
                    item
                    for item in app_catalog
                    if str(item.get("app_id") or "").strip().lower() == explicit_app_id.lower()
                ),
                None,
            )
        matches: List[Dict[str, Any]] = []
        for item in app_catalog:
            normalized_name = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(item.get("app_name") or "").lower())
            if normalized_name and normalized_name in normalized_input:
                matches.append(item)
        if len(matches) == 1:
            return matches[0]
        return None

    @staticmethod
    def _match_exact_flow_from_input(
        *,
        user_input: str,
        flow_catalog: List[Dict[str, Any]],
        explicit_flow_id: str,
    ) -> Optional[Dict[str, Any]]:
        normalized_input = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(user_input or "").lower())
        if explicit_flow_id:
            return next(
                (
                    item
                    for item in flow_catalog
                    if str(item.get("flow_id") or "").strip().lower() == explicit_flow_id.lower()
                ),
                None,
            )
        matches: List[Dict[str, Any]] = []
        for item in flow_catalog:
            normalized_name = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(item.get("flow_name") or "").lower())
            if normalized_name and normalized_name in normalized_input:
                matches.append(item)
        if len(matches) == 1:
            return matches[0]
        return None

    @classmethod
    def _extract_explicit_flow_targets(
        cls,
        *,
        user_input: str,
        flow_catalog: List[Dict[str, Any]],
        exact_flow: Optional[Dict[str, Any]],
    ) -> List[ExplicitFlowTarget]:
        text = str(user_input or "")
        if not text:
            return []
        token_pattern = r"\b(?!imsi-)(?!app-)(?!flow-)[A-Za-z][A-Za-z0-9_]*_[A-Za-z0-9_]*\b"
        matched_tokens = [token.strip() for token in re.findall(token_pattern, text) if token.strip()]
        if exact_flow is not None and len(matched_tokens) <= 1:
            flow_name = str(exact_flow.get("flow_name") or "").strip()
            return [ExplicitFlowTarget(flow_name=flow_name)] if flow_name else []
        catalog_flow_names = {
            str(item.get("flow_name") or "").strip()
            for item in flow_catalog
            if isinstance(item, dict) and str(item.get("flow_name") or "").strip()
        }
        resolved_targets: List[ExplicitFlowTarget] = []
        seen_names: set[str] = set()
        for token in matched_tokens:
            if token in catalog_flow_names and token not in seen_names:
                resolved_targets.append(ExplicitFlowTarget(flow_name=token))
                seen_names.add(token)
        for token in matched_tokens:
            lowered = token.lower()
            if lowered in {"urllc", "embb", "mmtc"}:
                continue
            if token not in seen_names:
                resolved_targets.append(ExplicitFlowTarget(flow_name=token))
                seen_names.add(token)
        return resolved_targets

    @staticmethod
    def _summarize_am_context(am_context_payload: Any) -> Dict[str, Any]:
        if not isinstance(am_context_payload, dict):
            return {}

        access_mobility_context = am_context_payload.get("accessMobilityContext") or {}
        am_policy_context = am_context_payload.get("amPolicyContext") or {}
        mobility_summary = am_context_payload.get("mobilitySummary") or {}
        association_records = am_context_payload.get("associationRecords") or []

        association_ids = [
            str(key or "").strip()
            for key in (am_policy_context.get("associations") or {}).keys()
            if str(key or "").strip()
        ]
        if not association_ids and isinstance(association_records, list):
            association_ids = [
                str(item.get("polAssoId") or "").strip()
                for item in association_records
                if isinstance(item, dict) and str(item.get("polAssoId") or "").strip()
            ]

        summary: Dict[str, Any] = {}
        if association_ids:
            summary["association_ids"] = association_ids
        if am_policy_context.get("allowedSnssais"):
            summary["allowed_snssais"] = am_policy_context.get("allowedSnssais") or []
        if am_policy_context.get("targetSnssais"):
            summary["target_snssais"] = am_policy_context.get("targetSnssais") or []
        if am_policy_context.get("mappingSnssais"):
            summary["mapping_snssais"] = am_policy_context.get("mappingSnssais") or []

        current_association_id = str(mobility_summary.get("currentAssociationId") or "").strip()
        if current_association_id:
            summary["current_association_id"] = current_association_id

        access_type = str(access_mobility_context.get("accessType") or "").strip()
        if access_type:
            summary["access_type"] = access_type

        rfsp = am_policy_context.get("rfsp")
        if rfsp is None:
            rfsp = mobility_summary.get("currentRfsp")
        if rfsp is not None:
            summary["rfsp"] = rfsp

        service_area_restriction = (
            am_policy_context.get("servAreaRes")
            or am_policy_context.get("wlServAreaRes")
            or mobility_summary.get("currentServAreaRes")
        )
        if service_area_restriction:
            summary["service_area_restriction"] = service_area_restriction
        return summary

    @staticmethod
    def _normalize_am_policy_candidates(candidates: Any) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if not isinstance(candidates, list):
            return normalized

        for item in candidates:
            if not isinstance(item, dict):
                continue
            candidate = dict(item)
            candidate["supi"] = str(candidate.get("supi") or "").strip()
            candidate["association_ids"] = [
                str(value or "").strip()
                for value in (candidate.get("association_ids") or [])
                if str(value or "").strip()
            ]
            candidate["match_reasons"] = [
                str(value or "").strip()
                for value in (candidate.get("match_reasons") or [])
                if str(value or "").strip()
            ]
            try:
                candidate["match_score"] = float(candidate.get("match_score") or 0.0)
            except (TypeError, ValueError):
                candidate["match_score"] = 0.0
            normalized.append(candidate)
        return normalized
