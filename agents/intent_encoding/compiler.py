from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from domain.policy_plan import FlowSelector, OperationIntent

from .contracts import FlowCandidateEvidence, IntentAdvisorDecision, IntentEvidence


class IntentCompiler:
    VALID_DOMAINS = {"qos", "mobility"}
    SM_GROUNDING_TOOLS = {"get_sm_ue_context", "get_sm_ue_flow_catalog", "search_sm_flow_targets"}
    AM_GROUNDING_TOOLS = {"get_am_policy_context", "search_am_policy_targets"}

    @classmethod
    def uses_sm_grounding(cls, requested_domains: List[str] | None) -> bool:
        normalized = {str(item or "").strip().lower() for item in (requested_domains or []) if str(item or "").strip()}
        return not normalized or "qos" in normalized

    @classmethod
    def uses_am_grounding(cls, requested_domains: List[str] | None) -> bool:
        normalized = {str(item or "").strip().lower() for item in (requested_domains or []) if str(item or "").strip()}
        return not normalized or "mobility" in normalized

    def extract_main_directives(self, context: str) -> Dict[str, Any]:
        text = str(context or "").strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}

        main_intent = payload.get("main_intent")
        if not isinstance(main_intent, dict):
            return {}

        requested_domains = self._normalize_requested_domains(main_intent.get("requested_domains"))
        domain_evidence = self._normalize_domain_evidence(main_intent.get("domain_evidence"))
        objective_profile = main_intent.get("objective_profile")
        objective_profile_hint = ""
        if isinstance(objective_profile, dict):
            objective_profile_hint = str(objective_profile.get("profile_name") or "").strip()

        return {
            "requested_domains": requested_domains,
            "domain_evidence": domain_evidence,
            "objective_profile_hint": objective_profile_hint,
        }

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
        requested_domains = self._normalize_requested_domains(main_directives.get("requested_domains"))
        # 关键步骤：根据 qos/mobility 只接收对应域的 grounding 证据，避免 SM/AM 结果互相污染。
        resolved_catalog_payload = dict(catalog_payload or {}) if self.uses_sm_grounding(requested_domains) else {}
        resolved_semantic_candidates = list(semantic_candidates or []) if self.uses_sm_grounding(requested_domains) else []
        resolved_am_context = dict(am_context_payload or {}) if self.uses_am_grounding(requested_domains) else {}
        resolved_am_policy_candidates = list(am_policy_candidates or []) if self.uses_am_grounding(requested_domains) else []

        app_catalog = resolved_catalog_payload.get("app_catalog") or []
        flow_catalog = resolved_catalog_payload.get("flow_catalog") or []
        explicit_flow_id = flow_match.group(1) if flow_match else ""
        explicit_app_id = app_match.group(1) if app_match else ""

        exact_app = self._match_exact_app_from_input(
            user_input=user_input,
            app_catalog=app_catalog,
            explicit_app_id=explicit_app_id,
        )

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
        elif exact_app is not None:
            matched_app_id = str(exact_app.get("app_id") or "").strip()
            for item in flow_catalog:
                if str(item.get("app_id") or "").strip() == matched_app_id:
                    candidate_flows.append(self._flow_candidate_from_catalog(item, score=0.9))

        if resolved_semantic_candidates:
            candidate_flows.extend(self._flow_candidates_from_semantic_matches(resolved_semantic_candidates))

        candidate_flows = self._deduplicate_flow_candidates(candidate_flows)
        candidate_apps = self._deduplicate_candidate_apps(
            candidate_apps=candidate_apps,
            candidate_flows=candidate_flows,
            app_catalog=app_catalog,
            semantic_candidates=resolved_semantic_candidates,
        )

        ambiguities: List[str] = []
        cache_hits: List[str] = []
        mobility_only = requested_domains == ["mobility"]
        if resolved_catalog_payload:
            cache_hits.append("ue_flow_catalog")
        if resolved_semantic_candidates:
            cache_hits.append("semantic_flow_search")
        if resolved_am_context:
            cache_hits.append("am_policy_context")
        if resolved_am_policy_candidates:
            cache_hits.append("am_policy_search")
        if not supi:
            ambiguities.append("missing supi")
        if not requested_domains:
            ambiguities.append("missing requested domains")
        if not mobility_only:
            if not candidate_flows:
                ambiguities.append("no candidate flow resolved")
            elif len(candidate_flows) > 1:
                ambiguities.append("multiple candidate flows remain")

        return IntentEvidence(
            user_input=user_input,
            supi=str(supi or "").strip(),
            requested_domains=requested_domains,
            explicit_app_id=explicit_app_id,
            explicit_flow_id=explicit_flow_id,
            candidate_flows=candidate_flows,
            candidate_apps=candidate_apps,
            ambiguities=ambiguities,
            cache_hits=cache_hits,
            operation_type_hint="modify",
            mobility_intent_hint={},
            objective_profile_hint=str(main_directives.get("objective_profile_hint") or "").strip(),
            domain_evidence=self._normalize_domain_evidence(main_directives.get("domain_evidence")),
            am_context_summary=self._summarize_am_context(resolved_am_context),
            am_policy_candidates=self._normalize_am_policy_candidates(resolved_am_policy_candidates),
            cached_catalog=resolved_catalog_payload,
            cached_semantic_candidates=resolved_semantic_candidates,
            cached_am_context=resolved_am_context,
            cached_am_policy_candidates=resolved_am_policy_candidates,
        )

    def validate_intent_grounding(
        self,
        *,
        evidence: IntentEvidence,
        grounding_tools: List[str],
    ) -> List[str]:
        errors: List[str] = []
        requested_domains = {str(item or "").strip().lower() for item in (evidence.requested_domains or []) if str(item or "").strip()}
        used_grounding_tools = {str(item or "").strip() for item in (grounding_tools or []) if str(item or "").strip()}
        if requested_domains == {"mobility"} and used_grounding_tools & self.SM_GROUNDING_TOOLS:
            errors.append("mobility-only intent must not call SM grounding tools")
        if requested_domains == {"qos"} and used_grounding_tools & self.AM_GROUNDING_TOOLS:
            errors.append("QoS-only intent must not call AM grounding tools")
        if list(evidence.requested_domains or []) == ["mobility"]:
            return errors
        has_grounding_source = bool(grounding_tools or evidence.cache_hits)
        if evidence.ambiguities and not has_grounding_source:
            errors.append("unresolved QoS intent requires cached evidence or at least one grounding tool call")
        return errors

    def validate_advisor_decision(
        self,
        *,
        evidence: IntentEvidence,
        decision: IntentAdvisorDecision,
    ) -> List[str]:
        errors: List[str] = []
        requested_domains = {str(item or "").strip().lower() for item in (evidence.requested_domains or []) if str(item or "").strip()}
        if requested_domains == {"mobility"} and decision.flows:
            errors.append("Mobility-only advisor decision must not include QoS flows.")
        if "qos" not in requested_domains:
            return errors

        if not decision.flows:
            errors.append("QoS advisor decision must include flows with the target SLA fields.")
            return errors

        for index, flow in enumerate(decision.flows):
            resolution_status = str(flow.resolution_status or "resolved").strip().lower() or "resolved"
            flow_id = str(flow.flow_id or "").strip()
            if resolution_status == "resolved" and not flow_id:
                errors.append(f"QoS advisor flow[{index}] is resolved but missing flow_id.")
        return errors

    def compile_operation_intent(
        self,
        *,
        evidence: IntentEvidence,
        advisor_decision: Optional[IntentAdvisorDecision],
        user_input: str,
        session_id: str,
        snapshot_id: str,
        main_directives: Optional[Dict[str, Any]] = None,
    ) -> OperationIntent:
        decision = advisor_decision or self._build_default_decision(evidence)
        directives = main_directives or {}

        catalog_payload = dict(evidence.cached_catalog or {})
        app_catalog = catalog_payload.get("app_catalog") or []
        flow_catalog = catalog_payload.get("flow_catalog") or []
        semantic_candidates = list(evidence.cached_semantic_candidates or [])

        selected_flow = self._resolve_selected_flow(
            evidence=evidence,
            decision=decision,
            flow_catalog=flow_catalog,
            semantic_candidates=semantic_candidates,
        )

        selected_app_id = str(decision.selected_app_id or "").strip()
        if not selected_app_id and decision.flows:
            for flow in decision.flows:
                candidate_app_id = str(flow.app_id or "").strip()
                if candidate_app_id:
                    selected_app_id = candidate_app_id
                    break
        if not selected_app_id and selected_flow is not None:
            selected_app_id = str(selected_flow.get("app_id") or "").strip()
        if not selected_app_id and len(evidence.candidate_apps) == 1:
            selected_app_id = str(evidence.candidate_apps[0].get("app_id") or "").strip()

        selected_app = next(
            (item for item in app_catalog if str(item.get("app_id") or "").strip() == selected_app_id),
            None,
        )
        app_name = None
        if selected_app is not None:
            app_name = str(selected_app.get("app_name") or "").strip() or None

        flows: List[FlowSelector] = []
        requested_domains = list(evidence.requested_domains or [])
        if requested_domains != ["mobility"]:
            if decision.flows:
                flows = self._build_operation_flows_from_advisor_decision(
                    advisor_flows=decision.flows,
                    evidence=evidence,
                    selected_app_id=selected_app_id,
                    flow_catalog=flow_catalog,
                    semantic_candidates=semantic_candidates,
                )
            elif selected_flow is not None:
                flow_selector = self._build_flow_selector_from_catalog(selected_flow)
                flow_selector.supi = evidence.supi
                flow_selector.app_id = selected_app_id or flow_selector.app_id
                flows.append(flow_selector)
            elif len(evidence.candidate_flows) > 1:
                for candidate in evidence.candidate_flows[:5]:
                    flows.append(
                        FlowSelector(
                            supi=candidate.supi or evidence.supi,
                            app_id=candidate.app_id,
                            flow_id=candidate.flow_id or None,
                            target_type="flow",
                            name=candidate.flow_name,
                            service_type=candidate.service_type,
                            service_type_id=candidate.service_type_id,
                            resolution_status="ambiguous",
                            resolution_candidates=[f"{candidate.app_id}/{candidate.flow_id}:{candidate.flow_name}"],
                        )
                    )

        resolution_status = "resolved"
        if requested_domains != ["mobility"] and not flows:
            resolution_status = "unmatched" if not evidence.candidate_flows else "ambiguous"
        if flows and any(flow.resolution_status == "ambiguous" for flow in flows):
            resolution_status = "ambiguous"

        resolved_mobility_intent: Dict[str, Any] = {}
        if isinstance(decision.mobility_intent, dict) and decision.mobility_intent:
            resolved_mobility_intent = dict(decision.mobility_intent)
        elif isinstance(evidence.mobility_intent_hint, dict) and evidence.mobility_intent_hint:
            resolved_mobility_intent = dict(evidence.mobility_intent_hint)

        return OperationIntent(
            session_id=str(session_id or "").strip(),
            snapshot_id=str(snapshot_id or "").strip(),
            supi=str(evidence.supi or "").strip(),
            app_id=selected_app_id,
            app_name=app_name,
            operation_type=str(decision.operation_type or evidence.operation_type_hint or "modify").strip() or "modify",
            urgency="Normal",
            raw_input=str(user_input or "").strip(),
            raw_intent_summary=str(decision.raw_intent_summary or user_input or "").strip(),
            resolution_status=resolution_status,
            requested_domains=self._normalize_requested_domains(
                evidence.requested_domains,
                authoritative_domains=directives.get("requested_domains"),
            ),
            domain_evidence=self._normalize_domain_evidence(
                directives.get("domain_evidence") or evidence.domain_evidence,
            ),
            mobility_intent=resolved_mobility_intent,
            objective_profile_hint=str(
                directives.get("objective_profile_hint")
                or decision.objective_profile_hint
                or evidence.objective_profile_hint
                or ""
            ).strip(),
            flows=flows,
        )

    @staticmethod
    def parse_json_payload_from_tool_result(content: Any, *, marker: str) -> Dict[str, Any]:
        text = str(content or "").strip()
        if not text:
            return {}
        payload_text = text
        marker_index = text.find(marker)
        if marker_index >= 0:
            payload_text = text[marker_index + len(marker):].strip()
        start = payload_text.find("{")
        end = payload_text.rfind("}")
        if start < 0 or end < start:
            return {}
        try:
            payload = json.loads(payload_text[start : end + 1])
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _build_default_decision(self, evidence: IntentEvidence) -> IntentAdvisorDecision:
        selected_app_id = evidence.explicit_app_id
        if not selected_app_id and len(evidence.candidate_apps) == 1:
            selected_app_id = str(evidence.candidate_apps[0].get("app_id") or "").strip()
        selected_flow_id = evidence.explicit_flow_id
        if not selected_flow_id and len(evidence.candidate_flows) == 1:
            selected_flow_id = str(evidence.candidate_flows[0].flow_id or "").strip()

        return IntentAdvisorDecision(
            selected_app_id=selected_app_id,
            selected_flow_id=selected_flow_id,
            operation_type=evidence.operation_type_hint,
            raw_intent_summary=evidence.user_input,
            rationale="",
            mobility_intent=dict(evidence.mobility_intent_hint or {}),
            objective_profile_hint=evidence.objective_profile_hint,
        )

    def _resolve_selected_flow(
        self,
        *,
        evidence: IntentEvidence,
        decision: IntentAdvisorDecision,
        flow_catalog: List[Dict[str, Any]],
        semantic_candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        selected_flow_id = str(decision.selected_flow_id or "").strip()
        if selected_flow_id:
            selected_flow = next(
                (item for item in flow_catalog if str(item.get("flow_id") or "").strip() == selected_flow_id),
                None,
            )
            if selected_flow is not None:
                return selected_flow
            return next(
                (item for item in semantic_candidates if str(item.get("flow_id") or "").strip() == selected_flow_id),
                None,
            )

        if len(evidence.candidate_flows) != 1:
            return None

        candidate_flow_id = evidence.candidate_flows[0].flow_id
        selected_flow = next(
            (item for item in flow_catalog if str(item.get("flow_id") or "").strip() == candidate_flow_id),
            None,
        )
        if selected_flow is not None:
            return selected_flow
        return next(
            (item for item in semantic_candidates if str(item.get("flow_id") or "").strip() == candidate_flow_id),
            None,
        )

    def _build_operation_flows_from_advisor_decision(
        self,
        *,
        advisor_flows: List[FlowSelector],
        evidence: IntentEvidence,
        selected_app_id: str,
        flow_catalog: List[Dict[str, Any]],
        semantic_candidates: List[Dict[str, Any]],
    ) -> List[FlowSelector]:
        resolved: List[FlowSelector] = []
        for advisor_flow in advisor_flows:
            catalog_flow = self._lookup_flow_record(
                flow_id=str(advisor_flow.flow_id or "").strip(),
                flow_catalog=flow_catalog,
                semantic_candidates=semantic_candidates,
            )
            merged_payload: Dict[str, Any] = {}
            if catalog_flow is not None:
                merged_payload = self._build_flow_selector_from_catalog(catalog_flow).model_dump(mode="json")

            advisor_payload = advisor_flow.model_dump(mode="json")
            for key, value in advisor_payload.items():
                if isinstance(value, str):
                    if value.strip():
                        merged_payload[key] = value
                elif isinstance(value, list):
                    if value:
                        merged_payload[key] = list(value)
                elif value is not None:
                    merged_payload[key] = value

            merged_payload["supi"] = str(
                merged_payload.get("supi")
                or evidence.supi
                or ""
            ).strip()
            merged_payload["app_id"] = str(
                merged_payload.get("app_id")
                or selected_app_id
                or ""
            ).strip()
            if not str(merged_payload.get("name") or "").strip():
                merged_payload["name"] = str(merged_payload.get("flow_id") or "").strip()
            if not str(merged_payload.get("resolution_status") or "").strip():
                merged_payload["resolution_status"] = "resolved" if str(merged_payload.get("flow_id") or "").strip() else "unmatched"

            resolved.append(FlowSelector.model_validate(merged_payload))
        return resolved

    @staticmethod
    def _lookup_flow_record(
        *,
        flow_id: str,
        flow_catalog: List[Dict[str, Any]],
        semantic_candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        normalized_flow_id = str(flow_id or "").strip()
        if not normalized_flow_id:
            return None
        catalog_match = next(
            (item for item in flow_catalog if str(item.get("flow_id") or "").strip() == normalized_flow_id),
            None,
        )
        if catalog_match is not None:
            return catalog_match
        return next(
            (item for item in semantic_candidates if str(item.get("flow_id") or "").strip() == normalized_flow_id),
            None,
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
        return FlowCandidateEvidence(
            supi=str(flow.get("supi") or "").strip(),
            app_id=str(flow.get("app_id") or "").strip(),
            app_name=str(flow.get("app_name") or "").strip() or None,
            flow_id=str(flow.get("flow_id") or "").strip(),
            flow_name=str(flow.get("flow_name") or "").strip(),
            service_type=str(flow.get("service_type") or "").strip() or None,
            service_type_id=flow.get("service_type_id"),
            score=score,
        )

    @staticmethod
    def _build_flow_selector_from_catalog(flow: Dict[str, Any]) -> FlowSelector:
        five_tuple = flow.get("five_tuple")
        return FlowSelector(
            supi=str(flow.get("supi") or "").strip(),
            app_id=str(flow.get("app_id") or "").strip(),
            flow_id=str(flow.get("flow_id") or "").strip() or None,
            target_type="flow",
            name=str(flow.get("flow_name") or flow.get("flow_id") or "").strip(),
            service_type=str(flow.get("service_type") or "").strip() or None,
            service_type_id=flow.get("service_type_id"),
            bw_ul=flow.get("bw_ul"),
            bw_dl=flow.get("bw_dl"),
            gbr_ul=flow.get("gbr_ul"),
            gbr_dl=flow.get("gbr_dl"),
            lat=flow.get("lat"),
            loss_req=flow.get("loss_req"),
            jitter_req=flow.get("jitter_req"),
            priority=flow.get("priority"),
            description=str(flow.get("flow_name") or "").strip() or None,
            five_tuple=list(five_tuple) if isinstance(five_tuple, (list, tuple)) else None,
            current_bw_ul=flow.get("current_bw_ul"),
            current_bw_dl=flow.get("current_bw_dl"),
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

    def _normalize_requested_domains(
        self,
        requested_domains: Any,
        *,
        authoritative_domains: Optional[List[str]] = None,
    ) -> List[str]:
        authoritative = [
            str(item or "").strip().lower()
            for item in (authoritative_domains or [])
            if str(item or "").strip()
        ]
        authoritative = [item for item in authoritative if item in self.VALID_DOMAINS]
        if authoritative:
            return list(dict.fromkeys(authoritative))

        normalized = [
            str(item or "").strip().lower()
            for item in (requested_domains or [])
            if str(item or "").strip()
        ]
        valid = [item for item in normalized if item in self.VALID_DOMAINS]
        return list(dict.fromkeys(valid))

    @staticmethod
    def _normalize_domain_evidence(domain_evidence: Any) -> Dict[str, List[str]]:
        normalized: Dict[str, List[str]] = {}
        if not isinstance(domain_evidence, dict):
            return normalized
        for key, values in domain_evidence.items():
            items = [str(item or "").strip() for item in (values or []) if str(item or "").strip()]
            if items:
                normalized[str(key).strip().lower()] = items
        return normalized
