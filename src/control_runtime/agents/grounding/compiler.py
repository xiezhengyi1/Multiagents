from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ...domain.control_plane import ControlSemantics, ControlStage, SemanticTarget, SemanticTargetType
from ...domain.policy_plan import FlowSelector, GroundingEvidenceBundle, OperationIntent, QosTargetEnvelope

from .contracts import FlowCandidateEvidence, IntentAdvisorDecision, IntentEvidence


class IntentCompiler:
    VALID_DOMAINS = {"qos", "mobility"}
    SM_GROUNDING_TOOLS = {"get_sm_ue_context", "get_sm_ue_flow_catalog", "search_sm_flow_targets", "get_ue_flow_catalog", "search_flow_targets_by_name"}
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
            "control_semantics": main_intent.get("control_semantics") if isinstance(main_intent.get("control_semantics"), dict) else {},
            "objective_profile_hint": objective_profile_hint,
            "supi": str(main_intent.get("supi") or "").strip(),
            "retry_scope": str(main_intent.get("retry_scope") or "").strip(),
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
        explicit_app_name = str(exact_app.get("app_name") or "").strip() if exact_app is not None else ""
        exact_flow = self._match_exact_flow_from_input(
            user_input=user_input,
            flow_catalog=flow_catalog,
            explicit_flow_id=explicit_flow_id,
        )
        explicit_flow_name = self._extract_explicit_flow_name(
            user_input=user_input,
            flow_catalog=flow_catalog,
            exact_flow=exact_flow,
        )
        strict_explicit_flow_request = bool(explicit_flow_id or explicit_flow_name)

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
        elif exact_flow is not None:
            candidate_flows.append(self._flow_candidate_from_catalog(exact_flow, score=1.0))
        elif exact_app is not None and not explicit_flow_name:
            matched_app_id = str(exact_app.get("app_id") or "").strip()
            for item in flow_catalog:
                if str(item.get("app_id") or "").strip() == matched_app_id:
                    candidate_flows.append(self._flow_candidate_from_catalog(item, score=0.9))

        if resolved_semantic_candidates and not strict_explicit_flow_request:
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
                if explicit_flow_name:
                    ambiguities.append(f"explicit flow name not grounded: {explicit_flow_name}")
            elif len(candidate_flows) > 1:
                ambiguities.append("multiple candidate flows remain")

        return IntentEvidence(
            user_input=user_input,
            supi=str(supi or "").strip(),
            requested_domains=requested_domains,
            retry_scope=str(main_directives.get("retry_scope") or "").strip(),
            explicit_app_id=explicit_app_id,
            explicit_app_name=explicit_app_name,
            explicit_flow_id=explicit_flow_id,
            explicit_flow_name=explicit_flow_name,
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
        if requested_domains == {"mobility"} and (used_grounding_tools & self.SM_GROUNDING_TOOLS):
            errors.append("mobility-only intent must not call SM grounding tools")
        if requested_domains == {"qos"} and (used_grounding_tools & self.AM_GROUNDING_TOOLS):
            errors.append("QoS-only intent must not call AM grounding tools")
        if list(evidence.requested_domains or []) == ["mobility"]:
            if not evidence.am_context_summary and "am_policy_context" not in {str(item or "").strip().lower() for item in (evidence.cache_hits or [])}:
                errors.append("mobility-only intent requires grounded AM policy context before returning final intent")
            if self._mobility_request_mentions_specific_targets(evidence.user_input):
                has_am_target_evidence = bool(evidence.am_policy_candidates) or (
                    "am_policy_search" in {str(item or "").strip().lower() for item in (evidence.cache_hits or [])}
                )
                if not has_am_target_evidence:
                    errors.append(
                        "mobility intent that names association/RFSP/NSSAI/service-area/access targets requires search_am_policy_targets evidence"
                    )
            return errors
        named_flow_request = (
            "/" in str(evidence.user_input or "")
            and not str(evidence.explicit_app_id or "").strip()
            and not str(evidence.explicit_flow_id or "").strip()
            and not str(evidence.explicit_flow_name or "").strip()
        )
        if "qos" in requested_domains and str(evidence.explicit_flow_name or "").strip() and not evidence.candidate_flows:
            errors.append(
                f"explicitly named QoS flow '{evidence.explicit_flow_name}' was not grounded by catalog/search evidence"
            )
        if (
            "qos" in requested_domains
            and named_flow_request
            and not evidence.candidate_flows
            and "search_sm_flow_targets" not in used_grounding_tools
            and "semantic_flow_search" not in {str(item or "").strip().lower() for item in (evidence.cache_hits or [])}
        ):
            errors.append("named QoS flow request requires search_sm_flow_targets before returning final intent")
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
            errors.append("QoS advisor decision must include grounded target flows.")
            return errors

        for index, flow in enumerate(decision.flows):
            resolution_status = str(flow.resolution_status or "resolved").strip().lower() or "resolved"
            flow_id = str(flow.flow_id or "").strip()
            if resolution_status == "resolved" and not flow_id:
                errors.append(f"QoS advisor flow[{index}] is resolved but missing flow_id.")
            if resolution_status == "resolved" and flow_id and not self._flow_id_is_grounded(
                flow_id=flow_id,
                evidence=evidence,
            ):
                errors.append(
                    f"QoS advisor flow[{index}] resolved flow_id={flow_id} is not grounded by catalog/search evidence."
                )
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
        selected_app_id = self._resolve_selected_app_id(
            evidence=evidence,
            decision=decision,
            selected_flow=selected_flow,
        )
        app_name = self._resolve_selected_app_name(app_catalog=app_catalog, selected_app_id=selected_app_id)
        flows = self._build_compiled_flows(
            evidence=evidence,
            decision=decision,
            selected_app_id=selected_app_id,
            selected_flow=selected_flow,
            flow_catalog=flow_catalog,
            semantic_candidates=semantic_candidates,
        )
        resolution_status = self._resolve_operation_status(
            requested_domains=evidence.requested_domains,
            candidate_flows=evidence.candidate_flows,
            flows=flows,
        )
        resolved_mobility_intent = self._resolve_mobility_intent(
            decision=decision,
            evidence=evidence,
        )
        qos_target_envelopes = self._build_qos_target_envelopes(
            user_input=user_input,
            flows=flows,
            domain_evidence=self._normalize_domain_evidence(
                directives.get("domain_evidence") or evidence.domain_evidence,
            ),
        )
        control_semantics = self._ground_control_semantics(
            directives=directives,
            flows=flows,
            flow_catalog=flow_catalog,
            app_catalog=app_catalog,
        )

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
            control_semantics=control_semantics,
            mobility_intent=resolved_mobility_intent,
            objective_profile_hint=str(
                directives.get("objective_profile_hint")
                or decision.objective_profile_hint
                or evidence.objective_profile_hint
                or ""
            ).strip(),
            grounding_evidence=self._build_grounding_evidence(
                evidence=evidence,
                directives=directives,
                selected_app_id=selected_app_id,
                flows=flows,
            ),
            flows=flows,
            qos_target_envelopes=qos_target_envelopes,
        )

    def _ground_control_semantics(
        self,
        *,
        directives: Dict[str, Any],
        flows: List[FlowSelector],
        flow_catalog: List[Dict[str, Any]],
        app_catalog: List[Dict[str, Any]],
    ) -> ControlSemantics:
        raw_semantics = directives.get("control_semantics")
        if not isinstance(raw_semantics, dict):
            return ControlSemantics()
        semantics = ControlSemantics.model_validate(raw_semantics)
        if not semantics.stages:
            return semantics

        flow_rows = [self._flow_selector_to_row(flow) for flow in flows]
        all_flow_rows = flow_rows + [
            item for item in (flow_catalog or []) if isinstance(item, dict)
        ]
        app_rows = [item for item in (app_catalog or []) if isinstance(item, dict)]

        grounded_stages: List[ControlStage] = []
        prior_active_flow_ids: List[str] = []
        for stage in semantics.stages:
            grounded_targets: List[SemanticTarget] = []
            active_flow_ids: List[str] = []
            active_app_ids: List[str] = []
            for target in stage.targets:
                grounded_target = target.model_copy(deep=True)
                matched_flows = self._match_semantic_target_flows(
                    target=target,
                    flow_rows=all_flow_rows,
                )
                matched_apps = self._match_semantic_target_apps(
                    target=target,
                    app_rows=app_rows,
                )
                grounded_target.matched_flow_ids = [
                    str(item.get("flow_id") or "").strip()
                    for item in matched_flows
                    if str(item.get("flow_id") or "").strip()
                ]
                grounded_target.matched_app_ids = [
                    str(item.get("app_id") or "").strip()
                    for item in matched_apps
                    if str(item.get("app_id") or "").strip()
                ]
                if grounded_target.target_type == SemanticTargetType.SCOPE and grounded_target.semantic_name == "其余业务":
                    grounded_target.matched_flow_ids = [
                        str(item.get("flow_id") or "").strip()
                        for item in all_flow_rows
                        if str(item.get("flow_id") or "").strip()
                        and str(item.get("flow_id") or "").strip() not in prior_active_flow_ids
                    ]
                if len(grounded_target.matched_flow_ids) == 1:
                    flow_row = matched_flows[0]
                    grounded_target.flow_id = grounded_target.matched_flow_ids[0]
                    grounded_target.flow_name = str(flow_row.get("flow_name") or grounded_target.semantic_name).strip()
                    grounded_target.app_id = str(flow_row.get("app_id") or "").strip()
                    grounded_target.app_name = str(flow_row.get("app_name") or "").strip()
                    grounded_target.supi = str(flow_row.get("supi") or "").strip()
                    grounded_target.resolution_status = "grounded_flow"
                elif len(grounded_target.matched_app_ids) == 1:
                    app_row = matched_apps[0]
                    grounded_target.app_id = grounded_target.matched_app_ids[0]
                    grounded_target.app_name = str(app_row.get("app_name") or grounded_target.semantic_name).strip()
                    grounded_target.resolution_status = "grounded_app"
                    grounded_target.matched_flow_ids = [
                        str(item.get("flow_id") or "").strip()
                        for item in all_flow_rows
                        if str(item.get("app_id") or "").strip() == grounded_target.app_id
                        and str(item.get("flow_id") or "").strip()
                    ]
                else:
                    grounded_target.resolution_status = "semantic"
                grounded_targets.append(grounded_target)
                for flow_id in grounded_target.matched_flow_ids:
                    if flow_id and flow_id not in active_flow_ids:
                        active_flow_ids.append(flow_id)
                for app_id in grounded_target.matched_app_ids:
                    if app_id and app_id not in active_app_ids:
                        active_app_ids.append(app_id)
            grounded_stages.append(
                stage.model_copy(
                    update={
                        "targets": grounded_targets,
                        "active_flow_ids": active_flow_ids,
                        "active_app_ids": active_app_ids,
                    },
                    deep=True,
                )
            )
            for flow_id in active_flow_ids:
                if flow_id and flow_id not in prior_active_flow_ids:
                    prior_active_flow_ids.append(flow_id)
        return semantics.model_copy(update={"stages": grounded_stages}, deep=True)

    @staticmethod
    def _flow_selector_to_row(flow: FlowSelector) -> Dict[str, Any]:
        return {
            "flow_id": str(flow.flow_id or "").strip(),
            "flow_name": str(flow.name or "").strip(),
            "app_id": str(flow.app_id or "").strip(),
            "app_name": "",
            "supi": str(flow.supi or "").strip(),
        }

    def _match_semantic_target_flows(
        self,
        *,
        target: SemanticTarget,
        flow_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        semantic_name = str(target.semantic_name or "").strip()
        if not semantic_name:
            return []
        normalized_name = self._normalize_semantic_name(semantic_name)
        matches: List[Dict[str, Any]] = []
        for row in flow_rows:
            flow_name = str(row.get("flow_name") or row.get("name") or "").strip()
            if not flow_name:
                continue
            normalized_flow = self._normalize_semantic_name(flow_name)
            if normalized_flow == normalized_name:
                matches.append(row)
        return self._deduplicate_flow_rows(matches)

    def _match_semantic_target_apps(
        self,
        *,
        target: SemanticTarget,
        app_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        semantic_name = str(target.semantic_name or "").strip()
        if not semantic_name:
            return []
        normalized_name = self._normalize_semantic_name(semantic_name)
        matches: List[Dict[str, Any]] = []
        for row in app_rows:
            app_name = str(row.get("app_name") or row.get("name") or "").strip()
            if not app_name:
                continue
            normalized_app = self._normalize_semantic_name(app_name)
            if normalized_app == normalized_name:
                matches.append(row)
        return self._deduplicate_app_rows(matches)

    @staticmethod
    def _normalize_semantic_name(value: str) -> str:
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").lower())

    @staticmethod
    def _deduplicate_flow_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str]] = set()
        for row in rows:
            key = (str(row.get("app_id") or "").strip(), str(row.get("flow_id") or "").strip())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped

    @staticmethod
    def _deduplicate_app_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            app_id = str(row.get("app_id") or "").strip()
            if not app_id or app_id in seen:
                continue
            seen.add(app_id)
            deduped.append(row)
        return deduped

    def _build_qos_target_envelopes(
        self,
        *,
        user_input: str,
        flows: List[FlowSelector],
        domain_evidence: Dict[str, List[str]],
    ) -> List[QosTargetEnvelope]:
        if not flows:
            return []
        request_signals = self._extract_qos_request_signals(
            user_input=user_input,
            domain_evidence=domain_evidence,
        )
        envelopes: List[QosTargetEnvelope] = []
        for flow in flows:
            flow_id = str(flow.flow_id or "").strip()
            if not flow_id or str(flow.resolution_status or "").strip().lower() != "resolved":
                continue
            envelopes.append(
                QosTargetEnvelope(
                    flow_id=flow_id,
                    app_id=str(flow.app_id or "").strip(),
                    flow_name=str(flow.name or flow_id).strip(),
                    baseline_priority=flow.priority,
                    baseline_latency_ms=flow.lat,
                    baseline_jitter_ms=flow.jitter_req,
                    baseline_packet_error_rate=flow.loss_req,
                    baseline_max_br_ul_mbps=flow.bw_ul,
                    baseline_max_br_dl_mbps=flow.bw_dl,
                    baseline_gbr_ul_mbps=flow.gbr_ul,
                    baseline_gbr_dl_mbps=flow.gbr_dl,
                    strictest_priority=flow.priority,
                    strictest_latency_ms=self._derive_strictest_latency(flow.lat, request_signals),
                    strictest_jitter_ms=self._derive_strictest_jitter(flow.jitter_req, request_signals),
                    strictest_packet_error_rate=self._derive_strictest_loss(flow.loss_req, request_signals),
                    strictest_max_br_ul_mbps=self._derive_strictest_bandwidth(flow.bw_ul, request_signals, direction="ul"),
                    strictest_max_br_dl_mbps=self._derive_strictest_bandwidth(flow.bw_dl, request_signals, direction="dl"),
                    strictest_gbr_ul_mbps=self._derive_strictest_gbr(flow.gbr_ul, request_signals, direction="ul"),
                    strictest_gbr_dl_mbps=self._derive_strictest_gbr(flow.gbr_dl, request_signals, direction="dl"),
                    rationale=self._build_qos_objective_rationale(flow=flow, request_signals=request_signals),
                )
            )
        return envelopes

    @staticmethod
    def _extract_qos_request_signals(
        *,
        user_input: str,
        domain_evidence: Dict[str, List[str]],
    ) -> Dict[str, bool]:
        qos_evidence = " ".join(domain_evidence.get("qos") or [])
        text = f"{user_input} {qos_evidence}".lower()
        return {
            "latency": any(token in text for token in ("latency", "delay", "时延", "延迟", "低时延", "urlcc")),
            "jitter": any(token in text for token in ("jitter", "抖动", "稳定", "连续性", "continuity")),
            "reliability": any(token in text for token in ("reliability", "稳定", "可靠", "控制稳定", "loss", "丢包", "packet error")),
            "throughput": any(token in text for token in ("throughput", "bandwidth", "吞吐", "带宽")),
            "uplink_only": any(token in text for token in ("uplink", "上行", "ul")),
            "downlink_only": any(token in text for token in ("downlink", "下行", "dl")),
            "strong_control": any(token in text for token in ("control", "控制", "urllc", "industrial", "robot", "drone", "medical", "telemedicine")),
        }

    @staticmethod
    def _derive_strictest_latency(baseline: Optional[float], request_signals: Dict[str, bool]) -> Optional[float]:
        if baseline is None:
            return None
        factor = 1.0
        if request_signals.get("latency"):
            factor = 0.9
        if request_signals.get("strong_control"):
            factor = min(factor, 0.85)
        return round(max(baseline * factor, 1.0), 3)

    @staticmethod
    def _derive_strictest_jitter(baseline: Optional[float], request_signals: Dict[str, bool]) -> Optional[float]:
        if baseline is None:
            return None
        factor = 1.0
        if request_signals.get("jitter") or request_signals.get("reliability"):
            factor = 0.9
        if request_signals.get("strong_control"):
            factor = min(factor, 0.85)
        return round(max(baseline * factor, 0.0), 3)

    @staticmethod
    def _derive_strictest_loss(baseline: Optional[float], request_signals: Dict[str, bool]) -> Optional[float]:
        if baseline is None:
            return None
        factor = 1.0
        if request_signals.get("reliability") or request_signals.get("jitter"):
            factor = 0.9
        if request_signals.get("strong_control"):
            factor = min(factor, 0.85)
        return round(max(baseline * factor, 0.0), 6)

    @staticmethod
    def _derive_strictest_bandwidth(
        baseline: Optional[float],
        request_signals: Dict[str, bool],
        *,
        direction: str,
    ) -> Optional[float]:
        if baseline is None:
            return None
        if not request_signals.get("throughput"):
            return round(max(baseline, 0.0), 3)
        if request_signals.get("uplink_only") and direction == "dl":
            return round(max(baseline, 0.0), 3)
        if request_signals.get("downlink_only") and direction == "ul":
            return round(max(baseline, 0.0), 3)
        factor = 1.1
        if request_signals.get("strong_control"):
            factor = 1.05
        return round(max(baseline * factor, 0.0), 3)

    @classmethod
    def _derive_strictest_gbr(
        cls,
        baseline: Optional[float],
        request_signals: Dict[str, bool],
        *,
        direction: str,
    ) -> Optional[float]:
        if baseline is None:
            return None
        return cls._derive_strictest_bandwidth(baseline, request_signals, direction=direction)

    @staticmethod
    def _build_qos_objective_rationale(
        *,
        flow: FlowSelector,
        request_signals: Dict[str, bool],
    ) -> List[str]:
        reasons = [f"grounded_from_flow:{str(flow.flow_id or '').strip()}"]
        if request_signals.get("latency"):
            reasons.append("user_requests_lower_latency")
        if request_signals.get("jitter"):
            reasons.append("user_requests_lower_jitter_or_stability")
        if request_signals.get("reliability"):
            reasons.append("user_requests_higher_reliability")
        if request_signals.get("throughput"):
            reasons.append("user_requests_higher_throughput")
        if request_signals.get("strong_control"):
            reasons.append("request_has_control_or_medical_criticality")
        if len(reasons) == 1:
            reasons.append("preserve_grounded_baseline_without_intensification")
        return reasons

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

    @staticmethod
    def _resolve_selected_app_name(*, app_catalog: List[Dict[str, Any]], selected_app_id: str) -> Optional[str]:
        selected_app = next(
            (item for item in app_catalog if str(item.get("app_id") or "").strip() == selected_app_id),
            None,
        )
        if selected_app is None:
            return None
        return str(selected_app.get("app_name") or "").strip() or None

    @staticmethod
    def _resolve_mobility_intent(
        *,
        decision: IntentAdvisorDecision,
        evidence: IntentEvidence,
    ) -> Dict[str, Any]:
        if isinstance(decision.mobility_intent, dict) and decision.mobility_intent:
            return dict(decision.mobility_intent)
        if isinstance(evidence.mobility_intent_hint, dict) and evidence.mobility_intent_hint:
            return dict(evidence.mobility_intent_hint)
        return {}

    def _resolve_selected_app_id(
        self,
        *,
        evidence: IntentEvidence,
        decision: IntentAdvisorDecision,
        selected_flow: Optional[Dict[str, Any]],
    ) -> str:
        selected_app_id = str(decision.selected_app_id or "").strip()
        if not selected_app_id and decision.flows:
            for flow in decision.flows:
                candidate_app_id = str(flow.app_id or "").strip()
                if candidate_app_id:
                    return candidate_app_id
        if not selected_app_id and selected_flow is not None:
            return str(selected_flow.get("app_id") or "").strip()
        if not selected_app_id and len(evidence.candidate_apps) == 1:
            return str(evidence.candidate_apps[0].get("app_id") or "").strip()
        return selected_app_id

    def _build_compiled_flows(
        self,
        *,
        evidence: IntentEvidence,
        decision: IntentAdvisorDecision,
        selected_app_id: str,
        selected_flow: Optional[Dict[str, Any]],
        flow_catalog: List[Dict[str, Any]],
        semantic_candidates: List[Dict[str, Any]],
    ) -> List[FlowSelector]:
        if list(evidence.requested_domains or []) == ["mobility"]:
            return []
        if decision.flows:
            return self._build_operation_flows_from_advisor_decision(
                advisor_flows=decision.flows,
                evidence=evidence,
                selected_app_id=selected_app_id,
                flow_catalog=flow_catalog,
                semantic_candidates=semantic_candidates,
            )
        if selected_flow is not None:
            flow_selector = self._build_flow_selector_from_catalog(selected_flow)
            flow_selector.supi = evidence.supi
            flow_selector.app_id = selected_app_id or flow_selector.app_id
            return [flow_selector]
        if len(evidence.candidate_flows) <= 1:
            return []
        return [
            FlowSelector(
                supi=candidate.supi or evidence.supi,
                app_id=candidate.app_id,
                app_name=candidate.app_name,
                flow_id=candidate.flow_id or None,
                target_type="flow",
                name=candidate.flow_name,
                service_type=candidate.service_type,
                service_type_id=candidate.service_type_id,
                resolution_status="ambiguous",
                resolution_candidates=[f"{candidate.app_id}/{candidate.flow_id}:{candidate.flow_name}"],
            )
            for candidate in evidence.candidate_flows[:5]
        ]

    @staticmethod
    def _resolve_operation_status(
        *,
        requested_domains: List[str],
        candidate_flows: List[FlowCandidateEvidence],
        flows: List[FlowSelector],
    ) -> str:
        if list(requested_domains or []) == ["mobility"]:
            return "resolved"
        if any(flow.resolution_status == "ambiguous" for flow in flows):
            return "ambiguous"
        if flows:
            return "resolved"
        return "unmatched" if not candidate_flows else "ambiguous"

    def _build_grounding_evidence(
        self,
        *,
        evidence: IntentEvidence,
        directives: Dict[str, Any],
        selected_app_id: str,
        flows: List[FlowSelector],
    ) -> GroundingEvidenceBundle:
        selected_flow_ids = {
            str(flow.flow_id or "").strip()
            for flow in flows
            if str(flow.flow_id or "").strip()
        }
        evidence_sources = {
            "domain_routing": list(self._normalize_domain_evidence(directives.get("domain_evidence") or evidence.domain_evidence).keys()),
            "cache_hits": list(evidence.cache_hits or []),
            "grounding_tools": [
                item
                for item in [
                    "sm_grounding" if self.uses_sm_grounding(evidence.requested_domains) else "",
                    "am_grounding" if self.uses_am_grounding(evidence.requested_domains) else "",
                ]
                if item
            ],
        }
        evidence_sources = {
            key: [str(item).strip() for item in values if str(item).strip()]
            for key, values in evidence_sources.items()
            if values
        }
        return GroundingEvidenceBundle(
            grounded_supi=str(evidence.supi or "").strip(),
            grounded_apps=[
                {
                    "app_id": str(item.get("app_id") or "").strip(),
                    "app_name": str(item.get("app_name") or "").strip(),
                    "selected": str(item.get("app_id") or "").strip() == selected_app_id,
                }
                for item in (evidence.candidate_apps or [])
            ],
            grounded_flows=[
                {
                    "supi": candidate.supi,
                    "app_id": candidate.app_id,
                    "app_name": candidate.app_name,
                    "flow_id": str(candidate.flow_id or "").strip(),
                    "flow_name": candidate.flow_name,
                    "service_type": candidate.service_type,
                    "service_type_id": candidate.service_type_id,
                    "score": candidate.score,
                    "selected": str(candidate.flow_id or "").strip() in selected_flow_ids,
                }
                for candidate in evidence.candidate_flows
            ],
            grounded_mobility_targets={
                "summary": dict(evidence.am_context_summary or {}),
                "candidates": list(evidence.am_policy_candidates or []),
            },
            evidence_sources=evidence_sources,
            unresolved_ambiguities=list(evidence.ambiguities or []),
            rejected_hypotheses=[],
            cache_hits=list(evidence.cache_hits or []),
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
            return self._lookup_flow_record(
                flow_id=selected_flow_id,
                flow_catalog=flow_catalog,
                semantic_candidates=semantic_candidates,
            )

        if len(evidence.candidate_flows) != 1:
            return None

        return self._lookup_flow_record(
            flow_id=evidence.candidate_flows[0].flow_id,
            flow_catalog=flow_catalog,
            semantic_candidates=semantic_candidates,
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
            advisor_flow_id = str(advisor_flow.flow_id or "").strip()
            resolution_status = str(advisor_flow.resolution_status or "resolved").strip().lower() or "resolved"
            if resolution_status == "resolved" and advisor_flow_id and not self._flow_id_is_grounded(
                flow_id=advisor_flow_id,
                evidence=evidence,
            ):
                raise ValueError(f"resolved advisor flow_id is not grounded by evidence: {advisor_flow_id}")
            catalog_flow = self._lookup_flow_record(
                flow_id=advisor_flow_id,
                flow_catalog=flow_catalog,
                semantic_candidates=semantic_candidates,
            )
            resolved.append(
                FlowSelector.model_validate(
                    self._merge_advisor_flow_payload(
                        advisor_flow=advisor_flow,
                        catalog_flow=catalog_flow,
                        supi=evidence.supi,
                        selected_app_id=selected_app_id,
                    )
                )
            )
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

    def _merge_advisor_flow_payload(
        self,
        *,
        advisor_flow: FlowSelector,
        catalog_flow: Optional[Dict[str, Any]],
        supi: str,
        selected_app_id: str,
    ) -> Dict[str, Any]:
        merged_payload: Dict[str, Any] = {}
        if catalog_flow is not None:
            merged_payload = self._build_flow_selector_from_catalog(catalog_flow).model_dump(mode="json")
        semantic_keys = {
            "supi",
            "app_id",
            "app_name",
            "flow_id",
            "target_type",
            "name",
            "service_type",
            "service_type_id",
            "description",
            "resolution_status",
            "resolution_candidates",
        }
        for key, value in advisor_flow.model_dump(mode="json").items():
            if key not in semantic_keys:
                continue
            if isinstance(value, str):
                if value.strip():
                    merged_payload[key] = value
            elif isinstance(value, list):
                if value:
                    merged_payload[key] = list(value)
            elif value is not None:
                merged_payload[key] = value
        merged_payload["supi"] = str(merged_payload.get("supi") or supi or "").strip()
        merged_payload["app_id"] = str(merged_payload.get("app_id") or selected_app_id or "").strip()
        if not str(merged_payload.get("name") or "").strip():
            merged_payload["name"] = str(merged_payload.get("flow_id") or "").strip()
        if not str(merged_payload.get("resolution_status") or "").strip():
            merged_payload["resolution_status"] = "resolved" if str(merged_payload.get("flow_id") or "").strip() else "unmatched"
        return merged_payload

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
    def _extract_explicit_flow_name(
        cls,
        *,
        user_input: str,
        flow_catalog: List[Dict[str, Any]],
        exact_flow: Optional[Dict[str, Any]],
    ) -> str:
        if exact_flow is not None:
            return str(exact_flow.get("flow_name") or "").strip()

        text = str(user_input or "")
        if not text:
            return ""
        catalog_flow_names = {
            str(item.get("flow_name") or "").strip()
            for item in flow_catalog
            if isinstance(item, dict) and str(item.get("flow_name") or "").strip()
        }
        token_pattern = r"\b(?!imsi-)(?!app-)(?!flow-)[A-Za-z][A-Za-z0-9_]*_[A-Za-z0-9_]*\b"
        matched_tokens = [token.strip() for token in re.findall(token_pattern, text) if token.strip()]
        for token in matched_tokens:
            if token in catalog_flow_names:
                return token
        for token in matched_tokens:
            lowered = token.lower()
            if lowered in {"urllc", "embb", "mmtc"}:
                continue
            return token
        return ""

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

    @staticmethod
    def _mobility_request_mentions_specific_targets(user_input: str) -> bool:
        text = str(user_input or "").strip().lower()
        if not text:
            return False
        patterns = (
            r"\bassociation\b",
            r"\brfsp\b",
            r"\bnssai\b",
            r"\bs-nssai\b",
            r"\ballowed\s*nssai\b",
            r"\btarget\s*nssai\b",
            r"\bservice[-\s]?area\b",
            r"\bpresence\s*area\b",
            r"\baccess\s*type\b",
            r"\b3gpp_access\b",
            r"\bnon_3gpp_access\b",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    def _flow_id_is_grounded(self, *, flow_id: str, evidence: IntentEvidence) -> bool:
        normalized_flow_id = str(flow_id or "").strip()
        if not normalized_flow_id:
            return False
        if any(str(item.flow_id or "").strip() == normalized_flow_id for item in (evidence.candidate_flows or [])):
            return True
        cached_catalog = evidence.cached_catalog or {}
        flow_catalog = cached_catalog.get("flow_catalog") or []
        if any(str(item.get("flow_id") or "").strip() == normalized_flow_id for item in flow_catalog if isinstance(item, dict)):
            return True
        semantic_candidates = evidence.cached_semantic_candidates or []
        return any(
            str(item.get("flow_id") or "").strip() == normalized_flow_id
            for item in semantic_candidates
            if isinstance(item, dict)
        )
