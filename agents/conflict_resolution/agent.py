from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agent_runtime import AgentWorkspace, ArtifactCache, ArtifactEnvelope, ArtifactStore
from agents.worker import ArtifactWorkerMixin
from domain.control_plane import ControlDomain, RevisionRequest, UnifiedConstraintSet

from .contracts import ConflictResolutionRequest, ConflictResolutionResult


class ConflictResolutionTool(ArtifactWorkerMixin):
    AM_POLICY_TYPE = "PcfAmPolicyControlPolicyAssociation"

    def __init__(self) -> None:
        self.agent_name = "conflict_resolution"
        self.init_worker_runtime()

    def expected_request_type(self) -> str:
        return "ConflictResolutionRequest"

    def response_artifact_type(self) -> str:
        return "ConflictResolutionResult"

    def handle_artifact(self, envelope: ArtifactEnvelope) -> ConflictResolutionResult:
        request = ConflictResolutionRequest.model_validate(envelope.payload)
        return self.run(request)

    @staticmethod
    def _binding_key(policy: Dict[str, Any]) -> str:
        supi = str(policy.get("supi") or "").strip()
        app_id = str(policy.get("app_id") or "").strip()
        flow_id = str(policy.get("flow_id") or "").strip()
        target_type = str(policy.get("target_type") or "flow").strip()
        return "|".join([supi, app_id, flow_id, target_type])

    @staticmethod
    def _resource_keys(policy: Dict[str, Any]) -> List[str]:
        keys: List[str] = []
        for raw_key in policy.get("resource_keys") or []:
            normalized = str(raw_key or "").strip()
            if normalized:
                keys.append(normalized)

        details = policy.get("policy_details") or {}
        if isinstance(details, dict):
            route_sets = details.get("routeSelParamSets")
            if isinstance(route_sets, list):
                for route_set in route_sets:
                    if not isinstance(route_set, dict):
                        continue
                    dnn = str(route_set.get("dnn") or "").strip()
                    if dnn:
                        keys.append(f"dnn:{dnn}")
                    snssai = route_set.get("snssai")
                    if snssai not in (None, "", {}, []):
                        keys.append(f"snssai:{json.dumps(snssai, sort_keys=True, ensure_ascii=False)}")

        return sorted(set(keys))

    @staticmethod
    def _canonical_policy(policy: Dict[str, Any]) -> str:
        return json.dumps(policy.get("policy_details") or {}, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _extract_snssai_keys(raw_items: Any) -> List[str]:
        keys: List[str] = []
        if not isinstance(raw_items, list):
            return keys
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            sst = item.get("sst")
            sd = item.get("sd")
            if sst is None:
                continue
            keys.append(f"{sst}:{sd or ''}")
        return keys

    @staticmethod
    def _extract_snssai_keys_from_resource_keys(raw_items: Any) -> List[str]:
        keys: List[str] = []
        if not isinstance(raw_items, list):
            return keys
        for item in raw_items:
            text = str(item or "").strip()
            if not text.startswith("snssai:"):
                continue
            payload = text.split(":", 1)[1]
            try:
                parsed = json.loads(payload)
            except Exception:
                continue
            if isinstance(parsed, dict) and parsed.get("sst") is not None:
                keys.append(f"{parsed.get('sst')}:{parsed.get('sd') or ''}")
        return keys

    @staticmethod
    def _domains_for_conflict(conflict: Dict[str, Any]) -> List[ControlDomain]:
        conflict_type = str(conflict.get("type") or "").strip()
        domain = str(conflict.get("domain") or "").strip().lower()
        if domain == ControlDomain.QOS.value:
            return [ControlDomain.QOS]
        if domain == ControlDomain.MOBILITY.value:
            return [ControlDomain.MOBILITY]
        if conflict_type in {
            "cross_domain_snssai_mismatch",
            "qos_mobility_slice_inconsistency",
            "ambr_qos_uplink_inconsistency",
            "ambr_qos_downlink_inconsistency",
            "service_area_missing_location_context",
            "planner_cross_domain_conflict",
        }:
            return [ControlDomain.QOS, ControlDomain.MOBILITY]
        if conflict_type in {"binding_conflict", "duplicate_binding", "resource_conflict", "duplicate_resource_claim"}:
            return [ControlDomain.QOS]
        return []

    @staticmethod
    def _hard_constraint_messages(conflict: Dict[str, Any]) -> List[str]:
        detail = str(conflict.get("detail") or "").strip()
        if detail:
            return [detail]
        conflict_type = str(conflict.get("type") or "").strip()
        if conflict_type:
            return [conflict_type]
        return []

    def _build_revision_requests(self, conflicts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], UnifiedConstraintSet, List[str]]:
        revision_requests: List[Dict[str, Any]] = []
        hard_constraints: List[str] = []
        evidence: List[str] = []
        affected_domains: List[str] = []

        for conflict in conflicts:
            if str(conflict.get("resolution_status") or "").strip().lower() != "unresolved":
                continue
            domains = self._domains_for_conflict(conflict)
            target_policy_ids = [str(item).strip() for item in conflict.get("policy_ids") or [] if str(item).strip()]
            target_objects: List[str] = []
            for key in ("binding_key", "resource_key"):
                value = str(conflict.get(key) or "").strip()
                if value:
                    target_objects.append(value)
            if not target_objects:
                target_objects.extend(target_policy_ids)
            constraints = self._hard_constraint_messages(conflict)
            detail = str(conflict.get("detail") or conflict.get("type") or "revise conflicting plan").strip()
            evidence_line = json.dumps(conflict, ensure_ascii=False, sort_keys=True)
            evidence.append(evidence_line)
            hard_constraints.extend(constraints)
            for domain in domains:
                affected_domains.append(domain.value)
                revision_requests.append(
                    RevisionRequest(
                        target_domain=domain,
                        conflict_type=str(conflict.get("type") or "").strip(),
                        target_policy_ids=target_policy_ids,
                        target_objects=target_objects,
                        reason=detail,
                        suggested_actions=[f"Revise the {domain.value} plan to satisfy: {detail}"],
                        hard_constraints=constraints,
                        evidence=[evidence_line],
                    ).model_dump(mode="json")
                )

        unified_constraints = UnifiedConstraintSet(
            hard_constraints=sorted(set(item for item in hard_constraints if item)),
            evidence=evidence,
        )
        return revision_requests, unified_constraints, sorted(set(item for item in affected_domains if item))

    def _collect_cross_domain_conflicts(self, request: ConflictResolutionRequest) -> List[Dict[str, Any]]:
        domains = {str(item or "").strip().lower() for item in (request.conflict_scope or {}).get("domains", [])}
        if not {"qos", "mobility"}.issubset(domains):
            return []

        qos_policies = [
            policy for policy in request.candidate_policies
            if str(policy.get("policy_type") or "").strip() != self.AM_POLICY_TYPE
        ]
        mobility_policies = [
            policy for policy in request.candidate_policies
            if str(policy.get("policy_type") or "").strip() == self.AM_POLICY_TYPE
        ]
        if not qos_policies or not mobility_policies:
            return []

        conflicts: List[Dict[str, Any]] = []
        ue_context = (request.upstream_context or {}).get("ue_context") if isinstance(request.upstream_context, dict) else {}
        user_loc = ((ue_context or {}).get("accessMobilityContext") or {}).get("userLoc") if isinstance(ue_context, dict) else None

        qos_snssais: List[str] = []
        qos_ul = 0.0
        qos_dl = 0.0
        for policy in qos_policies:
            details = policy.get("policy_details") or {}
            if isinstance(details.get("routeSelParamSets"), list):
                for route_set in details.get("routeSelParamSets") or []:
                    if isinstance(route_set, dict) and isinstance(route_set.get("snssai"), dict):
                        qos_snssais.extend(self._extract_snssai_keys([route_set.get("snssai")]))
            qos_snssais.extend(self._extract_snssai_keys_from_resource_keys(policy.get("resource_keys")))
            qos_decs = details.get("qosDecs") or {}
            if isinstance(qos_decs, dict):
                for qos in qos_decs.values():
                    if not isinstance(qos, dict):
                        continue
                    try:
                        qos_ul += float(qos.get("gbrUl") or qos.get("maxbrUl") or 0.0)
                    except (TypeError, ValueError):
                        pass
                    try:
                        qos_dl += float(qos.get("gbrDl") or qos.get("maxbrDl") or 0.0)
                    except (TypeError, ValueError):
                        pass

        for policy in mobility_policies:
            details = policy.get("policy_details") or {}
            request_payload = details.get("request") or {}
            allowed = set(self._extract_snssai_keys(request_payload.get("allowedSnssais")))
            target = set(self._extract_snssai_keys(request_payload.get("targetSnssais")))
            policy_id = str(policy.get("policy_id") or "").strip()

            if target and not target.issubset(allowed):
                conflicts.append(
                    {
                        "type": "cross_domain_snssai_mismatch",
                        "policy_ids": [policy_id],
                        "resolution_status": "unresolved",
                        "detail": "targetSnssais must be covered by allowedSnssais",
                    }
                )

            uncovered_qos = sorted(set(qos_snssais) - allowed) if allowed else sorted(set(qos_snssais))
            if uncovered_qos:
                conflicts.append(
                    {
                        "type": "qos_mobility_slice_inconsistency",
                        "policy_ids": [policy_id],
                        "resolution_status": "unresolved",
                        "detail": f"QoS-selected S-NSSAI values are not covered by mobility allowedSnssais: {uncovered_qos}",
                    }
                )

            ue_ambr = details.get("ueAmbr") or request_payload.get("ueAmbr") or {}
            try:
                ambr_ul = float(str(ue_ambr.get("uplink") or "0").split()[0])
            except (TypeError, ValueError, AttributeError):
                ambr_ul = 0.0
            try:
                ambr_dl = float(str(ue_ambr.get("downlink") or "0").split()[0])
            except (TypeError, ValueError, AttributeError):
                ambr_dl = 0.0
            if ambr_ul and ambr_ul < qos_ul:
                conflicts.append(
                    {
                        "type": "ambr_qos_uplink_inconsistency",
                        "policy_ids": [policy_id],
                        "resolution_status": "unresolved",
                        "detail": "UE AMBR uplink is lower than QoS-assigned uplink demand",
                    }
                )
            if ambr_dl and ambr_dl < qos_dl:
                conflicts.append(
                    {
                        "type": "ambr_qos_downlink_inconsistency",
                        "policy_ids": [policy_id],
                        "resolution_status": "unresolved",
                        "detail": "UE AMBR downlink is lower than QoS-assigned downlink demand",
                    }
                )

            if details.get("servAreaRes") and user_loc in (None, {}, []):
                conflicts.append(
                    {
                        "type": "service_area_missing_location_context",
                        "policy_ids": [policy_id],
                        "resolution_status": "unresolved",
                        "detail": "service-area restriction requires user location context",
                    }
                )

        return conflicts

    def run(self, request: ConflictResolutionRequest) -> ConflictResolutionResult:
        binding_groups: Dict[str, List[Dict[str, Any]]] = {}
        resource_groups: Dict[str, List[Dict[str, Any]]] = {}

        for policy in request.candidate_policies:
            binding_groups.setdefault(self._binding_key(policy), []).append(policy)
            for resource_key in self._resource_keys(policy):
                resource_groups.setdefault(resource_key, []).append(policy)

        conflicts: List[Dict[str, Any]] = []
        affected_policy_ids: List[str] = []
        affected_objects: List[str] = []
        recommendations: List[str] = []
        unresolved = False
        planner_verdicts = (request.upstream_context or {}).get("planner_cross_domain_verdicts") if isinstance(request.upstream_context, dict) else []
        if isinstance(planner_verdicts, list):
            for verdict in planner_verdicts:
                if not isinstance(verdict, dict):
                    continue
                status = str(verdict.get("status") or "").strip().lower()
                if status not in {"rejected", "incomplete_context", "failed"}:
                    continue
                hard_conflicts = [str(item) for item in verdict.get("hard_conflicts") or [] if str(item).strip()]
                infeasible_reasons = [str(item) for item in verdict.get("infeasible_reasons") or [] if str(item).strip()]
                detail_parts = hard_conflicts + infeasible_reasons
                if not detail_parts:
                    continue
                unresolved = True
                conflicts.append(
                    {
                        "type": "planner_cross_domain_conflict",
                        "domain": str(verdict.get("domain") or "").strip(),
                        "resolution_status": "unresolved",
                        "detail": "; ".join(detail_parts),
                    }
                )
                affected_objects.append(f"planner:{verdict.get('domain')}")
                recommendations.append("Revise the bounded planner output before execution because optimizer cross-domain checks already reported a blocking issue.")

        for binding_key, policies in binding_groups.items():
            if len(policies) < 2:
                continue
            signatures = {self._canonical_policy(policy) for policy in policies}
            policy_ids = [str(policy.get("policy_id") or "") for policy in policies if str(policy.get("policy_id") or "").strip()]
            affected_policy_ids.extend(policy_ids)
            affected_objects.append(f"binding:{binding_key}")
            if len(signatures) == 1:
                conflicts.append(
                    {
                        "type": "duplicate_binding",
                        "binding_key": binding_key,
                        "policy_ids": policy_ids,
                        "resolution_status": "resolved",
                    }
                )
                recommendations.append(f"Deduplicate identical policies for binding {binding_key}.")
            else:
                unresolved = True
                conflicts.append(
                    {
                        "type": "binding_conflict",
                        "binding_key": binding_key,
                        "policy_ids": policy_ids,
                        "resolution_status": "unresolved",
                    }
                )
                recommendations.append(f"Escalate conflicting policy intents for binding {binding_key}.")

        for resource_key, policies in resource_groups.items():
            if len(policies) < 2:
                continue
            signatures = {self._canonical_policy(policy) for policy in policies}
            policy_ids = [str(policy.get("policy_id") or "") for policy in policies if str(policy.get("policy_id") or "").strip()]
            affected_policy_ids.extend(policy_ids)
            affected_objects.append(f"resource:{resource_key}")
            if len(signatures) == 1:
                conflicts.append(
                    {
                        "type": "duplicate_resource_claim",
                        "resource_key": resource_key,
                        "policy_ids": policy_ids,
                        "resolution_status": "resolved",
                    }
                )
                recommendations.append(f"Deduplicate identical claims on resource {resource_key}.")
            else:
                unresolved = True
                conflicts.append(
                    {
                        "type": "resource_conflict",
                        "resource_key": resource_key,
                        "policy_ids": policy_ids,
                        "resolution_status": "unresolved",
                    }
                )
                recommendations.append(f"Review conflicting resource claims on {resource_key}.")

        cross_domain_conflicts = self._collect_cross_domain_conflicts(request)
        for conflict in cross_domain_conflicts:
            conflicts.append(conflict)
            affected_policy_ids.extend([str(item) for item in conflict.get("policy_ids") or [] if str(item).strip()])
            affected_objects.append(f"cross-domain:{conflict.get('type')}")
            unresolved = True
            detail = str(conflict.get("detail") or conflict.get("type") or "cross-domain conflict").strip()
            recommendations.append(f"Resolve cross-domain conflict: {detail}.")

        revision_requests, unified_constraints, affected_domains = self._build_revision_requests(conflicts)

        if not conflicts:
            return ConflictResolutionResult(
                status="no_conflict",
                mediator_status="approved",
                reason_summary="No policy or resource conflicts detected in the candidate set.",
                unified_constraints=UnifiedConstraintSet(),
            )

        status = "unresolved" if unresolved else "resolved"
        mediator_status = "revise" if unresolved else "approved"
        reason_summary = (
            "Detected blocking conflicts that require planner revision before execution."
            if unresolved
            else "Detected duplicate conflicts that can be resolved by deduplication."
        )
        return ConflictResolutionResult(
            status=status,
            mediator_status=mediator_status,
            conflicts=conflicts,
            affected_policy_ids=sorted(set(affected_policy_ids)),
            affected_objects=sorted(set(affected_objects)),
            affected_domains=affected_domains,
            resolution_recommendations=recommendations,
            reason_summary=reason_summary,
            revision_requests=revision_requests,
            unified_constraints=unified_constraints,
        )

    def run_from_artifact(self, request_path: Path) -> Path:
        return self.consume_request_artifact(request_path)
