from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Set

from shared.runtime import ArtifactEnvelope
from shared.runtime import ArtifactWorkerMixin
from ...domain.control_plane import ControlDomain, RevisionRequest, UnifiedConstraintSet

from .contracts import ConflictResolutionRequest, ConflictResolutionResult


class ConflictResolutionTool(ArtifactWorkerMixin):
    AM_POLICY_TYPE = "PcfAmPolicyControlPolicyAssociation"

    def __init__(self) -> None:
        self.agent_name = "conflict_resolution"
        self.init_worker_runtime()

    def handle_artifact(self, envelope: ArtifactEnvelope) -> ConflictResolutionResult:
        request = ConflictResolutionRequest.model_validate(envelope.payload)
        return self.run(request)

    @staticmethod
    def _dedupe(values: Iterable[str]) -> List[str]:
        seen: Set[str] = set()
        result: List[str] = []
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    @staticmethod
    def _domain_for_policy(policy_type: str) -> str:
        return ControlDomain.MOBILITY.value if policy_type == ConflictResolutionTool.AM_POLICY_TYPE else ControlDomain.QOS.value

    @staticmethod
    def _canonical_payload(policy: Dict[str, Any]) -> str:
        return json.dumps(policy.get("policy_details") or {}, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _parse_rate(raw_value: Any) -> float:
        if raw_value in (None, "", {}, []):
            return 0.0
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
        text = str(raw_value).strip()
        if not text:
            return 0.0
        try:
            return float(text.split()[0])
        except (TypeError, ValueError, IndexError):
            return 0.0

    @classmethod
    def _extract_snssai_keys(cls, raw_items: Any) -> List[str]:
        keys: List[str] = []
        if not isinstance(raw_items, list):
            return keys
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            sst = item.get("sst")
            if sst is None:
                continue
            keys.append(f"{sst}:{item.get('sd') or ''}")
        return keys

    @classmethod
    def _extract_snssai_keys_from_resource_keys(cls, raw_items: Any) -> List[str]:
        keys: List[str] = []
        if not isinstance(raw_items, list):
            return keys
        for raw_item in raw_items:
            text = str(raw_item or "").strip()
            if not text.startswith("snssai:"):
                continue
            try:
                payload = json.loads(text.split(":", 1)[1])
            except Exception:
                continue
            if isinstance(payload, dict) and payload.get("sst") is not None:
                keys.append(f"{payload.get('sst')}:{payload.get('sd') or ''}")
        return keys

    @classmethod
    def _normalize_valid_snssai_values(cls, raw_items: Any) -> Set[str]:
        values: Set[str] = set()
        if not isinstance(raw_items, list):
            return values
        for raw_item in raw_items:
            text = str(raw_item or "").strip()
            if not text.startswith("snssai:"):
                continue
            try:
                payload = json.loads(text.split(":", 1)[1])
            except Exception:
                continue
            if isinstance(payload, dict) and payload.get("sst") is not None:
                values.add(f"{payload.get('sst')}:{payload.get('sd') or ''}")
        return values

    @classmethod
    def _extract_qos_snssai_keys(cls, policy: Dict[str, Any]) -> List[str]:
        details = policy.get("policy_details") if isinstance(policy.get("policy_details"), dict) else {}
        keys: List[str] = []
        route_sets = details.get("routeSelParamSets")
        if isinstance(route_sets, list):
            for route_set in route_sets:
                if isinstance(route_set, dict) and isinstance(route_set.get("snssai"), dict):
                    keys.extend(cls._extract_snssai_keys([route_set.get("snssai")]))
        keys.extend(cls._extract_snssai_keys_from_resource_keys(policy.get("resource_keys")))
        return cls._dedupe(keys)

    @classmethod
    def _extract_qos_slice_keys(cls, policy: Dict[str, Any]) -> List[str]:
        keys: List[str] = []
        for raw_item in policy.get("resource_keys") or []:
            text = str(raw_item or "").strip()
            if text.startswith("slice:"):
                keys.append(text)
        return cls._dedupe(keys)

    @classmethod
    def _extract_qos_demand(cls, policy: Dict[str, Any]) -> Dict[str, float]:
        details = policy.get("policy_details") if isinstance(policy.get("policy_details"), dict) else {}
        qos_decs = details.get("qosDecs") if isinstance(details.get("qosDecs"), dict) else {}
        ul_total = 0.0
        dl_total = 0.0
        for qos_payload in qos_decs.values():
            if not isinstance(qos_payload, dict):
                continue
            ul_total += cls._parse_rate(qos_payload.get("gbrUl") or qos_payload.get("maxbrUl"))
            dl_total += cls._parse_rate(qos_payload.get("gbrDl") or qos_payload.get("maxbrDl"))
        return {"ul": ul_total, "dl": dl_total}

    @classmethod
    def _summarize_policy(cls, policy: Dict[str, Any]) -> Dict[str, Any]:
        policy_type = str(policy.get("policy_type") or "").strip()
        policy_id = str(policy.get("policy_id") or "").strip()
        supi = str(policy.get("supi") or "").strip()
        app_id = str(policy.get("app_id") or "").strip()
        flow_id = str(policy.get("flow_id") or "").strip()
        target_type = str(policy.get("target_type") or "flow").strip() or "flow"
        domain = cls._domain_for_policy(policy_type)
        details = policy.get("policy_details") if isinstance(policy.get("policy_details"), dict) else {}
        request_payload = details.get("request") if isinstance(details.get("request"), dict) else {}
        resource_keys = cls._dedupe(str(item or "").strip() for item in (policy.get("resource_keys") or []))
        qos_snssai_keys = cls._extract_qos_snssai_keys(policy)
        qos_slice_keys = cls._extract_qos_slice_keys(policy)
        qos_demand = cls._extract_qos_demand(policy)
        binding_key = "|".join(
            [
                domain,
                policy_type,
                supi,
                app_id,
                flow_id,
                target_type,
            ]
        )
        return {
            "policy": policy,
            "policy_id": policy_id,
            "policy_type": policy_type,
            "domain": domain,
            "supi": supi,
            "app_id": app_id,
            "flow_id": flow_id,
            "target_type": target_type,
            "binding_key": binding_key,
            "resource_keys": resource_keys,
            "payload_signature": cls._canonical_payload(policy),
            "qos_slice_keys": qos_slice_keys,
            "qos_snssai_keys": qos_snssai_keys,
            "qos_demand": qos_demand,
            "allowed_snssai_keys": cls._dedupe(cls._extract_snssai_keys(request_payload.get("allowedSnssais"))),
            "target_snssai_keys": cls._dedupe(cls._extract_snssai_keys(request_payload.get("targetSnssais"))),
            "ue_ambr_ul": cls._parse_rate((details.get("ueAmbr") or request_payload.get("ueAmbr") or {}).get("uplink")),
            "ue_ambr_dl": cls._parse_rate((details.get("ueAmbr") or request_payload.get("ueAmbr") or {}).get("downlink")),
            "requires_location_context": bool(details.get("servAreaRes")),
        }

    def _collect_planner_conflicts(self, request: ConflictResolutionRequest) -> List[Dict[str, Any]]:
        conflicts: List[Dict[str, Any]] = []
        upstream_context = request.upstream_context if isinstance(request.upstream_context, dict) else {}
        verdicts = upstream_context.get("planner_cross_domain_verdicts")
        if not isinstance(verdicts, list):
            return conflicts
        for verdict in verdicts:
            if not isinstance(verdict, dict):
                continue
            status = str(verdict.get("status") or "").strip().lower()
            if status not in {"rejected", "failed", "incomplete_context"}:
                continue
            detail_items = [
                str(item).strip()
                for item in list(verdict.get("hard_conflicts") or []) + list(verdict.get("infeasible_reasons") or [])
                if str(item).strip()
            ]
            if not detail_items:
                continue
            domain = str(verdict.get("domain") or "").strip().lower()
            domains = [domain] if domain in {ControlDomain.QOS.value, ControlDomain.MOBILITY.value} else [ControlDomain.QOS.value, ControlDomain.MOBILITY.value]
            conflicts.append(
                {
                    "type": "planner_blocking_conflict",
                    "domain": domain,
                    "domains": domains,
                    "policy_ids": [],
                    "objects": [f"planner:{domain or 'unknown'}"],
                    "resolution_status": "unresolved",
                    "detail": "; ".join(detail_items),
                }
            )
        return conflicts

    def _collect_binding_conflicts(self, policy_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for summary in policy_summaries:
            groups.setdefault(summary["binding_key"], []).append(summary)

        conflicts: List[Dict[str, Any]] = []
        for binding_key, summaries in groups.items():
            if len(summaries) < 2:
                continue
            policy_ids = self._dedupe(summary["policy_id"] for summary in summaries)
            payload_signatures = {summary["payload_signature"] for summary in summaries}
            is_duplicate = len(payload_signatures) == 1
            conflicts.append(
                {
                    "type": "duplicate_policy" if is_duplicate else "binding_conflict",
                    "domain": summaries[0]["domain"],
                    "domains": [summaries[0]["domain"]],
                    "policy_ids": policy_ids,
                    "objects": [f"binding:{binding_key}"],
                    "resolution_status": "resolved" if is_duplicate else "unresolved",
                    "detail": (
                        f"Multiple identical policies target the same binding {binding_key}."
                        if is_duplicate
                        else f"Different policy payloads target the same binding {binding_key}."
                    ),
                }
            )
        return conflicts

    def _collect_unknown_resource_conflicts(
        self,
        policy_summaries: List[Dict[str, Any]],
        resource_view: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        valid_slice_keys = {
            str(item or "").strip()
            for item in (resource_view.get("valid_slice_keys") or [])
            if str(item or "").strip()
        }
        valid_snssai_keys = {
            str(item or "").strip()
            for item in (resource_view.get("valid_snssai_keys") or [])
            if str(item or "").strip()
        }
        valid_snssai_values = self._normalize_valid_snssai_values(resource_view.get("valid_snssai_keys"))
        if not valid_slice_keys and not valid_snssai_keys:
            return []

        conflicts: List[Dict[str, Any]] = []
        for summary in policy_summaries:
            unknown_items: List[str] = []
            for item in summary["qos_slice_keys"]:
                if valid_slice_keys and item not in valid_slice_keys:
                    unknown_items.append(item)
            for item in summary["qos_snssai_keys"]:
                if valid_snssai_values and item not in valid_snssai_values:
                    unknown_items.append(item)
            if not unknown_items:
                continue
            conflicts.append(
                {
                    "type": "unknown_resource_reference",
                    "domain": summary["domain"],
                    "domains": [summary["domain"]],
                    "policy_ids": [summary["policy_id"]],
                    "objects": [f"policy:{summary['policy_id']}"] + [f"resource:{item}" for item in self._dedupe(unknown_items)],
                    "resolution_status": "unresolved",
                    "detail": f"Policy references unknown slice or S-NSSAI resources: {self._dedupe(unknown_items)}.",
                }
            )
        return conflicts

    def _collect_resource_conflicts(
        self,
        policy_summaries: List[Dict[str, Any]],
        resource_view: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        exclusive_keys = {
            str(item or "").strip()
            for item in (resource_view.get("exclusive_resource_keys") or [])
            if str(item or "").strip()
        }
        if not exclusive_keys:
            return []

        claims: Dict[str, List[Dict[str, Any]]] = {}
        for summary in policy_summaries:
            for resource_key in summary["resource_keys"]:
                if resource_key in exclusive_keys:
                    claims.setdefault(resource_key, []).append(summary)

        conflicts: List[Dict[str, Any]] = []
        for resource_key, summaries in claims.items():
            distinct_bindings = {summary["binding_key"] for summary in summaries}
            if len(distinct_bindings) < 2:
                continue
            conflicts.append(
                {
                    "type": "exclusive_resource_conflict",
                    "domain": summaries[0]["domain"],
                    "domains": sorted({summary["domain"] for summary in summaries}),
                    "policy_ids": self._dedupe(summary["policy_id"] for summary in summaries),
                    "objects": [f"resource:{resource_key}"],
                    "resolution_status": "unresolved",
                    "detail": f"Exclusive resource {resource_key} is claimed by multiple bindings.",
                }
            )
        return conflicts

    @staticmethod
    def _parse_snssai_key_to_value(resource_key: str) -> str:
        text = str(resource_key or "").strip()
        if text.startswith("slice:"):
            return text.split(":", 1)[1].strip()
        if not text.startswith("snssai:"):
            return ""
        try:
            payload = json.loads(text.split(":", 1)[1])
        except Exception:
            return ""
        if not isinstance(payload, dict) or payload.get("sst") is None:
            return ""
        return f"{int(payload['sst']):02X}{str(payload.get('sd') or '')}"

    def _collect_slice_capacity_conflicts(
        self,
        policy_summaries: List[Dict[str, Any]],
        resource_view: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        slice_capacity_by_snssai = (
            resource_view.get("slice_capacity_by_snssai")
            if isinstance(resource_view.get("slice_capacity_by_snssai"), dict)
            else {}
        )
        flow_allocations = (
            resource_view.get("flow_allocations")
            if isinstance(resource_view.get("flow_allocations"), dict)
            else {}
        )
        if not slice_capacity_by_snssai:
            return []

        # Do not double count repeated policies for the same binding.
        unique_qos_summaries: List[Dict[str, Any]] = []
        seen_bindings: Set[str] = set()
        for summary in policy_summaries:
            if summary["domain"] != ControlDomain.QOS.value:
                continue
            binding_key = str(summary["binding_key"] or "").strip()
            if not binding_key or binding_key in seen_bindings:
                continue
            seen_bindings.add(binding_key)
            unique_qos_summaries.append(summary)

        per_slice_delta: Dict[str, Dict[str, Any]] = {}
        for summary in unique_qos_summaries:
            target_snssai = ""
            if summary["qos_slice_keys"]:
                target_snssai = self._parse_snssai_key_to_value(summary["qos_slice_keys"][0])
            elif summary["qos_snssai_keys"]:
                target_snssai = self._parse_snssai_key_to_value(summary["qos_snssai_keys"][0])
            if not target_snssai or target_snssai not in slice_capacity_by_snssai:
                continue

            demand_ul = float(summary["qos_demand"]["ul"])
            demand_dl = float(summary["qos_demand"]["dl"])
            current_allocation = flow_allocations.get(summary["flow_id"]) if summary["flow_id"] else {}
            current_allocation = current_allocation if isinstance(current_allocation, dict) else {}
            current_snssai = str(current_allocation.get("current_slice_snssai") or "").strip()
            current_ul = float(current_allocation.get("allocated_bandwidth_ul", 0.0) or 0.0)
            current_dl = float(current_allocation.get("allocated_bandwidth_dl", 0.0) or 0.0)

            if target_snssai == current_snssai:
                delta_ul = max(0.0, demand_ul - current_ul)
                delta_dl = max(0.0, demand_dl - current_dl)
            else:
                delta_ul = max(0.0, demand_ul)
                delta_dl = max(0.0, demand_dl)
            if delta_ul <= 0.0 and delta_dl <= 0.0:
                continue

            entry = per_slice_delta.setdefault(
                target_snssai,
                {"delta_ul": 0.0, "delta_dl": 0.0, "policy_ids": [], "flow_ids": []},
            )
            entry["delta_ul"] += delta_ul
            entry["delta_dl"] += delta_dl
            entry["policy_ids"].append(summary["policy_id"])
            if summary["flow_id"]:
                entry["flow_ids"].append(summary["flow_id"])

        conflicts: List[Dict[str, Any]] = []
        for snssai, delta in per_slice_delta.items():
            capacity = slice_capacity_by_snssai.get(snssai)
            if not isinstance(capacity, dict):
                continue
            remaining_ul = float(capacity.get("remaining_ul", 0.0) or 0.0)
            remaining_dl = float(capacity.get("remaining_dl", 0.0) or 0.0)
            exceeds_ul = delta["delta_ul"] > remaining_ul + 1e-9
            exceeds_dl = delta["delta_dl"] > remaining_dl + 1e-9
            if not exceeds_ul and not exceeds_dl:
                continue
            detail_parts: List[str] = []
            if exceeds_ul:
                detail_parts.append(
                    f"uplink delta {delta['delta_ul']:.3f} Mbps exceeds remaining {remaining_ul:.3f} Mbps"
                )
            if exceeds_dl:
                detail_parts.append(
                    f"downlink delta {delta['delta_dl']:.3f} Mbps exceeds remaining {remaining_dl:.3f} Mbps"
                )
            conflicts.append(
                {
                    "type": "slice_capacity_conflict",
                    "domain": ControlDomain.QOS.value,
                    "domains": [ControlDomain.QOS.value],
                    "policy_ids": self._dedupe(delta["policy_ids"]),
                    "objects": [f"slice:{snssai}"] + [f"flow:{item}" for item in self._dedupe(delta["flow_ids"])],
                    "resolution_status": "unresolved",
                    "detail": f"Target slice {snssai} cannot absorb the requested incremental demand: {'; '.join(detail_parts)}.",
                }
            )
        return conflicts

    def _collect_cross_domain_conflicts(
        self,
        policy_summaries: List[Dict[str, Any]],
        ue_context: Dict[str, Any],
        requested_domains: Set[str],
    ) -> List[Dict[str, Any]]:
        if {ControlDomain.QOS.value, ControlDomain.MOBILITY.value} - requested_domains:
            return []

        qos_summaries = [summary for summary in policy_summaries if summary["domain"] == ControlDomain.QOS.value]
        mobility_summaries = [summary for summary in policy_summaries if summary["domain"] == ControlDomain.MOBILITY.value]
        if not qos_summaries or not mobility_summaries:
            return []

        conflicts: List[Dict[str, Any]] = []
        user_loc = ((ue_context.get("accessMobilityContext") or {}).get("userLoc")) if isinstance(ue_context, dict) else None

        # Only raise slice-alignment conflicts when the candidate policies carry explicit S-NSSAI evidence.
        qos_snssai_keys = self._dedupe(
            key
            for summary in qos_summaries
            for key in summary["qos_snssai_keys"]
        )
        total_qos_ul = sum(float(summary["qos_demand"]["ul"]) for summary in qos_summaries)
        total_qos_dl = sum(float(summary["qos_demand"]["dl"]) for summary in qos_summaries)

        for summary in mobility_summaries:
            allowed = set(summary["allowed_snssai_keys"])
            target = set(summary["target_snssai_keys"])
            if target and not target.issubset(allowed):
                conflicts.append(
                    {
                        "type": "target_subset_allowed_violation",
                        "domain": "cross_domain",
                        "domains": [ControlDomain.MOBILITY.value],
                        "policy_ids": [summary["policy_id"]],
                        "objects": [f"policy:{summary['policy_id']}"],
                        "resolution_status": "unresolved",
                        "detail": "targetSnssais must be a subset of allowedSnssais.",
                    }
                )

            if qos_snssai_keys and allowed:
                uncovered = sorted(set(qos_snssai_keys) - allowed)
                if uncovered:
                    conflicts.append(
                        {
                            "type": "snssai_alignment_conflict",
                            "domain": "cross_domain",
                            "domains": [ControlDomain.QOS.value, ControlDomain.MOBILITY.value],
                            "policy_ids": [summary["policy_id"]] + self._dedupe(item["policy_id"] for item in qos_summaries),
                            "objects": [f"policy:{summary['policy_id']}"] + [f"snssai:{item}" for item in uncovered],
                            "resolution_status": "unresolved",
                            "detail": f"QoS-selected S-NSSAI values are not covered by mobility allowedSnssais: {uncovered}.",
                        }
                    )

            if summary["ue_ambr_ul"] and summary["ue_ambr_ul"] < total_qos_ul:
                conflicts.append(
                    {
                        "type": "ambr_qos_uplink_inconsistency",
                        "domain": "cross_domain",
                        "domains": [ControlDomain.QOS.value, ControlDomain.MOBILITY.value],
                        "policy_ids": [summary["policy_id"]] + self._dedupe(item["policy_id"] for item in qos_summaries),
                        "objects": [f"policy:{summary['policy_id']}"],
                        "resolution_status": "unresolved",
                        "detail": "UE AMBR uplink is lower than the total QoS uplink demand.",
                    }
                )

            if summary["ue_ambr_dl"] and summary["ue_ambr_dl"] < total_qos_dl:
                conflicts.append(
                    {
                        "type": "ambr_qos_downlink_inconsistency",
                        "domain": "cross_domain",
                        "domains": [ControlDomain.QOS.value, ControlDomain.MOBILITY.value],
                        "policy_ids": [summary["policy_id"]] + self._dedupe(item["policy_id"] for item in qos_summaries),
                        "objects": [f"policy:{summary['policy_id']}"],
                        "resolution_status": "unresolved",
                        "detail": "UE AMBR downlink is lower than the total QoS downlink demand.",
                    }
                )

            if summary["requires_location_context"] and user_loc in (None, {}, []):
                conflicts.append(
                    {
                        "type": "service_area_missing_location_context",
                        "domain": "cross_domain",
                        "domains": [ControlDomain.MOBILITY.value],
                        "policy_ids": [summary["policy_id"]],
                        "objects": [f"policy:{summary['policy_id']}"],
                        "resolution_status": "unresolved",
                        "detail": "service-area restriction requires user location context.",
                    }
                )

        return conflicts

    def _build_revision_bundle(
        self,
        conflicts: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], UnifiedConstraintSet, List[str]]:
        revision_requests: List[Dict[str, Any]] = []
        hard_constraints: List[str] = []
        evidence: List[str] = []
        affected_domains: List[str] = []

        for conflict in conflicts:
            if str(conflict.get("resolution_status") or "").strip().lower() != "unresolved":
                continue
            detail = str(conflict.get("detail") or conflict.get("type") or "revise conflicting plan").strip()
            policy_ids = self._dedupe(conflict.get("policy_ids") or [])
            objects = self._dedupe(conflict.get("objects") or [])
            domains = [
                domain
                for domain in self._dedupe(conflict.get("domains") or [])
                if domain in {ControlDomain.QOS.value, ControlDomain.MOBILITY.value}
            ]
            evidence_line = json.dumps(conflict, ensure_ascii=False, sort_keys=True)
            evidence.append(evidence_line)
            hard_constraints.append(detail)
            affected_domains.extend(domains)
            for domain in domains:
                revision_requests.append(
                    RevisionRequest(
                        target_domain=ControlDomain(domain),
                        conflict_type=str(conflict.get("type") or "").strip(),
                        target_policy_ids=policy_ids,
                        target_objects=objects or policy_ids,
                        reason=detail,
                        suggested_actions=[f"Revise the {domain} plan to satisfy: {detail}"],
                        hard_constraints=[detail],
                        evidence=[evidence_line],
                    ).model_dump(mode="json")
                )

        return (
            revision_requests,
            UnifiedConstraintSet(
                hard_constraints=self._dedupe(hard_constraints),
                evidence=evidence,
            ),
            self._dedupe(affected_domains),
        )

    def run(self, request: ConflictResolutionRequest) -> ConflictResolutionResult:
        policy_summaries = [
            self._summarize_policy(policy)
            for policy in request.candidate_policies
            if isinstance(policy, dict)
        ]
        requested_domains = {
            str(item or "").strip().lower()
            for item in ((request.conflict_scope or {}).get("domains") or [])
            if str(item or "").strip()
        }
        resource_view = request.resource_view if isinstance(request.resource_view, dict) else {}
        upstream_context = request.upstream_context if isinstance(request.upstream_context, dict) else {}
        ue_context = upstream_context.get("ue_context") if isinstance(upstream_context.get("ue_context"), dict) else {}

        conflicts: List[Dict[str, Any]] = []
        conflicts.extend(self._collect_planner_conflicts(request))
        conflicts.extend(self._collect_binding_conflicts(policy_summaries))
        conflicts.extend(self._collect_unknown_resource_conflicts(policy_summaries, resource_view))
        conflicts.extend(self._collect_resource_conflicts(policy_summaries, resource_view))
        conflicts.extend(self._collect_slice_capacity_conflicts(policy_summaries, resource_view))
        conflicts.extend(self._collect_cross_domain_conflicts(policy_summaries, ue_context, requested_domains))

        unresolved_conflicts = [
            conflict
            for conflict in conflicts
            if str(conflict.get("resolution_status") or "").strip().lower() == "unresolved"
        ]
        revision_requests, unified_constraints, affected_domains = self._build_revision_bundle(conflicts)
        affected_policy_ids = self._dedupe(
            str(item).strip()
            for conflict in conflicts
            for item in (conflict.get("policy_ids") or [])
        )
        affected_objects = self._dedupe(
            str(item).strip()
            for conflict in conflicts
            for item in (conflict.get("objects") or [])
        )

        if not conflicts:
            return ConflictResolutionResult(
                status="no_conflict",
                mediator_status="approved",
                reason_summary="No policy conflicts were detected from explicit planner, binding, or cross-domain evidence.",
                unified_constraints=UnifiedConstraintSet(),
            )

        mediator_status = "revise" if unresolved_conflicts else "approved"
        status = "unresolved" if unresolved_conflicts else "resolved"
        if unresolved_conflicts:
            reason_summary = f"Detected {len(unresolved_conflicts)} blocking conflict(s) that require planner revision before execution."
        else:
            reason_summary = "Detected non-blocking duplicate policies that can be safely deduplicated."

        recommendations = self._dedupe(
            (
                f"Revise candidate policies to resolve: {str(conflict.get('detail') or conflict.get('type') or '').strip()}"
                if str(conflict.get("resolution_status") or "").strip().lower() == "unresolved"
                else f"Deduplicate repeated policy claims for {', '.join(conflict.get('objects') or [])}."
            )
            for conflict in conflicts
        )

        return ConflictResolutionResult(
            status=status,
            mediator_status=mediator_status,
            conflicts=conflicts,
            affected_policy_ids=affected_policy_ids,
            affected_objects=affected_objects,
            affected_domains=affected_domains,
            resolution_recommendations=recommendations,
            reason_summary=reason_summary,
            revision_requests=revision_requests,
            unified_constraints=unified_constraints,
        )
