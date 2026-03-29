from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agent_runtime import AgentWorkspace, ArtifactCache, ArtifactEnvelope, ArtifactStore

from .contracts import ConflictResolutionRequest, ConflictResolutionResult


class ConflictResolutionAgent:
    def __init__(self) -> None:
        self.agent_name = "conflict_resolution"
        self.workspace = AgentWorkspace.for_agent(self.agent_name)
        self.cache = ArtifactCache(self.workspace)
        self.artifact_store = ArtifactStore()

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

        if not conflicts:
            return ConflictResolutionResult(
                status="no_conflict",
                reason_summary="No policy or resource conflicts detected in the candidate set.",
            )

        status = "unresolved" if unresolved else "resolved"
        reason_summary = (
            "Detected conflicts that require operator review."
            if unresolved
            else "Detected duplicate conflicts that can be resolved by deduplication."
        )
        return ConflictResolutionResult(
            status=status,
            conflicts=conflicts,
            affected_policy_ids=sorted(set(affected_policy_ids)),
            affected_objects=sorted(set(affected_objects)),
            resolution_recommendations=recommendations,
            reason_summary=reason_summary,
        )

    def run_from_artifact(self, request_path: Path) -> Path:
        envelope = self.artifact_store.read_artifact(request_path)
        if envelope.target_agent != self.agent_name:
            raise ValueError(f"artifact target_agent mismatch: expected {self.agent_name}, got {envelope.target_agent}")

        self.cache.cache_received(envelope)
        request = ConflictResolutionRequest.model_validate(envelope.payload)
        result = self.run(request)

        response_envelope = ArtifactEnvelope(
            artifact_type="ConflictResolutionResult",
            source_agent=self.agent_name,
            target_agent=envelope.source_agent,
            session_id=request.session_id,
            snapshot_id=request.snapshot_id,
            correlation_id=envelope.correlation_id,
            upstream_artifact_ids=[envelope.artifact_id],
            payload=result.model_dump(mode="json"),
        )
        response_path = self.artifact_store.write_response(response_envelope)
        self.cache.cache_produced(response_envelope)
        return response_path
