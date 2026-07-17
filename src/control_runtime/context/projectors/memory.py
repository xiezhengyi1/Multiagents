from __future__ import annotations

from typing import Any

from .base import json_mapping, without_empty_values
from .evidence import IntentEvidenceProjector
from .global_intent import GlobalControlIntentProjector
from .grounding_decision import GroundingDecisionProjector
from .planning_context import PlanningContextProjector
from .policy_plan import PolicyPlanDraftProjector


def project_grounding_decision_for_prompt(grounding_decision: Any) -> dict[str, Any]:
    return GroundingDecisionProjector.project(grounding_decision)


def project_global_intent_for_prompt(global_intent: Any) -> dict[str, Any]:
    return GlobalControlIntentProjector.project(global_intent)


def project_collaboration_context_for_prompt(context: Any) -> dict[str, Any]:
    return PlanningContextProjector.project(context)


def project_intent_evidence_for_prompt(evidence: Any) -> dict[str, Any]:
    return IntentEvidenceProjector.project(evidence)


def project_memory_payload(role: str, payload: Any) -> dict[str, Any]:
    normalized_role = str(role or "").strip().upper()
    if normalized_role == "IEA":
        return project_grounding_decision_for_prompt(payload)
    if normalized_role == "MAIN":
        return project_global_intent_for_prompt(payload)
    if normalized_role == "OSA":
        return PolicyPlanDraftProjector.project(payload)
    if normalized_role == "AD":
        raw = json_mapping(payload)
        return without_empty_values(
            {
                "status": raw.get("status"),
                "root_cause_category": raw.get("root_cause_category"),
                "root_cause": raw.get("root_cause"),
                "reason_summary": raw.get("reason_summary"),
                "affected_policy_ids": raw.get("affected_policy_ids"),
                "affected_flow_ids": raw.get("affected_flow_ids"),
                "recommended_actions": raw.get("recommended_actions"),
            }
        )
    raw = json_mapping(payload)
    return without_empty_values(
        {
            "session_id": raw.get("session_id"),
            "snapshot_id": raw.get("snapshot_id"),
            "supi": raw.get("supi"),
            "status": raw.get("status"),
            "summary": raw.get("summary"),
        }
    )
