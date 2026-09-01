from __future__ import annotations

from typing import Literal

from pydantic import JsonValue

from .common import ContractModel, ExecutionScope


ArtifactType = Literal[
    "EnvironmentGenerationRequest",
    "ScenarioCandidateArtifact",
    "GroundingDecisionRequest",
    "GroundingDecision",
    "PlanningRequest",
    "PolicyPlanDraft",
    "FeedbackReport",
]


def artifact_payload_model(artifact_type: ArtifactType) -> type[ContractModel]:
    from control_runtime.agents.dispatch.contracts import FeedbackReport
    from control_runtime.agents.grounding.contracts import (
        GroundingDecision,
        GroundingDecisionRequest,
    )
    from control_runtime.agents.planning.contracts import PlanningRequest, PolicyPlanDraft
    from environment.contracts import EnvironmentGenerationRequest, ScenarioCandidateArtifact

    registry: dict[ArtifactType, type[ContractModel]] = {
        "EnvironmentGenerationRequest": EnvironmentGenerationRequest,
        "ScenarioCandidateArtifact": ScenarioCandidateArtifact,
        "GroundingDecisionRequest": GroundingDecisionRequest,
        "GroundingDecision": GroundingDecision,
        "PlanningRequest": PlanningRequest,
        "PolicyPlanDraft": PolicyPlanDraft,
        "FeedbackReport": FeedbackReport,
    }
    if artifact_type not in registry:
        raise ValueError(f"unregistered artifact_type: {artifact_type}")
    return registry[artifact_type]


def validate_artifact_payload(
    artifact_type: ArtifactType,
    kind: Literal["request", "response"],
    payload: dict[str, JsonValue],
) -> tuple[dict[str, JsonValue], ExecutionScope]:
    allowed_kinds: dict[ArtifactType, set[str]] = {
        "EnvironmentGenerationRequest": {"request"},
        "ScenarioCandidateArtifact": {"response"},
        "GroundingDecisionRequest": {"request"},
        "GroundingDecision": {"response"},
        "PlanningRequest": {"request"},
        "PolicyPlanDraft": {"request", "response"},
        "FeedbackReport": {"response"},
    }
    if artifact_type not in allowed_kinds:
        raise ValueError(f"unregistered artifact_type: {artifact_type}")
    if kind not in allowed_kinds[artifact_type]:
        raise ValueError(f"artifact_type {artifact_type} cannot be used as kind={kind}")
    payload_model = artifact_payload_model(artifact_type).model_validate(payload)
    payload_scope = getattr(payload_model, "scope")
    if not isinstance(payload_scope, ExecutionScope):
        raise TypeError(f"registered artifact {artifact_type} must expose an ExecutionScope")
    return payload_model.model_dump(mode="json"), payload_scope


__all__ = ["ArtifactType", "artifact_payload_model", "validate_artifact_payload"]
