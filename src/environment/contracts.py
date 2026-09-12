from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from shared.contracts import ContractModel, NonEmptyStr
from shared.contracts.coordination import HandoffPayload


class EnvironmentGenerationRequest(HandoffPayload):
    scenario_id: NonEmptyStr
    objective: NonEmptyStr
    complexity: Literal["low", "medium", "high"] = "medium"
    target_flow_count: int = Field(default=8, ge=1)
    topology_mode: NonEmptyStr = "ulcl"
    stress_mode: NonEmptyStr = "slice_resource_contention"


class ScenarioCandidate(ContractModel):
    scenario_id: NonEmptyStr
    name: NonEmptyStr
    scenario: dict[str, JsonValue]
    split_mode_overlay: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> ScenarioCandidate:
        if str(self.scenario.get("scenario_id") or "").strip() != self.scenario_id:
            raise ValueError("scenario.scenario_id must match scenario_id")
        if str(self.scenario.get("name") or "").strip() != self.name:
            raise ValueError("scenario.name must match name")
        return self


class EnvironmentValidationReport(ContractModel):
    scenario_id: NonEmptyStr
    ok: bool
    errors: tuple[NonEmptyStr, ...] = ()
    warnings: tuple[NonEmptyStr, ...] = ()


class EnvironmentArtifact(HandoffPayload):
    candidate: ScenarioCandidate
    compilation_report: dict[str, JsonValue]


__all__ = [
    "EnvironmentArtifact",
    "EnvironmentGenerationRequest",
    "EnvironmentValidationReport",
    "ScenarioCandidate",
]
