from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from shared.contracts import (
    ContractModel,
    ExecutionScope,
    NonEmptyStr,
)


class EnvironmentGenerationRequest(ContractModel):
    scope: ExecutionScope
    scenario_id: NonEmptyStr
    objective: NonEmptyStr
    complexity: Literal["low", "medium", "high"] = "medium"
    target_flow_count: int = Field(default=8, ge=1)
    topology_mode: NonEmptyStr = "ulcl"
    stress_mode: NonEmptyStr = "slice_resource_contention"
    output_dir: Path = Path("generated")

    @model_validator(mode="after")
    def validate_scope(self) -> "EnvironmentGenerationRequest":
        if self.scope.snapshot_id is None:
            raise ValueError("environment generation requires scope.snapshot_id")
        return self


class ScenarioCandidate(ContractModel):
    scenario_id: NonEmptyStr
    name: NonEmptyStr
    scenario: dict[str, JsonValue]
    split_mode_overlay: dict[str, JsonValue] | None = None
    source_path: Path | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> "ScenarioCandidate":
        payload_scenario_id = str(self.scenario.get("scenario_id") or "").strip()
        payload_name = str(self.scenario.get("name") or "").strip()
        if payload_scenario_id != self.scenario_id:
            raise ValueError("scenario.scenario_id must match ScenarioCandidate.scenario_id")
        if payload_name != self.name:
            raise ValueError("scenario.name must match ScenarioCandidate.name")
        return self


class ScenarioCandidateArtifact(ContractModel):
    scope: ExecutionScope
    candidate: ScenarioCandidate


class EnvironmentValidationReport(ContractModel):
    scenario_id: NonEmptyStr
    ok: bool
    errors: tuple[NonEmptyStr, ...] = ()
    warnings: tuple[NonEmptyStr, ...] = ()

    @property
    def error_count(self) -> int:
        return len(self.errors)


class LaunchPlan(ContractModel):
    cwd: Path
    argv: list[NonEmptyStr] = Field(min_length=1)
    graph_db_url: NonEmptyStr
    live_graph_snapshot_id: NonEmptyStr
    manifest_path: Path
    healthcheck_host: NonEmptyStr = "127.0.0.1"
    healthcheck_port: int = 18080
    healthcheck_path: NonEmptyStr = "/policy-executions/launch-healthcheck"
    timeout_seconds: float = Field(default=180.0, gt=0.0)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
