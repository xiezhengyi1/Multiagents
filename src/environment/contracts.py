from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EnvironmentGenerationRequest:
    scenario_id: str
    objective: str
    complexity: str = "medium"
    target_flow_count: int = 8
    topology_mode: str = "ulcl"
    stress_mode: str = "slice_resource_contention"
    output_dir: Path = Path("")


@dataclass(frozen=True)
class ScenarioCandidate:
    scenario_id: str
    name: str
    scenario: dict[str, Any]
    split_mode_overlay: dict[str, Any] | None = None
    source_path: Path | None = None


@dataclass(frozen=True)
class EnvironmentValidationReport:
    scenario_id: str
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def error_count(self) -> int:
        return len(self.errors)


@dataclass(frozen=True)
class LaunchPlan:
    cwd: Path
    argv: list[object]
    graph_db_url: str
    live_graph_snapshot_id: str
    manifest_path: Path
    healthcheck_host: str = "127.0.0.1"
    healthcheck_port: int = 18080
    healthcheck_path: str = "/policy-executions/launch-healthcheck"
    timeout_seconds: float = 180.0
    metadata: dict[str, Any] = field(default_factory=dict)
