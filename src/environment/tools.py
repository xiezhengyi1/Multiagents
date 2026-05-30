from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .compiler import EnvironmentAgentCompiler
from .contracts import ScenarioCandidate
from .launcher import EnvironmentLauncher
from .specs import ExistingScenarioSpecExplorer


@dataclass
class SimpleEnvironmentTool:
    name: str
    description: str
    func: Callable[..., str]

    def invoke(self, payload: dict[str, Any]) -> str:
        return self.func(**payload)


def _make_tool(name: str, description: str, func: Callable[..., str]) -> Any:
    try:
        from langchain.tools import StructuredTool

        return StructuredTool.from_function(func=func, name=name, description=description)
    except Exception:
        return SimpleEnvironmentTool(name=name, description=description, func=func)


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _dump_mapping(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_environment_tools(
    *,
    compiler: EnvironmentAgentCompiler,
    launcher: EnvironmentLauncher,
    scenario_root: Path,
    execute_simulator: bool = False,
    explorer: ExistingScenarioSpecExplorer | None = None,
) -> list[Any]:
    explorer = explorer or ExistingScenarioSpecExplorer()
    feedback_log: list[dict[str, Any]] = []

    def list_existing_environment_specs(reason: str = "", limit: int = 50) -> str:
        """Return compact specs for existing scenario YAML files before generating a new env."""
        specs = explorer.discover_specs(Path(scenario_root), limit=limit)
        return _json({"status": "ok", "reason": reason, "spec_count": len(specs), "specs": specs})

    def write_candidate_environment_yaml(
        reason: str = "",
        scenario_id: str = "",
        scenario: dict[str, Any] | None = None,
        split_mode_overlay: dict[str, Any] | None = None,
        output_dir: str = "",
    ) -> str:
        """Write candidate base and optional split-mode YAML under the scenario root.

        Base scenario is written to <scenario_root>/<filename>.yaml (or
        <scenario_root>/<output_dir>/<filename>.yaml when output_dir is given).
        Split-mode overlay is always written to <scenario_root>/split_mode/<filename>.yaml.
        """
        normalized_id = str(scenario_id or (scenario or {}).get("scenario_id") or "").strip()
        if not normalized_id:
            raise ValueError("scenario_id is required")
        if not isinstance(scenario, dict):
            raise TypeError("scenario must be a mapping")
        filename = normalized_id.lower().replace("-", "_") + ".yaml"
        base_dir = Path(scenario_root) / output_dir if output_dir else Path(scenario_root)
        base_path = base_dir / filename
        _dump_mapping(base_path, scenario)
        overlay_path = ""
        if split_mode_overlay is not None:
            overlay_target = Path(scenario_root) / "split_mode" / filename
            _dump_mapping(overlay_target, split_mode_overlay)
            overlay_path = str(overlay_target)
        return _json(
            {
                "status": "ok",
                "reason": reason,
                "scenario_id": normalized_id,
                "scenario_path": str(base_path),
                "split_mode_overlay_path": overlay_path,
            }
        )

    def validate_candidate_environment(
        reason: str = "",
        scenario_id: str = "",
        name: str = "",
        scenario: dict[str, Any] | None = None,
        split_mode_overlay: dict[str, Any] | None = None,
    ) -> str:
        """Run static schema and cross-reference validation for a candidate env."""
        candidate = ScenarioCandidate(
            scenario_id=str(scenario_id or (scenario or {}).get("scenario_id") or "").strip(),
            name=str(name or (scenario or {}).get("name") or "").strip(),
            scenario=scenario or {},
            split_mode_overlay=split_mode_overlay,
        )
        report = compiler.validate_candidate(candidate)
        return _json(
            {
                "status": "ok" if report.ok else "failed",
                "reason": reason,
                "scenario_id": report.scenario_id,
                "errors": list(report.errors),
                "warnings": list(report.warnings),
            }
        )

    def simulate_candidate_environment(
        reason: str = "",
        scenario_path: str = "",
        run_id: str = "",
        live_graph_snapshot_id: str = "",
        graph_db_url: str = "",
    ) -> str:
        """Build or execute the direct simulator validation plan for a candidate env."""
        if not scenario_path:
            raise ValueError("scenario_path is required")
        plan = launcher.build_direct_launch_plan(
            scenario_path=Path(scenario_path),
            run_id=run_id or "environment-agent-validation",
            live_graph_snapshot_id=live_graph_snapshot_id or "live-environment-agent",
            graph_db_url=graph_db_url,
        )
        payload = {
            "status": "planned",
            "reason": reason,
            "execute_simulator": bool(execute_simulator),
            "cwd": str(plan.cwd),
            "argv": [str(item) for item in plan.argv],
            "manifest_path": str(plan.manifest_path),
            "healthcheck": {
                "host": plan.healthcheck_host,
                "port": plan.healthcheck_port,
                "path": plan.healthcheck_path,
            },
            "graph_db_url": plan.graph_db_url,
            "live_graph_snapshot_id": plan.live_graph_snapshot_id,
        }
        if execute_simulator:
            payload["status"] = "not_executed"
            payload["error"] = "execute_simulator=True requires a process runner integration; current tool returns the verified launch contract."
        return _json(payload)

    def record_validation_feedback(
        reason: str = "",
        scenario_id: str = "",
        phase: str = "",
        feedback: str = "",
    ) -> str:
        """Record validation feedback so the next generation attempt can adjust logic."""
        item = {
            "reason": reason,
            "scenario_id": scenario_id,
            "phase": phase,
            "feedback": feedback,
        }
        feedback_log.append(item)
        return _json({"status": "ok", "feedback_count": len(feedback_log), "latest_feedback": item})

    return [
        _make_tool("list_existing_environment_specs", list_existing_environment_specs.__doc__ or "", list_existing_environment_specs),
        _make_tool("write_candidate_environment_yaml", write_candidate_environment_yaml.__doc__ or "", write_candidate_environment_yaml),
        _make_tool("validate_candidate_environment", validate_candidate_environment.__doc__ or "", validate_candidate_environment),
        _make_tool("simulate_candidate_environment", simulate_candidate_environment.__doc__ or "", simulate_candidate_environment),
        _make_tool("record_validation_feedback", record_validation_feedback.__doc__ or "", record_validation_feedback),
    ]
