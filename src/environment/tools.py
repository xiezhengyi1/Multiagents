from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .compiler import EnvironmentAgentCompiler
from .draft import ScenarioDraftStore
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
        from langchain_core.tools import StructuredTool
    except ImportError:
        try:
            from langchain.tools import StructuredTool
        except ImportError:
            return SimpleEnvironmentTool(name=name, description=description, func=func)

    return StructuredTool.from_function(func=func, name=name, description=description)


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _dump_mapping(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_candidate_output_dir(scenario_root: Path, output_dir: str) -> Path:
    root = Path(scenario_root)
    requested = Path(str(output_dir or "").strip())
    if not requested.parts:
        return root
    if requested.is_absolute():
        return requested
    for prefix_length in range(min(len(root.parts), len(requested.parts)), 0, -1):
        if root.parts[-prefix_length:] == requested.parts[:prefix_length]:
            return root.joinpath(*requested.parts[prefix_length:])
    return root / requested


def build_environment_tools(
    *,
    compiler: EnvironmentAgentCompiler,
    launcher: EnvironmentLauncher,
    scenario_root: Path,
    execute_simulator: bool = True,
    explorer: ExistingScenarioSpecExplorer | None = None,
) -> list[Any]:
    explorer = explorer or ExistingScenarioSpecExplorer()
    draft = ScenarioDraftStore()
    feedback_log: list[dict[str, Any]] = []

    def list_existing_environment_specs(reason: str = "", limit: int = 50) -> str:
        """Return compact specs for existing scenario YAML files before generating a new env."""
        specs = explorer.discover_specs(Path(scenario_root), limit=limit)
        return _json({"status": "ok", "reason": reason, "spec_count": len(specs), "specs": specs})

    def initialize_environment_draft(
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Initialize one in-memory scenario draft with metadata only."""
        return _json({"status": "ok", "reason": reason, "draft": draft.initialize(metadata or {})})

    def replace_draft_section(reason: str = "", section: str = "", payload: Any = None) -> str:
        """Replace one draft section; first writes must follow the declared stage order."""
        return _json({"status": "ok", "reason": reason, "draft": draft.replace_section(section, payload)})

    def patch_draft_entity(
        reason: str = "",
        section: str = "",
        entity_id: str = "",
        changes: dict[str, Any] | None = None,
    ) -> str:
        """Patch one existing collection entity by stable ID without adding duplicates."""
        return _json(
            {
                "status": "ok",
                "reason": reason,
                "draft": draft.patch_entity(section, entity_id, changes or {}),
            }
        )

    def inspect_draft_section(reason: str = "", section: str = "") -> str:
        """Return the requested draft section in full plus a compact draft summary."""
        return _json({"reason": reason, **draft.inspect_section(section)})

    def validate_environment_draft(reason: str = "") -> str:
        """Assemble and statically validate the complete in-memory scenario draft."""
        candidate = draft.assemble_candidate()
        report = compiler.validate_candidate(candidate)
        draft.validation_passed = report.ok
        return _json(
            {
                "status": "ok" if report.ok else "failed",
                "reason": reason,
                "scenario_id": report.scenario_id,
                "errors": list(report.errors),
                "warnings": list(report.warnings),
                "draft": draft.summary(),
            }
        )

    def write_validated_environment_yaml(reason: str = "", output_dir: str = "") -> str:
        """Write YAML only after the latest complete draft passes static validation."""
        if not draft.validation_passed:
            raise ValueError("draft must pass validation before writing YAML")
        candidate = draft.assemble_candidate()
        filename = candidate.scenario_id.lower().replace("-", "_") + ".yaml"
        base_path = _resolve_candidate_output_dir(Path(scenario_root), output_dir) / filename
        _dump_mapping(base_path, candidate.scenario)
        overlay_path = None
        if candidate.split_mode_overlay is not None:
            overlay_path = Path(scenario_root) / "split_mode" / filename
            _dump_mapping(overlay_path, candidate.split_mode_overlay)
        draft.record_written_paths(scenario_path=base_path, split_mode_overlay_path=overlay_path)
        return _json(
            {
                "status": "ok",
                "reason": reason,
                "scenario_id": candidate.scenario_id,
                "scenario_path": draft.scenario_path,
                "split_mode_overlay_path": draft.split_mode_overlay_path,
            }
        )

    def simulate_candidate_environment(
        reason: str = "",
        run_id: str = "",
        live_graph_snapshot_id: str = "",
        graph_db_url: str = "",
    ) -> str:
        """Execute direct simulator validation for the latest written draft YAML."""
        if not draft.scenario_path:
            raise ValueError("validated YAML must be written before simulation")
        plan = launcher.build_direct_launch_plan(
            scenario_path=Path(draft.scenario_path),
            run_id=run_id or "environment-agent-validation",
            live_graph_snapshot_id=live_graph_snapshot_id or "live-environment-agent",
            graph_db_url=graph_db_url,
        )
        payload: dict[str, Any] = {
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
            payload = {
                "reason": reason,
                "execute_simulator": True,
                **launcher.validate_simulator_startup(plan),
            }
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
        _make_tool("initialize_environment_draft", initialize_environment_draft.__doc__ or "", initialize_environment_draft),
        _make_tool("replace_draft_section", replace_draft_section.__doc__ or "", replace_draft_section),
        _make_tool("patch_draft_entity", patch_draft_entity.__doc__ or "", patch_draft_entity),
        _make_tool("inspect_draft_section", inspect_draft_section.__doc__ or "", inspect_draft_section),
        _make_tool("validate_environment_draft", validate_environment_draft.__doc__ or "", validate_environment_draft),
        _make_tool("write_validated_environment_yaml", write_validated_environment_yaml.__doc__ or "", write_validated_environment_yaml),
        _make_tool("simulate_candidate_environment", simulate_candidate_environment.__doc__ or "", simulate_candidate_environment),
        _make_tool("record_validation_feedback", record_validation_feedback.__doc__ or "", record_validation_feedback),
    ]
