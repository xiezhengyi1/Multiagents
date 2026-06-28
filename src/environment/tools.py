from __future__ import annotations

from copy import deepcopy
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .compiler import EnvironmentAgentCompiler
from .draft import ScenarioDraftStore
from .launcher import EnvironmentLauncher
from .specs import ExistingScenarioSpecExplorer
from shared.logging import log_event


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


def _load_mapping(path: Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        payload = yaml.safe_load(text)
    except Exception:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a mapping")
    return payload


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


def _repair_section_for_error(error: str) -> str:
    if error.startswith(("slice ", "duplicate slice ")):
        return "slices"
    if error.startswith(("upf ", "duplicate upf ")):
        return "upfs"
    if error.startswith(("gNB ", "duplicate gNB ")):
        return "gnbs"
    if error.startswith(("UE ", "duplicate UE ")):
        return "ues"
    if error.startswith(("app ", "duplicate app ")):
        return "apps"
    if error.startswith(("flow ", "duplicate flow ")):
        return "flows"
    if error.startswith(("free5gc ", "ns3 ", "writer ", "topology ", "bridge ")):
        return "runtime_config"
    if error.startswith("split_mode_overlay "):
        return "split_mode_overlay"
    return "unknown"


def _build_repair_plan(errors: list[str]) -> list[dict[str, Any]]:
    counts = Counter(_repair_section_for_error(error) for error in errors)
    plan: list[dict[str, Any]] = []
    for section, error_count in counts.items():
        collection = section in {"slices", "upfs", "gnbs", "ues", "apps", "flows"}
        action = "patch_draft_entity" if collection and error_count == 1 else "replace_draft_section"
        plan.append(
            {
                "section": section,
                "error_count": error_count,
                "action": action,
                "guidance": (
                    f"Replace the complete {section} section in one call; do not patch entities one by one."
                    if action == "replace_draft_section"
                    else f"Patch the single affected {section} entity."
                ),
            }
        )
    return plan


def _build_topology_graph(scenario: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    slice_ids: dict[str, str] = {}
    upf_ids: dict[str, str] = {}
    gnb_ids: dict[str, str] = {}

    for index, item in enumerate(scenario.get("slices") or [], start=1):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        node_id = f"slice-node-{index}"
        slice_ids[label] = node_id
        nodes.append({"id": node_id, "type": "slice", "label": label, "attributes": deepcopy(item)})

    for index, item in enumerate(scenario.get("upfs") or [], start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        node_id = f"core-node-{index}"
        upf_ids[name] = node_id
        nodes.append({"id": node_id, "type": "core_node", "label": name, "attributes": deepcopy(item)})

    for index, item in enumerate(scenario.get("gnbs") or [], start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        node_id = f"ran-node-{index}"
        gnb_ids[name] = node_id
        attributes = deepcopy(item)
        attributes["slices"] = [{"slice_ref": value} for value in item.get("slices") or []]
        nodes.append({"id": node_id, "type": "ran_node", "label": name, "attributes": attributes})
        for slice_ref in item.get("slices") or []:
            if str(slice_ref) in slice_ids:
                links.append(
                    {"source": node_id, "target": slice_ids[str(slice_ref)], "type": "serves_slice", "attributes": {}}
                )
        backhaul_upf = str(item.get("backhaul_upf") or "").strip()
        if backhaul_upf in upf_ids:
            links.append(
                {"source": node_id, "target": upf_ids[backhaul_upf], "type": "tunneled_via", "attributes": {}}
            )

    for index, item in enumerate(scenario.get("ues") or [], start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        node_id = f"ue-node-{index}"
        nodes.append({"id": node_id, "type": "ue", "label": name, "attributes": deepcopy(item)})
        gnb = str(item.get("gnb") or "").strip()
        if gnb in gnb_ids:
            links.append({"source": node_id, "target": gnb_ids[gnb], "type": "attached_to", "attributes": {}})

    return {"nodes": nodes, "links": links}


def _payload_count(payload: Any) -> int:
    return len(payload) if isinstance(payload, list) else int(payload is not None)


def _field_diff(draft: dict[str, Any], on_disk: dict[str, Any]) -> dict[str, Any]:
    """Compare two scenario mappings and report missing / extra / mismatched keys."""
    draft_keys = set(draft.keys())
    disk_keys = set(on_disk.keys())
    missing_on_disk = sorted(draft_keys - disk_keys)
    extra_on_disk = sorted(disk_keys - draft_keys)
    type_mismatches: list[dict[str, Any]] = []
    value_mismatches: list[dict[str, Any]] = []
    for key in sorted(draft_keys & disk_keys):
        draft_val = draft[key]
        disk_val = on_disk[key]
        dt = type(draft_val).__name__
        vt = type(disk_val).__name__
        if dt != vt:
            type_mismatches.append({"key": key, "draft_type": dt, "disk_type": vt})
        elif isinstance(draft_val, list) and isinstance(disk_val, list):
            if len(draft_val) != len(disk_val):
                value_mismatches.append({
                    "key": key,
                    "draft_count": len(draft_val),
                    "disk_count": len(disk_val),
                })
        elif draft_val != disk_val:
            value_mismatches.append({
                "key": key,
                "draft_value": str(draft_val)[:200],
                "disk_value": str(disk_val)[:200],
            })
    has_divergence = bool(missing_on_disk or type_mismatches or value_mismatches)
    return {
        "identical": not has_divergence and not extra_on_disk,
        "missing_on_disk": missing_on_disk,
        "extra_on_disk": extra_on_disk,
        "type_mismatches": type_mismatches,
        "value_mismatches": value_mismatches,
    }


def _simulation_failure_class(payload: dict[str, Any]) -> str:
    text = " ".join(
        str(payload.get(field) or "")
        for field in ("error", "log_tail")
    ).lower()
    if "filenotfounderror" in text and ("graph" in text or "topology" in text):
        return "missing_topology_graph"
    if "exited before readiness" in text or "launcher failed" in text:
        return "launcher_failed"
    if not (payload.get("graph_snapshot") or {}).get("ok", False):
        return "graph_snapshot_not_ready"
    if not (payload.get("gateway_health") or {}).get("ok", False):
        return "gateway_not_ready"
    if not (payload.get("sla_initialization") or {}).get("ok", False):
        return "sla_initialization_failed"
    return "unknown"


_TRACEBACK_SECTION_PATTERNS: list[tuple[tuple[str, ...], str, str]] = [
    (("resource when ns3.slice_isolation is true", "must define resource"), "slices",
     "Every slice must define a resource mapping with capacity_dl_mbps, capacity_ul_mbps, "
     "guaranteed_dl_mbps, and guaranteed_ul_mbps. Verify the written YAML still contains "
     "`resource:` under each slice entry. If the draft already has resource fields, the "
     "issue may be a YAML serialization mismatch — try adjusting field ordering or re-writing."),
    (("slice_id", "unknown slice"), "slices",
     "A slice_ref in ues or flows references a slice label that does not exist. "
     "Check all slice_ref values against the slice labels list."),
    (("upf", "backhaul"), "upfs",
     "A gNB references a UPF that does not exist. Verify backhaul_upf fields in gnbs match upf names."),
    (("gnb does not advertise", "advertise slice"), "ues",
     "A UE is attached to a gNB that does NOT serve one of the UE's session slices. "
     "This is a topology mismatch — you must EITHER: (1) change the UE's gnb field to "
     "point to a gNB that DOES serve the required slice_ref, OR (2) add the missing "
     "slice label to the gNB's slices list. Check which gNB(s) serve each slice and "
     "ensure every UE is attached to a compatible gNB. Always prefer moving the UE "
     "rather than altering the gNB's slice assignments."),
    (("gnb", "attach"), "gnbs",
     "A UE references a gNB name that does not exist. Verify gnb fields in ues match gNB names."),
    (("supi", "imsi"), "ues",
     "A UE supi is invalid or conflicts. Verify SUPI values are unique and correctly formatted."),
    (("app_id", "missing app"), "apps",
     "A UE session or flow references an app_id that does not exist. "
     "If you renamed app_ids, you MUST also replace the ues and flows sections."),
    (("flow", "sla"), "flows",
     "A flow SLA target is missing or invalid. Verify every flow has an sla_target mapping."),
    (("filenotfounderror", "graph", "topology"), "topology",
     "The topology graph file was not found. Ensure the graph_file path under topology is correct."),
    (("filenotfounderror",), "runtime_config",
     "A file referenced in runtime_config (compose_file, ns3_root, etc.) does not exist. "
     "Copy paths exactly from an existing working scenario."),
    (("policy_reload_ms", "member_descriptor"), "split_mode_overlay",
     "The split_mode_overlay ns3 section is missing policy_reload_ms field. "
     "Add `policy_reload_ms: 100` and `activation_poll_ms: 200` to the overlay's ns3 block. "
     "Copy the full ns3 shape from an existing split_mode overlay."),
    (("policy_reload_ms",), "split_mode_overlay",
     "The split_mode_overlay ns3 section is missing or has an invalid policy_reload_ms field. "
     "Add `policy_reload_ms: 100` and `activation_poll_ms: 200` to the overlay's ns3 block."),
]


def _extract_diagnostic_from_log(log_tail: str) -> dict[str, Any]:
    """Parse simulator traceback and suggest which draft section needs repair."""
    normalized = log_tail.lower() if log_tail else ""
    for keywords, section, guidance in _TRACEBACK_SECTION_PATTERNS:
        if all(keyword in normalized for keyword in keywords):
            return {"section": section, "guidance": guidance, "matched_keywords": list(keywords)}
    return {"section": "unknown", "guidance": "Examine the full log_tail to identify the failing entity.", "matched_keywords": []}


def build_environment_tools(
    *,
    compiler: EnvironmentAgentCompiler,
    launcher: EnvironmentLauncher,
    scenario_root: Path,
    execute_simulator: bool = True,
    explorer: ExistingScenarioSpecExplorer | None = None,
    logger: logging.Logger | None = None,
) -> list[Any]:
    explorer = explorer or ExistingScenarioSpecExplorer()
    draft = ScenarioDraftStore()
    feedback_log: list[dict[str, Any]] = []
    sections_requiring_replacement: set[str] = set()

    def list_existing_environment_specs(reason: str = "", limit: int = 50) -> str:
        """Return compact specs for existing scenario YAML files before generating a new env."""
        specs = explorer.discover_specs(Path(scenario_root), limit=limit)
        return _json({"status": "ok", "reason": reason, "spec_count": len(specs), "specs": specs})

    def initialize_environment_draft(
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Initialize one in-memory scenario draft with metadata only."""
        sections_requiring_replacement.clear()
        summary = draft.initialize(metadata or {})
        if logger:
            log_event(logger, "environment_draft_initialized", scenario_id=summary["scenario_id"])
        return _json({"status": "ok", "reason": reason, "draft": draft.summary(compact=True)})

    def replace_draft_section(reason: str = "", section: str = "", payload: Any = None) -> str:
        """Replace one draft section; first writes must follow the declared stage order."""
        summary = draft.replace_section(section, payload)
        sections_requiring_replacement.discard(section)
        if logger:
            log_event(
                logger,
                "environment_section_replaced",
                scenario_id=summary["scenario_id"],
                section=section,
                item_count=_payload_count(payload),
                next_section=summary["next_section"],
            )
        return _json({"status": "ok", "reason": reason, "draft": draft.summary(compact=True)})

    def patch_draft_entity(
        reason: str = "",
        section: str = "",
        entity_id: str = "",
        changes: dict[str, Any] | None = None,
    ) -> str:
        """Patch one existing collection entity by stable ID without adding duplicates."""
        if section in sections_requiring_replacement:
            raise ValueError(
                f"validation requires replace complete section {section}; "
                "use replace_draft_section instead of patch_draft_entity"
            )
        summary = draft.patch_entity(section, entity_id, changes or {})
        if logger:
            log_event(
                logger,
                "environment_entity_patched",
                scenario_id=summary["scenario_id"],
                section=section,
                entity_id=entity_id,
                changed_fields=",".join(sorted((changes or {}).keys())),
            )
        return _json(
            {
                "status": "ok",
                "reason": reason,
                "draft": draft.summary(compact=True),
            }
        )

    def inspect_draft_section(reason: str = "", section: str = "") -> str:
        """Return the requested draft section in full plus a compact draft summary."""
        return _json({"reason": reason, **draft.inspect_section(section)})

    def validate_environment_draft(reason: str = "") -> str:
        """Validate the draft and return a section-level repair plan for any errors."""
        candidate = draft.assemble_candidate()
        report = compiler.validate_candidate(candidate)
        draft.validation_passed = report.ok
        repair_plan = _build_repair_plan(list(report.errors))
        sections_requiring_replacement.clear()
        sections_requiring_replacement.update(
            item["section"]
            for item in repair_plan
            if item["action"] == "replace_draft_section"
        )
        if logger:
            log_event(
                logger,
                "environment_static_validation",
                scenario_id=report.scenario_id,
                status="ok" if report.ok else "failed",
                error_count=len(report.errors),
                warning_count=len(report.warnings),
                repair_sections=",".join(
                    f"{item['section']}:{item['action']}" for item in repair_plan
                ),
            )
        return _json(
            {
                "status": "ok" if report.ok else "failed",
                "reason": reason,
                "scenario_id": report.scenario_id,
                "repair_plan": repair_plan,
                "errors": list(report.errors),
                "warnings": list(report.warnings),
                "draft": draft.summary(compact=True),
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
        topology_graph_path = None
        topology = candidate.scenario.get("topology")
        graph_file = str((topology or {}).get("graph_file") or "").strip() if isinstance(topology, dict) else ""
        if graph_file:
            topology_graph_path = (base_path.parent / graph_file).resolve()
            topology_graph_created = not topology_graph_path.is_file()
            topology_graph_owned = topology_graph_path.stem.lower() == base_path.stem.lower()
            if topology_graph_owned or topology_graph_created:
                _dump_mapping(topology_graph_path, _build_topology_graph(candidate.scenario))
            if logger:
                log_event(
                    logger,
                    "environment_topology_graph_written",
                    scenario_id=candidate.scenario_id,
                    path=topology_graph_path,
                    action=(
                        "created"
                        if topology_graph_created
                        else "refreshed"
                        if topology_graph_owned
                        else "reused"
                    ),
                )
        overlay_path = None
        if candidate.split_mode_overlay is not None:
            overlay_path = Path(scenario_root) / "split_mode" / filename
            overlay = deepcopy(candidate.split_mode_overlay)
            overlay["base_scenario"] = os.path.relpath(
                base_path.resolve(),
                start=overlay_path.parent.resolve(),
            ).replace("\\", "/")
            _dump_mapping(overlay_path, overlay)
        draft.record_written_paths(scenario_path=base_path, split_mode_overlay_path=overlay_path)
        if logger:
            log_event(
                logger,
                "environment_yaml_written",
                scenario_id=candidate.scenario_id,
                scenario_path=draft.scenario_path,
                split_mode_overlay_path=draft.split_mode_overlay_path,
                topology_graph_path=str(topology_graph_path) if topology_graph_path else "",
            )
        return _json(
            {
                "status": "ok",
                "reason": reason,
                "scenario_id": candidate.scenario_id,
                "scenario_path": draft.scenario_path,
                "split_mode_overlay_path": draft.split_mode_overlay_path,
                "topology_graph_path": str(topology_graph_path) if topology_graph_path else "",
            }
        )

    def simulate_candidate_environment(
        reason: str = "",
        run_id: str = "",
        live_graph_snapshot_id: str = "",
        graph_db_url: str = "",
    ) -> str:
        """Execute direct simulator validation for the latest written overlay or base YAML."""
        if not draft.scenario_path:
            raise ValueError("validated YAML must be written before simulation")
        scenario_path = draft.split_mode_overlay_path or draft.scenario_path
        plan = launcher.build_direct_launch_plan(
            scenario_path=Path(scenario_path),
            run_id=run_id or "environment-agent-validation",
            live_graph_snapshot_id=live_graph_snapshot_id or "live-environment-agent",
            graph_db_url=graph_db_url,
        )
        if logger:
            log_event(
                logger,
                "environment_simulation_start",
                scenario_id=draft.summary()["scenario_id"],
                scenario_path=scenario_path,
                run_id=run_id or "environment-agent-validation",
                live_graph_snapshot_id=live_graph_snapshot_id or "live-environment-agent",
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
        if logger:
            fields: dict[str, Any] = {
                "scenario_id": draft.summary()["scenario_id"],
                "status": payload.get("status"),
                "simulator_started": payload.get("simulator_started", False),
            }
            if payload.get("status") != "ok":
                fields["failure_class"] = _simulation_failure_class(payload)
                fields["error"] = str(payload.get("error") or "")
                fields["log_tail"] = str(payload.get("log_tail") or "")[-1200:].replace("\n", "\\n")
            log_event(logger, "environment_simulation_result", **fields)
        if payload.get("status") != "ok":
            log_tail = str(payload.get("log_tail") or "")
            diagnostic = _extract_diagnostic_from_log(log_tail)
            payload["failure_analysis"] = {
                "class": _simulation_failure_class(payload),
                "suggested_section": diagnostic["section"],
                "suggested_action": "replace_draft_section" if diagnostic["section"] != "unknown" else "inspect_draft_section",
                "guidance": diagnostic["guidance"],
                "matched_keywords": diagnostic.get("matched_keywords", []),
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
        if logger:
            log_event(
                logger,
                "environment_feedback_recorded",
                scenario_id=scenario_id,
                phase=phase,
                feedback_count=len(feedback_log),
                feedback=feedback[-500:].replace("\n", "\\n"),
            )
        return _json({"status": "ok", "feedback_count": len(feedback_log), "latest_feedback": item})

    def read_back_written_yaml(reason: str = "") -> str:
        """Read the last written YAML and overlay from disk, returning both raw payloads
        and a field-level diff against the current in-memory draft.

        Use this when a simulation fails but the draft validates — it reveals
        serialization gaps between what the draft tools hold and what the simulator sees.
        """
        if not draft.scenario_path:
            raise ValueError("no YAML has been written yet — call write_validated_environment_yaml first")
        scenario_path = Path(draft.scenario_path)
        overlay_path = Path(draft.split_mode_overlay_path) if draft.split_mode_overlay_path else None

        scenario_on_disk = None
        scenario_load_error = None
        if scenario_path.is_file():
            try:
                scenario_on_disk = _load_mapping(scenario_path)
            except Exception as exc:
                scenario_load_error = f"failed to load scenario YAML: {exc}"

        overlay_on_disk = None
        overlay_load_error = None
        if overlay_path and overlay_path.is_file():
            try:
                overlay_on_disk = _load_mapping(overlay_path)
            except Exception as exc:
                overlay_load_error = f"failed to load overlay YAML: {exc}"

        candidate = draft.assemble_candidate()
        draft_scenario = candidate.scenario
        draft_overlay = candidate.split_mode_overlay

        comparison: dict[str, Any] = {"scenario": _field_diff(draft_scenario, scenario_on_disk or {})}
        if draft_overlay is not None and overlay_on_disk is not None:
            comparison["overlay"] = _field_diff(draft_overlay, overlay_on_disk)

        return _json({
            "status": "ok",
            "reason": reason,
            "scenario_path": draft.scenario_path,
            "split_mode_overlay_path": draft.split_mode_overlay_path,
            "scenario_on_disk": scenario_on_disk,
            "scenario_load_error": scenario_load_error,
            "overlay_on_disk": overlay_on_disk,
            "overlay_load_error": overlay_load_error,
            "comparison": comparison,
            "draft": draft.summary(compact=True),
        })

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
        _make_tool("read_back_written_yaml", read_back_written_yaml.__doc__ or "", read_back_written_yaml),
    ]
