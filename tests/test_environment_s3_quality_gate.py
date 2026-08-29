from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from environment.compiler import EnvironmentAgentCompiler, S3_QUALITY_PROFILE
from environment.contracts import EnvironmentGenerationRequest, ScenarioCandidate
from environment.agent import EnvironmentGenerationAgent
from environment.tools import build_environment_tools


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_ROOT = ROOT / "experiments" / "scenarios"


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _s3_candidate() -> ScenarioCandidate:
    scenario = _load_yaml(SCENARIO_ROOT / "s3_high_complexity.yaml")
    overlay = _load_yaml(SCENARIO_ROOT / "split_mode" / "s3_high_complexity.yaml")
    return ScenarioCandidate(
        scenario_id=str(scenario["scenario_id"]),
        name=str(scenario["name"]),
        scenario=scenario,
        split_mode_overlay=overlay,
    )


def _tool_by_name(tools: list[object], name: str) -> object:
    return next(tool for tool in tools if getattr(tool, "name", "") == name)


def _populate_s3_draft(tools: list[object], candidate: ScenarioCandidate) -> None:
    scenario = candidate.scenario
    initialize = _tool_by_name(tools, "initialize_environment_draft")
    initialize.invoke(
        {
            "metadata": {
                key: scenario[key]
                for key in ("name", "scenario_id", "tick_ms", "seed")
            }
        }
    )
    for section in ("slices", "upfs", "gnbs", "ues", "apps", "flows"):
        _tool_by_name(tools, "replace_draft_section").invoke(
            {"section": section, "payload": deepcopy(scenario[section])}
        )
    runtime_config = {
        key: deepcopy(scenario[key])
        for key in ("free5gc", "ns3", "writer", "topology", "bridge")
    }
    _tool_by_name(tools, "replace_draft_section").invoke(
        {"section": "runtime_config", "payload": runtime_config}
    )
    _tool_by_name(tools, "replace_draft_section").invoke(
        {"section": "split_mode_overlay", "payload": deepcopy(candidate.split_mode_overlay)}
    )


def test_s3_seed_meets_the_production_quality_floor() -> None:
    report = EnvironmentAgentCompiler().validate_candidate(
        _s3_candidate(),
        quality_profile=S3_QUALITY_PROFILE,
    )

    assert report.ok, report.errors


def test_environment_agent_enforces_s3_by_default_without_starting_runtime() -> None:
    agent = EnvironmentGenerationAgent(initialize_runtime=False)

    assert agent.quality_profile == S3_QUALITY_PROFILE


def test_s3_gate_rejects_a_candidate_below_the_flow_floor() -> None:
    candidate = _s3_candidate()
    candidate = ScenarioCandidate(
        scenario_id=candidate.scenario_id,
        name=candidate.name,
        scenario=deepcopy(candidate.scenario),
        split_mode_overlay=deepcopy(candidate.split_mode_overlay),
    )
    candidate.scenario["flows"].pop()

    report = EnvironmentAgentCompiler().validate_candidate(candidate, quality_profile=S3_QUALITY_PROFILE)

    assert not report.ok
    assert "quality profile s3 requires at least 15 flows; found 14" in report.errors


def test_schema_gate_rejects_allocation_below_a_flow_guarantee() -> None:
    candidate = _s3_candidate()
    candidate = ScenarioCandidate(
        scenario_id=candidate.scenario_id,
        name=candidate.name,
        scenario=deepcopy(candidate.scenario),
        split_mode_overlay=deepcopy(candidate.split_mode_overlay),
    )
    candidate.scenario["flows"][0]["allocated_bandwidth_dl_mbps"] = 0

    report = EnvironmentAgentCompiler().validate_candidate(candidate)

    assert not report.ok
    assert any("allocated DL bandwidth is below its guarantee" in error for error in report.errors)


def test_production_draft_tools_apply_the_s3_gate_by_default(tmp_path: Path) -> None:
    candidate = _s3_candidate()
    tools = build_environment_tools(
        compiler=EnvironmentAgentCompiler(),
        launcher=object(),
        scenario_root=tmp_path,
        execute_simulator=False,
    )
    _populate_s3_draft(tools, candidate)

    result = json.loads(_tool_by_name(tools, "validate_environment_draft").invoke({}))

    assert result["status"] == "ok"
    assert result["quality_profile"] == S3_QUALITY_PROFILE
    assert "at least 5 slices" in result["quality_contract"]


def test_production_artifacts_are_isolated_under_generated_and_rewritable_for_repairs(tmp_path: Path) -> None:
    candidate = _s3_candidate()
    tools = build_environment_tools(
        compiler=EnvironmentAgentCompiler(),
        launcher=object(),
        scenario_root=tmp_path,
        execute_simulator=False,
    )
    _populate_s3_draft(tools, candidate)
    validation = json.loads(_tool_by_name(tools, "validate_environment_draft").invoke({}))

    assert validation["status"] == "ok"
    written = json.loads(_tool_by_name(tools, "write_validated_environment_yaml").invoke({}))
    written_again = json.loads(_tool_by_name(tools, "write_validated_environment_yaml").invoke({}))

    assert Path(written["scenario_path"]).parent == tmp_path / "generated"
    assert Path(written["split_mode_overlay_path"]).parent == tmp_path / "split_mode" / "generated"
    assert Path(written["topology_graph_path"]).parent == tmp_path / "generated" / "graphs"
    assert written_again["scenario_path"] == written["scenario_path"]


def test_existing_overlay_is_rejected_before_the_base_scenario_is_written(tmp_path: Path) -> None:
    candidate = _s3_candidate()
    tools = build_environment_tools(
        compiler=EnvironmentAgentCompiler(),
        launcher=object(),
        scenario_root=tmp_path,
        execute_simulator=False,
    )
    _populate_s3_draft(tools, candidate)
    validation = json.loads(_tool_by_name(tools, "validate_environment_draft").invoke({}))
    filename = candidate.scenario_id.lower().replace("-", "_") + ".yaml"
    base_path = tmp_path / "generated" / filename
    overlay_path = tmp_path / "split_mode" / "generated" / filename
    overlay_path.parent.mkdir(parents=True)
    overlay_path.write_text("existing overlay", encoding="utf-8")

    assert validation["status"] == "ok"
    with pytest.raises(FileExistsError, match="split-mode overlay"):
        _tool_by_name(tools, "write_validated_environment_yaml").invoke({})
    assert not base_path.exists()


def test_generation_prompt_makes_the_s3_floor_explicit() -> None:
    prompt = EnvironmentAgentCompiler().build_generation_prompt(
        EnvironmentGenerationRequest(scenario_id="generated-s3", objective="exercise all slices"),
        quality_profile=S3_QUALITY_PROFILE,
    )

    assert "Required quality profile: s3" in prompt
    assert "at least 5 slices" in prompt
    assert "at least 15 flows" in prompt
