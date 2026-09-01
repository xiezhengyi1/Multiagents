from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from environment.compiler import EnvironmentAgentCompiler, S3_QUALITY_PROFILE
from environment.contracts import EnvironmentGenerationRequest, ScenarioCandidate
from environment.agent import EnvironmentGenerationAgent
from shared.contracts import ExecutionScope


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_ROOT = ROOT / "experiments" / "scenarios"


def _environment_scope() -> ExecutionScope:
    return ExecutionScope(
        session_id="environment-test",
        thread_id="environment-test",
        snapshot_id="live-environment-test",
    )


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


def test_generation_prompt_makes_the_s3_floor_explicit() -> None:
    prompt = EnvironmentAgentCompiler().build_semantic_generation_prompt(
        EnvironmentGenerationRequest(
            scope=_environment_scope(),
            scenario_id="generated-s3",
            objective="exercise all slices",
        ),
        quality_profile=S3_QUALITY_PROFILE,
    )

    assert "required quality profile: s3" in prompt
    assert "at least 5 slices" in prompt
    assert "at least 15 flows" in prompt
