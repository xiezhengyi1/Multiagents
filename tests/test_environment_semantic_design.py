from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from environment.compiler import EnvironmentAgentCompiler, S3_QUALITY_PROFILE
from environment.agent import EnvironmentGenerationAgent
from environment.contracts import EnvironmentGenerationRequest
from environment.design import (
    EnvironmentCatalog,
    EnvironmentDesign,
    EnvironmentDesignCompiler,
    EnvironmentDesignError,
)
from environment.tools import build_environment_tools


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_ROOT = ROOT / "experiments" / "scenarios"


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _s3_semantic_fixture() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    scenario = _load_yaml(SCENARIO_ROOT / "s3_high_complexity.yaml")
    overlay = _load_yaml(SCENARIO_ROOT / "split_mode" / "s3_high_complexity.yaml")
    objective = "Exercise an explainable S3 multi-slice ULCL environment"
    catalog = {
        "scenario_id": scenario["scenario_id"],
        "objective": objective,
        "rationale": "The catalog preserves the complete S3 entity manifest before details are compiled.",
        "expected_counts": {
            section: len(scenario[section])
            for section in ("slices", "upfs", "gnbs", "ues", "apps", "flows")
        },
        "slice_keys": [item["label"] for item in scenario["slices"]],
        "upf_keys": [item["name"] for item in scenario["upfs"]],
        "gnb_keys": [item["name"] for item in scenario["gnbs"]],
        "ue_keys": [item["name"] for item in scenario["ues"]],
        "app_keys": [item["app_id"] for item in scenario["apps"]],
        "flow_keys": [item["flow_id"] for item in scenario["flows"]],
    }
    design = {
        "scenario_id": scenario["scenario_id"],
        "name": scenario["name"],
        "objective": objective,
        "rationale": "The topology combines multi-UPF ULCL routing with mixed eMBB, URLLC, and mMTC demand.",
        "tick_ms": scenario["tick_ms"],
        "seed": scenario["seed"],
        "slices": [
            {
                "key": item["label"],
                "sst": item["sst"],
                "sd": item["sd"],
                "resource": deepcopy(item["resource"]),
                "qos": deepcopy(item["qos"]),
                "rationale": f"{item['label']} carries one coherent service class and its explicit capacity budget.",
            }
            for item in scenario["slices"]
        ],
        "upfs": [
            {
                "key": item["name"],
                "name": item["name"],
                "role": item["role"],
                "dnn": item["dnn"],
                "rationale": f"{item['name']} provides the requested {item['role']} role in the ULCL topology.",
            }
            for item in scenario["upfs"]
        ],
        "gnbs": [
            {
                "key": item["name"],
                "name": item["name"],
                "alias": item["alias"],
                "tac": item["tac"],
                "nci": item["nci"],
                "served_slice_keys": list(item["slices"]),
                "backhaul_upf_key": item["backhaul_upf"],
                "rationale": f"{item['name']} exposes overlapping slices for resilient UE placement.",
            }
            for item in scenario["gnbs"]
        ],
        "ues": [
            {
                "key": item["name"],
                "name": item["name"],
                "attached_gnb_key": item["gnb"],
                "target_gnb_key": item["free5gc_policy"]["target_gnb"],
                "preferred_gnb_keys": list(item["free5gc_policy"]["preferred_gnbs"]),
                "rationale": f"{item['name']} is placed where all of its application slices are served.",
            }
            for item in scenario["ues"]
        ],
        "apps": [
            {
                "key": item["app_id"],
                "name": item["name"],
                "ue_key": item["ue_name"],
                "apn": "internet",
                "pdu_session_type": "IPv4",
                "rationale": f"{item['name']} represents a distinct workload assigned to {item['ue_name']}.",
            }
            for item in scenario["apps"]
        ],
        "flows": [
            {
                "key": item["flow_id"],
                "name": item["name"],
                "app_key": item["app_id"],
                "slice_key": item["slice_ref"],
                "five_qi": item["five_qi"],
                "service_type": item["service_type"],
                "traffic": {
                    key: item[key]
                    for key in (
                        "packet_size_bytes",
                        "arrival_rate_pps",
                        "dl_packet_size_bytes",
                        "ul_packet_size_bytes",
                        "dl_arrival_rate_pps",
                        "ul_arrival_rate_pps",
                    )
                },
                "sla": deepcopy(item["sla_target"]),
                "allocated_bandwidth_dl_mbps": item["allocated_bandwidth_dl_mbps"],
                "allocated_bandwidth_ul_mbps": item["allocated_bandwidth_ul_mbps"],
                "optimize_requested": item["optimize_requested"],
                "rationale": f"{item['name']} keeps its LLM-authored traffic and SLA values unchanged.",
            }
            for item in scenario["flows"]
        ],
        "runtime": {
            "free5gc": deepcopy(scenario["free5gc"]),
            "ns3": deepcopy(scenario["ns3"]),
            "writer": deepcopy(scenario["writer"]),
            "bridge": deepcopy(scenario["bridge"]),
            "split_mode_ns3": deepcopy(overlay["ns3"]),
            "split_mode_runtime": deepcopy(overlay["runtime"]),
            "split_mode_radio": deepcopy(overlay["radio"]),
        },
    }
    return catalog, design, scenario


def _tool_by_name(tools: list[object], name: str) -> object:
    return next(tool for tool in tools if getattr(tool, "name", "") == name)


def test_semantic_s3_design_compiles_deterministically_without_changing_flow_metrics() -> None:
    catalog_payload, design_payload, source = _s3_semantic_fixture()
    catalog = EnvironmentCatalog.model_validate(catalog_payload)
    design = EnvironmentDesign.model_validate(design_payload)
    compiler = EnvironmentDesignCompiler()

    first = compiler.compile(catalog, design)
    second = compiler.compile(catalog, design)
    report = EnvironmentAgentCompiler().validate_candidate(
        first.candidate,
        quality_profile=S3_QUALITY_PROFILE,
    )

    assert first == second
    assert report.ok, report.errors
    for source_flow, compiled_flow in zip(source["flows"], first.candidate.scenario["flows"], strict=True):
        assert compiled_flow["sla_target"] == source_flow["sla_target"]
        assert compiled_flow["allocated_bandwidth_dl_mbps"] == source_flow["allocated_bandwidth_dl_mbps"]
        assert compiled_flow["allocated_bandwidth_ul_mbps"] == source_flow["allocated_bandwidth_ul_mbps"]
        assert compiled_flow["packet_size_bytes"] == source_flow["packet_size_bytes"]
        assert compiled_flow["dl_arrival_rate_pps"] == source_flow["dl_arrival_rate_pps"]
        assert compiled_flow["ul_arrival_rate_pps"] == source_flow["ul_arrival_rate_pps"]


def test_environment_agent_defaults_to_the_semantic_design_protocol() -> None:
    agent = EnvironmentGenerationAgent(initialize_runtime=False)
    prompt = agent.compiler.build_semantic_generation_prompt(
        EnvironmentGenerationRequest(
            scenario_id="semantic-s3",
            objective="Design an explainable S3 environment",
        )
    )

    assert "submit_environment_catalog" in agent.system_prompt
    assert "compile_environment_design" in agent.system_prompt
    assert "revise_environment_design" in agent.system_prompt
    assert "prepare_environment_design_context" in agent.system_prompt
    assert "Only gNB entities have an alias" not in agent.system_prompt
    assert "`alias` exists only on a gNB" in agent.system_prompt
    assert "replace_draft_section" not in agent.system_prompt
    assert "exact flow metrics" in " ".join(prompt.split())


def test_catalog_makes_a_missing_flow_detectable_before_yaml_assembly() -> None:
    catalog_payload, design_payload, _ = _s3_semantic_fixture()
    design_payload["flows"].pop()

    with pytest.raises(EnvironmentDesignError) as raised:
        EnvironmentDesignCompiler().compile(
            EnvironmentCatalog.model_validate(catalog_payload),
            EnvironmentDesign.model_validate(design_payload),
        )

    assert any("is missing catalog keys" in error for error in raised.value.errors)
    assert catalog_payload["flow_keys"][-1] in " ".join(raised.value.errors)


def test_semantic_toolchain_excludes_legacy_yaml_mutation_tools_and_writes_design_report(tmp_path: Path) -> None:
    catalog, design, _ = _s3_semantic_fixture()
    tools = build_environment_tools(
        compiler=EnvironmentAgentCompiler(),
        launcher=object(),
        scenario_root=tmp_path,
        execute_simulator=False,
        semantic_design_mode=True,
    )
    tool_names = {getattr(tool, "name", "") for tool in tools}

    assert "compile_environment_design" in tool_names
    assert "revise_environment_design" in tool_names
    assert "prepare_environment_design_context" in tool_names
    assert "submit_environment_catalog" in tool_names
    assert "record_validation_feedback" not in tool_names
    assert "replace_draft_section" not in tool_names
    assert "patch_draft_entity" not in tool_names

    compile_schema = _tool_by_name(tools, "compile_environment_design").args_schema.model_json_schema()
    assert "EnvironmentDesign" in compile_schema["$defs"]
    assert "FlowSlaDesign" in compile_schema["$defs"]
    catalog_result = json.loads(_tool_by_name(tools, "submit_environment_catalog").invoke({"catalog": catalog}))
    compile_result = json.loads(_tool_by_name(tools, "compile_environment_design").invoke({"design": design}))
    validation_result = json.loads(_tool_by_name(tools, "validate_environment_draft").invoke({}))
    written = json.loads(_tool_by_name(tools, "write_validated_environment_yaml").invoke({}))

    assert catalog_result["status"] == "ok"
    assert compile_result["status"] == "ok"
    assert validation_result["status"] == "ok"
    design_report = json.loads(Path(written["design_report_path"]).read_text(encoding="utf-8"))
    assert design_report["scenario_rationale"] == design["rationale"]
    assert design_report["design"]["flows"][0]["rationale"] == design["flows"][0]["rationale"]


def test_semantic_revision_repairs_only_affected_entity_and_blocks_premature_yaml_validation(
    tmp_path: Path,
) -> None:
    catalog, design, _ = _s3_semantic_fixture()
    original_flow = deepcopy(design["flows"][0])
    app_by_key = {item["key"]: item for item in design["apps"]}
    ue_by_key = {item["key"]: item for item in design["ues"]}
    gnb_by_key = {item["key"]: item for item in design["gnbs"]}
    app = app_by_key[original_flow["app_key"]]
    ue = ue_by_key[app["ue_key"]]
    served = set(gnb_by_key[ue["attached_gnb_key"]]["served_slice_keys"])
    incompatible_slice = next(key for key in catalog["slice_keys"] if key not in served)
    design["flows"][0]["slice_key"] = incompatible_slice

    tools = build_environment_tools(
        compiler=EnvironmentAgentCompiler(),
        launcher=object(),
        scenario_root=tmp_path,
        execute_simulator=False,
        semantic_design_mode=True,
    )
    _tool_by_name(tools, "submit_environment_catalog").invoke({"catalog": catalog})
    failed = json.loads(
        _tool_by_name(tools, "compile_environment_design").invoke({"design": design})
    )
    guarded = json.loads(_tool_by_name(tools, "validate_environment_draft").invoke({}))

    assert failed["status"] == "failed"
    assert failed["required_action"] == "revise_environment_design"
    assert "flow_app_and_slice" in failed["repair_context"]
    assert guarded["status"] == "blocked"
    assert guarded["phase"] == "workflow_guard"
    assert not any("missing required top-level sections" in error for error in guarded["errors"])

    repaired = json.loads(
        _tool_by_name(tools, "revise_environment_design").invoke(
            {"revision": {"flows": [original_flow]}}
        )
    )

    assert repaired["status"] == "ok"
    assert repaired["revision_applied"] == [f"flows.{original_flow['key']}"]
    assert json.loads(_tool_by_name(tools, "validate_environment_draft").invoke({}))["status"] == "ok"


def test_combined_design_context_loads_runtime_only_from_the_trusted_scenario_root() -> None:
    tools = build_environment_tools(
        compiler=EnvironmentAgentCompiler(),
        launcher=object(),
        scenario_root=SCENARIO_ROOT,
        execute_simulator=False,
        semantic_design_mode=True,
    )

    runtime_reference = json.loads(_tool_by_name(tools, "prepare_environment_design_context").invoke({}))

    assert runtime_reference["runtime"]["free5gc"]["mode"] == "ulcl"
    assert runtime_reference["runtime"]["split_mode_ns3"]["policy_reload_ms"] == 100
    assert runtime_reference["spec_count"] > 0


def test_infeasible_llm_metric_is_reported_without_silent_repair(tmp_path: Path) -> None:
    catalog, design, _ = _s3_semantic_fixture()
    design["flows"][0]["allocated_bandwidth_dl_mbps"] = 0
    tools = build_environment_tools(
        compiler=EnvironmentAgentCompiler(),
        launcher=object(),
        scenario_root=tmp_path,
        execute_simulator=False,
        semantic_design_mode=True,
    )
    _tool_by_name(tools, "submit_environment_catalog").invoke({"catalog": catalog})

    result = json.loads(_tool_by_name(tools, "compile_environment_design").invoke({"design": design}))

    assert result["status"] == "failed"
    assert result["phase"] == "compiled_candidate_validation"
    assert any("allocated DL bandwidth is below its guarantee" in error for error in result["errors"])
    assert result["metric_policy"].startswith("The compiler did not change")
    assert design["flows"][0]["allocated_bandwidth_dl_mbps"] == 0


def test_s3_catalog_floor_is_rejected_before_detailed_design(tmp_path: Path) -> None:
    catalog, _, _ = _s3_semantic_fixture()
    catalog["expected_counts"]["flows"] = 14
    catalog["flow_keys"].pop()
    tools = build_environment_tools(
        compiler=EnvironmentAgentCompiler(),
        launcher=object(),
        scenario_root=tmp_path,
        execute_simulator=False,
        semantic_design_mode=True,
    )

    result = json.loads(_tool_by_name(tools, "submit_environment_catalog").invoke({"catalog": catalog}))

    assert result["status"] == "failed"
    assert any("requires at least 15 flows" in error for error in result["errors"])
