from pathlib import Path

import pytest
from pydantic import ValidationError

from shared.contracts import AgentMessage, AgentRunResult, ExecutionScope, ToolCall
from control_runtime.agents.grounding.contracts import FlowSelector, GroundingDecision
from control_runtime.agents.main.contracts import ControlDomain, GlobalControlIntent
from control_runtime.agents.planning.compiler import OptimizationStrategyCompiler
from control_runtime.agents.planning.contracts import (
    PlanningContext,
    PlanningRequest,
    SharedControlContext,
)
from control_runtime.agents.planning.planning_validation import PlanningAdvisorValidator
from control_runtime.agents.planning.response_models import OsaAdvisorOutput
from control_runtime.agents.single import agent as single_agent


def _planning_request() -> PlanningRequest:
    intent = GlobalControlIntent(
        scope=ExecutionScope(
            session_id="session-1",
            thread_id="session-1",
            snapshot_id="snapshot-1",
            round_index=1,
            supi="imsi-1",
        ),
        next_agent="optimization_strategy",
        requested_domains=[ControlDomain.QOS],
    )
    return PlanningRequest(
        grounding_decision=GroundingDecision(
            flows=[
                FlowSelector(
                    supi="imsi-1",
                    app_id="app-1",
                    flow_id="flow-1",
                    name="video",
                    service_type_id=1,
                )
            ]
        ),
        context=PlanningContext(
            session_id="session-1",
            snapshot_id="snapshot-1",
            shared_context=SharedControlContext(main_intent=intent),
        ),
    )


def _optimizer_evidence() -> dict:
    return {
        "latest_optimizer_preview": {
            "status": "approved",
            "qos_flow_assignments": [
                {
                    "flow_id": "flow-1",
                    "new_slice": "01000001",
                    "bw_ul": 10,
                    "bw_dl": 20,
                }
            ],
        }
    }


def _valid_osa_wire() -> dict:
    return {
        "planning_status": "executable_plan",
        "rationale": "optimizer allocation is feasible",
        "sm_policies": [
            {
                "flow_id": "flow-1",
                "app_id": "app-1",
                "priority": 1,
                "target_latency_ms": 10,
                "packet_error_rate": 0.001,
                "max_br_ul_mbps": 10,
                "max_br_dl_mbps": 20,
            }
        ],
    }


def _result(payload: dict, messages: list[AgentMessage] | None = None) -> AgentRunResult:
    return AgentRunResult(
        messages=messages or [AgentMessage(role="assistant", content="{}")],
        structured_response=payload,
    )


def test_single_rejects_legacy_osa_aliases_without_runtime_normalization() -> None:
    legacy_single_wire = _result({"smPolicies": []})

    assert not hasattr(single_agent, "_SingleAgentOutputNormalizingLlm")
    assert not hasattr(single_agent, "SingleAgentRoundDecision")
    with pytest.raises(ValidationError, match="smPolicies"):
        single_agent.coerce_structured_response(
            legacy_single_wire,
            OsaAdvisorOutput,
            error_message="Single control agent returned no OsaAdvisorOutput",
        )


def test_single_osa_wire_compiles_equivalently_to_direct_planning_wire() -> None:
    raw_wire = _valid_osa_wire()
    single_output = single_agent.coerce_structured_response(
        _result(raw_wire),
        OsaAdvisorOutput,
        error_message="Single control agent returned no OsaAdvisorOutput",
    )
    direct_output = OsaAdvisorOutput.model_validate(raw_wire)

    compiler = OptimizationStrategyCompiler()
    single_plan = compiler.assemble_policy_plan(
        advisor_output=single_output,
        planning_request=_planning_request(),
        planning_tool_evidence=_optimizer_evidence(),
    )
    direct_plan = compiler.assemble_policy_plan(
        advisor_output=direct_output,
        planning_request=_planning_request(),
        planning_tool_evidence=_optimizer_evidence(),
    )

    assert single_plan.model_dump(mode="json") == direct_plan.model_dump(mode="json")
    assert single_plan.all_policies[0].policy_id == "smp-app-1-flow-1"


def test_single_hidden_planning_request_preserves_catalog_allocation_for_slice_migration() -> None:
    agent = object.__new__(single_agent.SingleControlAgent)
    planning_request = agent._build_single_planning_request(
        advisor_output=OsaAdvisorOutput.model_validate(_valid_osa_wire()),
        advisor_result=_result(
            _valid_osa_wire(),
            messages=[
                AgentMessage(role="assistant", tool_calls=[ToolCall(id="call-1", name="get_sm_ue_flow_catalog", arguments={})]),
                AgentMessage(role="tool", name="get_sm_ue_flow_catalog", tool_call_id="call-1", tool_status="success", content='{"flow_catalog":[{"supi":"imsi-1","app_id":"app-1","flow_id":"flow-1","flow_name":"video","allocation":{"current_slice_snssai":"01000001","allocated_bandwidth_ul":10,"allocated_bandwidth_dl":20}}]}'),
                AgentMessage(role="assistant", tool_calls=[ToolCall(id="call-2", name="get_ue_slice_subscription", arguments={})]),
                AgentMessage(role="tool", name="get_ue_slice_subscription", tool_call_id="call-2", tool_status="success", content='{"authority":"subscription","authorized_snssais":["01000002"]}'),
                AgentMessage(role="assistant", content="{}"),
            ],
        ),
        user_input="move off the current slice for imsi-1 video",
        supi="imsi-1",
        session_id="session-1",
        snapshot_id="snapshot-1",
        round_index=1,
        feedback_context="",
        round_traces=[],
        previous_mediator_decision={},
    )

    flow = planning_request.grounding_decision.flows[0]
    assert flow.current_slice_snssai == "01000001"
    assert flow.current_bw_ul == 10
    assert flow.current_bw_dl == 20
    constraint = planning_request.grounding_decision.qos_operation_constraints[0]
    assert constraint.source_slice_snssai is None
    assert PlanningAdvisorValidator._required_slice_change_constraints(planning_request) == [
        {"flow_id": "flow-1", "source_slice_snssai": "01000001"}
    ]


def test_removed_compatibility_modules_are_not_packaged() -> None:
    project_root = Path(__file__).resolve().parents[1]
    for relative_path in (
        "src/control_runtime/agents/single/contracts.py",
        "src/control_runtime/agents/planning/planning_artifact_compiler.py",
        "src/control_runtime/agents/planning/tool_result_adapter.py",
        "src/control_runtime/agents/planning/identifiers.py",
        "src/control_runtime/agents/planning/retry.py",
    ):
        assert not (project_root / relative_path).exists()
