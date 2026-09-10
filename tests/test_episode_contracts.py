from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import ValidationError

from control_runtime.actors import (
    ControlRequest,
    MainActor,
    MainProposal,
    MainValidator,
    PpaProposal,
    PpaValidator,
    SbaArtifact,
    SbaCompiler,
    SbaProposal,
    SbaValidator,
)
from control_runtime.actors.main import (
    MAIN_SYSTEM_PROMPT,
    ControlSemanticMode,
    MainControlSemantics,
    MainControlStage,
    MainRetryScope,
    MainSemanticTarget,
    SemanticGoal,
    SemanticTargetType,
    StageTrigger,
    validate_main_proposal,
)
from control_runtime.actors.ppa import PPA_SYSTEM_PROMPT
from control_runtime.actors.sba import (
    SBA_SYSTEM_PROMPT,
    GroundedFlowBinding,
    QosOperationConstraint,
    SliceMigrationDecision,
)
from control_runtime.contracts import ControlDomain, ObjectiveProfile
from control_runtime.graph import (
    AllowedEdge,
    ContextBuilder,
    EpisodeGraph,
    EventEmission,
    EventRouter,
    RepairBudget,
    RepairLimits,
)
from control_runtime.simulation import (
    SBA_OBSERVATION_TOOLS,
    EnvironmentObservation,
    EnvironmentSnapshot,
    ExecutionFeedback,
    FunctionSimulatorAdapter,
    ObservationRequest,
    OperationReceipt,
    SimulatorObservationSink,
    SnapshotEntity,
    SnapshotMetric,
    SnapshotRelation,
)
from shared.contracts.coordination import (
    CoordinationDecision,
    CoordinationEvent,
    HandoffPayload,
    ObservationPayload,
    ProposalPayload,
    RepairRequestPayload,
    TerminalPayload,
    ValidationIssue,
    ValidationResultPayload,
)
from shared.contracts.model_trace import ModelMessage, ModelToolCall, ModelToolSpec
from shared.runtime.journal import EpisodeJournal
from shared.runtime.langchain_trace import (
    AgentRunContext,
    AtomicTraceMiddleware,
    snapshot_langchain_tool,
)


class SampleProposal(ProposalPayload):
    decision: str


class SampleObservation(ObservationPayload):
    policy_revision: int


class SampleHandoff(HandoffPayload):
    status: str


def test_coordination_payload_must_match_message_type() -> None:
    with pytest.raises(ValidationError, match="only supports message types"):
        CoordinationEvent[SampleProposal](
            event_id="episode-1:event:000000",
            event_index=0,
            scope={"episode_id": "episode-1", "round_index": 0},
            source_agent="main",
            target_agent="sba",
            message_type="observation",
            payload=SampleProposal(decision="ground"),
        )


def test_structural_repair_references_failed_trace_without_fake_proposal() -> None:
    event = CoordinationEvent[RepairRequestPayload](
        event_id="episode-1:event:000001",
        event_index=1,
        scope={"episode_id": "episode-1", "round_index": 0},
        source_agent="sba_validator",
        target_agent="sba",
        message_type="repair_request",
        in_reply_to="episode-1:event:000000",
        payload=RepairRequestPayload(
            issues=[ValidationIssue(code="invalid_tool_args", message="missing flows")],
            attempt=1,
            max_attempts=3,
            failed_trace_id="episode-1:trace:000000",
        ),
    )

    assert event.invalidated_artifact_ids == []
    assert event.payload.failed_trace_id == "episode-1:trace:000000"


def test_episode_journal_writes_only_events_and_atomic_traces(tmp_path: Path) -> None:
    journal = EpisodeJournal(root=tmp_path, episode_id="episode-1")
    trigger = journal.append_event(
        round_index=0,
        source_agent="episode_controller",
        target_agent="main",
        message_type="observation",
        payload=SampleObservation(policy_revision=7),
    )
    tool_spec = ModelToolSpec(
        name="inspect_pcf",
        description="Read current PCF state",
        input_schema={"type": "object", "properties": {}},
    )
    first = journal.append_trace(
        trigger_event_id=trigger.event_id,
        round_index=0,
        agent_name="main",
        model_name="test-model",
        input_messages=[ModelMessage(role="user", content="inspect policy")],
        output_message=ModelMessage(
            role="assistant",
            content="",
            tool_calls=[
                ModelToolCall(
                    tool_call_id="call-1",
                    name="inspect_pcf",
                    arguments={},
                )
            ],
        ),
        tools=[tool_spec],
        status="success",
    )
    second = journal.append_trace(
        trigger_event_id=trigger.event_id,
        round_index=0,
        agent_name="main",
        model_name="test-model",
        input_messages=[
            ModelMessage(role="user", content="inspect policy"),
            ModelMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ModelToolCall(
                        tool_call_id="call-1",
                        name="inspect_pcf",
                        arguments={},
                    )
                ],
            ),
            ModelMessage(role="tool", content={"revision": 7}, tool_call_id="call-1"),
        ],
        output_message=ModelMessage(role="assistant", content={"decision": "continue"}),
        tools=[tool_spec],
        status="success",
    )
    terminal = journal.append_event(
        round_index=0,
        source_agent="episode_controller",
        target_agent="episode_controller",
        message_type="terminal",
        in_reply_to=trigger.event_id,
        input_artifact_ids=[trigger.event_id],
        payload=TerminalPayload(
            outcome="success",
            reason="objective reached",
            reward_total=1.0,
            reward_components={"objective": 1.0},
        ),
    )

    assert first.call_index == 0
    assert second.call_index == 1
    assert terminal.event_index == 1
    assert sorted(path.name for path in journal.directory.iterdir()) == [
        "events.jsonl",
        "traces.jsonl",
    ]
    event_rows = [
        json.loads(row)
        for row in journal.events_path.read_text(encoding="utf-8").splitlines()
    ]
    trace_rows = [
        json.loads(row)
        for row in journal.traces_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event_index"] for row in event_rows] == [0, 1]
    assert [row["call_index"] for row in trace_rows] == [0, 1]
    assert trace_rows[0]["output_message"]["tool_calls"][0]["name"] == "inspect_pcf"


def test_terminal_reward_total_is_reproducible_from_components() -> None:
    with pytest.raises(ValidationError, match="sum of reward_components"):
        TerminalPayload(
            outcome="failure",
            reason="budget exhausted",
            reward_total=-1.0,
            reward_components={"invalid_output": -0.5},
        )


def test_langchain_middleware_records_each_real_model_call(tmp_path: Path) -> None:
    journal = EpisodeJournal(root=tmp_path, episode_id="episode-trace")
    trigger = journal.append_event(
        round_index=0,
        source_agent="episode_controller",
        target_agent="main",
        message_type="observation",
        payload=SampleObservation(policy_revision=3),
    )

    @tool
    def inspect_policy(policy_id: str) -> dict[str, int]:
        """Read one policy revision."""

        return {"revision": 3}

    context = AgentRunContext(
        journal=journal,
        trigger_event_id=trigger.event_id,
        round_index=0,
        agent_name="main",
        model_name="test-model",
    )
    request = ModelRequest(
        model=cast(BaseChatModel, object()),
        messages=[HumanMessage(content="inspect p-1")],
        system_message=SystemMessage(content="You are Main."),
        tools=[inspect_policy],
        runtime=cast(Any, SimpleNamespace(context=context)),
    )
    response = ModelResponse(
        result=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-policy",
                        "name": "inspect_policy",
                        "args": {"policy_id": "p-1"},
                        "type": "tool_call",
                    }
                ],
                usage_metadata={
                    "input_tokens": 12,
                    "output_tokens": 5,
                    "total_tokens": 17,
                },
            )
        ]
    )

    returned = AtomicTraceMiddleware().wrap_model_call(request, lambda _: response)

    assert returned is response
    row = json.loads(journal.traces_path.read_text(encoding="utf-8"))
    assert row["input_messages"][0] == {
        "role": "system",
        "content": "You are Main.",
        "tool_calls": [],
        "tool_call_id": None,
    }
    assert row["output_message"]["tool_calls"][0]["arguments"] == {"policy_id": "p-1"}
    assert row["tools"][0]["name"] == "inspect_policy"
    assert row["usage"] == {"input_tokens": 12, "output_tokens": 5}


def test_episode_journal_serializes_parallel_events(tmp_path: Path) -> None:
    journal = EpisodeJournal(root=tmp_path, episode_id="episode-parallel-events")

    def append(value: int) -> None:
        journal.append_event(
            round_index=0,
            source_agent="simulator_adapter",
            target_agent="sba",
            message_type="observation",
            payload=SampleObservation(policy_revision=value),
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append, range(32)))

    rows = [
        json.loads(line)
        for line in journal.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event_index"] for row in rows] == list(range(32))
    assert len({row["event_id"] for row in rows}) == 32


def test_langchain_middleware_hides_structured_response_tool(tmp_path: Path) -> None:
    journal = EpisodeJournal(root=tmp_path, episode_id="episode-structured-trace")
    trigger = journal.append_event(
        round_index=0,
        source_agent="episode_controller",
        target_agent="main",
        message_type="observation",
        payload=SampleObservation(policy_revision=3),
    )
    context = AgentRunContext(
        journal=journal,
        trigger_event_id=trigger.event_id,
        round_index=0,
        agent_name="main",
        model_name="test-model",
    )
    schema = CoordinationDecision[SampleProposal]
    inspect_policy = {
        "type": "function",
        "function": {
            "name": "inspect_policy",
            "description": "Inspect one policy.",
            "parameters": {
                "type": "object",
                "properties": {"policy_id": {"type": "string"}},
                "required": ["policy_id"],
            },
        },
    }
    response_tool = {
        "type": "function",
        "function": {
            "name": schema.__name__,
            "description": "Return the structured response.",
            "parameters": schema.model_json_schema(),
        },
    }
    request = ModelRequest(
        model=cast(BaseChatModel, object()),
        messages=[HumanMessage(content="decide")],
        tools=[inspect_policy, response_tool],
        runtime=cast(Any, SimpleNamespace(context=context)),
    )
    arguments = {
        "target_agent": "main_validator",
        "message_type": "proposal",
        "payload": {"decision": "accept"},
    }
    response = ModelResponse(
        result=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-response",
                        "name": schema.__name__,
                        "args": arguments,
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )

    AtomicTraceMiddleware(response_schema=schema).wrap_model_call(
        request, lambda _: response
    )

    row = json.loads(journal.traces_path.read_text(encoding="utf-8"))
    assert [tool["name"] for tool in row["tools"]] == ["inspect_policy"]
    assert row["output_message"]["content"] == arguments
    assert row["output_message"]["tool_calls"] == []


def test_langgraph_routes_bidirectional_repairs_as_ordered_events(
    tmp_path: Path,
) -> None:
    journal = EpisodeJournal(root=tmp_path, episode_id="episode-graph")
    initial = journal.append_event(
        round_index=0,
        source_agent="episode_controller",
        target_agent="main",
        message_type="observation",
        payload=SampleObservation(policy_revision=1),
    )
    main_calls = 0

    def main_handler(state: Any) -> EventEmission:
        nonlocal main_calls
        current = state["current_event"]
        main_calls += 1
        message_type = "proposal" if main_calls == 1 else "repair_response"
        return EventEmission(
            decision=CoordinationDecision[SampleProposal](
                target_agent="main_validator",
                message_type=message_type,
                payload=SampleProposal(decision=f"attempt-{main_calls}"),
            ),
            in_reply_to=current.event_id,
            input_artifact_ids=(current.event_id,),
            invalidated_artifact_ids=(),
        )

    def validator_handler(state: Any) -> EventEmission:
        current = state["current_event"]
        if current.message_type == "proposal":
            return EventEmission(
                decision=CoordinationDecision[RepairRequestPayload](
                    target_agent="main",
                    message_type="repair_request",
                    payload=RepairRequestPayload(
                        issues=[
                            ValidationIssue(
                                code="missing_evidence", message="inspect PCF"
                            )
                        ],
                        attempt=1,
                        max_attempts=2,
                    ),
                ),
                in_reply_to=current.event_id,
                input_artifact_ids=(),
                invalidated_artifact_ids=(current.event_id,),
            )
        return EventEmission(
            decision=CoordinationDecision[ValidationResultPayload](
                target_agent="episode_controller",
                message_type="validation_result",
                payload=ValidationResultPayload(notes=["accepted"]),
            ),
            in_reply_to=current.event_id,
            input_artifact_ids=(current.event_id,),
            invalidated_artifact_ids=(),
        )

    def controller_handler(state: Any) -> EventEmission:
        current = state["current_event"]
        return EventEmission(
            decision=CoordinationDecision[TerminalPayload](
                target_agent="episode_controller",
                message_type="terminal",
                payload=TerminalPayload(
                    outcome="success",
                    reason="validated",
                    reward_total=1.0,
                    reward_components={"validation": 1.0},
                ),
            ),
            in_reply_to=current.event_id,
            input_artifact_ids=(current.event_id,),
            invalidated_artifact_ids=(),
        )

    edges = [
        AllowedEdge("episode_controller", "main", "observation"),
        AllowedEdge("main", "main_validator", "proposal"),
        AllowedEdge("main_validator", "main", "repair_request"),
        AllowedEdge("main", "main_validator", "repair_response"),
        AllowedEdge("main_validator", "episode_controller", "validation_result"),
        AllowedEdge("episode_controller", "episode_controller", "terminal"),
    ]
    episode_graph = EpisodeGraph(
        journal=journal,
        router=EventRouter(edges, controller_actor="episode_controller"),
        handlers={
            "main": main_handler,
            "main_validator": validator_handler,
            "episode_controller": controller_handler,
        },
        retry_policies={
            "main": None,
            "main_validator": None,
            "episode_controller": None,
        },
    )

    terminal = episode_graph.invoke(initial)

    assert terminal.message_type == "terminal"
    rows = [
        json.loads(row)
        for row in journal.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["message_type"] for row in rows] == [
        "observation",
        "proposal",
        "repair_request",
        "repair_response",
        "validation_result",
        "terminal",
    ]
    assert [row["event_index"] for row in rows] == list(range(6))


def test_execution_feedback_preserves_partial_success_without_rollback() -> None:
    observed_at = datetime.now(UTC)
    snapshot = EnvironmentSnapshot(
        snapshot_id="snapshot-2",
        observed_at=observed_at,
        entities=[
            SnapshotEntity(
                entity_id="ue:1", entity_type="ue", attributes={"supi": "imsi-1"}
            ),
            SnapshotEntity(
                entity_id="flow:1", entity_type="flow", attributes={"latency": 9.0}
            ),
        ],
        relations=[
            SnapshotRelation(
                relation_id="ue-flow",
                relation_type="owns",
                source_entity_id="ue:1",
                target_entity_id="flow:1",
            )
        ],
        metrics=[
            SnapshotMetric(
                owner_entity_id="flow:1",
                metric_name="latency_ms",
                value=9.0,
                observed_at=observed_at,
            )
        ],
    )

    feedback = ExecutionFeedback(
        status="partial_success",
        receipts=[
            OperationReceipt(
                operation_id="op-1", status="success", result={"revision": 2}
            ),
            OperationReceipt(operation_id="op-2", status="failed", error="rejected"),
        ],
        snapshot=snapshot,
        objective_metrics={"sla_satisfaction": 0.5},
    )

    assert [receipt.status for receipt in feedback.receipts] == ["success", "failed"]
    assert feedback.snapshot.snapshot_id == "snapshot-2"


def _main_proposal(*, evidence: bool) -> MainProposal:
    return MainProposal(
        supi="imsi-12345",
        requested_domains=[ControlDomain.QOS],
        serving_slice_change_requested=False,
        domain_evidence={
            "qos": ["explicit qos request"],
            "mobility": [],
        }
        if evidence
        else {},
        control_semantics=MainControlSemantics(
            mode=ControlSemanticMode.SINGLE_STEP,
            current_stage=1,
            stages=[
                MainControlStage(
                    stage_index=1,
                    name="protect-flow",
                    trigger=StageTrigger.INITIAL,
                    summary="protect the named flow",
                    targets=[
                        MainSemanticTarget(
                            semantic_name="video",
                            target_type=SemanticTargetType.APP,
                            goal=SemanticGoal.PROTECT,
                            metric_focus="latency",
                            note="preserve latency",
                            supi="imsi-12345",
                        )
                    ],
                )
            ],
            notes=[],
        ),
        objective_profile=ObjectiveProfile(
            profile_name="latency",
            sla_violation_cost=1.0,
            mobility_risk_cost=0.0,
            control_churn_cost=0.2,
            resource_pressure_cost=0.5,
            fairness_cost=0.2,
        ),
        investigation_targets=[],
        uncertainty_flags=[],
        retry_scope=MainRetryScope.FULL_REGROUND,
        required_evidence=["flow binding"],
        forbidden_assumptions=["invented flow id"],
        intent_encoding_guidance="",
    )


def test_main_accepts_a_scoped_multi_supi_current_stage() -> None:
    original = _main_proposal(evidence=True)
    target = original.control_semantics.stages[0].targets[0].model_copy(
        update={"supi": "imsi-208930000000001"}
    )
    stage = original.control_semantics.stages[0].model_copy(update={"targets": [target]})
    semantics = original.control_semantics.model_copy(update={"stages": [stage]})
    proposal = original.model_copy(
        update={"supi": "imsi-208930000000001", "control_semantics": semantics}
    )

    issues = validate_main_proposal(
        proposal=proposal,
        target_agent="sba",
        message_type="proposal",
        trigger_message_type="control_request",
        user_input=(
            "Protect imsi-208930000000001 first, then imsi-208930000000006."
        ),
        round_index=0,
    )

    assert issues == []


def test_main_rejects_multi_supi_target_outside_active_scope() -> None:
    original = _main_proposal(evidence=True)
    target = original.control_semantics.stages[0].targets[0].model_copy(
        update={"supi": "imsi-208930000000006"}
    )
    stage = original.control_semantics.stages[0].model_copy(update={"targets": [target]})
    semantics = original.control_semantics.model_copy(update={"stages": [stage]})
    proposal = original.model_copy(
        update={"supi": "imsi-208930000000001", "control_semantics": semantics}
    )

    issues = validate_main_proposal(
        proposal=proposal,
        target_agent="sba",
        message_type="proposal",
        trigger_message_type="control_request",
        user_input=(
            "Protect imsi-208930000000001 first, then imsi-208930000000006."
        ),
        round_index=0,
    )

    assert issues == []


def test_main_accepts_multi_supi_targets_without_a_top_level_scope() -> None:
    original = _main_proposal(evidence=True)
    first_target = original.control_semantics.stages[0].targets[0].model_copy(
        update={"supi": "imsi-208930000000001"}
    )
    second_target = first_target.model_copy(
        update={"semantic_name": "medical", "supi": "imsi-208930000000006"}
    )
    stage = original.control_semantics.stages[0].model_copy(
        update={"targets": [first_target, second_target]}
    )
    semantics = original.control_semantics.model_copy(update={"stages": [stage]})
    proposal = original.model_copy(update={"supi": None, "control_semantics": semantics})

    issues = validate_main_proposal(
        proposal=proposal,
        target_agent="sba",
        message_type="proposal",
        trigger_message_type="control_request",
        user_input=(
            "Protect imsi-208930000000001 first, then imsi-208930000000006."
        ),
        round_index=0,
    )

    assert issues == []


def test_main_proposal_is_intercepted_repaired_and_delivered(tmp_path: Path) -> None:
    journal = EpisodeJournal(root=tmp_path, episode_id="episode-main")
    snapshot = journal.append_event(
        round_index=0,
        source_agent="simulator",
        target_agent="episode_controller",
        message_type="observation",
        payload=EnvironmentSnapshot(
            snapshot_id="snapshot-main",
            observed_at=datetime.now(UTC),
            entities=[],
            relations=[],
            metrics=[],
        ),
    )
    request = journal.append_event(
        round_index=0,
        source_agent="episode_controller",
        target_agent="main",
        message_type="handoff",
        input_artifact_ids=[snapshot.event_id],
        payload=ControlRequest(user_input="protect video for imsi-12345"),
    )

    class FakeMainAgent:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _: Any, *, context: Any) -> dict[str, Any]:
            self.calls += 1
            return {
                "structured_response": CoordinationDecision[MainProposal](
                    target_agent="sba",
                    message_type="proposal" if self.calls == 1 else "repair_response",
                    payload=_main_proposal(evidence=self.calls > 1),
                )
            }

    main_actor = MainActor(
        agent=FakeMainAgent(),
        model_name="fake-main",
        journal=journal,
        context_builder=ContextBuilder(
            journal=journal,
            token_budget=100_000,
            count_tokens=len,
        ),
    )
    main_validator = MainValidator(
        journal=journal,
        repair_budget=RepairBudget(
            RepairLimits(
                local_limits={"main": 2},
                upstream={"main": "episode_controller"},
                total_limit=4,
                controller_actor="episode_controller",
            )
        ),
    )

    def sba_handler(state: Any) -> EventEmission:
        current = state["current_event"]
        return EventEmission(
            decision=CoordinationDecision[SampleHandoff](
                target_agent="episode_controller",
                message_type="handoff",
                payload=SampleHandoff(status="received"),
            ),
            in_reply_to=current.event_id,
            input_artifact_ids=(current.event_id,),
            invalidated_artifact_ids=(),
        )

    def controller_handler(state: Any) -> EventEmission:
        current = state["current_event"]
        return EventEmission(
            decision=CoordinationDecision[TerminalPayload](
                target_agent="episode_controller",
                message_type="terminal",
                payload=TerminalPayload(
                    outcome="success",
                    reason="main delivered",
                    reward_total=1.0,
                    reward_components={"delivery": 1.0},
                ),
            ),
            in_reply_to=current.event_id,
            input_artifact_ids=(current.event_id,),
            invalidated_artifact_ids=(),
        )

    edges = [
        AllowedEdge("episode_controller", "main", "handoff"),
        AllowedEdge("main", "sba", "proposal"),
        AllowedEdge("main", "sba", "repair_response"),
        AllowedEdge("main_validator", "main", "repair_request"),
        AllowedEdge("main_validator", "sba", "validation_result"),
        AllowedEdge("sba", "episode_controller", "handoff"),
        AllowedEdge("episode_controller", "episode_controller", "terminal"),
    ]
    graph = EpisodeGraph(
        journal=journal,
        router=EventRouter(
            edges,
            controller_actor="episode_controller",
            interceptors={
                ("main", "proposal"): "main_validator",
                ("main", "repair_response"): "main_validator",
            },
        ),
        handlers={
            "main": main_actor,
            "main_validator": main_validator,
            "sba": sba_handler,
            "episode_controller": controller_handler,
        },
        retry_policies={
            "main": None,
            "main_validator": None,
            "sba": None,
            "episode_controller": None,
        },
    )

    terminal = graph.invoke(request)

    assert terminal.message_type == "terminal"
    rows = [
        json.loads(row)
        for row in journal.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["message_type"] for row in rows[1:]] == [
        "handoff",
        "proposal",
        "repair_request",
        "repair_response",
        "validation_result",
        "handoff",
        "terminal",
    ]
    assert rows[2]["invalidated_artifact_ids"] == []
    assert rows[3]["invalidated_artifact_ids"] == [rows[2]["event_id"]]


def test_original_agent_prompts_are_preserved_with_only_contract_adapter() -> None:
    assert "Priority staging:" in MAIN_SYSTEM_PROMPT
    assert "Retry decision table, in precedence order:" in MAIN_SYSTEM_PROMPT
    assert "NON-NEGOTIABLE SUBSCRIPTION EVIDENCE BOUNDARY" in SBA_SYSTEM_PROMPT
    assert "NON-NEGOTIABLE QoS BINDING COMPLETION PROCEDURE" in SBA_SYSTEM_PROMPT
    assert "QoS slice migration rules:" in PPA_SYSTEM_PROMPT
    assert "Migration/domain decision table:" in PPA_SYSTEM_PROMPT
    for prompt in (MAIN_SYSTEM_PROMPT, SBA_SYSTEM_PROMPT, PPA_SYSTEM_PROMPT):
        assert "CoordinationDecision" in prompt
    assert "exactly one MainControlDecision JSON object" not in MAIN_SYSTEM_PROMPT
    assert "one raw IEA decision JSON object" not in SBA_SYSTEM_PROMPT
    assert "one complete OsaAdvisorOutput JSON object" not in PPA_SYSTEM_PROMPT
    assert "MainProposal payload structure-only few-shot" in MAIN_SYSTEM_PROMPT
    assert "Few-shot SbaProposal payload shapes" in SBA_SYSTEM_PROMPT
    assert "Few-shot PpaProposal payload shapes" in PPA_SYSTEM_PROMPT


def test_simulator_tool_read_is_both_observation_event_and_traceable_schema(
    tmp_path: Path,
) -> None:
    journal = EpisodeJournal(root=tmp_path, episode_id="episode-tool-observation")
    trigger = journal.append_event(
        round_index=0,
        source_agent="episode_controller",
        target_agent="sba",
        message_type="handoff",
        payload=ControlRequest(user_input="inspect imsi-12345"),
    )

    class ObservationAdapter:
        def observe(self, request: ObservationRequest) -> EnvironmentObservation:
            return EnvironmentObservation(
                base_snapshot_id="snapshot-1",
                observed_at=datetime.now(UTC),
                query=request,
                result={"flow_catalog": []},
            )

    sink = SimulatorObservationSink(
        adapter=cast(Any, ObservationAdapter()),
        journal=journal,
        trigger_event=trigger,
        target_agent="sba",
    )
    result = sink.record(
        query_name="get_sm_ue_flow_catalog",
        arguments={"supi": "imsi-12345"},
    )
    event = journal.get_event(sink.event_ids[0])
    tool_schema = snapshot_langchain_tool(SBA_OBSERVATION_TOOLS[0]).input_schema

    assert result["observation_event_id"] == event.event_id
    assert event.source_agent == "simulator_adapter"
    assert event.message_type == "observation"
    assert tool_schema["properties"] == {"supi": {"title": "Supi", "type": "string"}}
    assert "runtime" not in tool_schema["properties"]


def test_sba_validation_and_compiler_attach_main_owned_supi(tmp_path: Path) -> None:
    journal = EpisodeJournal(root=tmp_path, episode_id="episode-sba")
    request = journal.append_event(
        round_index=0,
        source_agent="episode_controller",
        target_agent="main",
        message_type="handoff",
        payload=ControlRequest(user_input="protect video for imsi-12345"),
    )
    main_event = journal.append_event(
        round_index=0,
        source_agent="main",
        target_agent="sba",
        message_type="proposal",
        in_reply_to=request.event_id,
        input_artifact_ids=[request.event_id],
        payload=_main_proposal(evidence=True),
    )
    main_validation = journal.append_event(
        round_index=0,
        source_agent="main_validator",
        target_agent="sba",
        message_type="validation_result",
        in_reply_to=main_event.event_id,
        input_artifact_ids=[main_event.event_id],
        payload=ValidationResultPayload(notes=["accepted"]),
    )
    observation = journal.append_event(
        round_index=0,
        source_agent="simulator_adapter",
        target_agent="sba",
        message_type="observation",
        in_reply_to=main_validation.event_id,
        input_artifact_ids=[main_validation.event_id],
        payload=EnvironmentObservation(
            base_snapshot_id="snapshot-1",
            observed_at=datetime.now(UTC),
            query=ObservationRequest(
                query_name="get_sm_ue_flow_catalog",
                arguments={"supi": "imsi-12345"},
            ),
            result={"flow_catalog": [{"app_id": "app-1", "flow_id": "flow-1"}]},
        ),
    )
    proposal = journal.append_event(
        round_index=0,
        source_agent="sba",
        target_agent="sba_compiler",
        message_type="proposal",
        in_reply_to=main_validation.event_id,
        input_artifact_ids=[main_validation.event_id, observation.event_id],
        payload=SbaProposal(
            flows=[
                GroundedFlowBinding(
                    app_id="app-1",
                    flow_id="flow-1",
                    name="video",
                    resolution_status="resolved",
                )
            ],
            mobility_intent=None,
            qos_operation_constraints=[
                QosOperationConstraint(
                    flow_id="flow-1",
                    app_id="app-1",
                    operation_type="qos_reallocation",
                    require_slice_change=False,
                    target_slice_preference="01000001",
                    no_op_allowed=True,
                    rationale=["same-slice tuning"],
                )
            ],
            slice_migration_authorization=SliceMigrationDecision(
                decision="not_applicable",
                target_snssais=[],
                subscription_change_required=False,
                rationale=[],
            ),
            open_questions=[],
        ),
    )
    budget = RepairBudget(
        RepairLimits(
            local_limits={"sba": 2},
            upstream={"sba": "main"},
            total_limit=4,
            controller_actor="episode_controller",
        )
    )
    validator_result = SbaValidator(journal=journal, repair_budget=budget)(
        {"current_event": proposal, "pending_model_failure": None}
    )
    validation_event = journal.append_event(
        round_index=0,
        source_agent="sba_validator",
        target_agent=validator_result.decision.target_agent,
        message_type=validator_result.decision.message_type,
        in_reply_to=validator_result.in_reply_to,
        input_artifact_ids=validator_result.input_artifact_ids,
        invalidated_artifact_ids=validator_result.invalidated_artifact_ids,
        payload=validator_result.decision.payload,
    )
    compiler_result = SbaCompiler(journal=journal)(
        {"current_event": validation_event, "pending_model_failure": None}
    )

    assert compiler_result.decision.message_type == "handoff"
    assert compiler_result.decision.target_agent == "ppa"
    assert compiler_result.decision.payload.supi == "imsi-12345"
    assert compiler_result.decision.payload.flows[0].flow_id == "flow-1"


def test_ppa_can_request_sba_reground_without_invalidating_valid_feedback(
    tmp_path: Path,
) -> None:
    journal = EpisodeJournal(root=tmp_path, episode_id="episode-ppa-reground")
    sba_event = journal.append_event(
        round_index=1,
        source_agent="sba_compiler",
        target_agent="ppa",
        message_type="handoff",
        payload=SbaArtifact(
            supi="imsi-12345",
            flows=[
                GroundedFlowBinding(
                    app_id="app-1",
                    flow_id="flow-1",
                    name="video",
                    resolution_status="resolved",
                )
            ],
            mobility_intent=None,
            qos_operation_constraints=[],
            slice_migration_authorization=SliceMigrationDecision(
                decision="not_applicable",
                target_snssais=[],
                subscription_change_required=False,
                rationale=[],
            ),
            open_questions=[],
        ),
    )
    proposal = journal.append_event(
        round_index=1,
        source_agent="ppa",
        target_agent="sba",
        message_type="proposal",
        in_reply_to=sba_event.event_id,
        input_artifact_ids=[sba_event.event_id],
        payload=PpaProposal(
            planning_status="needs_upstream_reground",
            rationale="flow evidence is stale",
            missing_evidence=["fresh flow binding"],
            blocked_targets=["flow-1"],
            upstream_requests=["rebind flow-1"],
            planner_conflicts=[],
            open_questions=[],
            sm_policies=[],
            am_policy=None,
            ursp_policies=[],
            partial_policies=[],
        ),
    )
    validator = PpaValidator(
        journal=journal,
        repair_budget=RepairBudget(
            RepairLimits(
                local_limits={"sba": 2, "ppa": 2},
                upstream={"sba": "main", "ppa": "sba"},
                total_limit=6,
                controller_actor="episode_controller",
            )
        ),
    )

    result = validator({"current_event": proposal, "pending_model_failure": None})

    assert result.decision.message_type == "repair_request"
    assert result.decision.target_agent == "sba"
    assert result.input_artifact_ids == (proposal.event_id,)
    assert result.invalidated_artifact_ids == ()


def test_function_simulator_adapter_exposes_one_strict_boundary() -> None:
    observed_at = datetime.now(UTC)
    first_snapshot = EnvironmentSnapshot(
        snapshot_id="snapshot-1",
        observed_at=observed_at,
        entities=[],
        relations=[],
        metrics=[],
    )
    next_snapshot = EnvironmentSnapshot(
        snapshot_id="snapshot-2",
        observed_at=observed_at,
        entities=[],
        relations=[],
        metrics=[],
    )
    adapter = FunctionSimulatorAdapter(
        snapshot_reader=lambda: first_snapshot,
        observation_handlers={
            "get_sm_ue_flow_catalog": lambda arguments: {
                "supi": arguments["supi"],
                "flow_catalog": [],
            }
        },
        execution_handler=lambda action: ExecutionFeedback(
            status="success",
            receipts=[
                OperationReceipt(
                    operation_id=operation.operation_id,
                    status="success",
                    result={"applied": True},
                )
                for operation in action.operations
            ],
            snapshot=next_snapshot,
            objective_metrics={"sla_satisfaction": 1.0},
        ),
    )

    assert adapter.capture_snapshot().snapshot_id == "snapshot-1"
    observation = adapter.observe(
        ObservationRequest(
            query_name="get_sm_ue_flow_catalog",
            arguments={"supi": "imsi-12345"},
        )
    )
    assert observation.base_snapshot_id == "snapshot-1"
    assert observation.result == {"supi": "imsi-12345", "flow_catalog": []}
