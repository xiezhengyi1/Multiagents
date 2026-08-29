from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://offline:offline@127.0.0.1:5432/offline_test",
)

_IEA_TEST_CONTEXT = '{"main_intent":{"requested_domains":[],"domain_evidence":{}}}'

from control_runtime.orchestrators.loop_state import (
    ControlRoundSnapshot,
    OrchestratorLoopState,
    append_round_trace,
)
from control_runtime.orchestrators.contracts import ControlRoundTrace
from control_runtime.orchestrators.round_context import (
    build_feedback_context_from_snapshots,
    build_main_context,
    build_memory_context,
)


def _main_intent(*, next_agent: str = "optimization_strategy"):
    from control_runtime.agents.main.contracts import ControlDomain, GlobalControlIntent

    return GlobalControlIntent(
        session_id="session-1",
        snapshot_id="snapshot-1",
        supi="imsi-001",
        next_agent=next_agent,
        requested_domains=[ControlDomain.QOS],
        routing_decision="protect qos flow",
        routing_rationale="offline characterization",
    )


def _grounding_decision():
    from control_runtime.agents.grounding.contracts import FlowSelector, GroundingDecision

    return GroundingDecision(
        flows=[
            FlowSelector(
                supi="imsi-001",
                app_id="app-1",
                flow_id="flow-1",
                name="video",
            )
        ]
    )


def _qos_plan(*, planning_status: str = "executable_plan"):
    from control_runtime.agents.planning.contracts import PolicyDraft, PolicyPlanDraft, PlanningRationale

    return PolicyPlanDraft(
        session_id="session-1",
        snapshot_id="snapshot-1",
        supi="imsi-001",
        planning_status=planning_status,
        planning_rationale=PlanningRationale(explanation="offline plan rationale"),
        all_policies=[
            PolicyDraft(
                policy_id="sm-1",
                policy_type="SmPolicyDecision",
                supi="imsi-001",
                app_id="app-1",
                flow_id="flow-1",
                target_type="flow",
                policy_details={"flow_id": "flow-1"},
            )
        ],
    )


def _bare_iea_for_public_contract(monkeypatch):
    """Build the smallest public IEA surface needed for retry characterization."""

    import control_runtime.agents.grounding.agent as grounding_agent
    from control_runtime.agents.grounding.agent import IntentEncodingAgent

    class BareIntentEncodingAgent(IntentEncodingAgent):
        pass

    evidence = SimpleNamespace(
        supi="",
        requested_domains=[],
        serving_slice_change_requested=False,
        subscription_summary={},
        model_dump=lambda **_kwargs: {"evidence": "stub"},
    )
    agent = object.__new__(BareIntentEncodingAgent)
    agent.logger = SimpleNamespace(error=lambda *_args, **_kwargs: None)
    agent.validator = object()
    agent.tools = []
    agent.last_failure_debug = {}
    monkeypatch.setattr(agent, "ensure_worker_runtime_initialized", lambda: None)
    monkeypatch.setattr(agent, "_extract_intent_evidence", lambda **_kwargs: evidence)
    monkeypatch.setattr(agent, "_resolve_token_context", lambda: (None, None))
    monkeypatch.setattr(agent, "build_runtime_context", lambda **_kwargs: object())
    monkeypatch.setattr(agent, "_create_advisor_agent", lambda _tools: object())
    monkeypatch.setattr(agent, "_build_advisor_prompt", lambda **_kwargs: "base prompt")
    monkeypatch.setattr(
        grounding_agent,
        "build_grounding_validation_retry_prompt",
        lambda **_kwargs: "retry prompt",
    )
    monkeypatch.setattr(grounding_agent, "log_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(grounding_agent, "log_timing", lambda *_args, **_kwargs: None)
    return agent, grounding_agent, evidence


def _bare_main_for_public_contract(monkeypatch):
    """Build the smallest public Main surface needed for retry characterization."""

    import control_runtime.agents.main.agent as main_agent
    from control_runtime.agents.main.agent import MainControlAgent

    class BareMainControlAgent(MainControlAgent):
        pass

    agent = object.__new__(BareMainControlAgent)
    agent.agent_name = "main_control"
    agent.logger = SimpleNamespace()
    agent._schema_projection_count = 0
    agent._schema_projected_fields = []
    monkeypatch.setattr(agent, "_resolve_token_context", lambda: (None, None))
    monkeypatch.setattr(agent, "build_runtime_context", lambda **_kwargs: object())
    return agent, main_agent


def _main_control_decision_result(*, include_compiled_field: bool = False) -> dict:
    from control_runtime.agents.main.contracts import ControlDomain, MainControlDecision

    decision = MainControlDecision(
        next_agent="intent_encoding",
        requested_domains=[ControlDomain.QOS],
        domain_evidence={},
        control_semantics={"stages": [], "notes": []},
        objective_profile={},
        investigation_targets=[],
        uncertainty_flags=[],
        required_evidence=[],
        forbidden_assumptions=[],
    ).model_dump(mode="json")
    if include_compiled_field:
        decision["routing_decision"] = "compiler-owned field"
    return {"structured_response": decision}


def test_canonical_runtime_paths_stay_bound_to_the_repository_layout() -> None:
    from shared.runtime.io.paths import project_root, runtime_root, trace_root

    assert project_root() == ROOT
    assert runtime_root() == ROOT / "data" / "runtime"
    assert trace_root("optimization_strategy") == (
        ROOT / "training" / "optimization_strategy" / "raw_traces"
    )
    assert "src/shared" not in project_root().as_posix()


def test_runtime_imports_only_through_its_canonical_shared_package() -> None:
    import importlib

    canonical_runtime = importlib.import_module("shared.runtime")

    assert canonical_runtime.__name__ == "shared.runtime"
    with pytest.raises(ImportError):
        importlib.import_module("src.shared.runtime")


def test_removed_runtime_compatibility_modules_stay_unimportable() -> None:
    import importlib

    from shared.runtime import ArtifactWorkerMixin

    assert ArtifactWorkerMixin.__name__ == "ArtifactWorkerMixin"
    for module_name in ("shared.runtime.api", "shared.runtime.trace.legacy_migration"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_retry_policy_rejects_mutable_retry_configuration_containers() -> None:
    from shared.agents import RetryPhase, RetryPolicy

    with pytest.raises(TypeError, match=r"retryable_exceptions.*tuple"):
        RetryPolicy(retryable_exceptions=[RuntimeError])

    with pytest.raises(TypeError, match=r"retryable_phases.*frozenset"):
        RetryPolicy(retryable_phases={RetryPhase.INVOKE})


def test_retry_policy_rejects_mutable_or_duplicate_phase_exception_mappings() -> None:
    from shared.agents import RetryPhase, RetryPolicy

    with pytest.raises(TypeError, match=r"retryable_exceptions_by_phase.*tuple"):
        RetryPolicy(retryable_exceptions_by_phase=[(RetryPhase.INVOKE, (RuntimeError,))])

    with pytest.raises(ValueError, match="duplicate phases"):
        RetryPolicy(
            retryable_exceptions_by_phase=(
                (RetryPhase.INVOKE, (RuntimeError,)),
                (RetryPhase.INVOKE, (ValueError,)),
            )
        )


class _RecordingConflictTool:
    def __init__(self, result):
        self.result = result
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        return self.result


class _RecordingDispatchAgent:
    def __init__(self, report):
        self.report = report
        self.calls = []

    def execute_and_evaluate(self, policy_plan, *, trace_metadata=None):
        self.calls.append((policy_plan, trace_metadata))
        return self.report


class _RecordingDiagnosisTool:
    def __init__(self, result):
        self.result = result
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        return self.result


class OrchestratorContextRefactorTest(unittest.TestCase):
    def test_loop_state_keeps_snapshots_and_derives_previous_fields(self) -> None:
        state = OrchestratorLoopState()

        for index in range(1, 5):
            append_round_trace(
                state,
                trace=ControlRoundTrace(
                    round_index=index,
                    global_intent={"round": index},
                    grounding_decision={"grounding": index},
                    policy_plan={"plan": index},
                    pda_feedback={"execution_status": "Failed", "violation_details": f"failure-{index}"},
                    diagnosis={"root_cause_category": f"category-{index}"},
                    negotiation_request={"summary": f"negotiation-{index}"},
                ),
                feedback_added=f"[Round Feedback]\nround_index: {index}",
            )

        self.assertEqual([snap.trace.round_index for snap in state.rounds], [2, 3, 4])
        self.assertEqual(state.previous_diagnosis, {"root_cause_category": "category-4"})
        self.assertEqual(state.previous_report_payload["violation_details"], "failure-4")
        self.assertEqual(state.previous_negotiation_request, {"summary": "negotiation-4"})
        with self.assertRaises(Exception):
            state.rounds[-1].trace.diagnosis = {}

    def test_feedback_context_summarizes_older_rounds_and_keeps_recent_full(self) -> None:
        snapshots = [
            ControlRoundSnapshot(
                trace=ControlRoundTrace(
                    round_index=1,
                    diagnosis={"root_cause_category": "old_binding", "reason_summary": "wrong flow binding"},
                ),
                feedback_added="[Round Feedback]\nround_index: 1\nvery old verbose detail",
            ),
            ControlRoundSnapshot(
                trace=ControlRoundTrace(
                    round_index=2,
                    diagnosis={"root_cause_category": "recent_policy", "reason_summary": "policy failed"},
                ),
                feedback_added="[Round Feedback]\nround_index: 2\nrecent detail",
            ),
            ControlRoundSnapshot(
                trace=ControlRoundTrace(
                    round_index=3,
                    diagnosis={"root_cause_category": "latest_dispatch", "reason_summary": "dispatch failed"},
                ),
                feedback_added="[Round Feedback]\nround_index: 3\nlatest detail",
            ),
        ]

        context = build_feedback_context_from_snapshots(snapshots)

        self.assertIn("[Older Feedback Summary]", context)
        self.assertIn("old_binding", context)
        self.assertIn("round_index: 2", context)
        self.assertIn("round_index: 3", context)
        self.assertNotIn("very old verbose detail", context)

    def test_memory_context_prefers_entries_matching_diagnosis_and_routing_hints(self) -> None:
        class Memory:
            def retrieve(self, _: str) -> dict:
                return {
                    "short_term": [
                        {"role": "MAIN", "content": "generic context"},
                        {"role": "AD", "content": "planning_blocked optimizer preview missing"},
                    ],
                    "long_term": [
                        "general success case",
                        "optimization_strategy target_stable retry fixed planner gap",
                    ],
                }

        context = build_memory_context(
            "QoS request",
            memory_manager=Memory(),
            diagnosis_hint="planning_blocked",
            routing_hint="optimization_strategy target_stable",
        )

        self.assertLess(context.index("planning_blocked"), context.index("generic context"))
        self.assertIn("optimization_strategy target_stable", context)

    def test_main_context_is_markdown_sections_not_json_blob(self) -> None:
        from unittest.mock import patch

        with patch(
            "control_runtime.integrations.storage.get_snapshot_data_by_id",
            return_value={"apps": [], "slices": [], "nodes": [], "mobility": [], "flows": []},
        ):
            context = build_main_context(
                "snap-1",
                round_index=2,
                memory_context="[Memory] relevant",
                feedback_context="[Round Feedback] previous",
                previous_diagnosis={"root_cause_category": "planning_blocked"},
            )

        self.assertTrue(context.startswith("## Snapshot Summary"))
        self.assertIn("## Previous Round Diagnosis", context)
        self.assertIn("## Retry Hints", context)
        self.assertIn("## Memory Context", context)
        self.assertNotEqual(context[:1], "{")

    def test_round_trace_uses_current_grounding_contract(self) -> None:
        snapshot = ControlRoundSnapshot(
            trace=ControlRoundTrace.from_payload(
                {
                "round_index": 1,
                "global_intent": {"requested_domains": ["qos"]},
                "grounding_decision": {"flows": [{"flow_id": "flow-1"}]},
                "policy_plan": {"planning_status": "executable_plan"},
                "diagnosis": {},
                }
            )
        )

        trace = snapshot.trace.to_payload()
        self.assertEqual(trace["grounding_decision"]["flows"][0]["flow_id"], "flow-1")
        self.assertNotIn("operation_intent", trace)


def test_execute_planned_round_returns_a_planning_blocker_without_external_execution(monkeypatch) -> None:
    import control_runtime.orchestrators.round_execution as round_execution
    from control_runtime.agents.main.contracts import ControlDomain
    from control_runtime.agents.planning.contracts import PolicyPlanDraft, PlanningRationale

    monkeypatch.setattr(
        round_execution,
        "get_snapshot_data_by_id",
        lambda _: {"snapshot_id": "snapshot-1"},
    )
    blocked_plan = PolicyPlanDraft(
        session_id="session-1",
        snapshot_id="snapshot-1",
        supi="imsi-001",
        planning_status="needs_upstream_reground",
        planning_rationale=PlanningRationale(explanation="catalog evidence is missing"),
        missing_evidence=["SM flow catalog"],
        upstream_requests=["reground flow-1"],
    )

    artifacts = round_execution.execute_planned_round(
        session_id="session-1",
        snapshot_id="snapshot-1",
        round_index=1,
        policy_plan=blocked_plan,
        global_intent=_main_intent(),
        grounding_decision=_grounding_decision(),
        cr_tool=SimpleNamespace(run=lambda *_args, **_kwargs: pytest.fail("mediator must not run")),
        pd_agent=SimpleNamespace(execute_and_evaluate=lambda *_args, **_kwargs: pytest.fail("dispatch must not run")),
        ad_tool=SimpleNamespace(run=lambda *_args, **_kwargs: pytest.fail("diagnosis must not run")),
    )

    assert artifacts.completed is False
    assert artifacts.report is None
    assert artifacts.diagnosis["root_cause_category"] == "planning_blocked"
    assert artifacts.planning_blocker is not None
    assert artifacts.planning_blocker.upstream_requests == ["reground flow-1"]
    assert artifacts.unified_plan.blocked_domains == [ControlDomain.QOS]
    assert artifacts.trace.mediator_decision == {}
    assert artifacts.trace.planning_blocker == artifacts.planning_blocker.model_dump(mode="json")
    assert artifacts.trace.planning_blocker["planning_status"] == "needs_upstream_reground"


def test_execute_planned_round_returns_success_contract_from_stubbed_boundaries(monkeypatch) -> None:
    import control_runtime.orchestrators.round_execution as round_execution
    from control_runtime.agents.dispatch.contracts import FeedbackReport
    from control_runtime.diagnostics.diagnosis.contracts import AssuranceDiagnosisResult
    from control_runtime.diagnostics.mediation.contracts import ConflictResolutionResult
    from control_runtime.agents.main.contracts import ControlDomain

    monkeypatch.setattr(
        round_execution,
        "get_snapshot_data_by_id",
        lambda _: {"snapshot_id": "snapshot-1", "apps": [], "slices": [], "nodes": []},
    )
    monkeypatch.setattr(round_execution, "get_ue_context_by_supi", lambda *_args, **_kwargs: {})
    report = FeedbackReport(
        execution_status="Success",
        performance_metrics=json.dumps(
            {
                "dispatch_results": [
                    {"status": "success", "policy_id": "sm-1", "policy_type": "SmPolicyDecision"}
                ],
                "assurance_results": [
                    {"status": "satisfied", "policy_id": "sm-1", "policy_type": "SmPolicyDecision"}
                ],
            }
        ),
        violation_details="none",
    )
    conflict_tool = _RecordingConflictTool(ConflictResolutionResult())
    dispatch_agent = _RecordingDispatchAgent(report)
    diagnosis_tool = _RecordingDiagnosisTool(
        AssuranceDiagnosisResult(status="healthy", reason_summary="SLA satisfied")
    )

    artifacts = round_execution.execute_planned_round(
        session_id="session-1",
        snapshot_id="snapshot-1",
        round_index=1,
        policy_plan=_qos_plan(),
        global_intent=_main_intent(),
        grounding_decision=_grounding_decision(),
        cr_tool=conflict_tool,
        pd_agent=dispatch_agent,
        ad_tool=diagnosis_tool,
        trace_metadata={"scenario_id": "offline-case"},
    )

    assert artifacts.completed is True
    assert artifacts.report == report
    assert artifacts.execution_reentry is None
    assert artifacts.qos_feedback["execution_status"] == "Success"
    assert artifacts.unified_plan.execution_order == [ControlDomain.QOS]
    assert artifacts.unified_plan.approved_policies == [
        {"domain": "qos", "policy_id": "sm-1", "policy_type": "SmPolicyDecision"}
    ]
    assert artifacts.trace.pda_feedback["execution_status"] == "Success"
    assert artifacts.trace.mediator_decision == artifacts.unified_plan.mediator_decision.model_dump(mode="json")
    assert conflict_tool.requests[0].snapshot_id == "snapshot-1"
    assert dispatch_agent.calls[0][1] == {"scenario_id": "offline-case"}
    assert diagnosis_tool.requests[0].execution_feedback["pda"]["execution_status"] == "Success"


def test_execute_planned_round_emits_reentry_contract_after_dispatch_failure(monkeypatch) -> None:
    import control_runtime.orchestrators.round_execution as round_execution
    from control_runtime.agents.dispatch.contracts import FeedbackReport
    from control_runtime.diagnostics.diagnosis.contracts import AssuranceDiagnosisResult
    from control_runtime.diagnostics.mediation.contracts import ConflictResolutionResult

    monkeypatch.setattr(
        round_execution,
        "get_snapshot_data_by_id",
        lambda _: {"snapshot_id": "snapshot-1", "apps": [], "slices": [], "nodes": []},
    )
    monkeypatch.setattr(round_execution, "get_ue_context_by_supi", lambda *_args, **_kwargs: {})
    report = FeedbackReport(
        execution_status="Failed",
        performance_metrics=json.dumps(
            {"dispatch_results": [{"status": "failed", "policy_id": "sm-1", "policy_type": "SmPolicyDecision"}]}
        ),
        violation_details="policy gateway rejected sm-1",
        failure_scope="qos",
        feedback_payload={
            "target_bindings_at_risk": ["flow-1"],
            "policy_objects_at_risk": ["sm-1"],
            "reason_by_domain": {"qos": "gateway rejected policy"},
            "failures": [{"phase": "dispatch", "policy_id": "sm-1"}],
        },
    )

    artifacts = round_execution.execute_planned_round(
        session_id="session-1",
        snapshot_id="snapshot-1",
        round_index=2,
        policy_plan=_qos_plan(),
        global_intent=_main_intent(),
        grounding_decision=_grounding_decision(),
        cr_tool=_RecordingConflictTool(ConflictResolutionResult()),
        pd_agent=_RecordingDispatchAgent(report),
        ad_tool=_RecordingDiagnosisTool(
            AssuranceDiagnosisResult(root_cause_category="dispatch_failure", reason_summary="gateway rejected")
        ),
    )

    assert artifacts.completed is False
    assert artifacts.report == report
    assert artifacts.qos_feedback["execution_status"] == "Failed"
    assert artifacts.execution_reentry is not None
    assert artifacts.execution_reentry.round_index == 2
    assert artifacts.execution_reentry.failure_scope == "qos"
    assert artifacts.execution_reentry.target_bindings_at_risk == ["flow-1"]
    assert artifacts.execution_reentry.policy_objects_at_risk == ["sm-1"]
    assert artifacts.trace.execution_reentry["reason_by_domain"] == {"qos": "gateway rejected policy"}


def test_osa_public_generate_strategy_rejects_wrong_input_and_returns_reground_plan_without_llm(monkeypatch) -> None:
    from control_runtime.agents.planning.agent import OptimizationStrategyAgent
    from control_runtime.agents.grounding.contracts import GroundingDecision
    from control_runtime.agents.planning.contracts import (
        PlanningContext,
        PlanningRequest,
        PlanningRationale,
        PolicyPlanDraft,
        SharedControlContext,
    )

    class BareOptimizationStrategyAgent(OptimizationStrategyAgent):
        pass

    agent = object.__new__(BareOptimizationStrategyAgent)
    monkeypatch.setattr(agent, "ensure_worker_runtime_initialized", lambda: None)

    with pytest.raises(TypeError, match="PlanningRequest"):
        agent.generate_strategy({"not": "a PlanningRequest"})

    expected = PolicyPlanDraft(
        planning_status="needs_upstream_reground",
        planning_rationale=PlanningRationale(explanation="upstream evidence required"),
        upstream_requests=["reground QoS flow"],
    )
    compiler_calls = []

    class Compiler:
        def build_upstream_reground_plan(self, *, planning_request, reason):
            compiler_calls.append((planning_request, reason))
            return expected

    agent.compiler = Compiler()
    monkeypatch.setattr(agent, "_cache_received_request", lambda _request: object())
    request = PlanningRequest(
        grounding_decision=GroundingDecision(),
        context=PlanningContext(shared_context=SharedControlContext(main_intent=_main_intent())),
    )

    assert agent.generate_strategy(request) == expected
    assert compiler_calls[0][0] == request
    assert "QoS" in compiler_calls[0][1]


def test_osa_stage_retries_invocation_failures_and_reraises_the_original_exception(monkeypatch) -> None:
    import control_runtime.agents.planning.agent as planning_agent
    from control_runtime.agents.planning.agent import OptimizationStrategyAgent
    from control_runtime.agents.planning.contracts import PlanningContext, PlanningRequest, SharedControlContext

    class BareOptimizationStrategyAgent(OptimizationStrategyAgent):
        pass

    agent = object.__new__(BareOptimizationStrategyAgent)
    agent.logger = SimpleNamespace()
    agent.last_failure_debug = {}
    attempts = []
    terminal_error = RuntimeError("advisor transport unavailable")
    request = PlanningRequest(
        grounding_decision=_grounding_decision(),
        context=PlanningContext(shared_context=SharedControlContext(main_intent=_main_intent())),
    )
    monkeypatch.setattr(agent, "_cached_tool_evidence_for_retry", lambda: {})

    def fail_invocation(**_kwargs):
        attempts.append("invoke")
        raise terminal_error

    monkeypatch.setattr(agent, "_invoke_strategy_advisor", fail_invocation)
    monkeypatch.setattr(planning_agent, "log_event", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError) as raised:
        agent._run_advisor_stage(
            planning_request=request,
            base_prompt="initial prompt",
            trace_metadata={"scenario_id": "offline-case"},
            state=planning_agent._OsaAdvisorStageState(current_prompt="initial prompt"),
        )

    assert raised.value is terminal_error
    assert attempts == ["invoke", "invoke", "invoke"]


def test_osa_stage_does_not_retry_or_record_non_retryable_invocation_errors(monkeypatch) -> None:
    import control_runtime.agents.planning.agent as planning_agent
    from control_runtime.agents.planning.agent import OptimizationStrategyAgent
    from control_runtime.agents.planning.contracts import PlanningContext, PlanningRequest, SharedControlContext

    class BareOptimizationStrategyAgent(OptimizationStrategyAgent):
        pass

    agent = object.__new__(BareOptimizationStrategyAgent)
    agent.logger = SimpleNamespace()
    initial_debug = {"preserved": "non-retryable invocation"}
    agent.last_failure_debug = initial_debug.copy()
    attempts = []
    cached_evidence_calls = []
    terminal_error = ValueError("advisor request is malformed")
    request = PlanningRequest(
        grounding_decision=_grounding_decision(),
        context=PlanningContext(shared_context=SharedControlContext(main_intent=_main_intent())),
    )

    def fail_invocation(**_kwargs):
        attempts.append("invoke")
        raise terminal_error

    monkeypatch.setattr(agent, "_invoke_strategy_advisor", fail_invocation)
    monkeypatch.setattr(
        agent,
        "_cached_tool_evidence_for_retry",
        lambda: cached_evidence_calls.append("cached") or {},
    )
    monkeypatch.setattr(planning_agent, "log_event", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError) as raised:
        agent._run_advisor_stage(
            planning_request=request,
            base_prompt="initial prompt",
            trace_metadata=None,
            state=planning_agent._OsaAdvisorStageState(current_prompt="initial prompt"),
        )

    assert raised.value is terminal_error
    assert attempts == ["invoke"]
    assert cached_evidence_calls == []
    assert agent.last_failure_debug == initial_debug


def test_osa_public_generate_strategy_preserves_stage_failure_cleanup_and_trace(monkeypatch) -> None:
    import control_runtime.agents.planning.agent as planning_agent
    from control_runtime.agents.planning.agent import OptimizationStrategyAgent
    from control_runtime.agents.planning.contracts import PlanningContext, PlanningRequest, SharedControlContext

    class BareOptimizationStrategyAgent(OptimizationStrategyAgent):
        pass

    agent = object.__new__(BareOptimizationStrategyAgent)
    agent.logger = SimpleNamespace(exception=lambda *_args, **_kwargs: None)
    agent.compiler = SimpleNamespace(build_planning_evidence=lambda _request: {})
    terminal_error = RuntimeError("strategy advisor unavailable")
    trace_writes = []
    request = PlanningRequest(
        grounding_decision=_grounding_decision(),
        context=PlanningContext(shared_context=SharedControlContext(main_intent=_main_intent())),
    )

    def fail_stage(*, state, **_kwargs):
        agent._pending_invoke_messages = [{"role": "user", "content": "initial prompt"}]
        state.advisor_trace = {"path_label": "strategy_advisor"}
        raise terminal_error

    monkeypatch.setattr(agent, "ensure_worker_runtime_initialized", lambda: None)
    monkeypatch.setattr(agent, "_cache_received_request", lambda _request: object())
    monkeypatch.setattr(agent, "_include_knowledge_tools", lambda _request: False)
    monkeypatch.setattr(agent, "_run_advisor_stage", fail_stage)
    monkeypatch.setattr(agent, "_write_advisor_trace", lambda **kwargs: trace_writes.append(kwargs))
    monkeypatch.setattr(planning_agent, "_build_lean_grounding_decision", lambda _decision: {})
    monkeypatch.setattr(planning_agent, "project_collaboration_context_for_prompt", lambda _context: {})
    monkeypatch.setattr(planning_agent, "build_request_tools", lambda _request: ([], {}))
    monkeypatch.setattr(planning_agent, "build_planning_user_prompt", lambda **_kwargs: "initial prompt")
    monkeypatch.setattr(planning_agent, "log_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(planning_agent, "log_timing", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError) as raised:
        agent.generate_strategy(request, trace_metadata={"scenario_id": "offline-case"})

    assert raised.value is terminal_error
    assert trace_writes[0]["status"] == "error"
    assert trace_writes[0]["error"] == "strategy advisor unavailable"
    assert not hasattr(agent, "_pending_invoke_messages")


def test_osa_stage_retries_contract_validation_and_keeps_terminal_message_stable(monkeypatch) -> None:
    import control_runtime.agents.planning.agent as planning_agent
    from control_runtime.agents.planning.agent import OptimizationStrategyAgent
    from control_runtime.agents.planning.response_models import OsaAdvisorOutput
    from control_runtime.agents.planning.contracts import PlanningContext, PlanningRequest, SharedControlContext

    class BareOptimizationStrategyAgent(OptimizationStrategyAgent):
        pass

    agent = object.__new__(BareOptimizationStrategyAgent)
    agent.logger = SimpleNamespace()
    agent.last_failure_debug = {}
    invocations = []
    validations = []
    request = PlanningRequest(
        grounding_decision=_grounding_decision(),
        context=PlanningContext(shared_context=SharedControlContext(main_intent=_main_intent())),
    )
    agent.compiler = SimpleNamespace(
        validate_advisor_output=lambda **_kwargs: validations.append("validate") or [
            "optimizer preview missing",
            "policy binding invalid",
        ]
    )
    monkeypatch.setattr(agent, "_cached_tool_evidence_for_retry", lambda: {})

    def invoke_advisor(**_kwargs):
        invocations.append("invoke")
        return OsaAdvisorOutput(), {}, {}

    monkeypatch.setattr(agent, "_invoke_strategy_advisor", invoke_advisor)
    monkeypatch.setattr(planning_agent, "extract_planning_tool_evidence", lambda **_kwargs: {})
    monkeypatch.setattr(planning_agent, "extract_grounding_tool_names", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(planning_agent, "log_event", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError) as raised:
        agent._run_advisor_stage(
            planning_request=request,
            base_prompt="initial prompt",
            trace_metadata=None,
            state=planning_agent._OsaAdvisorStageState(current_prompt="initial prompt"),
        )

    assert str(raised.value) == (
        "OSA advisor contract validation failed: optimizer preview missing; policy binding invalid"
    )
    assert invocations == ["invoke", "invoke", "invoke"]
    assert validations == ["validate", "validate", "validate"]


def test_iea_public_entrypoint_returns_grounding_decision_and_persists_artifact_boundary(monkeypatch) -> None:
    from control_runtime.agents.grounding.agent import IntentEncodingAgent

    class BareIntentEncodingAgent(IntentEncodingAgent):
        pass

    agent = object.__new__(BareIntentEncodingAgent)
    decision = _grounding_decision()
    captured = {}
    request_envelope = object()
    monkeypatch.setattr(agent, "ensure_worker_runtime_initialized", lambda: None)

    def cache_request(**kwargs):
        captured["request"] = kwargs
        return request_envelope

    def analyze_intent(user_input, **kwargs):
        captured["analysis"] = {"user_input": user_input, **kwargs}
        return decision

    def cache_result(**kwargs):
        captured["result"] = kwargs

    monkeypatch.setattr(agent, "_cache_received_request", cache_request)
    monkeypatch.setattr(agent, "analyze_intent", analyze_intent)
    monkeypatch.setattr(agent, "_cache_produced_result", cache_result)

    result = agent.analyze_grounding_decision(
        "protect flow-1",
        context="ground current stage",
        session_id="session-1",
        snapshot_id="snapshot-1",
        trace_metadata={"scenario_id": "offline-case"},
    )

    assert result == decision
    assert captured["request"]["session_id"] == "session-1"
    assert captured["analysis"]["trace_metadata"] == {"scenario_id": "offline-case"}
    assert captured["result"] == {
        "request_envelope": request_envelope,
        "grounding_decision": decision,
    }


def test_iea_retries_the_public_analysis_contract_three_times_before_raising(monkeypatch) -> None:
    import control_runtime.agents.grounding.agent as grounding_agent
    from control_runtime.agents.grounding.agent import IntentEncodingAgent

    class BareIntentEncodingAgent(IntentEncodingAgent):
        pass

    agent = object.__new__(BareIntentEncodingAgent)
    attempts = []
    agent.logger = SimpleNamespace(error=lambda *_args, **_kwargs: None)
    agent.tools = []
    monkeypatch.setattr(agent, "ensure_worker_runtime_initialized", lambda: None)
    monkeypatch.setattr(
        agent,
        "_extract_intent_evidence",
        lambda **_kwargs: SimpleNamespace(
            supi="",
            requested_domains=[],
            serving_slice_change_requested=False,
            subscription_summary={},
        ),
    )
    monkeypatch.setattr(agent, "_resolve_token_context", lambda: (None, None))
    monkeypatch.setattr(agent, "build_runtime_context", lambda **_kwargs: object())
    monkeypatch.setattr(agent, "_create_advisor_agent", lambda _tools: object())
    monkeypatch.setattr(agent, "_build_advisor_prompt", lambda **_kwargs: "base prompt")
    monkeypatch.setattr(
        grounding_agent,
        "build_grounding_validation_retry_prompt",
        lambda **_kwargs: "retry prompt",
    )

    def fail_invocation(**_kwargs):
        attempts.append("attempt")
        raise RuntimeError("advisor unavailable")

    monkeypatch.setattr(agent, "_invoke_intent_advisor", fail_invocation)
    monkeypatch.setattr(grounding_agent, "log_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(grounding_agent, "log_timing", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="advisor unavailable"):
        agent.analyze_intent(
            "protect flow-1",
            context='{"main_intent":{"requested_domains":[],"domain_evidence":{}}}',
            session_id="session-1",
            snapshot_id="snapshot-1",
        )

    assert attempts == ["attempt", "attempt", "attempt"]


def test_main_run_preserves_failed_round_feedback_for_the_next_round(monkeypatch) -> None:
    import control_runtime.orchestrators.main_control_orchestrator as main_orchestrator
    from control_runtime.agents.dispatch.contracts import FeedbackReport
    from control_runtime.orchestrators.contracts import ExecutionReentryRequest, UnifiedControlPlan
    from control_runtime.orchestrators.contracts import ControlRoundTrace
    from control_runtime.orchestrators.round_execution import RoundExecutionArtifacts

    class BareMainControlOrchestrator(main_orchestrator.MainControlOrchestrator):
        pass

    class MainAgent:
        logger = SimpleNamespace()
        schema_projection_metrics = {"count": 2, "field_count": 3, "fields": ["supi", "requested_domains"]}

        def reset_schema_projection_metrics(self):
            return None

    orchestrator = object.__new__(BareMainControlOrchestrator)
    orchestrator.main_agent = MainAgent()
    orchestrator.memory_manager = SimpleNamespace(bind_thread=lambda _session_id: None)
    orchestrator.max_rounds = 2
    orchestrator._token_counter = object()
    orchestrator._inject_token_context = lambda: None
    orchestrator._remember = lambda *_args, **_kwargs: None
    orchestrator.cr_tool = object()
    orchestrator.pd_agent = object()
    orchestrator.ad_tool = object()
    plan_calls = []
    finalization = []
    intent = _main_intent()
    grounding = _grounding_decision()
    policy_plan = _qos_plan()

    first_report = FeedbackReport(
        execution_status="Failed",
        performance_metrics='{"dispatch_results": []}',
        violation_details="gateway rejected sm-1",
        feedback_payload={"target_bindings_at_risk": ["flow-1"]},
    )
    first_reentry = ExecutionReentryRequest(
        round_index=1,
        target_bindings_at_risk=["flow-1"],
        policy_objects_at_risk=["sm-1"],
        failure_scope="qos",
        summary="gateway rejected sm-1",
    )
    second_report = FeedbackReport(
        execution_status="Success",
        performance_metrics='{"dispatch_results": []}',
        violation_details="none",
    )

    def make_artifacts(*, round_index, completed, report, diagnosis, reentry=None):
        return RoundExecutionArtifacts(
            completed=completed,
            report=report,
            diagnosis=diagnosis,
            qos_feedback={"execution_status": report.execution_status},
            mobility_feedback={},
            unified_plan=UnifiedControlPlan(
                session_id="session-1",
                snapshot_id="snapshot-1",
                supi="imsi-001",
                global_intent=intent,
            ),
            trace=ControlRoundTrace(
                round_index=round_index,
                global_intent=intent.model_dump(mode="json"),
                grounding_decision=grounding.model_dump(mode="json"),
                policy_plan=policy_plan.model_dump(mode="json"),
                pda_feedback=report.model_dump(mode="json"),
                diagnosis=diagnosis,
                execution_reentry=reentry.model_dump(mode="json") if reentry else {},
            ),
            execution_reentry=reentry,
        )

    artifacts_by_round = {
        1: make_artifacts(
            round_index=1,
            completed=False,
            report=first_report,
            diagnosis={"root_cause_category": "dispatch_failure", "reason_summary": "gateway rejected"},
            reentry=first_reentry,
        ),
        2: make_artifacts(
            round_index=2,
            completed=True,
            report=second_report,
            diagnosis={"root_cause_category": "", "reason_summary": "recovered"},
        ),
    }

    def plan_round(**kwargs):
        plan_calls.append(kwargs)
        return intent, grounding, policy_plan, None

    monkeypatch.setattr(orchestrator, "_plan_round", plan_round)
    monkeypatch.setattr(main_orchestrator, "start_control_session", lambda **_kwargs: ("session-1", "snapshot-1"))
    monkeypatch.setattr(main_orchestrator, "build_memory_context", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(main_orchestrator, "build_feedback_context_from_snapshots", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(main_orchestrator, "log_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main_orchestrator,
        "execute_planned_round",
        lambda **kwargs: artifacts_by_round[kwargs["round_index"]],
    )
    monkeypatch.setattr(
        main_orchestrator,
        "finish_control_session",
        lambda **kwargs: finalization.append(kwargs),
    )

    result = orchestrator.run("protect flow-1", snapshot_id="snapshot-1")

    assert result.completed is True
    assert result.round_count == 2
    assert result.retry_count == 1
    assert [trace["round_index"] for trace in result.round_traces] == [1, 2]
    assert plan_calls[1]["previous_report_payload"]["execution_status"] == "Failed"
    assert plan_calls[1]["previous_execution_reentry"]["target_bindings_at_risk"] == ["flow-1"]
    assert result.main_schema_projection_count == 2
    assert finalization[0]["state"].completed is True


def test_iea_public_analysis_reraises_final_invocation_error_without_final_retry_side_effects(monkeypatch) -> None:
    """The shared runner must retain the former public IEA failure contract."""

    agent, grounding_agent, evidence = _bare_iea_for_public_contract(monkeypatch)
    attempts = []
    retry_events = []
    retry_prompts = []
    trace_writes = []
    decision = _grounding_decision()
    root_cause = LookupError("transport root cause")
    terminal_error = RuntimeError("advisor unavailable")
    first_invocation = SimpleNamespace(
        advisor_result={},
        write_final_trace=lambda **kwargs: trace_writes.append(kwargs),
    )

    def invoke_advisor(**_kwargs):
        attempts.append("invoke")
        if len(attempts) == 1:
            return first_invocation
        raise terminal_error from root_cause

    def capture_event(_logger, event, **kwargs):
        if event == "iea_retry":
            retry_events.append(kwargs)

    monkeypatch.setattr(agent, "_invoke_intent_advisor", invoke_advisor)
    monkeypatch.setattr(grounding_agent, "coerce_structured_response", lambda *_args, **_kwargs: object())
    def refresh_evidence(*, state):
        state.refreshed_evidence = evidence
        state.grounding_tools = []
        return evidence

    monkeypatch.setattr(agent, "_refresh_advisor_stage_evidence", refresh_evidence)
    monkeypatch.setattr(agent, "_compile_advisor_output", lambda **_kwargs: decision)
    monkeypatch.setattr(agent, "_materialize_catalog_facts", lambda **kwargs: kwargs["grounding_decision"])
    monkeypatch.setattr(agent, "_materialize_qos_constraint_facts", lambda **kwargs: kwargs["grounding_decision"])
    monkeypatch.setattr(agent, "_attach_subscription_evidence", lambda **kwargs: kwargs["grounding_decision"])
    monkeypatch.setattr(
        grounding_agent,
            "validate_grounding",
        lambda **_kwargs: (["advisor decision invalid"], [], None),
    )
    monkeypatch.setattr(grounding_agent, "log_event", capture_event)
    monkeypatch.setattr(
        grounding_agent,
        "build_grounding_validation_retry_prompt",
        lambda **kwargs: retry_prompts.append(kwargs) or "retry prompt",
    )
    agent._pending_invoke_messages = [{"role": "user", "content": "stale prompt"}]

    with pytest.raises(RuntimeError) as raised:
        agent.analyze_intent("protect flow-1", context=_IEA_TEST_CONTEXT, session_id="session-1", snapshot_id="snapshot-1")

    assert raised.value is terminal_error
    assert raised.value.__cause__ is root_cause
    assert attempts == ["invoke", "invoke", "invoke"]
    assert [event["reason"] for event in retry_events] == ["validation_errors", "invocation_error"]
    assert len(retry_prompts) == 2
    assert agent.last_failure_debug["attempt_index"] == 1
    assert agent.last_failure_debug["advisor_validation_errors"] == ["advisor decision invalid"]
    assert trace_writes == [
        {
            "grounding_decision": decision,
            "status": "error",
            "error": "advisor unavailable",
        }
    ]
    assert not hasattr(agent, "_pending_invoke_messages")


def test_iea_public_analysis_keeps_terminal_parse_message_and_cause(monkeypatch) -> None:
    agent, grounding_agent, _evidence = _bare_iea_for_public_contract(monkeypatch)
    attempts = []
    retry_events = []
    captures = []
    trace_writes = []
    parse_error = RuntimeError("advisor response is malformed")

    def invoke_advisor(**_kwargs):
        attempts.append("invoke")
        return SimpleNamespace(
            advisor_result={},
            write_final_trace=lambda **kwargs: trace_writes.append(kwargs),
        )

    def capture_event(_logger, event, **kwargs):
        if event == "iea_retry":
            retry_events.append(kwargs)

    def fail_coercion(*_args, **_kwargs):
        raise parse_error

    monkeypatch.setattr(agent, "_invoke_intent_advisor", invoke_advisor)
    monkeypatch.setattr(grounding_agent, "coerce_structured_response", fail_coercion)
    monkeypatch.setattr(
        agent,
        "_capture_advisor_contract_failure",
        lambda **kwargs: captures.append(kwargs["attempt_index"]),
    )
    monkeypatch.setattr(grounding_agent, "log_event", capture_event)
    agent._pending_invoke_messages = [{"role": "user", "content": "stale prompt"}]

    with pytest.raises(RuntimeError) as raised:
        agent.analyze_intent("protect flow-1", context=_IEA_TEST_CONTEXT, session_id="session-1", snapshot_id="snapshot-1")

    assert str(raised.value) == "IEA advisor output contract validation failed: advisor response is malformed"
    assert raised.value.__cause__ is parse_error
    assert attempts == ["invoke", "invoke", "invoke"]
    assert captures == [1, 2, 3]
    assert [event["reason"] for event in retry_events] == [
        "advisor_output_contract",
        "advisor_output_contract",
    ]
    assert trace_writes[-1] == {
        "grounding_decision": None,
        "status": "error",
        "error": "IEA advisor output contract validation failed: advisor response is malformed",
    }
    assert not hasattr(agent, "_pending_invoke_messages")


@pytest.mark.parametrize(
    ("advisor_errors", "grounding_errors", "expected_message"),
    [
        (["advisor contract invalid"], ["grounding evidence missing"], "IEA advisor decision validation failed: advisor contract invalid"),
        ([], ["grounding evidence missing"], "IEA grounding validation failed: grounding evidence missing"),
    ],
)
def test_iea_public_analysis_keeps_distinct_terminal_validation_messages(
    monkeypatch,
    advisor_errors,
    grounding_errors,
    expected_message,
) -> None:
    agent, grounding_agent, evidence = _bare_iea_for_public_contract(monkeypatch)
    invocations = []
    decision = _grounding_decision()
    trace_writes = []

    def invoke_advisor(**_kwargs):
        invocations.append("invoke")
        return SimpleNamespace(
            advisor_result={},
            write_final_trace=lambda **kwargs: trace_writes.append(kwargs),
        )

    monkeypatch.setattr(agent, "_invoke_intent_advisor", invoke_advisor)
    monkeypatch.setattr(grounding_agent, "coerce_structured_response", lambda *_args, **_kwargs: object())
    def refresh_evidence(*, state):
        state.refreshed_evidence = evidence
        state.grounding_tools = []
        return evidence

    monkeypatch.setattr(agent, "_refresh_advisor_stage_evidence", refresh_evidence)
    monkeypatch.setattr(agent, "_compile_advisor_output", lambda **_kwargs: decision)
    monkeypatch.setattr(agent, "_materialize_catalog_facts", lambda **kwargs: kwargs["grounding_decision"])
    monkeypatch.setattr(agent, "_materialize_qos_constraint_facts", lambda **kwargs: kwargs["grounding_decision"])
    monkeypatch.setattr(agent, "_attach_subscription_evidence", lambda **kwargs: kwargs["grounding_decision"])
    monkeypatch.setattr(
        grounding_agent,
            "validate_grounding",
        lambda **_kwargs: (advisor_errors, grounding_errors, None),
    )

    with pytest.raises(RuntimeError) as raised:
        agent.analyze_intent("protect flow-1", context=_IEA_TEST_CONTEXT, session_id="session-1", snapshot_id="snapshot-1")

    assert str(raised.value) == expected_message
    assert raised.value.__cause__ is None
    assert invocations == ["invoke", "invoke", "invoke"]
    assert agent.last_failure_debug["advisor_validation_errors"] == advisor_errors
    assert agent.last_failure_debug["grounding_validation_errors"] == grounding_errors
    assert trace_writes[-1]["grounding_decision"] == decision
    assert trace_writes[-1]["status"] == "error"


def test_iea_public_analysis_does_not_retry_nonretryable_errors_and_cleans_pending_messages(monkeypatch) -> None:
    agent, grounding_agent, _evidence = _bare_iea_for_public_contract(monkeypatch)
    terminal_error = ValueError("advisor request is malformed")
    attempts = []
    retry_prompts = []
    retry_events = []

    def invoke_advisor(**_kwargs):
        attempts.append("invoke")
        raise terminal_error

    def capture_event(_logger, event, **kwargs):
        if event == "iea_retry":
            retry_events.append(kwargs)

    monkeypatch.setattr(agent, "_invoke_intent_advisor", invoke_advisor)
    monkeypatch.setattr(
        grounding_agent,
        "build_grounding_validation_retry_prompt",
        lambda **kwargs: retry_prompts.append(kwargs) or "retry prompt",
    )
    monkeypatch.setattr(grounding_agent, "log_event", capture_event)
    agent._pending_invoke_messages = [{"role": "user", "content": "stale prompt"}]

    with pytest.raises(ValueError) as raised:
        agent.analyze_intent("protect flow-1", context=_IEA_TEST_CONTEXT, session_id="session-1", snapshot_id="snapshot-1")

    assert raised.value is terminal_error
    assert attempts == ["invoke"]
    assert retry_events == []
    assert retry_prompts == []
    assert agent.last_failure_debug == {}
    assert not hasattr(agent, "_pending_invoke_messages")


def test_iea_public_analysis_does_not_retry_invocation_validation_errors(monkeypatch) -> None:
    """IEA only retries RuntimeError at invocation; schema errors belong to parse."""

    from pydantic import ValidationError

    agent, grounding_agent, _evidence = _bare_iea_for_public_contract(monkeypatch)
    terminal_error = ValidationError.from_exception_data(
        "GroundingAdvisorOutput",
        [
            {
                "type": "missing",
                "loc": ("flows",),
                "msg": "Field required",
                "input": {},
            }
        ],
    )
    attempts = []
    retry_prompts = []
    retry_events = []

    def invoke_advisor(**_kwargs):
        attempts.append("invoke")
        raise terminal_error

    def capture_event(_logger, event, **kwargs):
        if event == "iea_retry":
            retry_events.append(kwargs)

    monkeypatch.setattr(agent, "_invoke_intent_advisor", invoke_advisor)
    monkeypatch.setattr(
        grounding_agent,
        "build_grounding_validation_retry_prompt",
        lambda **kwargs: retry_prompts.append(kwargs) or "retry prompt",
    )
    monkeypatch.setattr(grounding_agent, "log_event", capture_event)
    agent._pending_invoke_messages = [{"role": "user", "content": "stale prompt"}]

    with pytest.raises(ValidationError) as raised:
        agent.analyze_intent("protect flow-1", context=_IEA_TEST_CONTEXT, session_id="session-1", snapshot_id="snapshot-1")

    assert raised.value is terminal_error
    assert attempts == ["invoke"]
    assert retry_events == []
    assert retry_prompts == []
    assert agent.last_failure_debug == {}
    assert not hasattr(agent, "_pending_invoke_messages")


def test_execute_planned_round_preserves_wrapper_loader_monkeypatch_seam_and_artifact_contract(monkeypatch) -> None:
    import control_runtime.orchestrators.round_execution as round_execution
    from control_runtime.agents.dispatch.contracts import FeedbackReport
    from control_runtime.diagnostics.diagnosis.contracts import AssuranceDiagnosisResult
    from control_runtime.diagnostics.mediation.contracts import ConflictResolutionResult
    from control_runtime.orchestrators.round_execution import RoundExecutionArtifacts

    loader_calls = []
    ue_context = {"accessMobilityContext": {"currentAssociationId": "am-1"}}
    monkeypatch.setattr(
        round_execution,
        "get_snapshot_data_by_id",
        lambda _: {"snapshot_id": "snapshot-1", "apps": [], "slices": [], "nodes": []},
    )
    monkeypatch.setattr(
        round_execution,
        "get_ue_context_by_supi",
        lambda supi, *, snapshot_id: loader_calls.append((supi, snapshot_id)) or ue_context,
    )
    report = FeedbackReport(
        execution_status="Success",
        performance_metrics='{"dispatch_results":[{"status":"success","policy_id":"sm-1","policy_type":"SmPolicyDecision"}]}',
        violation_details="none",
    )
    conflict_tool = _RecordingConflictTool(ConflictResolutionResult())

    artifacts = round_execution.execute_planned_round(
        session_id="session-1",
        snapshot_id="snapshot-1",
        round_index=3,
        policy_plan=_qos_plan(),
        global_intent=_main_intent(),
        grounding_decision=_grounding_decision(),
        cr_tool=conflict_tool,
        pd_agent=_RecordingDispatchAgent(report),
        ad_tool=_RecordingDiagnosisTool(AssuranceDiagnosisResult(status="healthy")),
    )

    assert isinstance(artifacts, RoundExecutionArtifacts)
    assert loader_calls == [("imsi-001", "snapshot-1")]
    assert conflict_tool.requests[0].upstream_context["ue_context"] == ue_context
    assert artifacts.trace.round_index == 3
    assert artifacts.trace.policy_plan == _qos_plan().model_dump(mode="json")
    assert artifacts.trace.mediator_decision == {"status": "approved", "reason_summary": "", "affected_domains": [], "revision_requests": [], "unified_constraints": {"hard_constraints": [], "evidence": []}}


def test_execute_planned_round_keeps_mediator_blocked_artifacts_without_dispatch(monkeypatch) -> None:
    import control_runtime.orchestrators.round_execution as round_execution
    from control_runtime.diagnostics.diagnosis.contracts import AssuranceDiagnosisResult
    from control_runtime.diagnostics.mediation.contracts import ConflictResolutionResult

    monkeypatch.setattr(
        round_execution,
        "get_snapshot_data_by_id",
        lambda _: {"snapshot_id": "snapshot-1", "apps": [], "slices": [], "nodes": []},
    )
    monkeypatch.setattr(round_execution, "get_ue_context_by_supi", lambda *_args, **_kwargs: {})
    conflict_result = ConflictResolutionResult(
        mediator_status="revise",
        reason_summary="binding conflict requires revision",
    )
    diagnosis_tool = _RecordingDiagnosisTool(
        AssuranceDiagnosisResult(root_cause_category="mediation_blocked", reason_summary="revise plan")
    )

    artifacts = round_execution.execute_planned_round(
        session_id="session-1",
        snapshot_id="snapshot-1",
        round_index=2,
        policy_plan=_qos_plan(),
        global_intent=_main_intent(),
        grounding_decision=_grounding_decision(),
        cr_tool=_RecordingConflictTool(conflict_result),
        pd_agent=SimpleNamespace(execute_and_evaluate=lambda *_args, **_kwargs: pytest.fail("dispatch must not run")),
        ad_tool=diagnosis_tool,
    )

    assert artifacts.completed is False
    assert artifacts.report is None
    assert artifacts.execution_reentry is None
    assert artifacts.qos_feedback == {
        "execution_status": "Failed",
        "violation_details": "binding conflict requires revision",
        "revision_requests": [],
    }
    assert artifacts.mobility_feedback == {
        "status": "failed",
        "error": "binding conflict requires revision",
        "revision_requests": [],
    }
    assert artifacts.trace.mediator_decision["status"] == "revise"
    assert artifacts.trace.domain_verdicts[-1]["status"] == "needs_revision"
    assert artifacts.trace.pda_feedback == {}
    assert diagnosis_tool.requests[0].execution_feedback["qos"] == artifacts.qos_feedback


def test_execute_planned_round_keeps_partial_plan_dispatch_and_artifact_semantics(monkeypatch) -> None:
    import control_runtime.orchestrators.round_execution as round_execution
    from control_runtime.agents.dispatch.contracts import FeedbackReport
    from control_runtime.diagnostics.diagnosis.contracts import AssuranceDiagnosisResult
    from control_runtime.diagnostics.mediation.contracts import ConflictResolutionResult

    monkeypatch.setattr(
        round_execution,
        "get_snapshot_data_by_id",
        lambda _: {"snapshot_id": "snapshot-1", "apps": [], "slices": [], "nodes": []},
    )
    monkeypatch.setattr(round_execution, "get_ue_context_by_supi", lambda *_args, **_kwargs: {})
    plan = _qos_plan(planning_status="partial_plan")
    plan.blocked_targets = ["flow-2"]
    plan.upstream_requests = ["reground flow-2"]
    report = FeedbackReport(
        execution_status="Success",
        performance_metrics='{"dispatch_results":[{"status":"success","policy_id":"sm-1","policy_type":"SmPolicyDecision"}]}',
        violation_details="none",
    )

    artifacts = round_execution.execute_planned_round(
        session_id="session-1",
        snapshot_id="snapshot-1",
        round_index=4,
        policy_plan=plan,
        global_intent=_main_intent(),
        grounding_decision=_grounding_decision(),
        cr_tool=_RecordingConflictTool(ConflictResolutionResult()),
        pd_agent=_RecordingDispatchAgent(report),
        ad_tool=_RecordingDiagnosisTool(AssuranceDiagnosisResult(status="healthy", reason_summary="dispatch completed")),
    )

    assert artifacts.completed is True
    assert artifacts.report is not report
    assert artifacts.report.intent_status == "partially_satisfied"
    assert artifacts.report.feedback_payload == {
        "intent_status": "partially_satisfied",
        "intent_blockers": ["flow-2"],
        "upstream_requests": ["reground flow-2"],
        "planning_status": "partial_plan",
    }
    assert artifacts.diagnosis["root_cause_category"] == "intent_partially_satisfied"
    assert artifacts.diagnosis["intent_status"] == "partially_satisfied"
    assert artifacts.trace.pda_feedback["intent_status"] == "partially_satisfied"
    assert artifacts.unified_plan.approved_policies == [
        {"domain": "qos", "policy_id": "sm-1", "policy_type": "SmPolicyDecision"}
    ]


def test_main_public_analysis_retries_runtime_invocations_then_succeeds_with_retry_prompt_logs_and_projection(
    monkeypatch,
) -> None:
    agent, main_agent = _bare_main_for_public_contract(monkeypatch)
    first_error = RuntimeError("main transport unavailable 1")
    second_error = RuntimeError("main transport unavailable 2")
    invocation_prompts = []
    retry_prompts = []
    events = []
    trace_writes = []

    def invoke_global_intent(prompt, **_kwargs):
        invocation_prompts.append(prompt)
        if len(invocation_prompts) == 1:
            raise first_error
        if len(invocation_prompts) == 2:
            raise second_error
        return SimpleNamespace(
            raw_result=_main_control_decision_result(include_compiled_field=True),
            write_final_trace=lambda **kwargs: trace_writes.append(kwargs),
        )

    def build_retry_prompt(**kwargs):
        retry_prompts.append(kwargs)
        return f"retry prompt {len(retry_prompts)}"

    def capture_event(_logger, event, **kwargs):
        events.append((event, kwargs))

    monkeypatch.setattr(agent, "_invoke_global_intent_result", invoke_global_intent)
    monkeypatch.setattr(main_agent, "build_main_validation_retry_prompt", build_retry_prompt)
    monkeypatch.setattr(main_agent, "log_event", capture_event)
    agent._pending_invoke_messages = [{"role": "user", "content": "stale prompt"}]

    result = agent.analyze_global_intent(
        user_input="protect QoS",
        context="{}",
        session_id="session-1",
        snapshot_id="snapshot-1",
    )

    assert result.next_agent == "intent_encoding"
    assert len(invocation_prompts) == 3
    assert invocation_prompts[1:] == ["retry prompt 1", "retry prompt 2"]
    assert [prompt["base_prompt"] for prompt in retry_prompts] == [invocation_prompts[0]] * 2
    assert [prompt["validation_errors"] for prompt in retry_prompts] == [[], []]
    assert [prompt["invocation_error"] for prompt in retry_prompts] == [str(first_error), str(second_error)]
    assert [kwargs["attempt"] for event, kwargs in events if event == "main_control_attempt_start"] == [1, 2, 3]
    assert [
        (kwargs["attempt"], kwargs["validation_errors"], kwargs["supi"])
        for event, kwargs in events
        if event == "main_control_validation_failed"
    ] == [
        (1, str(first_error), "<empty>"),
        (2, str(second_error), "<empty>"),
    ]
    assert agent.schema_projection_metrics == {
        "count": 1,
        "field_count": 1,
        "fields": ["routing_decision"],
    }
    assert trace_writes == [
        {"status": "success", "structured_response": result.model_dump(mode="json")}
    ]
    assert not hasattr(agent, "_pending_invoke_messages")


def test_main_public_analysis_keeps_terminal_wrapped_parse_error_identity_cause_and_trace(monkeypatch) -> None:
    import control_runtime.agents.main.compiler as main_compiler

    agent, main_agent = _bare_main_for_public_contract(monkeypatch)
    compiler_error = RuntimeError("compiler rejected main decision")
    wrapped_parse_errors = []
    invocations = []
    retry_prompts = []
    trace_writes = []
    original_validate = main_compiler.validate_global_intent_result

    def invoke_global_intent(_prompt, **_kwargs):
        invocations.append("invoke")
        return SimpleNamespace(
            raw_result=_main_control_decision_result(),
            write_final_trace=lambda **kwargs: trace_writes.append(kwargs),
        )

    def capture_wrapped_parse_error(*args, **kwargs):
        try:
            return original_validate(*args, **kwargs)
        except RuntimeError as error:
            wrapped_parse_errors.append(error)
            raise

    def fail_compile(*_args, **_kwargs):
        raise compiler_error

    def build_retry_prompt(**kwargs):
        retry_prompts.append(kwargs)
        return f"retry prompt {len(retry_prompts)}"

    monkeypatch.setattr(agent, "_invoke_global_intent_result", invoke_global_intent)
    monkeypatch.setattr(main_agent, "validate_global_intent_result", capture_wrapped_parse_error)
    monkeypatch.setattr(main_compiler, "compile_global_intent", fail_compile)
    monkeypatch.setattr(main_agent, "build_main_validation_retry_prompt", build_retry_prompt)
    monkeypatch.setattr(main_agent, "log_event", lambda *_args, **_kwargs: None)
    agent._pending_invoke_messages = [{"role": "user", "content": "stale prompt"}]

    with pytest.raises(RuntimeError) as raised:
        agent.analyze_global_intent(
            user_input="protect QoS",
            context="{}",
            session_id="session-1",
            snapshot_id="snapshot-1",
        )

    expected_message = "Main Agent returned invalid MainControlDecision payload: compiler rejected main decision"
    assert str(raised.value) == expected_message
    assert raised.value is wrapped_parse_errors[-1]
    assert len(wrapped_parse_errors) == 3
    assert all(error.__cause__ is compiler_error for error in wrapped_parse_errors)
    assert raised.value.__cause__ is compiler_error
    assert invocations == ["invoke", "invoke", "invoke"]
    assert [prompt["validation_errors"] for prompt in retry_prompts] == [[], []]
    assert [prompt["invocation_error"] for prompt in retry_prompts] == [expected_message, expected_message]
    assert trace_writes == [
        {"status": "error", "structured_response": None, "error": expected_message}
    ]
    assert not hasattr(agent, "_pending_invoke_messages")


def test_main_public_analysis_keeps_terminal_semantic_validation_message_trace_and_projection(monkeypatch) -> None:
    agent, main_agent = _bare_main_for_public_contract(monkeypatch)
    intent = _main_intent(next_agent="intent_encoding")
    semantic_errors = ["semantic target is incomplete", "route must retain required evidence"]
    invocations = []
    retry_prompts = []
    events = []
    trace_writes = []

    def invoke_global_intent(_prompt, **_kwargs):
        invocations.append("invoke")
        return SimpleNamespace(
            raw_result={},
            write_final_trace=lambda **kwargs: trace_writes.append(kwargs),
        )

    def return_intent(_result, *, projected_fields, **_kwargs):
        projected_fields.append("routing_decision")
        return intent

    def build_retry_prompt(**kwargs):
        retry_prompts.append(kwargs)
        return f"retry prompt {len(retry_prompts)}"

    def capture_event(_logger, event, **kwargs):
        events.append((event, kwargs))

    monkeypatch.setattr(agent, "_invoke_global_intent_result", invoke_global_intent)
    monkeypatch.setattr(main_agent, "validate_global_intent_result", return_intent)
    monkeypatch.setattr(main_agent, "normalize_semantic_target_scope", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_agent, "validate_main_global_intent", lambda *_args, **_kwargs: semantic_errors)
    monkeypatch.setattr(main_agent, "build_main_validation_retry_prompt", build_retry_prompt)
    monkeypatch.setattr(main_agent, "log_event", capture_event)
    agent._pending_invoke_messages = [{"role": "user", "content": "stale prompt"}]

    with pytest.raises(RuntimeError) as raised:
        agent.analyze_global_intent(
            user_input="protect QoS",
            context="{}",
            session_id="session-1",
            snapshot_id="snapshot-1",
        )

    expected_message = (
        "Main Agent could not produce a valid MainControlDecision: "
        "semantic target is incomplete; route must retain required evidence"
    )
    assert str(raised.value) == expected_message
    assert raised.value.__cause__ is None
    assert invocations == ["invoke", "invoke", "invoke"]
    assert [prompt["validation_errors"] for prompt in retry_prompts] == [semantic_errors, semantic_errors]
    assert [prompt["invocation_error"] for prompt in retry_prompts] == ["", ""]
    assert [
        (kwargs["attempt"], kwargs["validation_errors"], kwargs["supi"])
        for event, kwargs in events
        if event == "main_control_validation_failed"
    ] == [
        (1, "semantic target is incomplete || route must retain required evidence", "imsi-001"),
        (2, "semantic target is incomplete || route must retain required evidence", "imsi-001"),
        (3, "semantic target is incomplete || route must retain required evidence", "imsi-001"),
    ]
    assert agent.schema_projection_metrics == {
        "count": 3,
        "field_count": 3,
        "fields": ["routing_decision"],
    }
    assert trace_writes == [
        {
            "status": "error",
            "structured_response": intent.model_dump(mode="json"),
            "error": expected_message,
        }
    ]
    assert not hasattr(agent, "_pending_invoke_messages")


def test_main_public_analysis_does_not_retry_nonretryable_parse_value_error_and_cleans_pending_messages(
    monkeypatch,
) -> None:
    agent, main_agent = _bare_main_for_public_contract(monkeypatch)
    terminal_error = ValueError("main response is malformed")
    invocations = []
    retry_prompts = []
    events = []
    trace_writes = []

    def invoke_global_intent(_prompt, **_kwargs):
        invocations.append("invoke")
        return SimpleNamespace(
            raw_result={},
            write_final_trace=lambda **kwargs: trace_writes.append(kwargs),
        )

    def fail_parse(*_args, **_kwargs):
        raise terminal_error

    def capture_event(_logger, event, **kwargs):
        events.append((event, kwargs))

    monkeypatch.setattr(agent, "_invoke_global_intent_result", invoke_global_intent)
    monkeypatch.setattr(main_agent, "validate_global_intent_result", fail_parse)
    monkeypatch.setattr(
        main_agent,
        "build_main_validation_retry_prompt",
        lambda **kwargs: retry_prompts.append(kwargs) or "retry prompt",
    )
    monkeypatch.setattr(main_agent, "log_event", capture_event)
    agent._pending_invoke_messages = [{"role": "user", "content": "stale prompt"}]

    with pytest.raises(ValueError) as raised:
        agent.analyze_global_intent(
            user_input="protect QoS",
            context="{}",
            session_id="session-1",
            snapshot_id="snapshot-1",
        )

    assert raised.value is terminal_error
    assert invocations == ["invoke"]
    assert retry_prompts == []
    assert [kwargs["attempt"] for event, kwargs in events if event == "main_control_attempt_start"] == [1]
    assert not [event for event, _kwargs in events if event == "main_control_validation_failed"]
    assert agent.schema_projection_metrics == {"count": 0, "field_count": 0, "fields": []}
    assert trace_writes == [
        {"status": "error", "structured_response": None, "error": str(terminal_error)}
    ]
    assert not hasattr(agent, "_pending_invoke_messages")


def test_stage_runner_reraises_before_attempt_error_without_running_callbacks_or_invoke() -> None:
    from shared.agents import RetryPolicy, StageRunner

    runner = StageRunner(retry_policy=RetryPolicy(max_attempts=3))
    before_error = RuntimeError("attempt setup failed")
    before_attempts = []
    invocations = []
    failures = []
    retries = []

    def before_attempt(attempt):
        before_attempts.append((attempt.attempt, attempt.max_attempts))
        raise before_error

    with pytest.raises(RuntimeError) as raised:
        runner.run(
            invoke=lambda _attempt: invocations.append("invoke"),
            before_attempt=before_attempt,
            on_failure=lambda context: failures.append(context),
            on_retry=lambda context: retries.append(context),
        )

    assert raised.value is before_error
    assert before_attempts == [(1, 3)]
    assert invocations == []
    assert failures == []
    assert retries == []
    assert runner.last_failure is None
    assert runner.failure_history == ()


def test_control_round_trace_round_trip_deeply_isolates_typed_detail_payloads() -> None:
    payload = {
        "round_index": 7,
        "mediator_decision": {"constraints": {"hard": [{"name": "latency"}]}},
        "agent_elapsed_ms": {"planning": 12.5},
        "planning_blocker": {"upstream_requests": [{"target": "flow-1"}]},
        "execution_reentry": {"reason_by_domain": {"qos": ["retry binding"]}},
        "single_agent_detail": {"attempts": [{"stage": "single_control"}]},
    }

    trace = ControlRoundTrace.from_payload(payload)
    payload["mediator_decision"]["constraints"]["hard"][0]["name"] = "mutated-input"
    payload["agent_elapsed_ms"]["planning"] = 999.0
    payload["planning_blocker"]["upstream_requests"][0]["target"] = "mutated-input"
    payload["execution_reentry"]["reason_by_domain"]["qos"][0] = "mutated-input"
    payload["single_agent_detail"]["attempts"][0]["stage"] = "mutated-input"

    round_trip = trace.to_payload()
    assert round_trip["mediator_decision"] == {"constraints": {"hard": [{"name": "latency"}]}}
    assert round_trip["agent_elapsed_ms"] == {"planning": 12.5}
    assert round_trip["planning_blocker"] == {"upstream_requests": [{"target": "flow-1"}]}
    assert round_trip["execution_reentry"] == {"reason_by_domain": {"qos": ["retry binding"]}}
    assert round_trip["single_agent_detail"] == {"attempts": [{"stage": "single_control"}]}

    round_trip["mediator_decision"]["constraints"]["hard"][0]["name"] = "mutated-output"
    round_trip["agent_elapsed_ms"]["planning"] = 0.0
    round_trip["planning_blocker"]["upstream_requests"][0]["target"] = "mutated-output"
    round_trip["execution_reentry"]["reason_by_domain"]["qos"][0] = "mutated-output"
    round_trip["single_agent_detail"]["attempts"][0]["stage"] = "mutated-output"

    assert trace.to_payload()["mediator_decision"]["constraints"]["hard"][0]["name"] == "latency"
    assert trace.to_payload()["agent_elapsed_ms"]["planning"] == 12.5
    assert trace.to_payload()["planning_blocker"]["upstream_requests"][0]["target"] == "flow-1"
    assert trace.to_payload()["execution_reentry"]["reason_by_domain"]["qos"] == ["retry binding"]
    assert trace.to_payload()["single_agent_detail"]["attempts"][0]["stage"] == "single_control"


def test_append_round_trace_is_typed_only_and_retains_complete_recent_trace_payloads() -> None:
    state = OrchestratorLoopState()

    with pytest.raises(TypeError, match="trace_payload"):
        append_round_trace(state, trace_payload={"round_index": 1})

    for round_index in range(1, 5):
        append_round_trace(
            state,
            trace=ControlRoundTrace(
                round_index=round_index,
                diagnosis={
                    "root_cause_category": f"cause-{round_index}",
                    "reason_summary": f"reason-{round_index}",
                },
                mediator_decision={"status": f"decision-{round_index}"},
                agent_elapsed_ms={"planning": float(round_index)},
            ),
            feedback_added=f"[Round Feedback]\\nround_index: {round_index}",
        )

    expected_keys = {
        "round_index",
        "global_intent",
        "grounding_decision",
        "policy_plan",
        "domain_verdicts",
        "pda_feedback",
        "qos_feedback",
        "mobility_feedback",
        "diagnosis",
        "mediator_decision",
        "negotiation_request",
        "planning_blocker",
        "execution_reentry",
        "single_agent_detail",
        "agent_elapsed_ms",
    }
    assert [snapshot.trace.round_index for snapshot in state.rounds] == [2, 3, 4]
    assert all(set(trace) == expected_keys for trace in state.round_traces)
    assert [trace["agent_elapsed_ms"] for trace in state.round_traces] == [
        {"planning": 2.0},
        {"planning": 3.0},
        {"planning": 4.0},
    ]

    feedback_context = build_feedback_context_from_snapshots(state.rounds)
    assert "[Older Feedback Summary]" in feedback_context
    assert "round 2: cause-2; reason-2" in feedback_context


def test_round_execution_does_not_export_execution_order_facade() -> None:
    import control_runtime.orchestrators.round_execution as round_execution

    assert "execution_order" not in round_execution.__all__
    assert not hasattr(round_execution, "execution_order")
