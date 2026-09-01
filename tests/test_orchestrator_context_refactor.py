from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytest

from shared.contracts import (
    AgentMessage,
    AgentRunResult,
    ExecutionScope,
    MemoryBundle,
    MemoryContextView,
    MemoryRecord,
)
from control_runtime.contracts import standard_objective_profile

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


def _execution_scope(round_index: int = 1) -> ExecutionScope:
    return ExecutionScope(
        session_id="session-1",
        thread_id="session-1",
        snapshot_id="snapshot-1",
        round_index=round_index,
        supi="imsi-001",
    )


def _dispatch_metrics(
    dispatch_results: list[dict[str, object]],
    *,
    scope: ExecutionScope,
    assurance_results: list[dict[str, object]] | None = None,
    dispatch_attempts: int,
    failure_summaries: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "metrics_kind": "dispatch",
        "policy_plan": {
            "scope": scope.model_dump(mode="json"),
            "policies": [],
        },
        "dispatch_results": dispatch_results,
        "assurance_results": assurance_results or [],
        "dispatch_attempts": dispatch_attempts,
        "domain_receipts": {
            "qos": [
                item
                for item in dispatch_results
                if item.get("policy_type") != "PcfAmPolicyControlPolicyAssociation"
            ],
            "mobility": [
                item
                for item in dispatch_results
                if item.get("policy_type") == "PcfAmPolicyControlPolicyAssociation"
            ],
        },
        "failure_summaries": failure_summaries or [],
    }


def _dispatch_failure_feedback(
    *,
    dispatch_attempts: int,
    failures: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "feedback_kind": "dispatch_failure",
        "supi": "imsi-001",
        "policy_id": "sm-1",
        "flow_id": "flow-1",
        "reason": "gateway rejected sm-1",
        "dispatch_attempts": dispatch_attempts,
        "target_bindings_at_risk": ["flow-1"],
        "policy_objects_at_risk": ["sm-1"],
        "reason_by_domain": {"qos": "gateway rejected policy"},
        "controller_feedback": {
            "phase": "execution",
            "failure_scope": "qos",
            "successful_policy_ids": [],
            "failed_policy_ids": ["sm-1"],
            "failures": failures,
            "retry_forbidden": False,
        },
        "failure_scope": "qos",
        "failures": failures,
        "retry_forbidden": False,
    }

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


def _main_intent(*, next_agent: str = "optimization_strategy", round_index: int = 1):
    from control_runtime.agents.main.contracts import ControlDomain, GlobalControlIntent

    return GlobalControlIntent(
        scope=ExecutionScope(
            session_id="session-1",
            thread_id="session-1",
            snapshot_id="snapshot-1",
            round_index=round_index,
            supi="imsi-001",
        ),
        next_agent=next_agent,
        requested_domains=[ControlDomain.QOS],
        objective_profile=standard_objective_profile("balanced"),
        routing_decision="protect qos flow",
        routing_rationale="offline characterization",
    )


def _grounding_decision(*, round_index: int = 1):
    from control_runtime.agents.grounding.contracts import (
        FlowSelector,
        GroundedMobilityEvidence,
        GroundingDecision,
        GroundingEvidenceBundle,
        SliceMigrationAuthorization,
    )

    return GroundingDecision(
        scope=_execution_scope(round_index),
        flows=[
            FlowSelector(
                supi="imsi-001",
                app_id="app-1",
                app_name="video-app",
                flow_id="flow-1",
                target_type="flow",
                name="video",
                service_type=None,
                service_type_id=None,
                bw_ul=None,
                bw_dl=None,
                gbr_ul=None,
                gbr_dl=None,
                lat=None,
                loss_req=None,
                jitter_req=None,
                priority=None,
                description=None,
                five_tuple=None,
                current_slice_snssai="01000001",
                current_bw_ul=None,
                current_bw_dl=None,
                resolution_status="resolved",
            )
        ],
        mobility_intent=None,
        grounding_evidence=GroundingEvidenceBundle(
            grounded_supi="imsi-001",
            grounded_apps=[],
            grounded_flows=[],
            mobility=GroundedMobilityEvidence(summary=None, subscription_entitlement=None),
            evidence_sources={},
        ),
        qos_operation_constraints=[],
        slice_migration_authorization=SliceMigrationAuthorization(
            decision="not_applicable",
            authority=None,
            authorized_snssais=[],
            target_snssais=[],
            subscription_change_required=False,
            rationale=[],
        ),
        open_questions=[],
    )


def _qos_plan(*, planning_status: str = "executable_plan", round_index: int = 1):
    from control_runtime.agents.planning.contracts import (
        OptimizerPlanEvidence,
        PolicyDraft,
        PolicyPlanDraft,
        PlanningRationale,
    )

    return PolicyPlanDraft(
        scope=_execution_scope(round_index),
        planning_status=planning_status,
        optimizer_result=OptimizerPlanEvidence(
            objective_breakdown={},
            cross_domain_verdicts=[],
        ),
        planning_rationale=PlanningRationale(
            selected_strategy_profile="offline",
            decisive_evidence=[],
            active_constraints=[],
            explanation="offline plan rationale",
            rejected_alternatives=[],
            main_constraints=[],
            iea_grounding_basis=[],
            osa_decision_basis=[],
            unresolved_gaps=["flow-2"] if planning_status == "partial_plan" else [],
        ),
        all_policies=[
            PolicyDraft(
                policy_id="sm-1",
                policy_type="SmPolicyDecision",
                supi="imsi-001",
                app_id="app-1",
                flow_id="flow-1",
                target_type="flow",
                resource_keys=["flow:flow-1"],
                policy_details={"flow_id": "flow-1"},
            )
        ],
        partial_policies=[],
        missing_evidence=[],
        blocked_targets=["flow-2"] if planning_status == "partial_plan" else [],
        upstream_requests=["reground flow-2"] if planning_status == "partial_plan" else [],
        planner_conflicts=[],
        open_questions=[],
    )


def _planning_context():
    from control_runtime.agents.planning.contracts import PlanningContext, SharedControlContext
    from control_runtime.diagnostics.mediation.contracts import UnifiedConstraintSet

    return PlanningContext(
        memory_context=MemoryContextView(
            query=None,
            selected_short_term=[],
            selected_long_term=[],
            prompt_text="",
        ),
        shared_context=SharedControlContext(main_intent=_main_intent()),
        feedback_context="",
        handoff_history=[],
        revision_requests=[],
        unified_constraints=UnifiedConstraintSet(hard_constraints=[], evidence=[]),
        qos_retry_constraints=[],
    )


def _reground_plan():
    from control_runtime.agents.planning.contracts import (
        OptimizerPlanEvidence,
        PolicyPlanDraft,
        PlanningRationale,
    )

    return PolicyPlanDraft(
        scope=_execution_scope(),
        planning_status="needs_upstream_reground",
        optimizer_result=OptimizerPlanEvidence(
            objective_breakdown={},
            cross_domain_verdicts=[],
        ),
        planning_rationale=PlanningRationale(
            selected_strategy_profile="reground",
            decisive_evidence=[],
            active_constraints=[],
            explanation="upstream evidence required",
            rejected_alternatives=[],
            main_constraints=[],
            iea_grounding_basis=[],
            osa_decision_basis=[],
            unresolved_gaps=["QoS grounding evidence"],
        ),
        all_policies=[],
        partial_policies=[],
        missing_evidence=["QoS grounding evidence"],
        blocked_targets=[],
        upstream_requests=["reground QoS flow"],
        planner_conflicts=[],
        open_questions=[],
    )


def _conflict_result(*, mediator_status: str = "approved", reason_summary: str = "no conflicts"):
    from control_runtime.diagnostics.mediation.contracts import (
        ConflictResolutionResult,
        UnifiedConstraintSet,
    )

    return ConflictResolutionResult(
        mediator_status=mediator_status,
        conflicts=[],
        affected_policy_ids=[],
        affected_objects=[],
        affected_domains=[],
        resolution_recommendations=[],
        reason_summary=reason_summary,
        revision_requests=[],
        unified_constraints=UnifiedConstraintSet(hard_constraints=[], evidence=[]),
    )


def _diagnosis_result(
    *,
    root_cause_category: str = "none",
    reason_summary: str = "no assurance violation",
):
    from control_runtime.diagnostics.diagnosis.contracts import AssuranceDiagnosisResult

    return AssuranceDiagnosisResult(
        status="diagnosed",
        root_cause_category=root_cause_category,
        root_cause=reason_summary,
        affected_policy_ids=[],
        affected_flow_ids=[],
        recommended_actions=[],
        reason_summary=reason_summary,
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
    monkeypatch.setattr(
        agent,
        "build_runtime_context",
        lambda **kwargs: SimpleNamespace(scope=kwargs["scope"]),
    )
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
    monkeypatch.setattr(agent, "_resolve_token_context", lambda: (None, None))
    monkeypatch.setattr(
        agent,
        "build_runtime_context",
        lambda **kwargs: SimpleNamespace(scope=kwargs["scope"]),
    )
    return agent, main_agent


def _main_control_decision_result(*, include_compiled_field: bool = False) -> AgentRunResult:
    from control_runtime.agents.main.contracts import ControlDomain, MainControlDecision

    decision = MainControlDecision(
        supi="",
        next_agent="intent_encoding",
        requested_domains=[ControlDomain.QOS],
        serving_slice_change_requested=False,
        domain_evidence={},
        control_semantics={
            "mode": "single_step",
            "current_stage": 1,
            "stages": [],
            "notes": [],
        },
        objective_profile=standard_objective_profile("balanced"),
        investigation_targets=[],
        uncertainty_flags=[],
        retry_scope=None,
        required_evidence=[],
        forbidden_assumptions=[],
        intent_encoding_guidance="",
    ).model_dump(mode="json")
    if include_compiled_field:
        decision["routing_decision"] = "compiler-owned field"
    return AgentRunResult(
        messages=[AgentMessage(role="assistant", content=json.dumps(decision))],
        structured_response=decision,
    )


def test_canonical_runtime_paths_stay_bound_to_the_repository_layout() -> None:
    from shared.runtime.io.paths import project_root, runtime_root, trace_runs_root

    assert project_root() == ROOT
    assert runtime_root() == ROOT / "data" / "runtime"
    assert trace_runs_root() == ROOT / "data" / "traces" / "runs"
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
        from control_runtime.orchestrators.contracts import (
            DomainNegotiationRequest,
            OrchestrationFailureReport,
        )

        state = OrchestratorLoopState()

        for index in range(1, 5):
            append_round_trace(
                state,
                trace=ControlRoundTrace(
                    scope=_execution_scope(index),
                    round_index=index,
                    global_intent=_main_intent(round_index=index),
                    grounding_decision=_grounding_decision(round_index=index),
                    policy_plan=_qos_plan(round_index=index),
                    orchestration_failure=OrchestrationFailureReport(
                        failure_stage="planning",
                        round_index=index,
                        violation_details=f"failure-{index}",
                        error_type="RuntimeError",
                        error=f"failure-{index}",
                        correction_suggestion="retry planning",
                        recommended_consumer="main_control",
                        debug_context={},
                    ),
                    diagnosis={"root_cause_category": f"category-{index}"},
                    negotiation_request=DomainNegotiationRequest(
                        round_index=index,
                        summary=f"negotiation-{index}",
                    ),
                ),
                feedback_added=f"[Round Feedback]\nround_index: {index}",
            )

        self.assertEqual([snap.trace.round_index for snap in state.rounds], [1, 2, 3, 4])
        self.assertEqual(state.previous_diagnosis, {"root_cause_category": "category-4"})
        self.assertEqual(state.previous_report_payload["violation_details"], "failure-4")
        self.assertEqual(state.previous_negotiation_request["summary"], "negotiation-4")
        with self.assertRaises(Exception):
            state.rounds[-1].trace.diagnosis = {}

    def test_feedback_context_summarizes_older_rounds_and_keeps_recent_full(self) -> None:
        snapshots = [
            ControlRoundSnapshot(
                trace=ControlRoundTrace(
                    scope=_execution_scope(1),
                    round_index=1,
                    diagnosis={"root_cause_category": "old_binding", "reason_summary": "wrong flow binding"},
                ),
                feedback_added="[Round Feedback]\nround_index: 1\nvery old verbose detail",
            ),
            ControlRoundSnapshot(
                trace=ControlRoundTrace(
                    scope=_execution_scope(2),
                    round_index=2,
                    diagnosis={"root_cause_category": "recent_policy", "reason_summary": "policy failed"},
                ),
                feedback_added="[Round Feedback]\nround_index: 2\nrecent detail",
            ),
            ControlRoundSnapshot(
                trace=ControlRoundTrace(
                    scope=_execution_scope(3),
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
        scope = ExecutionScope(session_id="session-1", thread_id="session-1")

        class Memory:
            def retrieve(self, query):
                return MemoryBundle(
                    query=query,
                    short_term=[
                        MemoryRecord(memory_id="m1", scope=scope, kind="event", source_agent="main", role="MAIN", content="generic context"),
                        MemoryRecord(memory_id="m2", scope=scope, kind="event", source_agent="ad", role="AD", content="planning_blocked optimizer preview missing"),
                    ],
                    long_term=[
                        MemoryRecord(memory_id="m3", scope=scope, kind="summary", source_agent="memory", role="summary", content="general success case"),
                        MemoryRecord(memory_id="m4", scope=scope, kind="summary", source_agent="memory", role="summary", content="optimization_strategy target_stable retry fixed planner gap"),
                    ],
                )

        context = build_memory_context(
            "QoS request",
            scope=scope,
            memory_manager=Memory(),
            diagnosis_hint="planning_blocked",
            routing_hint="optimization_strategy target_stable",
        )

        self.assertLess(
            context.prompt_text.index("planning_blocked"),
            context.prompt_text.index("generic context"),
        )
        self.assertIn("optimization_strategy target_stable", context.prompt_text)
        self.assertEqual(
            [record.memory_id for record in context.selected_short_term],
            ["m2", "m1"],
        )
        self.assertEqual(context.query.text, "QoS request")

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
        intent = _main_intent()
        grounding = _grounding_decision()
        policy_plan = _qos_plan()
        snapshot = ControlRoundSnapshot(
            trace=ControlRoundTrace.from_payload(
                {
                    "round_index": 1,
                    "scope": _execution_scope().model_dump(mode="json"),
                    "global_intent": intent.model_dump(mode="json"),
                    "grounding_decision": grounding.model_dump(mode="json"),
                    "policy_plan": policy_plan.model_dump(mode="json"),
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
    from control_runtime.agents.planning.contracts import PolicyPlanDraft

    monkeypatch.setattr(
        round_execution,
        "get_snapshot_data_by_id",
        lambda _: {"snapshot_id": "snapshot-1"},
    )
    blocked_payload = _reground_plan().model_dump(mode="json")
    blocked_payload["planning_rationale"]["explanation"] = "catalog evidence is missing"
    blocked_payload["missing_evidence"] = ["SM flow catalog"]
    blocked_payload["upstream_requests"] = ["reground flow-1"]
    blocked_plan = PolicyPlanDraft.model_validate(blocked_payload)

    artifacts = round_execution.execute_planned_round(
        scope=_execution_scope(),
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
    assert artifacts.trace.mediator_decision is None
    assert artifacts.trace.planning_blocker == artifacts.planning_blocker
    assert artifacts.trace.planning_blocker.planning_status == "needs_upstream_reground"


def test_execute_planned_round_returns_success_contract_from_stubbed_boundaries(monkeypatch) -> None:
    import control_runtime.orchestrators.round_execution as round_execution
    from control_runtime.agents.dispatch.contracts import FeedbackReport
    from control_runtime.agents.main.contracts import ControlDomain

    monkeypatch.setattr(
        round_execution,
        "get_snapshot_data_by_id",
        lambda _: {"snapshot_id": "snapshot-1", "apps": [], "slices": [], "nodes": []},
    )
    monkeypatch.setattr(round_execution, "get_ue_context_by_supi", lambda *_args, **_kwargs: {})
    report = FeedbackReport(
        scope=_execution_scope(),
        execution_status="Success",
        performance_metrics=_dispatch_metrics(
            [{"status": "success", "policy_id": "sm-1", "policy_type": "SmPolicyDecision"}],
            scope=_execution_scope(),
            assurance_results=[
                {"status": "satisfied", "policy_id": "sm-1", "policy_type": "SmPolicyDecision"}
            ],
            dispatch_attempts=1,
        ),
        violation_details="none",
        failure_scope="none",
        feedback_payload=None,
        dispatch_attempts=1,
    )
    conflict_tool = _RecordingConflictTool(_conflict_result())
    dispatch_agent = _RecordingDispatchAgent(report)
    diagnosis_tool = _RecordingDiagnosisTool(_diagnosis_result(reason_summary="SLA satisfied"))

    artifacts = round_execution.execute_planned_round(
        scope=_execution_scope(),
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
    assert artifacts.unified_plan.scope.session_id == "session-1"
    assert artifacts.unified_plan.scope.snapshot_id == "snapshot-1"
    assert artifacts.unified_plan.scope.supi == "imsi-001"
    assert artifacts.qos_feedback["execution_status"] == "Success"
    assert artifacts.unified_plan.execution_order == [ControlDomain.QOS]
    assert artifacts.unified_plan.approved_policies == [
        {"domain": "qos", "policy_id": "sm-1", "policy_type": "SmPolicyDecision"}
    ]
    assert artifacts.trace.dispatch_report.execution_status == "Success"
    assert artifacts.trace.mediator_decision == artifacts.unified_plan.mediator_decision
    assert conflict_tool.requests[0].scope.snapshot_id == "snapshot-1"
    assert dispatch_agent.calls[0][1] == {"scenario_id": "offline-case"}
    assert diagnosis_tool.requests[0].execution_feedback.pda["execution_status"] == "Success"


def test_execute_planned_round_emits_reentry_contract_after_dispatch_failure(monkeypatch) -> None:
    import control_runtime.orchestrators.round_execution as round_execution
    from control_runtime.agents.dispatch.contracts import FeedbackReport

    monkeypatch.setattr(
        round_execution,
        "get_snapshot_data_by_id",
        lambda _: {"snapshot_id": "snapshot-1", "apps": [], "slices": [], "nodes": []},
    )
    monkeypatch.setattr(round_execution, "get_ue_context_by_supi", lambda *_args, **_kwargs: {})
    failures = [{"phase": "dispatch", "policy_id": "sm-1"}]
    report = FeedbackReport(
        scope=_execution_scope(2),
        execution_status="Failed",
        performance_metrics=_dispatch_metrics(
            [{"status": "failed", "policy_id": "sm-1", "policy_type": "SmPolicyDecision"}],
            scope=_execution_scope(2),
            dispatch_attempts=1,
            failure_summaries=failures,
        ),
        violation_details="policy gateway rejected sm-1",
        failure_scope="qos",
        feedback_payload=_dispatch_failure_feedback(dispatch_attempts=1, failures=failures),
        dispatch_attempts=1,
    )

    artifacts = round_execution.execute_planned_round(
        scope=_execution_scope(2),
        policy_plan=_qos_plan(round_index=2),
        global_intent=_main_intent(round_index=2),
        grounding_decision=_grounding_decision(round_index=2),
        cr_tool=_RecordingConflictTool(_conflict_result()),
        pd_agent=_RecordingDispatchAgent(report),
        ad_tool=_RecordingDiagnosisTool(
            _diagnosis_result(
                root_cause_category="dispatch_failure",
                reason_summary="gateway rejected",
            )
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
    assert artifacts.trace.execution_reentry.reason_by_domain == {"qos": "gateway rejected policy"}


def test_osa_public_generate_strategy_rejects_wrong_input_and_returns_reground_plan_without_llm(monkeypatch) -> None:
    from control_runtime.agents.planning.agent import OptimizationStrategyAgent
    from control_runtime.agents.grounding.contracts import GroundingDecision
    from control_runtime.agents.planning.contracts import (
        PlanningRequest,
    )

    class BareOptimizationStrategyAgent(OptimizationStrategyAgent):
        pass

    agent = object.__new__(BareOptimizationStrategyAgent)
    monkeypatch.setattr(agent, "ensure_worker_runtime_initialized", lambda: None)

    with pytest.raises(TypeError, match="PlanningRequest"):
        agent.generate_strategy({"not": "a PlanningRequest"})

    expected = _reground_plan()
    compiler_calls = []

    class Compiler:
        def build_upstream_reground_plan(self, *, planning_request, reason):
            compiler_calls.append((planning_request, reason))
            return expected

    agent.compiler = Compiler()
    monkeypatch.setattr(agent, "_cache_received_request", lambda _request: object())
    empty_grounding_payload = _grounding_decision().model_dump(mode="json")
    empty_grounding_payload["flows"] = []
    request = PlanningRequest(
        scope=_execution_scope(),
        grounding_decision=GroundingDecision.model_validate(empty_grounding_payload),
        context=_planning_context(),
    )

    assert agent.generate_strategy(request) == expected
    assert compiler_calls[0][0] == request
    assert "QoS" in compiler_calls[0][1]


def test_osa_stage_retries_invocation_failures_and_reraises_the_original_exception(monkeypatch) -> None:
    import control_runtime.agents.planning.agent as planning_agent
    from control_runtime.agents.planning.agent import OptimizationStrategyAgent
    from control_runtime.agents.planning.contracts import PlanningRequest

    class BareOptimizationStrategyAgent(OptimizationStrategyAgent):
        pass

    agent = object.__new__(BareOptimizationStrategyAgent)
    agent.logger = SimpleNamespace()
    agent.last_failure_debug = {}
    attempts = []
    terminal_error = RuntimeError("advisor transport unavailable")
    request = PlanningRequest(
        scope=_execution_scope(),
        grounding_decision=_grounding_decision(),
        context=_planning_context(),
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
    from control_runtime.agents.planning.contracts import PlanningRequest

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
        scope=_execution_scope(),
        grounding_decision=_grounding_decision(),
        context=_planning_context(),
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
    from control_runtime.agents.planning.contracts import PlanningRequest

    class BareOptimizationStrategyAgent(OptimizationStrategyAgent):
        pass

    agent = object.__new__(BareOptimizationStrategyAgent)
    agent.logger = SimpleNamespace(exception=lambda *_args, **_kwargs: None)
    agent.compiler = SimpleNamespace(build_planning_evidence=lambda _request: {})
    terminal_error = RuntimeError("strategy advisor unavailable")
    trace_writes = []
    request = PlanningRequest(
        scope=_execution_scope(),
        grounding_decision=_grounding_decision(),
        context=_planning_context(),
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
    assert trace_writes[0]["error"] == "strategy advisor unavailable"
    assert trace_writes[0]["error"] == "strategy advisor unavailable"
    assert not hasattr(agent, "_pending_invoke_messages")


def test_osa_stage_retries_contract_validation_and_keeps_terminal_message_stable(monkeypatch) -> None:
    import control_runtime.agents.planning.agent as planning_agent
    from control_runtime.agents.planning.agent import OptimizationStrategyAgent
    from control_runtime.agents.planning.response_models import OsaAdvisorOutput
    from control_runtime.agents.planning.contracts import PlanningRequest

    class BareOptimizationStrategyAgent(OptimizationStrategyAgent):
        pass

    agent = object.__new__(BareOptimizationStrategyAgent)
    agent.logger = SimpleNamespace()
    agent.last_failure_debug = {}
    invocations = []
    validations = []
    trace_writes = []
    request = PlanningRequest(
        scope=_execution_scope(),
        grounding_decision=_grounding_decision(),
        context=_planning_context(),
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
        return OsaAdvisorOutput(
            planning_status="needs_upstream_reground",
            rationale="optimizer preview is missing",
            missing_evidence=["optimizer preview"],
            blocked_targets=[],
            upstream_requests=[],
            planner_conflicts=[],
            open_questions=[],
            sm_policies=[],
            am_policy=None,
            ursp_policies=[],
            partial_policies=[],
        ), {}, {}

    monkeypatch.setattr(agent, "_invoke_strategy_advisor", invoke_advisor)
    monkeypatch.setattr(
        agent,
        "_write_advisor_trace",
        lambda **kwargs: trace_writes.append(kwargs),
    )
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
    assert len(trace_writes) == 3
    assert all("contract validation failed" in item["error"] for item in trace_writes)


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
        scope=_execution_scope(),
        trace_metadata={"scenario_id": "offline-case"},
    )

    assert result == decision
    assert captured["request"]["request"].scope == _execution_scope()
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
    monkeypatch.setattr(
        agent,
        "build_runtime_context",
        lambda **kwargs: SimpleNamespace(scope=kwargs["scope"]),
    )
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
                scope=_execution_scope(),
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

    orchestrator = object.__new__(BareMainControlOrchestrator)
    orchestrator.main_agent = MainAgent()
    orchestrator.memory_manager = object()
    orchestrator.max_rounds = 2
    orchestrator._token_counter = object()
    orchestrator._inject_token_context = lambda: None
    orchestrator._remember = lambda *_args, **_kwargs: None
    orchestrator.cr_tool = object()
    orchestrator.pd_agent = object()
    orchestrator.ad_tool = object()
    plan_calls = []
    finalization = []
    first_failures = [{"phase": "dispatch", "policy_id": "sm-1"}]
    first_report = FeedbackReport(
        scope=_execution_scope(),
        execution_status="Failed",
        performance_metrics=_dispatch_metrics(
            [],
            scope=_execution_scope(),
            dispatch_attempts=0,
            failure_summaries=first_failures,
        ),
        violation_details="gateway rejected sm-1",
        failure_scope="qos",
        feedback_payload=_dispatch_failure_feedback(dispatch_attempts=0, failures=first_failures),
        dispatch_attempts=0,
    )
    first_reentry = ExecutionReentryRequest(
        round_index=1,
        target_bindings_at_risk=["flow-1"],
        policy_objects_at_risk=["sm-1"],
        failure_scope="qos",
        summary="gateway rejected sm-1",
    )
    second_report = FeedbackReport(
        scope=_execution_scope(2),
        execution_status="Success",
        performance_metrics=_dispatch_metrics([], scope=_execution_scope(2), dispatch_attempts=0),
        violation_details="none",
        failure_scope="none",
        feedback_payload=None,
        dispatch_attempts=0,
    )

    def make_artifacts(*, round_index, completed, report, diagnosis, reentry=None):
        return RoundExecutionArtifacts(
            completed=completed,
            report=report,
            diagnosis=diagnosis,
            qos_feedback={"execution_status": report.execution_status},
            mobility_feedback={},
            unified_plan=UnifiedControlPlan(
                scope=ExecutionScope(
                    session_id="session-1",
                    thread_id="session-1",
                    snapshot_id="snapshot-1",
                    round_index=round_index,
                    supi="imsi-001",
                ),
                global_intent=_main_intent(round_index=round_index),
                qos_proposal=None,
                mobility_proposal=None,
                domain_verdicts=[],
                mediator_decision=None,
                unified_constraints={"hard_constraints": [], "evidence": []},
                execution_order=[],
                approved_policies=[],
                blocked_domains=[],
                objective_breakdown={},
                control_churn_count=0,
                agent_contributions=[],
                handoff_records=[],
                open_questions=[],
            ),
            trace=ControlRoundTrace(
                scope=_execution_scope(round_index),
                round_index=round_index,
                global_intent=_main_intent(round_index=round_index),
                grounding_decision=_grounding_decision(round_index=round_index),
                policy_plan=_qos_plan(round_index=round_index),
                dispatch_report=report,
                diagnosis=diagnosis,
                execution_reentry=reentry,
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
        round_index = kwargs["round_index"]
        return (
            _main_intent(round_index=round_index),
            _grounding_decision(round_index=round_index),
            _qos_plan(round_index=round_index),
            None,
        )

    monkeypatch.setattr(orchestrator, "_plan_round", plan_round)
    monkeypatch.setattr(main_orchestrator, "start_control_session", lambda **_kwargs: ("session-1", "snapshot-1"))
    monkeypatch.setattr(
        main_orchestrator,
        "build_memory_context",
        lambda *_args, **_kwargs: MemoryContextView(
            query=None,
            selected_short_term=[],
            selected_long_term=[],
            prompt_text="",
        ),
    )
    monkeypatch.setattr(main_orchestrator, "build_feedback_context_from_snapshots", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(main_orchestrator, "log_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main_orchestrator,
        "execute_planned_round",
            lambda **kwargs: artifacts_by_round[kwargs["scope"].round_index],
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
    assert finalization[0]["state"].completed is True


def test_iea_public_analysis_records_rejected_output_before_later_invocation_failure(monkeypatch) -> None:
    """Every completed model response is traced once even when a later retry fails."""

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
        agent.analyze_intent("protect flow-1", context=_IEA_TEST_CONTEXT, scope=_execution_scope())

    assert raised.value is terminal_error
    assert raised.value.__cause__ is root_cause
    assert attempts == ["invoke", "invoke", "invoke"]
    assert [event["reason"] for event in retry_events] == ["validation_errors", "invocation_error"]
    assert len(retry_prompts) == 2
    assert agent.last_failure_debug["attempt_index"] == 1
    assert agent.last_failure_debug["advisor_validation_errors"] == ["advisor decision invalid"]
    assert trace_writes == [
        {
            "grounding_decision": None,
            "error": "IEA advisor decision validation failed: advisor decision invalid",
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
        agent.analyze_intent("protect flow-1", context=_IEA_TEST_CONTEXT, scope=_execution_scope())

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
        agent.analyze_intent("protect flow-1", context=_IEA_TEST_CONTEXT, scope=_execution_scope())

    assert str(raised.value) == expected_message
    assert raised.value.__cause__ is None
    assert invocations == ["invoke", "invoke", "invoke"]
    assert agent.last_failure_debug["advisor_validation_errors"] == advisor_errors
    assert agent.last_failure_debug["grounding_validation_errors"] == grounding_errors
    assert trace_writes[-1]["grounding_decision"] is None
    assert trace_writes[-1]["error"] == expected_message


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
        agent.analyze_intent("protect flow-1", context=_IEA_TEST_CONTEXT, scope=_execution_scope())

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
        agent.analyze_intent("protect flow-1", context=_IEA_TEST_CONTEXT, scope=_execution_scope())

    assert raised.value is terminal_error
    assert attempts == ["invoke"]
    assert retry_events == []
    assert retry_prompts == []
    assert agent.last_failure_debug == {}
    assert not hasattr(agent, "_pending_invoke_messages")


def test_execute_planned_round_preserves_wrapper_loader_monkeypatch_seam_and_artifact_contract(monkeypatch) -> None:
    import control_runtime.orchestrators.round_execution as round_execution
    from control_runtime.agents.dispatch.contracts import FeedbackReport
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
        scope=_execution_scope(3),
        execution_status="Success",
        performance_metrics=_dispatch_metrics(
            [{"status": "success", "policy_id": "sm-1", "policy_type": "SmPolicyDecision"}],
            scope=_execution_scope(3),
            dispatch_attempts=1,
        ),
        violation_details="none",
        failure_scope="none",
        feedback_payload=None,
        dispatch_attempts=1,
    )
    conflict_tool = _RecordingConflictTool(_conflict_result())

    artifacts = round_execution.execute_planned_round(
        scope=_execution_scope(3),
        policy_plan=_qos_plan(round_index=3),
        global_intent=_main_intent(round_index=3),
        grounding_decision=_grounding_decision(round_index=3),
        cr_tool=conflict_tool,
        pd_agent=_RecordingDispatchAgent(report),
        ad_tool=_RecordingDiagnosisTool(_diagnosis_result()),
    )

    assert isinstance(artifacts, RoundExecutionArtifacts)
    assert loader_calls == [("imsi-001", "snapshot-1")]
    assert conflict_tool.requests[0].upstream_context.ue_context == ue_context
    assert artifacts.trace.round_index == 3
    assert artifacts.trace.policy_plan == _qos_plan(round_index=3)
    assert artifacts.trace.mediator_decision.model_dump(mode="json") == {"status": "approved", "reason_summary": "no conflicts", "affected_domains": [], "revision_requests": [], "unified_constraints": {"hard_constraints": [], "evidence": []}}


def test_execute_planned_round_keeps_mediator_blocked_artifacts_without_dispatch(monkeypatch) -> None:
    import control_runtime.orchestrators.round_execution as round_execution

    monkeypatch.setattr(
        round_execution,
        "get_snapshot_data_by_id",
        lambda _: {"snapshot_id": "snapshot-1", "apps": [], "slices": [], "nodes": []},
    )
    monkeypatch.setattr(round_execution, "get_ue_context_by_supi", lambda *_args, **_kwargs: {})
    conflict_result = _conflict_result(
        mediator_status="revise",
        reason_summary="binding conflict requires revision",
    )
    diagnosis_tool = _RecordingDiagnosisTool(
        _diagnosis_result(
            root_cause_category="mediation_blocked",
            reason_summary="revise plan",
        )
    )

    artifacts = round_execution.execute_planned_round(
        scope=_execution_scope(2),
        policy_plan=_qos_plan(round_index=2),
        global_intent=_main_intent(round_index=2),
        grounding_decision=_grounding_decision(round_index=2),
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
    assert artifacts.trace.mediator_decision.status == "revise"
    assert artifacts.trace.domain_verdicts[-1].status.value == "needs_revision"
    assert artifacts.trace.dispatch_report is None
    assert diagnosis_tool.requests[0].execution_feedback.qos == artifacts.qos_feedback


def test_execute_planned_round_keeps_partial_plan_dispatch_and_artifact_semantics(monkeypatch) -> None:
    import control_runtime.orchestrators.round_execution as round_execution
    from control_runtime.agents.dispatch.contracts import FeedbackReport

    monkeypatch.setattr(
        round_execution,
        "get_snapshot_data_by_id",
        lambda _: {"snapshot_id": "snapshot-1", "apps": [], "slices": [], "nodes": []},
    )
    monkeypatch.setattr(round_execution, "get_ue_context_by_supi", lambda *_args, **_kwargs: {})
    plan = _qos_plan(planning_status="partial_plan", round_index=4)
    report = FeedbackReport(
        scope=_execution_scope(4),
        execution_status="Success",
        performance_metrics=_dispatch_metrics(
            [{"status": "success", "policy_id": "sm-1", "policy_type": "SmPolicyDecision"}],
            scope=_execution_scope(4),
            dispatch_attempts=1,
        ),
        violation_details="none",
        failure_scope="none",
        feedback_payload=None,
        dispatch_attempts=1,
    )

    artifacts = round_execution.execute_planned_round(
        scope=_execution_scope(4),
        policy_plan=plan,
        global_intent=_main_intent(round_index=4),
        grounding_decision=_grounding_decision(round_index=4),
        cr_tool=_RecordingConflictTool(_conflict_result()),
        pd_agent=_RecordingDispatchAgent(report),
        ad_tool=_RecordingDiagnosisTool(_diagnosis_result(reason_summary="dispatch completed")),
    )

    assert artifacts.completed is True
    assert artifacts.report is not report
    assert artifacts.report.intent_status == "partially_satisfied"
    assert artifacts.report.feedback_payload.model_dump(mode="json") == {
        "feedback_kind": "partial_intent",
        "intent_status": "partially_satisfied",
        "intent_blockers": ["flow-2"],
        "upstream_requests": ["reground flow-2"],
        "planning_status": "partial_plan",
    }
    assert artifacts.diagnosis["root_cause_category"] == "intent_partially_satisfied"
    assert artifacts.trace.dispatch_report.intent_status == "partially_satisfied"
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
            raw_result=_main_control_decision_result(),
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
        scope=_execution_scope(),
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
    assert trace_writes == [
        {"structured_response": result.model_dump(mode="json")}
    ]
    assert not hasattr(agent, "_pending_invoke_messages")


def test_main_public_analysis_traces_each_rejected_parse_attempt(monkeypatch) -> None:
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
            scope=_execution_scope(),
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
        {"structured_response": None, "error": expected_message},
        {"structured_response": None, "error": expected_message},
        {"structured_response": None, "error": expected_message},
    ]
    assert not hasattr(agent, "_pending_invoke_messages")


def test_main_public_analysis_traces_each_semantically_rejected_output(monkeypatch) -> None:
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

    def return_intent(_result, **_kwargs):
        return intent

    def build_retry_prompt(**kwargs):
        retry_prompts.append(kwargs)
        return f"retry prompt {len(retry_prompts)}"

    def capture_event(_logger, event, **kwargs):
        events.append((event, kwargs))

    monkeypatch.setattr(agent, "_invoke_global_intent_result", invoke_global_intent)
    monkeypatch.setattr(main_agent, "validate_global_intent_result", return_intent)
    monkeypatch.setattr(main_agent, "validate_main_global_intent", lambda *_args, **_kwargs: semantic_errors)
    monkeypatch.setattr(main_agent, "build_main_validation_retry_prompt", build_retry_prompt)
    monkeypatch.setattr(main_agent, "log_event", capture_event)
    agent._pending_invoke_messages = [{"role": "user", "content": "stale prompt"}]

    with pytest.raises(RuntimeError) as raised:
        agent.analyze_global_intent(
            user_input="protect QoS",
            context="{}",
            scope=_execution_scope(),
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
    expected_trace = {
        "structured_response": None,
        "error": expected_message,
    }
    assert trace_writes == [expected_trace, expected_trace, expected_trace]
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
            scope=_execution_scope(),
        )

    assert raised.value is terminal_error
    assert invocations == ["invoke"]
    assert retry_prompts == []
    assert [kwargs["attempt"] for event, kwargs in events if event == "main_control_attempt_start"] == [1]
    assert not [event for event, _kwargs in events if event == "main_control_validation_failed"]
    assert trace_writes == [
        {"structured_response": None, "error": str(terminal_error)}
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
        "scope": _execution_scope(7).model_dump(mode="json"),
        "round_index": 7,
        "mediator_decision": {
            "status": "approved",
            "reason_summary": "latency constraints approved",
            "affected_domains": ["qos"],
            "revision_requests": [],
            "unified_constraints": {
                "hard_constraints": ["latency <= 20 ms"],
                "evidence": ["telemetry.latency"],
            },
        },
        "agent_elapsed_ms": {"planning": 12.5},
        "planning_blocker": {
            "round_index": 7,
            "upstream_requests": ["reground flow-1"],
            "summary": "grounding required",
        },
        "execution_reentry": {
            "round_index": 7,
            "reason_by_domain": {"qos": "retry binding"},
            "summary": "retry flow-1",
        },
        "single_agent_detail": {"attempts": [{"stage": "single_control"}]},
    }

    trace = ControlRoundTrace.from_payload(payload)
    payload["mediator_decision"]["unified_constraints"]["hard_constraints"][0] = "mutated-input"
    payload["agent_elapsed_ms"]["planning"] = 999.0
    payload["planning_blocker"]["upstream_requests"][0] = "mutated-input"
    payload["execution_reentry"]["reason_by_domain"]["qos"] = "mutated-input"
    payload["single_agent_detail"]["attempts"][0]["stage"] = "mutated-input"

    round_trip = trace.to_payload()
    assert round_trip["mediator_decision"]["unified_constraints"]["hard_constraints"] == [
        "latency <= 20 ms"
    ]
    assert round_trip["agent_elapsed_ms"] == {"planning": 12.5}
    assert round_trip["planning_blocker"]["upstream_requests"] == ["reground flow-1"]
    assert round_trip["execution_reentry"]["reason_by_domain"] == {"qos": "retry binding"}
    assert round_trip["single_agent_detail"] == {"attempts": [{"stage": "single_control"}]}

    round_trip["mediator_decision"]["unified_constraints"]["hard_constraints"][0] = "mutated-output"
    round_trip["agent_elapsed_ms"]["planning"] = 0.0
    round_trip["planning_blocker"]["upstream_requests"][0] = "mutated-output"
    round_trip["execution_reentry"]["reason_by_domain"]["qos"] = "mutated-output"
    round_trip["single_agent_detail"]["attempts"][0]["stage"] = "mutated-output"

    assert trace.to_payload()["mediator_decision"]["unified_constraints"]["hard_constraints"] == [
        "latency <= 20 ms"
    ]
    assert trace.to_payload()["agent_elapsed_ms"]["planning"] == 12.5
    assert trace.to_payload()["planning_blocker"]["upstream_requests"] == ["reground flow-1"]
    assert trace.to_payload()["execution_reentry"]["reason_by_domain"] == {"qos": "retry binding"}
    assert trace.to_payload()["single_agent_detail"]["attempts"][0]["stage"] == "single_control"


def test_control_round_trace_rejects_malformed_payloads_instead_of_normalizing_them() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        ControlRoundTrace.from_payload(
            {"scope": _execution_scope().model_dump(mode="json"), "round_index": 0}
        )
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ControlRoundTrace.from_payload(
            {
                "scope": _execution_scope().model_dump(mode="json"),
                "round_index": 1,
                "legacy_trace": {},
            }
        )
    with pytest.raises(ValueError, match="Input should be a valid dictionary"):
        ControlRoundTrace.from_payload(
            {
                "scope": _execution_scope().model_dump(mode="json"),
                "round_index": 1,
                "domain_verdicts": ["invalid"],
            }
        )


def test_control_round_trace_rejects_cross_session_artifacts() -> None:
    foreign_intent = _main_intent().model_copy(
        update={
            "scope": _execution_scope().model_copy(update={"session_id": "session-2"})
        }
    )

    with pytest.raises(ValueError, match="session_id"):
        ControlRoundTrace(
            scope=_execution_scope(),
            round_index=1,
            global_intent=foreign_intent,
        )


def test_append_round_trace_is_typed_only_and_retains_complete_recent_trace_payloads() -> None:
    state = OrchestratorLoopState()

    with pytest.raises(TypeError, match="trace_payload"):
        append_round_trace(state, trace_payload={"round_index": 1})

    for round_index in range(1, 5):
        mediator_decision = _conflict_result(
            reason_summary=f"decision-{round_index}"
        ).to_mediator_decision()
        append_round_trace(
            state,
            trace=ControlRoundTrace(
                scope=_execution_scope(round_index),
                round_index=round_index,
                diagnosis={
                    "root_cause_category": f"cause-{round_index}",
                    "reason_summary": f"reason-{round_index}",
                },
                mediator_decision=mediator_decision,
                agent_elapsed_ms={"planning": float(round_index)},
            ),
            feedback_added=f"[Round Feedback]\\nround_index: {round_index}",
        )

    expected_keys = {
        "scope",
        "round_index",
        "global_intent",
        "grounding_decision",
        "policy_plan",
        "domain_verdicts",
        "dispatch_report",
        "orchestration_failure",
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
    assert [snapshot.trace.round_index for snapshot in state.rounds] == [1, 2, 3, 4]
    assert all(set(trace) == expected_keys for trace in state.round_traces)
    assert [trace["agent_elapsed_ms"] for trace in state.round_traces] == [
        {"planning": 1.0},
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
