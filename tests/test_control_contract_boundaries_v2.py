from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.contracts import ExecutionScope, MemoryContextView, MemoryQuery, MemoryRecord
from control_runtime.agents.grounding.contracts import MobilityIntent
from control_runtime.agents.planning.contracts import (
    OptimizerPlanEvidence,
    PlanningRationale,
    PolicyPlanDraft,
)
from control_runtime.agents.planning.response_models import OsaAdvisorOutput
from control_runtime.diagnostics.diagnosis.contracts import AssuranceDiagnosisRequest
from control_runtime.diagnostics.mediation.contracts import ConflictResolutionRequest
from control_runtime.orchestrators.contracts import ControlRoundResult, UnifiedControlPlan


def _scope() -> ExecutionScope:
    return ExecutionScope(
        session_id="session-1",
        thread_id="thread-1",
        snapshot_id="snapshot-1",
        round_index=1,
        supi="imsi-001",
    )


def _reground_plan() -> PolicyPlanDraft:
    return PolicyPlanDraft(
        scope=_scope(),
        planning_status="needs_upstream_reground",
        optimizer_result=OptimizerPlanEvidence(
            objective_breakdown={},
            cross_domain_verdicts=[],
        ),
        planning_rationale=PlanningRationale(
            selected_strategy_profile="balanced",
            decisive_evidence=[],
            active_constraints=[],
            explanation="flow evidence is missing",
            rejected_alternatives=[],
            main_constraints=[],
            iea_grounding_basis=[],
            osa_decision_basis=["missing flow evidence"],
            unresolved_gaps=["flow evidence"],
        ),
        all_policies=[],
        partial_policies=[],
        missing_evidence=["flow evidence"],
        blocked_targets=[],
        upstream_requests=["reground flow binding"],
        planner_conflicts=[],
        open_questions=[],
    )


def test_memory_query_requires_explicit_retrieval_limit() -> None:
    with pytest.raises(ValidationError, match="top_k"):
        MemoryQuery(scope=_scope(), text="policy evidence")


def test_memory_context_cannot_carry_unattributed_prompt_text() -> None:
    with pytest.raises(ValidationError, match="without a query"):
        MemoryContextView(
            query=None,
            selected_short_term=[],
            selected_long_term=[],
            prompt_text="remembered claim without provenance",
        )

    record = MemoryRecord(
        memory_id="memory-1",
        scope=_scope(),
        kind="event",
        source_agent="main",
        role="MAIN",
        content="grounded fact",
    )
    with pytest.raises(ValidationError, match="prompt_text presence"):
        MemoryContextView(
            query=MemoryQuery(scope=_scope(), text="policy evidence", top_k=3),
            selected_short_term=[record],
            selected_long_term=[],
            prompt_text="",
        )


def test_mobility_intent_rejects_legacy_alias_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MobilityIntent.model_validate(
            {
                "association_id": "association-1",
                "rfsp": 1,
                "allowed_snssais": ["01000001"],
                "target_snssais": ["01000002"],
                "target_snssai": "01000002",
                "access_type": "3GPP_ACCESS",
                "service_area_restriction": None,
            }
        )


def test_osa_output_requires_open_questions_at_the_wire_boundary() -> None:
    with pytest.raises(ValidationError, match="open_questions"):
        OsaAdvisorOutput.model_validate(
            {
                "planning_status": "needs_upstream_reground",
                "rationale": "flow evidence is missing",
                "missing_evidence": ["flow evidence"],
                "blocked_targets": [],
                "upstream_requests": ["reground flow binding"],
                "planner_conflicts": [],
                "sm_policies": [],
                "am_policy": None,
                "ursp_policies": [],
                "partial_policies": [],
            }
        )


def test_policy_plan_rejects_flat_scope_identity() -> None:
    payload = _reground_plan().model_dump(mode="json")
    payload["session_id"] = "session-1"
    payload["snapshot_id"] = "snapshot-1"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PolicyPlanDraft.model_validate(payload)


def test_mediation_request_rejects_flat_scope_identity() -> None:
    payload = ConflictResolutionRequest(
        scope=_scope(),
        candidate_policies=[],
        resource_view={
            "valid_slice_keys": [],
            "valid_snssai_keys": [],
            "slice_capacity_by_snssai": {},
            "flow_allocations": {},
            "exclusive_resource_keys": [],
        },
        conflict_scope={"domains": []},
        upstream_context={
            "ue_context": {},
            "planner_cross_domain_verdicts": [],
        },
    ).model_dump(mode="json")
    payload["session_id"] = "session-1"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ConflictResolutionRequest.model_validate(payload)


def test_diagnosis_request_rejects_flat_scope_identity() -> None:
    payload = AssuranceDiagnosisRequest(
        scope=_scope(),
        execution_feedback={"pda": {}, "qos": {}, "mobility": {}},
        dispatch_receipts=[],
        assurance_verdicts=[],
        telemetry_snapshot={},
        upstream_context={
            "global_intent": {},
            "grounding_decision": {},
            "conflict_result": {},
        },
    ).model_dump(mode="json")
    payload["snapshot_id"] = "snapshot-1"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AssuranceDiagnosisRequest.model_validate(payload)


def test_unified_control_plan_requires_every_aggregate_field() -> None:
    with pytest.raises(ValidationError, match="unified_constraints"):
        UnifiedControlPlan(scope=_scope(), global_intent=None)


def test_control_round_result_has_no_flat_identity_or_implicit_fields() -> None:
    with pytest.raises(ValidationError, match="qos_feedback"):
        ControlRoundResult(
            scope=_scope(),
            completed=False,
            global_intent={},
            unified_plan={},
        )

    result = ControlRoundResult(
        scope=_scope(),
        completed=False,
        global_intent={},
        unified_plan={},
        qos_feedback={},
        mobility_feedback={},
        diagnosis={},
        negotiation_request={},
        planning_blocker={},
        execution_reentry={},
        round_count=1,
        retry_count=0,
        round_traces=[],
        agent_elapsed_ms={},
        single_agent_details=[],
    )

    assert not hasattr(result, "session_id")
    assert not hasattr(result, "snapshot_id")
