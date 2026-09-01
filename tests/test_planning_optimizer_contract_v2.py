from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.contracts import ExecutionScope
from control_runtime.agents.main.contracts import GlobalControlIntent
from control_runtime.agents.planning.contracts import PlanningContext, SharedControlContext
from control_runtime.agents.planning.response_models import (
    AmPolicySpec,
    OsaAdvisorOutput,
    PartialPolicySpec,
    RouteSelectionParameterSetSpec,
    SmPolicySpec,
    SnssaiSpec,
    TrafficDescriptorSpec,
    UrspPolicySpec,
)
from control_runtime.contracts import ControlDomain, ObjectiveProfile, standard_objective_profile
from control_runtime.integrations.optimizer.contracts import (
    JointOptimizationRequest,
    JointOptimizationResult,
    OptimizationProblemConfig,
    OptimizerInput,
)


def _scope() -> ExecutionScope:
    return ExecutionScope(
        session_id="session-1",
        thread_id="session-1",
        snapshot_id="snapshot-1",
        round_index=1,
        supi="imsi-1",
    )


def _main_intent() -> GlobalControlIntent:
    return GlobalControlIntent(
        scope=_scope(),
        raw_input="optimize flow-1",
        next_agent="optimization_strategy",
        requested_domains=[ControlDomain.QOS],
        objective_profile=standard_objective_profile("balanced"),
    )


def test_planning_and_optimizer_wire_models_require_every_declared_field() -> None:
    models = (
        ObjectiveProfile,
        SnssaiSpec,
        TrafficDescriptorSpec,
        RouteSelectionParameterSetSpec,
        SmPolicySpec,
        AmPolicySpec,
        UrspPolicySpec,
        PartialPolicySpec,
        OsaAdvisorOutput,
        OptimizationProblemConfig,
        OptimizerInput,
        JointOptimizationRequest,
        JointOptimizationResult,
    )

    for model in models:
        schema = model.model_json_schema()
        assert set(schema["required"]) == set(model.model_fields), model.__name__


def test_objective_profile_rejects_missing_costs_instead_of_filling_defaults() -> None:
    with pytest.raises(ValidationError, match="sla_violation_cost"):
        ObjectiveProfile(profile_name="balanced")


def test_snssai_model_accepts_only_the_declared_object_shape() -> None:
    with pytest.raises(ValidationError):
        SnssaiSpec.model_validate("01000001")
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        SnssaiSpec(sst=1, sd="ABCDEF")


def test_osa_output_requires_explicit_nullable_and_empty_collection_fields() -> None:
    with pytest.raises(ValidationError, match="missing_evidence"):
        OsaAdvisorOutput(
            planning_status="executable_plan",
            rationale="optimizer approved the grounded flow",
        )


def test_planning_context_rejects_flat_identity_fields() -> None:
    payload = PlanningContext(
        shared_context=SharedControlContext(main_intent=_main_intent())
    ).model_dump(mode="json")
    payload["session_id"] = "session-1"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PlanningContext.model_validate(payload)


def test_optimizer_input_rejects_flat_identity_fields() -> None:
    payload = OptimizerInput(
        scope=_scope(),
        target_supis=["imsi-1"],
        requested_domains=[ControlDomain.QOS],
        grounding_decision={"flows": []},
        objective_profile=standard_objective_profile("balanced"),
        raw_intent="optimize flow-1",
    ).model_dump(mode="json")
    payload["snapshot_id"] = "snapshot-1"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        OptimizerInput.model_validate(payload)


def test_partial_policy_rejects_aliases_and_extra_blocker_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PartialPolicySpec.model_validate(
            {
                "app_id": "app-1",
                "flow_id": "flow-1",
                "target_type": "flow",
                "policy_type": "SmPolicyDecision",
                "resource_keys": [],
                "policy_details": {},
                "blocked_reason": "missing entitlement",
            }
        )
