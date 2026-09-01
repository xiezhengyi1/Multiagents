from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.contracts import ExecutionScope
from control_runtime.contracts import standard_objective_profile
from control_runtime.agents.main import contracts as main_contracts
from control_runtime.agents.main.compiler import compile_global_intent, validate_global_intent
from control_runtime.agents.main.contracts import (
    ControlDomain,
    MainControlDecision,
    MainControlSemantics,
    MainRetryScope,
)


def _scope(*, round_index: int = 1) -> ExecutionScope:
    return ExecutionScope(
        session_id="session-1",
        thread_id="session-1",
        snapshot_id="snapshot-1",
        round_index=round_index,
    )


def _decision(
    *,
    next_agent: str = "intent_encoding",
    retry_scope: MainRetryScope | None = None,
    target_supi: str = "imsi-208930000000001",
) -> MainControlDecision:
    return MainControlDecision(
        supi="imsi-208930000000001",
        next_agent=next_agent,
        requested_domains=[ControlDomain.QOS],
        serving_slice_change_requested=False,
        domain_evidence={ControlDomain.QOS: ["flow catalog"]},
        control_semantics={
            "mode": "single_step",
            "current_stage": 1,
            "stages": [
                {
                    "stage_index": 1,
                    "name": "protect flow",
                    "trigger": "initial",
                    "summary": "protect the grounded flow",
                    "targets": [
                        {
                            "semantic_name": "flow-1",
                            "target_type": "flow",
                            "goal": "protect",
                            "metric_focus": "latency",
                            "note": "user-selected flow",
                            "supi": target_supi,
                        }
                    ],
                }
            ],
            "notes": [],
        },
        objective_profile=standard_objective_profile("balanced"),
        investigation_targets=[],
        uncertainty_flags=[],
        retry_scope=retry_scope,
        required_evidence=["flow catalog"],
        forbidden_assumptions=[],
        intent_encoding_guidance=(
            "Re-ground the selected flow."
            if next_agent == "intent_encoding" and retry_scope is not None
            else ""
        ),
    )


def test_main_model_uses_one_strict_response_contract() -> None:
    assert not hasattr(main_contracts, "MainControlDecisionWire")
    required = set(MainControlDecision.model_json_schema()["required"])
    assert required == set(MainControlDecision.model_fields)
    objective_schema = MainControlDecision.model_json_schema()["$defs"]["ObjectiveProfile"]
    assert set(objective_schema["required"]) == set(main_contracts.ObjectiveProfile.model_fields)

    payload = _decision().model_dump(mode="json")
    payload["compiler_owned_field"] = "must be rejected"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MainControlDecision.model_validate(payload)


def test_main_control_semantics_rejects_invalid_stage_selection_instead_of_clamping() -> None:
    with pytest.raises(ValidationError, match="existing stage_index"):
        MainControlSemantics(
            mode="staged_priority",
            current_stage=9,
            stages=[
                {
                    "stage_index": 1,
                    "name": "first",
                    "trigger": "initial",
                    "summary": "first stage",
                    "targets": [],
                }
            ],
            notes=[],
        )


def test_main_compiler_rejects_missing_retry_scope() -> None:
    with pytest.raises(ValueError, match="requires retry_scope"):
        compile_global_intent(
            _decision(next_agent="optimization_strategy", retry_scope=None),
            user_input="Protect flow-1.",
            scope=_scope(round_index=2),
        )


def test_main_validation_reports_missing_target_supi_without_repairing_it() -> None:
    intent = compile_global_intent(
        _decision(target_supi=""),
        user_input="Protect flow-1 for imsi-208930000000001.",
        scope=_scope(),
    )

    errors = validate_global_intent(intent)

    assert any("single-SUPI semantic target must carry target.supi" in error for error in errors)
    assert intent.control_semantics.stages[0].targets[0].supi == ""
    assert intent.scope.supi == "imsi-208930000000001"


def test_global_control_intent_rejects_flat_scope_fields() -> None:
    payload = compile_global_intent(
        _decision(),
        user_input="Protect flow-1.",
        scope=_scope(),
    ).model_dump(mode="json")
    payload["session_id"] = payload["scope"]["session_id"]

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        main_contracts.GlobalControlIntent.model_validate(payload)
