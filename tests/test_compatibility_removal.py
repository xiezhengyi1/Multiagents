from __future__ import annotations

import inspect
import os
import re
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from shared.contracts import ExecutionScope

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

from control_runtime.agents.main.contracts import ObjectiveProfile
from control_runtime.contracts import ControlDomain, ObjectiveProfile as NeutralObjectiveProfile
from control_runtime.agents.grounding.contracts import IntentEncodingDirectives
from control_runtime.agents.planning.response_models import OsaAdvisorOutput
from control_runtime import execution as execution_package
from control_runtime.diagnostics.mediation.contracts import ConflictResolutionResult
from control_runtime.integrations.optimizer.contracts import (
    COUPLING_CONSTRAINTS,
    MOBILITY_CONSTRAINTS,
    MOBILITY_DECISION_VARIABLES,
    SESSION_CONSTRAINTS,
    SESSION_DECISION_VARIABLES,
    JointOptimizationRequest,
    OptimizationProblemConfig,
    OptimizationTemplate,
    OptimizerInput,
)
from control_runtime.integrations.optimizer import joint_control, request_factory
from control_runtime.integrations.optimizer.objective_weights import build_qos_optimizer_weights
from control_runtime.integrations.scenario.network_graph import NetworkGraph
from control_runtime.integrations.scenario import network_status
from control_runtime.integrations.storage.session_store import (
    create_session_context,
    update_session_context,
)
from database.models import SessionContext
from shared.memory import MemoryManager


REMOVED_ARCHITECTURE_FILES = (
    "control_runtime/domain/__init__.py",
    "control_runtime/domain/collaboration.py",
    "control_runtime/domain/control_loop.py",
    "control_runtime/domain/control_plane.py",
    "control_runtime/domain/intent_encoding.py",
    "control_runtime/domain/policy_compiler.py",
    "control_runtime/domain/policy_guard.py",
    "control_runtime/domain/policy_plan.py",
    "control_runtime/domain/runtime_contracts.py",
    "control_runtime/context/control_loop.py",
    "control_runtime/context/intent_context.py",
    "control_runtime/context/intent_encoding_context.py",
    "control_runtime/context/planning_context.py",
    "control_runtime/context/retry_constraints.py",
    "control_runtime/context/evidence/formatter.py",
    "control_runtime/context/projectors/registry.py",
    "control_runtime/context/prompts/builders/base.py",
    "control_runtime/execution/policy_guard.py",
    "control_runtime/execution/round_artifacts.py",
    "control_runtime/execution/round_assurance.py",
    "control_runtime/execution/round_dispatch.py",
    "control_runtime/execution/round_mediation.py",
)


def _objective_profile(**updates: float | str) -> ObjectiveProfile:
    payload: dict[str, float | str] = {
        "profile_name": "balanced",
        "sla_violation_cost": 1.0,
        "mobility_risk_cost": 0.8,
        "control_churn_cost": 0.5,
        "resource_pressure_cost": 0.7,
        "fairness_cost": 0.3,
    }
    payload.update(updates)
    return ObjectiveProfile.model_validate(payload)


def _problem_config(**updates: object) -> OptimizationProblemConfig:
    payload: dict[str, object] = {
        "template": OptimizationTemplate.JOINT_BALANCED,
        "solver_mode": "incremental",
        "active_objectives": [
            "sla_violation_cost",
            "mobility_risk_cost",
            "control_churn_cost",
            "resource_pressure_cost",
            "fairness_cost",
        ],
        "active_constraints": [
            *SESSION_CONSTRAINTS,
            *MOBILITY_CONSTRAINTS,
            *COUPLING_CONSTRAINTS,
        ],
        "decision_variables": [
            *SESSION_DECISION_VARIABLES,
            *MOBILITY_DECISION_VARIABLES,
        ],
        "qos_relaxation_ratio": 0.2,
        "slice_kpi_source": "qos",
        "qos_feasibility_mode": "hard",
        "enable_sla_constraints": True,
    }
    payload.update(updates)
    return OptimizationProblemConfig.model_validate(payload)


def test_misplaced_architecture_sources_and_imports_are_removed() -> None:
    assert all(not (SRC / path).exists() for path in REMOVED_ARCHITECTURE_FILES)
    legacy_import = re.compile(
        r"^\s*(?:from\s+(?:control_runtime\.)?domain\b|from\s+\.+domain\b|import\s+control_runtime\.domain\b)",
        re.MULTILINE,
    )
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in (SRC / "control_runtime").rglob("*.py")
        if legacy_import.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_execution_package_exports_only_public_execution_entrypoints() -> None:
    assert execution_package.__all__ == [
        "AssuranceEvaluator",
        "ExecutionController",
        "ExecutionOutcome",
    ]


def test_agent_wire_contracts_reject_invalid_shapes_instead_of_normalizing_them() -> None:
    with pytest.raises(TypeError, match="domain_evidence"):
        IntentEncodingDirectives.from_context(
            '{"main_intent":{"requested_domains":["qos"]}}'
        )

    with pytest.raises(ValidationError):
        OsaAdvisorOutput.model_validate({"sm_policies": None})


def test_memory_manager_does_not_accept_ignored_long_term_file() -> None:
    assert "long_term_file" not in inspect.signature(MemoryManager).parameters


def test_conflict_result_exposes_only_canonical_mediator_status() -> None:
    result = ConflictResolutionResult()

    assert result.mediator_status == "approved"
    assert not hasattr(result, "status")


def test_network_status_has_no_legacy_function_alias() -> None:
    assert not hasattr(network_status, "get_network_status")


def test_session_context_uses_only_canonical_stage_field() -> None:
    assert not hasattr(SessionContext, "current_step")
    assert "current_step" not in inspect.signature(create_session_context).parameters
    assert "current_step" not in inspect.signature(update_session_context).parameters
    assert "current_stage" in inspect.signature(create_session_context).parameters
    assert "current_stage" in inspect.signature(update_session_context).parameters


def test_database_bootstrap_migrates_then_drops_current_step() -> None:
    source = (ROOT / "database" / "init_db.py").read_text(encoding="utf-8")

    assert source.index("SET current_stage = current_step") < source.index(
        "DROP COLUMN current_step"
    )


def test_optimizer_weight_mapping_lives_at_integration_boundary() -> None:
    profile = _objective_profile(
        sla_violation_cost=1.0,
        control_churn_cost=0.5,
        resource_pressure_cost=0.7,
        fairness_cost=0.3,
    )

    weights = build_qos_optimizer_weights(profile)

    assert not hasattr(profile, "to_legacy_qos_weights")
    assert weights.solver_arguments() == (70.0, 25.0, 860.0, 225.0)
    assert weights.to_payload() == {
        "resource_pressure": 70.0,
        "control_churn": 25.0,
        "sla_and_fairness": 860.0,
        "fairness_and_sla_tail": 225.0,
    }


def test_optimizer_request_boundary_uses_neutral_contracts() -> None:
    assert ObjectiveProfile is NeutralObjectiveProfile

    integration_import = re.compile(
        r"^\s*(?:from\s+(?:control_runtime\.)?agents\b|from\s+\.+agents\b|import\s+control_runtime\.agents\b)",
        re.MULTILINE,
    )
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in (SRC / "control_runtime" / "integrations").rglob("*.py")
        if integration_import.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []
    assert "PlanningRequest" not in inspect.getsource(request_factory)


def test_optimizer_request_factory_accepts_explicit_integration_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        request_factory,
        "get_snapshot_data_by_id",
        lambda snapshot_id: {"apps": [], "slices": [], "nodes": []},
    )
    monkeypatch.setattr(
        request_factory,
        "get_ue_context_by_supi",
        lambda supi, *, snapshot_id: {},
    )

    request = request_factory.build_optimizer_request(
        OptimizerInput(
            scope=ExecutionScope(
                session_id="session-1",
                thread_id="session-1",
                snapshot_id="snapshot-1",
                round_index=1,
                supi="imsi-1",
            ),
            target_supis=["imsi-1"],
            requested_domains=[ControlDomain.QOS],
            grounding_decision={
                "flows": [
                    {"supi": "imsi-1", "app_id": "app-1", "flow_id": "flow-1"}
                ]
            },
            objective_profile=_objective_profile(),
            raw_intent="optimize QoS",
        ),
        template_name="joint_balanced",
        qos_relaxation_ratio=0.2,
        slice_kpi_source="qos",
        qos_feasibility_mode="hard",
    )

    assert request.scope.session_id == "session-1"
    assert request.scope.snapshot_id == "snapshot-1"
    assert request.target_ues == ["imsi-1"]
    assert request.requested_domains == [ControlDomain.QOS]


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        (_objective_profile(
            sla_violation_cost=0.0,
            control_churn_cost=0.0,
            resource_pressure_cost=0.0,
            fairness_cost=0.0,
        ), (1.0, 1.0, 1.0, 0.0)),
        (_objective_profile(
            sla_violation_cost=0.25,
            control_churn_cost=0.1,
            resource_pressure_cost=0.2,
            fairness_cost=0.4,
        ), (20.0, 5.0, 280.0, 137.5)),
    ],
)
def test_optimizer_weight_adapter_preserves_floors_and_formula(
    profile: ObjectiveProfile,
    expected: tuple[float, float, float, float],
) -> None:
    assert build_qos_optimizer_weights(profile).solver_arguments() == expected


def test_qos_solver_receives_adapter_weights_in_positional_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_optimize(
        target_app: dict[str, object],
        w1: float,
        w2: float,
        w3: float,
        w4: float,
        **kwargs: object,
    ) -> dict[str, object]:
        captured.update(target_app=target_app, weights=(w1, w2, w3, w4), kwargs=kwargs)
        return {}

    monkeypatch.setattr(joint_control, "optimize_network_slices", fake_optimize)
    profile = _objective_profile(
        sla_violation_cost=0.25,
        control_churn_cost=0.1,
        resource_pressure_cost=0.2,
        fairness_cost=0.4,
    )
    request = JointOptimizationRequest(
        scope=ExecutionScope(
            session_id="session-1",
            thread_id="session-1",
            snapshot_id="snapshot-1",
            round_index=1,
            supi="001",
        ),
        target_ues=["001"],
        requested_domains=[ControlDomain.QOS],
        grounding_decision={
            "flows": [
                {
                    "flow_id": "flow-1",
                    "app_id": "app-1",
                    "app_name": "video",
                    "supi": "001",
                }
            ]
        },
        traffic_state={},
        resource_state={},
        mobility_state={},
        policy_state={},
        objective_profile=profile,
        problem_config=_problem_config(decision_variables=["slice_assignment"]),
    )

    result, errors = joint_control._run_qos_subproblem(
        request,
        problem_config=request.problem_config,
        objective_profile=profile,
    )

    assert result == {}
    assert errors == []
    assert captured["weights"] == (20.0, 5.0, 280.0, 137.5)


def test_network_graph_exposes_only_canonical_scenario_projection() -> None:
    graph = NetworkGraph(snapshot_id="snapshot-1", trigger_event="test")
    graph.add_node(
        "app:001:video",
        "app",
        properties={"id": "video", "name": "Video", "supi": "001"},
    )
    graph.add_node(
        "flow:001:video:f1",
        "flow",
        properties={"id": "f1", "name": "stream", "supi": "001", "app_id": "video"},
    )
    graph.add_edge(
        "app:001:video",
        "flow:001:video:f1",
        "contains_flow",
        edge_key="app:001:video->flow:001:video:f1",
    )

    snapshot = graph.to_scenario_snapshot()

    assert not hasattr(graph, "to_compatibility_snapshot")
    assert snapshot["snapshot_id"] == "snapshot-1"
    assert snapshot["trigger_event"] == "test"
    assert snapshot["apps"] == [
        {
            "id": "video",
            "name": "Video",
            "supi": "001",
            "flows": [
                {"id": "f1", "name": "stream", "supi": "001", "app_id": "video"}
            ],
        }
    ]
