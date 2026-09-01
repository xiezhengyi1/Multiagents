from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from jinja2 import UndefinedError

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

from control_runtime.agents.grounding.contracts import (
    FlowSelector,
    GroundedMobilityEvidence,
    GroundingDecision,
    GroundingEvidenceBundle,
    QosOperationConstraint,
    SliceMigrationAuthorization,
)
from control_runtime.orchestrators.round_context import build_intent_encoding_context
from control_runtime.agents.main.contracts import (
    ControlSemanticMode,
    ControlDomain,
    GlobalControlIntent,
    MainControlSemantics,
    MainControlStage,
    StageTrigger,
)
from control_runtime.context.prompts.main import build_main_validation_retry_prompt
from control_runtime.agents.planning.contracts import (
    PlanningContext,
    PlanningRequest,
    SharedControlContext,
)
from control_runtime.agents.planning.evidence import build_planning_evidence
from control_runtime.context.prompts.engine import PromptEngine
from control_runtime.context.prompts.grounding import render_grounding_system_prompt
from control_runtime.diagnostics.mediation.contracts import UnifiedConstraintSet
from control_runtime.context.prompts.main import render_main_system_prompt
from control_runtime.context.prompts.planning import (
    build_planning_user_prompt,
    render_planning_system_prompt,
)
from control_runtime.context.prompts.single import render_single_system_prompt
from control_runtime.context.projectors import field, project_global_intent_for_prompt
from control_runtime.context.projectors.base import BaseProjector
from control_runtime.orchestrators.round_context import build_planning_context
from control_runtime.orchestrators.round_transitions import scope_global_intent_for_intent_encoding
from shared.contracts import ExecutionScope, MemoryContextView
from control_runtime.contracts import standard_objective_profile


LEGACY_CONTEXT_ENGINEERING_FILES = [
    "src/control_runtime/agents/common/context_projection.py",
    "src/control_runtime/agents/grounding/evidence_builder.py",
    "src/control_runtime/agents/grounding/prompts.py",
    "src/control_runtime/agents/planning/planning_evidence.py",
    "src/control_runtime/agents/planning/request_builder.py",
    "src/control_runtime/agents/planning/prompts.py",
    "src/control_runtime/agents/main/prompts.py",
    "src/control_runtime/agents/single/prompts.py",
    "src/control_runtime/agents/prompt_skills/knowledge_search.py",
]

CONSOLIDATED_AGENT_HELPER_FILES = [
    "src/control_runtime/agents/grounding/qos_envelope_builder.py",
    "src/control_runtime/agents/planning/policy_normalizer.py",
]

LEGACY_IMPORT_PATTERNS = [
    "agents.common.context_projection",
    "agents.grounding.evidence_builder",
    "agents.grounding.prompts",
    "agents.planning.planning_evidence",
    "agents.planning.request_builder",
    "agents.planning.prompts",
    "agents.main.prompts",
    "agents.single.prompts",
    "agents.prompt_skills.knowledge_search",
    "agents.grounding.qos_envelope_builder",
    "agents.planning.policy_normalizer",
    "from ..common import project_",
]


def _intent_scope(
    *,
    session_id: str = "s1",
    snapshot_id: str = "snap1",
    supi: str | None = "imsi-1",
    round_index: int = 1,
) -> ExecutionScope:
    return ExecutionScope(
        session_id=session_id,
        thread_id=session_id,
        snapshot_id=snapshot_id,
        round_index=round_index,
        supi=supi,
    )


def test_legacy_context_engineering_entrypoints_are_removed() -> None:
    for legacy_path in [*LEGACY_CONTEXT_ENGINEERING_FILES, *CONSOLIDATED_AGENT_HELPER_FILES]:
        assert not (ROOT / legacy_path).exists(), legacy_path

    scanned_files = [
        path
        for base in (ROOT / "src", ROOT / "tests")
        for path in base.rglob("*.py")
        if "__pycache__" not in path.parts
        and path.name != "test_context_engineering_refactor.py"
    ]
    offenders: list[str] = []
    for path in scanned_files:
        text = path.read_text(encoding="utf-8")
        for pattern in LEGACY_IMPORT_PATTERNS:
            if pattern in text:
                offenders.append(f"{path.relative_to(ROOT)}: {pattern}")
    assert offenders == []


def test_projector_rejects_unknown_model_fields_at_definition_time() -> None:
    with pytest.raises(TypeError, match="does not exist on FlowSelector"):

        class BadFlowProjector(BaseProjector):
            model = FlowSelector
            visible = (field("requested_domains"),)


def test_global_control_intent_projection_uses_declared_routing_and_flow_contracts() -> None:
    intent = GlobalControlIntent(
        scope=_intent_scope(),
        raw_input="protect flow",
        next_agent="intent_encoding",
        requested_domains=[ControlDomain.QOS],
        objective_profile=standard_objective_profile("balanced"),
        domain_evidence={"qos": ["flow catalog"]},
        required_evidence=["flow catalog"],
        routing_decision="ground qos flow",
    )

    projected = project_global_intent_for_prompt(intent)

    assert projected["scope"]["session_id"] == "s1"
    assert projected["scope"]["supi"] == "imsi-1"
    assert projected["requested_domains"] == ["qos"]
    assert projected["domain_evidence"] == {"qos": ["flow catalog"]}
    assert projected["routing_decision"] == "ground qos flow"
    assert "flows" not in projected


def test_routing_and_grounding_models_keep_ownership_boundaries() -> None:
    assert "requested_domains" in GlobalControlIntent.model_fields
    assert "flows" not in GlobalControlIntent.model_fields
    assert "requested_domains" not in FlowSelector.model_fields


def test_planning_evidence_matches_operation_constraint_shape() -> None:
    main_intent = GlobalControlIntent(
        scope=_intent_scope(),
        next_agent="optimization_strategy",
        requested_domains=[ControlDomain.QOS],
        objective_profile=standard_objective_profile("balanced"),
    )
    grounding_decision = GroundingDecision(
        flows=[
            FlowSelector(
                supi="imsi-1",
                app_id="app-1",
                app_name=None,
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
                current_slice_snssai="01020304",
                current_bw_ul=None,
                current_bw_dl=None,
                resolution_status="resolved",
            )
        ],
        mobility_intent=None,
        grounding_evidence=GroundingEvidenceBundle(
            grounded_supi="imsi-1",
            grounded_apps=[],
            grounded_flows=[],
            mobility=GroundedMobilityEvidence(summary=None, subscription_entitlement=None),
            evidence_sources={},
        ),
        qos_operation_constraints=[
            QosOperationConstraint(
                flow_id="flow-1",
                app_id="app-1",
                operation_type="slice_migration",
                require_slice_change=True,
                source_slice_snssai="01020304",
                excluded_slice_snssais=["01020304"],
                target_slice_preference="lower_latency",
                no_op_allowed=False,
                rationale=[],
            )
        ],
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
    planning_context = PlanningContext(
        memory_context=MemoryContextView(
            query=None,
            selected_short_term=[],
            selected_long_term=[],
            prompt_text="",
        ),
        shared_context=SharedControlContext(main_intent=main_intent),
        feedback_context="",
        handoff_history=[],
        revision_requests=[],
        unified_constraints=UnifiedConstraintSet(hard_constraints=[], evidence=[]),
        qos_retry_constraints=[],
    )

    evidence = build_planning_evidence(
        PlanningRequest(
            grounding_decision=grounding_decision,
            context=planning_context,
        )
    )

    assert evidence["derived_operation_constraints"] == [
        {
            "flow_id": "flow-1",
            "app_id": "app-1",
            "operation_type": "slice_migration",
            "require_slice_change": True,
            "source_slice_snssai": "01020304",
            "excluded_slice_snssais": ["01020304"],
            "target_slice_preference": "lower_latency",
            "no_op_allowed": False,
            "rationale": [],
        }
    ]


def test_main_retry_prompt_keeps_main_contract() -> None:
    retry_prompt = build_main_validation_retry_prompt(
        base_prompt="User input:\nexample",
        validation_errors=[],
        invocation_error="Input should be a valid dictionary, got list",
    )

    assert retry_prompt.startswith("User input:\nexample")
    assert "MainControlDecision" in retry_prompt
    assert "Input should be a valid dictionary, got list" in retry_prompt


def test_prompt_engine_raises_for_missing_template_variables() -> None:
    with pytest.raises(UndefinedError):
        PromptEngine().render("grounding/system.j2")


def test_system_prompt_renderers_render_jinja_templates() -> None:
    for rendered in (
        render_main_system_prompt(),
        render_grounding_system_prompt(),
        render_planning_system_prompt(),
        render_single_system_prompt(),
    ):
        assert "JSON" in rendered
        assert "{% block" not in rendered
        assert "{% include" not in rendered


def test_system_prompt_renderers_expose_nonempty_schema_guidance() -> None:
    rendered_prompts = (
        render_main_system_prompt(),
        render_grounding_system_prompt(),
        render_planning_system_prompt(),
        render_single_system_prompt(),
    )

    for rendered in rendered_prompts:
        assert isinstance(rendered, str)
        assert len(rendered.strip()) > 100
        assert "JSON" in rendered


def test_system_prompt_renderers_render_without_template_artifacts() -> None:
    for rendered in (
        render_main_system_prompt(),
        render_grounding_system_prompt(),
        render_planning_system_prompt(),
        render_single_system_prompt(),
    ):
        assert "{%" not in rendered
        assert "{{" not in rendered


def test_planning_user_prompt_is_rendered_from_jinja_template() -> None:
    prompt = build_planning_user_prompt(
        normalized_user_intent={"session_id": "s1", "app_id": "app_1"},
        coordination_context={"active_domains": ["qos"]},
        planning_evidence={"flows": [{"flow_id": "flow-1"}]},
        available_tool_names=["preview_qos_optimizer"],
    )

    assert isinstance(prompt, str)
    assert "s1" in prompt
    assert "flow-1" in prompt
    assert "preview_qos_optimizer" in prompt


def test_planning_user_prompt_carries_all_public_round_inputs() -> None:
    prompt = build_planning_user_prompt(
        normalized_user_intent={"session_id": "session-a", "app_id": "app-a"},
        coordination_context={"active_domains": ["mobility"]},
        planning_evidence={"flows": [{"flow_id": "flow-a"}]},
        available_tool_names=["inspect_mobility_ue_policies"],
    )

    for value in ("session-a", "app-a", "mobility", "flow-a", "inspect_mobility_ue_policies"):
        assert value in prompt


def test_intent_context_keeps_the_1800_character_retry_feedback_window() -> None:
    feedback = "obsolete-feedback-marker " + ("x" * 1800) + " tail-feedback-marker"
    payload = json.loads(
        build_intent_encoding_context(
            global_intent={
                "supi": "imsi-001",
                "requested_domains": ["qos"],
                "domain_evidence": {},
            },
            round_index=2,
            diagnosis={"root_cause_category": "dispatch_failure", "reason_summary": "retry"},
            feedback_context=feedback,
        )
    )

    summary = payload["retry_delta"]["feedback_summary"]
    assert summary.startswith("... [older feedback omitted]\n")
    assert "obsolete-feedback-marker" not in summary
    assert summary.endswith("tail-feedback-marker")
    assert len(summary) <= len("... [older feedback omitted]\n") + 1800
    assert "retry_delta" not in json.loads(
        build_intent_encoding_context(
            global_intent={
                "supi": "imsi-001",
                "requested_domains": ["qos"],
                "domain_evidence": {},
            },
            round_index=1,
            diagnosis={"root_cause_category": "dispatch_failure"},
            feedback_context=feedback,
        )
    )


def test_intent_context_scopes_one_stage_without_mutating_the_main_intent() -> None:
    intent = GlobalControlIntent(
        scope=_intent_scope(
            session_id="session-1",
            snapshot_id="snapshot-1",
            supi="imsi-001",
        ),
        next_agent="intent_encoding",
        requested_domains=[ControlDomain.QOS],
        objective_profile=standard_objective_profile("balanced"),
        control_semantics=MainControlSemantics(
            mode=ControlSemanticMode.STAGED_PRIORITY,
            current_stage=2,
            stages=[
                MainControlStage(
                    stage_index=2,
                    name="ground first flow",
                    trigger=StageTrigger.INITIAL,
                    summary="first stage",
                    targets=[],
                ),
                MainControlStage(
                    stage_index=5,
                    name="ground second flow",
                    trigger=StageTrigger.AFTER_PREVIOUS_STAGE,
                    summary="second stage",
                    targets=[],
                ),
            ],
            notes=[],
        ),
    )

    scoped = scope_global_intent_for_intent_encoding(
        global_intent=intent,
        round_index=2,
    )

    assert [stage.stage_index for stage in intent.control_semantics.stages] == [2, 5]
    assert scoped is not intent
    assert scoped.control_semantics.current_stage == 1
    assert [(stage.stage_index, stage.name) for stage in scoped.control_semantics.stages] == [
        (1, "ground second flow")
    ]


def test_planning_context_keeps_last_two_handoffs_and_sorted_qos_retry_constraints() -> None:
    intent = GlobalControlIntent(
        scope=_intent_scope(
            session_id="session-1",
            snapshot_id="snapshot-1",
            supi="imsi-001",
            round_index=3,
        ),
        next_agent="optimization_strategy",
        requested_domains=[ControlDomain.QOS],
        objective_profile=standard_objective_profile("balanced"),
    )
    previous_execution_feedback = {
        "feedback_payload": {
            "failures": [
                {
                    "phase": "assurance",
                    "flow_id": "flow-z",
                    "policy_id": "sm-z",
                    "assurance_result": {
                        "metrics": {
                            "latency_within_tolerance": False,
                            "latency": 150,
                            "latency_requirement": 100,
                            "tolerance_ratio": 0.1,
                        }
                    },
                },
                {
                    "phase": "assurance",
                    "flow_id": "flow-a",
                    "policy_id": "sm-a",
                    "assurance_result": {
                        "metrics": {
                            "latency_within_tolerance": False,
                            "latency": 130,
                            "latency_requirement": 100,
                            "tolerance_ratio": 0.1,
                        }
                    },
                },
                {
                    "phase": "assurance",
                    "flow_id": "flow-z",
                    "policy_id": "sm-z-retry",
                    "assurance_result": {
                        "metrics": {
                            "latency_within_tolerance": False,
                            "latency": 160,
                            "latency_requirement": 100,
                            "tolerance_ratio": 0.1,
                        }
                    },
                },
                {
                    "phase": "dispatch",
                    "flow_id": "ignored",
                    "assurance_result": {"metrics": {"latency_within_tolerance": False}},
                },
            ]
        }
    }

    context = build_planning_context(
        intent,
        memory_context=MemoryContextView(
            query=None,
            selected_short_term=[],
            selected_long_term=[],
            prompt_text="",
        ),
        feedback_context="",
        handoff_history=[{"round": 1}, {"round": 2}, {"round": 3}, {"round": 4}],
        revision_requests=[],
        unified_constraints={"hard_constraints": [], "evidence": []},
        previous_execution_feedback=previous_execution_feedback,
    )

    assert context.handoff_history == [{"round": 3}, {"round": 4}]
    assert [
        (constraint.flow_id, constraint.minimum_target_latency_ms, constraint.policy_id)
        for constraint in context.qos_retry_constraints
    ] == [
        ("flow-a", 119, "sm-a"),
        ("flow-z", 146, "sm-z-retry"),
    ]
    assert context.qos_retry_constraints[1].allowed_latency_ms == pytest.approx(110)
