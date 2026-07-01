from __future__ import annotations

import sys
from pathlib import Path

import pytest
from jinja2 import UndefinedError

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PROMPT_FIXTURES = ROOT / "tests" / "fixtures" / "context_prompts"
for candidate in (ROOT, SRC):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from control_runtime.context import (
    EvidenceFormatter,
    FlowSelectorProjector,
    MainPromptBuilder,
    OperationIntentProjector,
    ProjectorRegistry,
    RetryPromptBuilder,
    field,
    project_operation_intent_for_prompt,
)
from control_runtime.context.prompts import (
    GroundingPromptBuilder,
    IEA_SYSTEM_PROMPT,
    MAIN_CONTROL_SYSTEM_PROMPT,
    OSA_SYSTEM_PROMPT,
    PlanningPromptBuilder,
    SINGLE_AGENT_ROUND_PROMPT,
    SinglePromptBuilder,
)
from control_runtime.context.prompts.engine import PromptEngine
from control_runtime.context.projectors.base import BaseProjector
from control_runtime.domain.collaboration import PlanningContext, PlanningRequest
from control_runtime.domain.policy_plan import FlowSelector, OperationIntent, QosTargetEnvelope


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
    "src/control_runtime/agents/grounding/directives.py",
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
    "agents.grounding.directives",
    "agents.grounding.qos_envelope_builder",
    "agents.planning.policy_normalizer",
    "from ..common import project_",
]


def _prompt_fixture(name: str) -> str:
    return (PROMPT_FIXTURES / name).read_text(encoding="utf-8")


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


def test_operation_intent_projection_uses_declared_flow_fields_without_dead_or_wrong_fields() -> None:
    intent = OperationIntent(
        session_id="s1",
        snapshot_id="snap1",
        supi="imsi-1",
        app_id="app-1",
        app_name="RemoteDrive",
        urgency="Normal",
        raw_input="protect flow",
        requested_domains=["qos"],
        flows=[
            FlowSelector(
                supi="imsi-1",
                app_id="app-1",
                app_name="RemoteDrive",
                flow_id="flow-1",
                target_type="flow",
                name="video",
                service_type="urllc",
                service_type_id=1,
                bw_ul=10.0,
                current_bw_ul=3.0,
                current_bw_dl=4.0,
                five_tuple=["10.0.0.1", "10.0.0.2", 1000, 2000, "udp"],
                resolution_candidates=["stale-candidate"],
            )
        ],
        qos_target_envelopes=[
            QosTargetEnvelope(
                flow_id="flow-1",
                app_id="app-1",
                flow_name="video",
                baseline_latency_ms=10.0,
                rationale=["derived from SLA"],
            )
        ],
    )

    projected = project_operation_intent_for_prompt(intent)
    flow_payload = projected["flows"][0]

    assert projected["session_id"] == "s1"
    assert "urgency" not in projected
    assert flow_payload["app_name"] == "RemoteDrive"
    assert flow_payload["target_type"] == "flow"
    assert flow_payload["five_tuple"] == ["10.0.0.1", "10.0.0.2", 1000, 2000, "udp"]
    assert "requested_domains" not in flow_payload
    assert "dnn" not in flow_payload
    assert "current_bw_ul" not in flow_payload
    assert "current_bw_dl" not in flow_payload
    assert "resolution_candidates" not in flow_payload
    assert "rationale" in projected["qos_target_envelopes"][0]


def test_dead_policy_plan_fields_are_removed_from_models() -> None:
    assert "urgency" not in OperationIntent.model_fields
    assert "resolution_candidates" not in FlowSelector.model_fields


def test_projector_registry_resolves_known_models() -> None:
    assert ProjectorRegistry.for_model(FlowSelector) is FlowSelectorProjector
    assert ProjectorRegistry.for_model(OperationIntent) is OperationIntentProjector


def test_evidence_formatter_matches_planning_evidence_shape() -> None:
    request = PlanningRequest(
        operation_intent=OperationIntent(
            session_id="s1",
            snapshot_id="snap1",
            supi="imsi-1",
            flows=[
                FlowSelector(
                    supi="imsi-1",
                    app_id="app-1",
                    flow_id="flow-1",
                    name="video",
                    priority=1,
                    service_type_id=7,
                )
            ],
        ),
        context=PlanningContext(
            session_id="s1",
            snapshot_id="snap1",
            active_domains=["qos"],
            objective_profile={"profile_name": "latency"},
            required_evidence=["optimizer preview"],
        ),
    )

    evidence = EvidenceFormatter.for_osa(
        operation_intent=request.operation_intent,
        planning_context=request.context,
    )

    assert evidence["requested_domains"] == ["qos"]
    assert evidence["objective_profile"] == {"profile_name": "latency"}
    assert evidence["flows"] == [
        {
            "flow_id": "flow-1",
            "app_id": "app-1",
            "name": "video",
            "priority": 1,
            "service_type_id": 7,
            "current_slice_snssai": None,
        }
    ]


def test_main_prompt_builder_and_retry_builder_keep_compatibility_contracts() -> None:
    assert MainPromptBuilder().system_prompt() == MAIN_CONTROL_SYSTEM_PROMPT

    retry_prompt = RetryPromptBuilder().build_main(
        base_prompt="User input:\nexample",
        validation_errors=[],
        invocation_error="Input should be a valid dictionary, got list",
    )

    assert "Return exactly one GlobalControlIntent JSON object." in retry_prompt
    assert "Top-level output must be an object, never a list." in retry_prompt
    assert "Required GlobalControlIntent keys" in retry_prompt


def test_prompt_engine_raises_for_missing_template_variables() -> None:
    with pytest.raises(UndefinedError):
        PromptEngine().render("grounding/system.j2")


def test_prompt_builders_render_jinja_templates_as_compatibility_constants() -> None:
    assert MainPromptBuilder().system_prompt() == MAIN_CONTROL_SYSTEM_PROMPT
    assert GroundingPromptBuilder().system_prompt() == IEA_SYSTEM_PROMPT
    assert PlanningPromptBuilder().system_prompt() == OSA_SYSTEM_PROMPT
    assert SinglePromptBuilder().system_prompt() == SINGLE_AGENT_ROUND_PROMPT

    for rendered in (
        MainPromptBuilder().system_prompt(),
        GroundingPromptBuilder().system_prompt(),
        PlanningPromptBuilder().system_prompt(),
        SinglePromptBuilder().system_prompt(),
    ):
        assert "JSON" in rendered
        assert "{% block" not in rendered
        assert "{% include" not in rendered


def test_system_prompt_templates_preserve_existing_prompt_text_exactly() -> None:
    expected_prompts = {
        "main_system.txt": (MAIN_CONTROL_SYSTEM_PROMPT, MainPromptBuilder().system_prompt()),
        "grounding_system.txt": (IEA_SYSTEM_PROMPT, GroundingPromptBuilder().system_prompt()),
        "planning_system.txt": (OSA_SYSTEM_PROMPT, PlanningPromptBuilder().system_prompt()),
        "single_system.txt": (SINGLE_AGENT_ROUND_PROMPT, SinglePromptBuilder().system_prompt()),
    }

    for fixture_name, rendered_prompts in expected_prompts.items():
        expected = _prompt_fixture(fixture_name)
        for rendered in rendered_prompts:
            assert rendered == expected


def test_system_jinja_templates_compose_prompt_fragments() -> None:
    template_root = ROOT / "src" / "control_runtime" / "context" / "prompts" / "templates"

    for template_name in (
        "main/system.j2",
        "grounding/system.j2",
        "planning/system.j2",
        "single/system.j2",
    ):
        text = (template_root / template_name).read_text(encoding="utf-8")
        assert "{% include" in text, template_name

    grounding_system = (template_root / "grounding" / "system.j2").read_text(encoding="utf-8")
    planning_system = (template_root / "planning" / "system.j2").read_text(encoding="utf-8")

    assert "iea_knowledge_search_skill" in grounding_system
    assert "osa_knowledge_search_skill" in planning_system


def test_planning_user_prompt_is_rendered_from_jinja_template() -> None:
    from control_runtime.context.prompts import OSA_DYNAMIC_RULES
    from control_runtime.context.prompts.planning import _OUTPUT_FORMAT_RULES, _render_round_tool_policy

    prompt = PlanningPromptBuilder().advisor_user_prompt(
        normalized_user_intent={"session_id": "s1", "app_id": "app_1"},
        coordination_context={"active_domains": ["qos"]},
        planning_evidence={"flows": [{"flow_id": "flow-1"}]},
        available_tool_names=["preview_qos_optimizer"],
    )

    engine_rendered = PromptEngine().render(
        "planning/user.j2",
        normalized_user_intent={"session_id": "s1", "app_id": "app_1"},
        coordination_context={"active_domains": ["qos"]},
        planning_evidence={"flows": [{"flow_id": "flow-1"}]},
        tool_policy=_render_round_tool_policy(["preview_qos_optimizer"]),
        dynamic_rules=OSA_DYNAMIC_RULES.strip(),
        output_format_rules=_OUTPUT_FORMAT_RULES.strip(),
    )

    assert prompt == engine_rendered
    assert '"session_id": "s1"' in prompt
    assert "Callable tools in this round:" in prompt


def test_planning_user_prompt_preserves_existing_prompt_text_exactly() -> None:
    import json

    from control_runtime.context.prompts import OSA_DYNAMIC_RULES
    from control_runtime.context.prompts.planning import _OUTPUT_FORMAT_RULES, _render_round_tool_policy

    normalized_user_intent = {"session_id": "s1", "app_id": "app_1"}
    coordination_context = {"active_domains": ["qos"]}
    planning_evidence = {"flows": [{"flow_id": "flow-1"}]}
    tool_policy = _render_round_tool_policy(["preview_qos_optimizer"])

    prompt = PlanningPromptBuilder().advisor_user_prompt(
        normalized_user_intent=normalized_user_intent,
        coordination_context=coordination_context,
        planning_evidence=planning_evidence,
        available_tool_names=["preview_qos_optimizer"],
    )

    expected = (
        "Structured operation intent:\n"
        f"{json.dumps(normalized_user_intent, ensure_ascii=False, default=str)}\n\n"
        "Planning context:\n"
        f"{json.dumps(coordination_context, ensure_ascii=False, default=str)}\n\n"
        "Planning evidence:\n"
        f"{json.dumps(planning_evidence, ensure_ascii=False, default=str)}\n\n"
        f"{tool_policy}\n\n"
        f"{OSA_DYNAMIC_RULES.strip()}\n\n"
        "Task:\n"
        "- Inspect the evidence and return one complete grounded OsaAdvisorOutput.\n"
        "- If evidence is sufficient, return planning_status=\"executable_plan\" with all required fields grounded.\n"
        "- If evidence is insufficient or optimizer is infeasible/incomplete, return partial_plan or needs_upstream_reground.\n"
        "- Respect control_semantics.current_stage; optimize only the active stage flows.\n"
        "- Prefer optimizer sla values over telemetry values when filling final policy fields.\n\n"
        f"{_OUTPUT_FORMAT_RULES.strip()}\n\n"
        "Return one OsaAdvisorOutput JSON object only."
    )

    assert prompt == expected
