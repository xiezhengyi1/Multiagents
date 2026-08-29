from __future__ import annotations

from .engine import PromptEngine
from .retry import strip_retry_feedback


def render_main_system_prompt() -> str:
    return PromptEngine().render("main/system.j2")


def build_main_user_prompt(*, user_input: str, coordinator_context: str) -> str:
    return (
        f"User input:\n{user_input}\n\n"
        f"Coordinator context:\n{coordinator_context or 'N/A'}\n\n"
        "Dynamic routing rules for this round:\n"
        "- Read the coordinator context before deciding reuse or regrounding.\n"
        "- Round 1 must route to `intent_encoding`.\n"
        "- Retry rounds may route to `optimization_strategy` only when target bindings can be reused.\n"
        "- If retry evidence says binding is wrong, prefer `intent_encoding`.\n"
        "- If retry evidence says binding is stable but policy execution or assurance failed, prefer `optimization_strategy`.\n\n"
        "Resolve the staged semantic plan, round-level routing, retry scope, and explicit SUPI already present in the request. "
        "On a retry routed to intent_encoding, provide explicit intent_encoding_guidance; otherwise leave it empty unless needed."
    )


def build_main_validation_retry_prompt(
    *,
    base_prompt: str,
    validation_errors: list[str],
    invocation_error: str,
) -> str:
    issues = ([invocation_error] if invocation_error else []) + validation_errors
    required_keys = (
        "supi, next_agent, requested_domains, serving_slice_change_requested, domain_evidence, "
        "control_semantics, objective_profile, investigation_targets, uncertainty_flags, "
        "retry_scope, required_evidence, forbidden_assumptions, intent_encoding_guidance"
    )
    return (
        f"{strip_retry_feedback(base_prompt)}\n\n"
        "Retry feedback:\n"
        "Correct the output now.\n"
        "Return exactly one MainControlDecision JSON object.\n"
        "Return only the MainControlDecision fields directly at the top level.\n"
        "Do not emit runtime wrapper keys like messages or structured_response.\n"
        "Top-level output must be an object, never a list.\n"
        "Do not return markdown, prose, bullets, arrays, or partial sub-objects.\n"
        f"Required MainControlDecision keys: {required_keys}.\n"
        "Never emit session_id, snapshot_id, raw_input, round_strategy, reuse_contract, routing_decision, or routing_rationale; the compiler owns them.\n"
        "control_semantics.targets[].target_type must be one of flow, app, scope, named_object.\n"
        "UE/user-equipment/AM-policy targets must use scope; never use target_type=ue.\n"
        "control_semantics.targets[].goal must be one of protect, deprioritize, defer, observe.\n"
        "degrade/throttle/constrain must be rewritten as deprioritize; optimize/improve/boost must be rewritten as protect.\n"
        "The literal value optimize is forbidden in control_semantics.targets[].goal; retain it only in note.\n"
        "control_semantics.stages[].trigger must be one of initial, on_previous_failure, after_previous_stage, retry.\n"
        "For staged priority, stage_index=1 uses initial; stage_index>1 uses after_previous_stage.\n"
        "MainControlDecision control_semantics targets contain only semantic_name, target_type, goal, metric_focus, note, and supi; IEA owns resolved identifiers.\n"
        "MainControlDecision control_semantics is Main-owned; IEA never outputs it.\n"
        "For multi-SUPI requests, leave top-level supi empty and set each target.supi separately; never comma-join SUPIs.\n"
        "For exactly one explicit imsi-... in the user request, top-level supi and every semantic target.supi must equal that exact identifier; never leave either empty.\n"
        "Set serving_slice_change_requested=true only for a structured serving-S-NSSAI change and then include exactly qos and mobility; otherwise set it false.\n"
        "If validation says execution_failure with stable bindings, route to optimization_strategy and set retry_scope=target_stable; the compiler derives policy_revision and reuse_contract.\n"
        "If a previous draft was a list, convert the intended route into the top-level MainControlDecision object.\n\n"
        "Validation errors:\n- "
        + "\n- ".join(issues)
    )
