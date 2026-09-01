from __future__ import annotations

import re
from typing import List

from shared.contracts import AgentRunResult, ExecutionScope

from .contracts import (
    GlobalControlIntent,
    MainControlDecision,
    MainRetryScope,
    MainRoundStrategy,
    ReuseContract,
)


def validate_global_intent_result(
    result: AgentRunResult,
    *,
    user_input: str,
    scope: ExecutionScope,
) -> GlobalControlIntent:
    try:
        decision = MainControlDecision.model_validate(result.structured_response)
        return compile_global_intent(
            decision,
            user_input=user_input,
            scope=scope,
        )
    except Exception as exc:
        raise RuntimeError(f"Main Agent returned invalid MainControlDecision payload: {exc}") from exc


def compile_global_intent(
    decision: MainControlDecision,
    *,
    user_input: str,
    scope: ExecutionScope,
) -> GlobalControlIntent:
    round_index = scope.round_index
    next_agent = decision.next_agent
    if round_index <= 1:
        round_strategy = MainRoundStrategy.INITIAL_GROUNDING
        retry_scope = MainRetryScope.FULL_REGROUND
    elif next_agent == "optimization_strategy":
        round_strategy = MainRoundStrategy.POLICY_REVISION
        if decision.retry_scope is None:
            raise ValueError("retry MainControlDecision requires retry_scope")
        retry_scope = decision.retry_scope
    else:
        round_strategy = MainRoundStrategy.REGROUNDING
        if decision.retry_scope is None:
            raise ValueError("retry MainControlDecision requires retry_scope")
        retry_scope = decision.retry_scope

    reuse_allowed = next_agent == "optimization_strategy"
    domains = ",".join(item.value for item in decision.requested_domains) or "none"
    return GlobalControlIntent(
        scope=scope.model_copy(update={"supi": decision.supi or None}),
        raw_input=user_input,
        round_strategy=round_strategy,
        next_agent=decision.next_agent,
        requested_domains=decision.requested_domains,
        serving_slice_change_requested=decision.serving_slice_change_requested,
        domain_evidence=decision.domain_evidence,
        control_semantics=decision.control_semantics,
        objective_profile=decision.objective_profile,
        investigation_targets=decision.investigation_targets,
        uncertainty_flags=decision.uncertainty_flags,
        retry_scope=retry_scope,
        required_evidence=decision.required_evidence,
        forbidden_assumptions=decision.forbidden_assumptions,
        intent_encoding_guidance=decision.intent_encoding_guidance,
        routing_decision=f"{round_strategy.value}_{next_agent}_{domains}",
        routing_rationale=(
            f"Round {round_index}: route {domains} control through {next_agent}; "
            "preserve the Main-owned staged semantic plan."
        ),
        reuse_contract=ReuseContract(
            allowed=reuse_allowed,
            preserve_bindings=reuse_allowed,
            preserve_domains=reuse_allowed,
            preserve_stage_scope=reuse_allowed,
            invalidate_on=[] if reuse_allowed else ["reground_required"],
        ),
    )



def validate_global_intent(
    intent: GlobalControlIntent,
) -> List[str]:
    user_input = intent.raw_input
    round_index = intent.scope.round_index
    errors: List[str] = []
    if not intent.requested_domains:
        errors.append("requested_domains is empty")
    else:
        values = [item.value for item in intent.requested_domains]
        if any(item not in {"qos", "mobility"} for item in values):
            errors.append(f"requested_domains contains unsupported values: {values}")
        if str(intent.next_agent or "").strip() not in {"intent_encoding", "optimization_strategy"}:
            errors.append("next_agent must be either intent_encoding or optimization_strategy")
        if round_index <= 1 and str(intent.next_agent or "").strip() != "intent_encoding":
            errors.append("round-1 main routing must set next_agent=intent_encoding; optimization_strategy is retry-only")
    domain_values = {item.value for item in intent.requested_domains}
    if intent.serving_slice_change_requested and domain_values != {"qos", "mobility"}:
        errors.append("serving_slice_change_requested requires requested_domains=[qos,mobility]")
    explicit_supis = re.findall(r"(?i)(imsi-\d{5,})", str(user_input or ""))
    unique_explicit_supis = list(dict.fromkeys(explicit_supis))
    if len(unique_explicit_supis) == 1:
        explicit_supi = unique_explicit_supis[0]
        if intent.scope.supi != explicit_supi:
            errors.append(f"supi must equal explicit user-provided identifier {explicit_supi}")
    elif len(unique_explicit_supis) > 1 and intent.scope.supi is not None:
        errors.append("top-level supi must not comma-join multiple identifiers; leave it empty for multi-SUPI requests")
    if isinstance(intent.domain_evidence, dict) and any(intent.domain_evidence.values()):
        unknown_domains = [key for key in intent.domain_evidence.keys() if key not in {"qos", "mobility"}]
        if unknown_domains:
            errors.append(f"domain_evidence contains unsupported keys: {unknown_domains}")
        requested = {item.value for item in intent.requested_domains}
        evidence_keys = {str(key).strip().lower() for key, values in intent.domain_evidence.items() if values}
        if not requested.issubset(evidence_keys):
            errors.append(
                f"domain_evidence must cover every requested domain: requested={sorted(requested)} evidence={sorted(evidence_keys)}"
            )
    next_agent = str(intent.next_agent or "").strip()
    if next_agent == "intent_encoding" and not str(intent.intent_encoding_guidance or "").strip() and round_index > 1:
        errors.append("retry routing into intent_encoding requires explicit intent_encoding_guidance")
    if next_agent != "intent_encoding" and str(intent.intent_encoding_guidance or "").strip():
        errors.append("intent_encoding_guidance must stay empty unless next_agent=intent_encoding")
    if len(unique_explicit_supis) == 1:
        explicit_supi = unique_explicit_supis[0]
        for stage_index, stage in enumerate(intent.control_semantics.stages or [], start=1):
            for target_index, target in enumerate(stage.targets or [], start=1):
                if not str(target.supi or "").strip():
                    errors.append(
                        "single-SUPI semantic target must carry target.supi "
                        f"stage {stage_index} target {target_index}: expected {explicit_supi}"
                    )
    return errors


__all__ = [
    "compile_global_intent",
    "validate_global_intent",
    "validate_global_intent_result",
]
