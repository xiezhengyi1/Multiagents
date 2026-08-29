from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from shared.agents import coerce_structured_response

from .contracts import (
    GlobalControlIntent,
    MainControlDecision,
    MainRetryScope,
    MainRoundStrategy,
    ReuseContract,
)


_CONTROL_SEMANTICS_LEAK_FIELDS = (
    "objective_profile",
    "investigation_targets",
    "uncertainty_flags",
    "retry_scope",
    "required_evidence",
    "forbidden_assumptions",
    "intent_encoding_guidance",
    "routing_decision",
    "routing_rationale",
    "reuse_contract",
)
_DECISION_FIELDS = frozenset(
    {
        "supi", "next_agent", "requested_domains", "serving_slice_change_requested",
        "domain_evidence", "control_semantics", "objective_profile",
        "investigation_targets", "uncertainty_flags", "retry_scope",
        "required_evidence", "forbidden_assumptions", "intent_encoding_guidance",
    }
)
_COMPILED_FIELDS = frozenset(
    {
        "session_id", "snapshot_id", "raw_input", "round_strategy",
        "routing_decision", "routing_rationale", "reuse_contract",
    }
)


def validate_global_intent_result(
    result: Dict[str, Any],
    *,
    user_input: str = "",
    session_id: str = "",
    snapshot_id: str = "",
    round_index: int = 1,
    projected_fields: Optional[List[str]] = None,
) -> GlobalControlIntent:
    try:
        raw_payload = _raw_structured_payload(result)
        decision = coerce_structured_response(
            repair_global_intent_payload(
                result,
                projected_fields=projected_fields,
            ),
            MainControlDecision,
            error_message="Main Agent returned no structured MainControlDecision",
        )
        return compile_global_intent(
            decision,
            user_input=str(user_input or raw_payload.get("raw_input") or ""),
            session_id=session_id or str(raw_payload.get("session_id") or ""),
            snapshot_id=snapshot_id or str(raw_payload.get("snapshot_id") or ""),
            round_index=round_index,
        )
    except Exception as exc:
        raise RuntimeError(f"Main Agent returned invalid MainControlDecision payload: {exc}") from exc


def compile_global_intent(
    decision: MainControlDecision,
    *,
    user_input: str,
    session_id: str,
    snapshot_id: str,
    round_index: int,
) -> GlobalControlIntent:
    next_agent = str(decision.next_agent or "").strip()
    if round_index <= 1:
        round_strategy = MainRoundStrategy.INITIAL_GROUNDING
        retry_scope = MainRetryScope.FULL_REGROUND
    elif next_agent == "optimization_strategy":
        round_strategy = MainRoundStrategy.POLICY_REVISION
        retry_scope = decision.retry_scope or MainRetryScope.TARGET_STABLE
    else:
        round_strategy = MainRoundStrategy.REGROUNDING
        retry_scope = decision.retry_scope or MainRetryScope.FULL_REGROUND

    reuse_allowed = next_agent == "optimization_strategy"
    guidance = str(decision.intent_encoding_guidance or "").strip()
    if round_index > 1 and next_agent == "intent_encoding" and not guidance:
        guidance = "Re-ground the active stage using required_evidence; do not assume forbidden_assumptions."
    domains = ",".join(item.value for item in decision.requested_domains) or "none"
    return GlobalControlIntent(
        session_id=session_id,
        snapshot_id=snapshot_id,
        raw_input=user_input,
        supi=decision.supi,
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
        intent_encoding_guidance=guidance,
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


def repair_global_intent_payload(
    result: Dict[str, Any],
    *,
    projected_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Recover known Main/IEA ownership leaks before strict validation."""

    structured = result.get("structured_response")
    if isinstance(structured, BaseModel):
        structured = structured.model_dump(mode="json", exclude_unset=True)
    if not isinstance(structured, (dict, str)):
        return result
    if isinstance(structured, str):
        try:
            structured = json.loads(structured)
        except json.JSONDecodeError:
            return result
    if not isinstance(structured, dict):
        return result

    repaired = deepcopy(structured)
    for field_name in _COMPILED_FIELDS:
        if field_name in repaired:
            repaired.pop(field_name)
            if projected_fields is not None:
                projected_fields.append(field_name)
    semantics = repaired.get("control_semantics")
    if isinstance(semantics, dict):
        for field_name in _CONTROL_SEMANTICS_LEAK_FIELDS:
            if field_name not in semantics:
                continue
            leaked_value = semantics.pop(field_name)
            if field_name in _DECISION_FIELDS and field_name not in repaired:
                repaired[field_name] = leaked_value
            if projected_fields is not None:
                projected_fields.append(field_name)
    for stage in (repaired.get("control_semantics") or {}).get("stages") or []:
        if not isinstance(stage, dict):
            continue
        for target in stage.get("targets") or []:
            if not isinstance(target, dict):
                continue
            for field_name in ("app_id", "flow_id", "matched_flow_ids", "resolution_status"):
                target.pop(field_name, None)

    investigation_aliases = {
        "domain_ambiguous": "domain_boundary",
        "identifier_risk": "ue_binding",
        "runtime_evidence_missing": "assurance_gap",
        "execution_feedback_incomplete": "assurance_gap",
        "conflict_signal_present": "cross_domain_consistency",
    }
    allowed_investigations = {
        "domain_boundary",
        "ue_binding",
        "qos_flow_binding",
        "mobility_target_binding",
        "policy_feasibility",
        "cross_domain_consistency",
        "assurance_gap",
    }
    investigations = repaired.get("investigation_targets")
    if isinstance(investigations, list):
        repaired["investigation_targets"] = list(
            dict.fromkeys(
                normalized
                for item in investigations
                if (
                    normalized := investigation_aliases.get(str(item or "").strip(), str(item or "").strip())
                )
                in allowed_investigations
            )
        )

    normalized_result = dict(result)
    normalized_result["structured_response"] = repaired
    return normalized_result


def validate_global_intent(
    intent: GlobalControlIntent,
    *,
    user_input: str,
    round_index: int,
) -> List[str]:
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
        if str(intent.supi or "").strip() != explicit_supi:
            errors.append(f"supi must equal explicit user-provided identifier {explicit_supi}")
    elif len(unique_explicit_supis) > 1 and str(intent.supi or "").strip():
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


def normalize_semantic_target_scope(
    intent: GlobalControlIntent,
    *,
    user_input: str,
) -> None:
    explicit_supis = re.findall(r"(?i)(imsi-\d{5,})", str(user_input or ""))
    unique_explicit_supis = list(dict.fromkeys(explicit_supis))
    default_supi = ""
    if len(unique_explicit_supis) == 1:
        default_supi = unique_explicit_supis[0]
    elif len(unique_explicit_supis) == 0:
        default_supi = str(intent.supi or "").strip()
    if not default_supi:
        return
    for stage in intent.control_semantics.stages or []:
        for target in stage.targets or []:
            if not str(target.supi or "").strip():
                target.supi = default_supi


def _raw_structured_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    structured = result.get("structured_response")
    if isinstance(structured, BaseModel):
        return structured.model_dump(mode="json", exclude_unset=True)
    if isinstance(structured, str):
        try:
            structured = json.loads(structured)
        except json.JSONDecodeError:
            return {}
    return dict(structured) if isinstance(structured, dict) else {}


__all__ = [
    "compile_global_intent",
    "normalize_semantic_target_scope",
    "repair_global_intent_payload",
    "validate_global_intent",
    "validate_global_intent_result",
]
