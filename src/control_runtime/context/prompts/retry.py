from __future__ import annotations

import re
from typing import Any, Dict, List


class RetryPromptBuilder:
    """Unified retry prompt generation for agent validation failures."""

    def build_main(
        self,
        *,
        base_prompt: str,
        validation_errors: List[str],
        invocation_error: str,
    ) -> str:
        issues: List[str] = []
        if invocation_error:
            issues.append(invocation_error)
        if validation_errors:
            issues.extend(validation_errors)

        cleaned = _strip_retry_feedback(base_prompt)
        required_keys = (
            "supi, round_strategy, next_agent, requested_domains, domain_evidence, "
            "control_semantics, objective_profile, investigation_targets, uncertainty_flags, "
            "retry_scope, required_evidence, forbidden_assumptions, intent_encoding_guidance, "
            "routing_decision, routing_rationale, reuse_contract"
        )

        return (
            f"{cleaned}\n\n"
            "Retry feedback:\n"
            "Correct the output now.\n"
            "Return exactly one GlobalControlIntent JSON object.\n"
            "Return the GlobalControlIntent fields directly at the top level.\n"
            "Do not emit runtime wrapper keys like messages or structured_response.\n"
            "Top-level output must be an object, never a list.\n"
            "Do not return markdown, prose, bullets, arrays, or partial sub-objects.\n"
            f"Required GlobalControlIntent keys: {required_keys}.\n"
            "control_semantics.targets[].target_type must be one of flow, app, scope, named_object.\n"
            "UE/user-equipment/AM-policy targets must use scope; never use target_type=ue.\n"
            "control_semantics.targets[].goal must be one of protect, deprioritize, defer, observe.\n"
            "degrade/throttle/constrain must be rewritten as deprioritize.\n"
            "control_semantics.stages[].trigger must be one of initial, on_previous_failure, after_previous_stage, retry.\n"
            "For staged priority, stage_index=1 uses initial; stage_index>1 uses after_previous_stage.\n"
            "Do not populate control_semantics target app_id, flow_id, matched_flow_ids, or matched_app_ids; IEA owns resolved identifiers.\n"
            "For multi-SUPI requests, leave top-level supi empty and set each target.supi separately; never comma-join SUPIs.\n"
            "If validation says execution_failure with stable bindings, route to optimization_strategy, set round_strategy=policy_revision, retry_scope=target_stable, and set reuse_contract.allowed=true.\n"
            "If a previous draft was a list, convert the intended route into the top-level GlobalControlIntent object.\n"
            "If a previous draft only contained reuse_contract fields, keep those fields nested under reuse_contract and add all missing routing fields.\n\n"
            "Validation errors:\n- " + "\n- ".join(issues)
        )

    def build_grounding(
        self,
        *,
        base_prompt: str,
        advisor_validation_errors: List[str],
        grounding_validation_errors: List[str],
        invocation_error: str,
    ) -> str:
        issues: List[str] = []
        if invocation_error:
            issues.append(invocation_error)
        if advisor_validation_errors:
            issues.extend(advisor_validation_errors)
        if grounding_validation_errors:
            issues.extend(grounding_validation_errors)
        repair_rules: List[str] = [
            "Return one corrected IntentAdvisorDecision JSON object only.",
            "Do not guess missing identifiers, and do not rely on downstream compilation to fill them.",
            "Return raw JSON only, with no markdown fence and no prose outside the JSON object.",
            "`domain_resolution` must be a scalar string, not an object.",
        ]
        joined = " | ".join(issues)
        if "QoS advisor decision must include grounded target flows." in joined:
            repair_rules.extend(
                [
                    "This retry is specifically failing because your previous JSON omitted flows.",
                    "For the next answer, flows must be non-empty.",
                    "If you already have a grounded QoS candidate in evidence, copy it into flows and finalize.",
                    "If only some explicit QoS targets are grounded, return resolved entries for those grounded targets and unresolved entries for the remaining explicit targets.",
                    "If you still do not have a grounded QoS candidate, do not return an empty object; call the required SM grounding tool and then return either a resolved or explicitly unresolved flow entry.",
                    "Do not spend another tool call to reconfirm a single exact candidate that is already grounded in evidence.",
                ]
            )
        if "domain_resolution must be confirmed, narrowed, widened, or cannot_confirm" in joined:
            repair_rules.extend(
                [
                    "Set `domain_resolution` to exactly one of: confirmed, narrowed, widened, cannot_confirm.",
                    "Do not output a nested object under `domain_resolution`.",
                ]
            )
        if "cannot_confirm domain resolution requires domain_revision_rationale" in joined:
            repair_rules.extend(
                [
                    "If you set `domain_resolution` to `cannot_confirm`, you must include a non-empty `domain_revision_rationale`.",
                    "If you can confirm the domain boundary from evidence, use `confirmed` instead.",
                ]
            )
        if (
            "explicitly named QoS flow '" in joined
            and (
                "was not grounded by catalog/search evidence" in joined
                or "must appear in advisor decision flows as resolved or unresolved" in joined
            )
        ):
            repair_rules.extend(
                [
                    "For each explicitly named QoS flow, either ground it via catalog/search evidence or leave it unresolved.",
                    "When a flow is resolved, set `flows[].name` to the explicit flow name that the resolved binding satisfies.",
                    "Do not return a resolved flow binding for any name that is missing from catalog/search evidence.",
                ]
            )
        if "mobility-only intent must not call SM grounding tools" in joined:
            repair_rules.extend(
                [
                    "This retry is mobility-only.",
                    "Do not call search_sm_flow_targets, get_sm_ue_context, or get_sm_ue_flow_catalog.",
                ]
            )
        if "QoS-only intent must not call AM grounding tools" in joined:
            repair_rules.extend(
                [
                    "This retry is QoS-only.",
                    "Do not call get_am_policy_context or search_am_policy_targets.",
                ]
            )

        cleaned = _strip_retry_feedback(base_prompt)
        return (
            f"{cleaned}\n\n"
            "Retry feedback:\n"
            + "\n".join(f"- {rule}" for rule in repair_rules)
            + "\n\nValidation errors:\n- "
            + "\n- ".join(issues)
        )

    def build_osa(
        self,
        *,
        base_prompt: str,
        issues: list[str],
        cached_planning_evidence: Dict[str, Any] | None = None,
    ) -> str:
        # The OSA retry contract is still kept in the legacy module because its
        # examples are scanned directly by existing tests. This facade gives
        # callers a single import path while preserving that contract.
        from .planning import build_validation_retry_prompt

        return build_validation_retry_prompt(
            base_prompt=base_prompt,
            issues=issues,
            cached_planning_evidence=cached_planning_evidence,
        )


def _strip_retry_feedback(base_prompt: str) -> str:
    cleaned = re.sub(
        r"\n\nRetry feedback \(attempt \d+\).*$",
        "",
        base_prompt,
        flags=re.DOTALL,
    )
    cleaned = re.sub(
        r"\n\nRetry feedback:.*$",
        "",
        cleaned,
        flags=re.DOTALL,
    )
    cleaned = re.sub(
        r"\n\nYour previous draft failed validation.*$",
        "",
        cleaned,
        flags=re.DOTALL,
    )
    cleaned = re.sub(
        r"\n\nYour previous attempt failed validation.*$",
        "",
        cleaned,
        flags=re.DOTALL,
    )
    return cleaned
