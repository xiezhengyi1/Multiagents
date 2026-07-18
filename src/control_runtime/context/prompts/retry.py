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
            "supi, round_strategy, next_agent, requested_domains, serving_slice_change_requested, domain_evidence, "
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
            "degrade/throttle/constrain must be rewritten as deprioritize; optimize/improve/boost must be rewritten as protect.\n"
            "The literal value optimize is forbidden in control_semantics.targets[].goal; retain it only in note or routing_rationale.\n"
            "control_semantics.stages[].trigger must be one of initial, on_previous_failure, after_previous_stage, retry.\n"
            "For staged priority, stage_index=1 uses initial; stage_index>1 uses after_previous_stage.\n"
            "GlobalControlIntent control_semantics targets contain only semantic_name, target_type, goal, metric_focus, note, and supi; IEA owns resolved identifiers.\n"
            "GlobalControlIntent control_semantics is Main-owned; IEA never outputs it.\n"
            "For multi-SUPI requests, leave top-level supi empty and set each target.supi separately; never comma-join SUPIs.\n"
            "For exactly one explicit imsi-... in the user request, top-level supi and every semantic target.supi must equal that exact identifier; never leave either empty.\n"
            "Set serving_slice_change_requested=true only for a structured serving-S-NSSAI change and then include exactly qos and mobility; otherwise set it false.\n"
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
            "Return one corrected raw IEA decision JSON object only.",
            "Do not guess identifiers. Runtime materializes catalog SLA, provenance, and subscription-entitlement facts after parsing; select only catalog-grounded flow identities and IEA-owned decisions.",
            "Return raw JSON only, with no markdown fence and no prose outside the JSON object.",
            "`open_questions` must be an array of objects with owner_agent, question, blocking, and related_domains; never emit question strings.",
            "For unresolved targets use app_id='', flow_id='', and resolution_status='unresolved'.",
        ]
        joined = " | ".join(issues)

        # Tool-limit errors: tell the LLM to use cached data instead of re-querying.
        if "exceeded max_calls_per_tool" in joined or "max iterations" in joined.lower():
            repair_rules.extend(
                [
                    "CRITICAL: Your previous attempt collected tool results but exceeded the call limit before finalizing.",
                    "The evidence payload below already includes the data collected by the failed attempt.",
                    "Do NOT call get_sm_ue_flow_catalog again — use the catalog data already embedded in the evidence.",
                    "Do NOT call any tool you already called in the previous attempt — the results are in the evidence.",
                    "Finalize immediately from the embedded evidence. If a target is still ungrounded, mark it unresolved.",
                    "Use at most 1 additional tool call if absolutely required; then finalize.",
                ]
            )
        if "mobility_intent" in joined and "valid dictionary" in joined:
            repair_rules.append(
                "Set mobility_intent to {} when no mobility fact is needed; never use null, a string, or prose."
            )
        if "qos_operation_constraints" in joined and (
            "valid list" in joined
            or "valid dictionary" in joined
            or "input_type=str" in joined
        ):
            repair_rules.append(
                "qos_operation_constraints must be a JSON array: use [] or [{...}], never a single object, quoted JSON string, or prose. Every array element is a JSON object."
            )
        if "target_slice_preference" in joined and "valid string" in joined:
            repair_rules.append(
                "Use target_slice_preference=\"\" when there is no preference; never use null."
            )
        if "QoS GroundingDecision must include grounded target flows." in joined:
            repair_rules.extend(
                [
                    "This retry is specifically failing because your previous JSON omitted flows.",
                    "For the next answer, flows must be non-empty even when migration is blocked or the target is unresolved.",
                    "If you already have a grounded QoS candidate in evidence, return its compact resolved binding and finalize.",
                    "If only some explicit QoS targets are grounded, return resolved entries for those grounded targets and unresolved entries for the remaining explicit targets.",
                    "If you still do not have a grounded QoS candidate, do not return an empty object; return an explicitly unresolved flow entry: {\"app_id\":\"\",\"flow_id\":\"\",\"name\":\"<target>\",\"resolution_status\":\"unresolved\"}. Runtime attaches the Main-owned SUPI.",
                    "Do not spend another tool call to reconfirm a single exact candidate that is already grounded in evidence.",
                ]
            )
        if "requires get_sm_ue_flow_catalog" in joined or "catalog evidence" in joined:
            repair_rules.extend(
                [
                    "This QoS request has a known SUPI but no authoritative UE flow catalog in the visible evidence.",
                    "Call get_sm_ue_flow_catalog for that SUPI before finalizing. Use its result as the only authority for a resolved flow binding and SLA baseline.",
                    "If that catalog is empty or lacks the named target, preserve the target as an unresolved flows entry instead of omitting it.",
                ]
            )
        if (
            "slice_migration_authorization." in joined
            or "slice migration requires an explicit slice_migration_authorization decision" in joined
            or "slice migration authorization must preserve authorized S-NSSAI evidence" in joined
        ):
            repair_rules.extend(
                [
                    "Your raw `slice_migration_authorization` contains only decision, target_snssais, subscription_change_required, and rationale.",
                    "Runtime injects authority and authorized_snssais from read-only entitlement evidence; never output either field.",
                    "Choose exactly one decision: not_applicable, migration_authorized, blocked_by_subscription_entitlement, blocked_requires_subscription_provisioning, or evidence_missing. Never write migration_blocked.",
                    "Keep open_questions at the top level. Keep source-slice and target-slice preference only in qos_operation_constraints[].",
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
                    "For each explicitly named QoS flow, including a do-not-change or do-not-switch name, include exactly one flows entry.",
                    "For an excluded name, use the audit-only entry {\"name\":\"<excluded name>\",\"resolution_status\":\"unresolved\"}.",
                    "For each active name, either ground it via catalog/search evidence or leave it unresolved.",
                    "When a flow is resolved, set `flows[].name` to the explicit flow name that the resolved binding satisfies.",
                    "Do not return a resolved flow binding for any name that is missing from catalog/search evidence.",
                ]
            )
        if "mobility-only intent must not call SM grounding tools" in joined:
            repair_rules.extend(
                [
                    "This retry is mobility-only.",
                    "Do not call search_sm_flow_targets or get_sm_ue_flow_catalog.",
                ]
            )
        if "QoS-only intent must not call AM grounding tools" in joined:
            repair_rules.extend(
                [
                    "This retry is QoS-only.",
                    "Do not call get_am_policy_context or search_am_policy_targets.",
                ]
            )
        if (
            "missing QoS baseline values" in joined
            or "incomplete QoS baseline" in joined
            or "incomplete QoS flow selector" in joined
        ):
            repair_rules.extend(
                [
                    "Your resolved flow is not backed by the authoritative UE flow catalog.",
                    "Call get_sm_ue_flow_catalog if needed, then resolve only a returned flow_id; code will fill its SLA and envelope facts.",
                    "If no catalog flow matches the user's intent, return that target as unresolved instead of forcing a resolution.",
                ]
            )
        if "subscription provisioning" in joined or "unverified slice migration is blocked" in joined:
            repair_rules.extend(
                [
                    "The user did not explicitly request a subscription add/change. Do NOT request provisioning and do NOT set subscription_change_required=true.",
                    "Keep the requested target migration blocked with decision=\"blocked_by_subscription_entitlement\", target_snssais=[], subscription_change_required=false, and a rationale that the entitlement is absent. Runtime retains the observed authority and authorized S-NSSAIs.",
                    "Do not invent a target S-NSSAI or an AM/SM/URSP policy. Preserve only the grounded QoS intent for entitlement-limited best-effort planning.",
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
