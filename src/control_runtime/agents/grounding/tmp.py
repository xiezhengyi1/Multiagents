    @staticmethod
    def _build_advisor_prompt(*, evidence: IntentEvidence, context: str) -> str:
        requested_domains = [str(item or "").strip() for item in (evidence.requested_domains or []) if str(item or "").strip()]
        domain_mode = ",".join(requested_domains) or "<empty>"
        qos_required = "qos" in requested_domains
        mobility_only = requested_domains == ["mobility"]
        domain_specific_rules: List[str] = [
            f"- Domain mode for this request: {domain_mode}.",
            "- Final answer must be exactly one raw JSON object with no markdown fence and no surrounding prose.",
            "- `domain_resolution` must be one scalar string value, never an object.",
        ]
        if qos_required:
            domain_specific_rules.extend(
                [
                    "- This request includes QoS grounding. Final JSON must contain a non-empty flows array.",
                    "- Every resolved QoS flow must include grounded app_id and grounded flow_id.",
                    "- If SUPI is known and you plan to return any resolved QoS flow, prefer UE-catalog-backed grounding over a semantic search hit alone.",
                    "- If the current evidence does not already ground the QoS target, keep using SM grounding tools until flows is populated or the target is explicitly unresolved.",
                    "- Do not stop at selected_app_id / selected_flow_id alone; the grounded binding must appear inside flows.",
                ]
            )
            if evidence.candidate_flows:
                domain_specific_rules.extend(
                    [
                        "- Current evidence already contains candidate_flows. Reuse those grounded identifiers directly instead of searching again unless they are ambiguous.",
                        "- If candidate_flows contains a single exact match for the named QoS target and SUPI is known, check whether UE catalog truth is already present before finalizing.",
                        "- If candidate_flows contains a single exact match but no UE flow catalog truth is present yet, call get_sm_ue_flow_catalog once before returning the flow as resolved.",
                        "- Do not leave flows empty when candidate_flows is already non-empty.",
                        "- Do not call search_sm_flow_targets or get_sm_ue_context merely to reconfirm an already unique exact candidate.",
                        "- When SUPI is known, get_sm_ue_flow_catalog is not redundant if it is the missing source of baseline flow truth.",
                    ]
                )
            elif str(evidence.explicit_flow_name or "").strip():
                domain_specific_rules.extend(
                    [
                        f"- No grounded candidate_flows currently exist for the explicit QoS target '{evidence.explicit_flow_name}'.",
                        "- Before final JSON, call search_sm_flow_targets for that explicit flow target.",
                        "- After search returns a grounded exact match, if SUPI is known, fetch the UE flow catalog before returning the flow as resolved.",
                        "- Only finalize immediately after search when the structured evidence already includes the UE-catalog-backed flow truth you need.",
                    ]
                )
            explicit_target_names = [
                str(item.flow_name or "").strip()
                for item in (evidence.explicit_flow_targets or [])
                if str(item.flow_name or "").strip()
            ]
            if len(explicit_target_names) > 1:
                grounded_explicit_target_names = {
                    str(item.flow_name or "").strip()
                    for item in (evidence.candidate_flows or [])
                    if str(item.flow_name or "").strip() in explicit_target_names
                }
                unresolved_explicit_target_names = [
                    item for item in explicit_target_names
                    if item not in grounded_explicit_target_names
                ]
                domain_specific_rules.extend(
                    [
                        "- This request names multiple QoS flow targets.",
                        f"- Explicit QoS targets in this request: {json.dumps(explicit_target_names, ensure_ascii=False)}.",
                        "- Every resolved flow in `flows` must correspond to one of those explicit targets and be grounded by catalog/search evidence for that exact target.",
                        "- When a resolved flow corresponds to an explicit target, keep `flows[].name` equal to that explicit flow name.",
                        "- If candidate_flows does not already cover all explicit targets, search unresolved explicit targets individually before finalizing.",
                        "- If any explicit target remains ungrounded, do not substitute a nearby flow name.",
                    ]
                )
                if grounded_explicit_target_names and unresolved_explicit_target_names:
                    domain_specific_rules.extend(
                        [
                            f"- Evidence already grounds these explicit QoS targets: {json.dumps(sorted(grounded_explicit_target_names), ensure_ascii=False)}.",
                            f"- These explicit QoS targets are still unresolved in current evidence: {json.dumps(unresolved_explicit_target_names, ensure_ascii=False)}.",
                            "- The next answer must return a mixed flows array: resolved entries for grounded explicit targets, plus unresolved entries for still-unresolved explicit targets.",
                            "- Never leave flows empty when at least one explicit target is already grounded.",
                        ]
                    )
        if mobility_only:
            domain_specific_rules.extend(
                [
                    "- This is mobility-only grounding. Final JSON must keep flows empty.",
                    "- Do not call any SM grounding tool: search_sm_flow_targets, get_sm_ue_context, or get_sm_ue_flow_catalog.",
                    "- Use only AM grounding if more evidence is needed.",
                ]
            )
        return (
            "User request:\n"
            f"{evidence.user_input}\n\n"
            "Structured evidence:\n"
            f"{json.dumps(evidence.model_dump(mode='json'), ensure_ascii=False)}\n\n"
            "Coordinator context:\n"
            f"{context or 'N/A'}\n\n"
            "Task:\n"
            "- Resolve only the semantic choices that remain ambiguous.\n"
            "- Use tools only when the structured evidence does not already ground the required target.\n"
            "- You may revise Main's requested domain boundary when grounding evidence proves it is too narrow, too wide, or cannot be confirmed.\n"
            "- If you revise the domain boundary, populate grounded_requested_domains, domain_resolution, domain_revision_needed, and domain_revision_rationale explicitly.\n"
            "- For every QoS flow with resolution_status='resolved', include grounded flow_id and app_id in the final JSON.\n"
            "- If a QoS target is not fully grounded to flow_id + app_id, do not mark it resolved.\n"
            "- If the structured evidence already contains the grounded answer, finalize from that evidence without extra tool calls.\n"
            "- For QoS resolved flows, treat UE-specific flow catalog truth as the preferred final source of flow baseline fields whenever SUPI is known.\n"
            f"{chr(10).join(domain_specific_rules)}\n"
            "- Return one IntentAdvisorDecision JSON object only."
        )

    @staticmethod
    def _build_validation_retry_prompt(
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
                    "If SUPI is known and the grounded QoS candidate is only a semantic match, fetch get_sm_ue_flow_catalog before returning it as resolved.",
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
        return (
            f"{base_prompt}\n\n"
            "Your previous attempt failed validation.\n"
            "Validation errors:\n- "
            + "\n- ".join(issues)
            + "\n\n"
            "Re-ground the semantic target or correct tool usage before returning the next answer.\n"
            "Do not return any resolved QoS flow unless both flow_id and app_id are present and grounded.\n"
            + "\n".join(repair_rules)
        )
