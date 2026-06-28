COMMON_KNOWLEDGE_SEARCH_SKILL = """
Knowledge-base search skill:
- Use knowledge tools only when all three conditions are true:
  1. one required field or exact semantic fact is still missing after local runtime evidence,
  2. the missing fact is an exact external 3GPP object, clause, schema field, or enumerated token,
  3. the missing fact is necessary to finish this round's required JSON output.
- Before any knowledge-tool call, identify the single missing fact you are trying to resolve. One knowledge query must target one concrete unresolved fact.
- Prefer `get_knowledge_by_key` when the exact 3GPP object name or key is already known.
- Use `search_semantic_knowledge` only when the exact key is not yet known and you need the narrowest standards-level search to locate it.
- If a knowledge search does not resolve the fact exactly, stop and return an explicit unresolved / partial result. Do not guess, substitute a nearby concept, or continue searching for reassurance.

Never use knowledge tools:
- to decorate an answer with standards language,
- to reconfirm a field already grounded by runtime evidence,
- to infer local scenario semantics, local slice meaning, local app intent, or local business labels,
- to compensate for missing runtime bindings that should come from UE context, policy context, catalog search, or optimizer evidence.
"""


IEA_KNOWLEDGE_SEARCH_SKILL = (
    COMMON_KNOWLEDGE_SEARCH_SKILL
    + """
IEA-specific knowledge-search rules:
- Search only for exact 3GPP semantic ambiguity that runtime grounding cannot answer.
- If the unresolved fact is a local target binding, keep using runtime grounding tools or return the target as unresolved.
- Never use knowledge tools to decide what a local app, flow, slice label, service_type_id, or business phrase should mean.
- If runtime grounding cannot uniquely bind the named target, return unresolved rather than using the knowledge base to force an interpretation.
"""
)


OSA_KNOWLEDGE_SEARCH_SKILL = (
    COMMON_KNOWLEDGE_SEARCH_SKILL
    + """
OSA-specific knowledge-search rules:
- Knowledge tools never replace `preview_qos_optimizer`, `fetch_qos_network_status`, or `inspect_mobility_ue_policies`.
- Search only when one required executable field still depends on an exact external 3GPP object or enumerated token that local runtime evidence does not provide.
- If an executable field is blocked by missing optimizer or mobility evidence, return `partial_plan` or `needs_upstream_reground` instead of searching for a standards-side substitute.
- Never use knowledge tools to validate, justify, or soften values that are already fixed by optimizer output, live UE context, or local schema.
- You have at most 2 calls to `get_knowledge_by_key` per round. If the first call does not resolve the missing fact exactly, assume the second will not either — finalize from current evidence. Do not burn both calls without new evidence to guide the second query.
"""
)


__all__ = ["IEA_KNOWLEDGE_SEARCH_SKILL", "OSA_KNOWLEDGE_SEARCH_SKILL"]
