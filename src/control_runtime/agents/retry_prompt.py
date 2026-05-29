"""Structured retry prompt builder shared across Main, IEA, and OSA agents.

Replaces the append-based retry pattern (base_prompt + errors) with a
fixed-size error block that limits context inflation on repeated retries.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List


def _categorize_errors(errors: List[str]) -> Dict[str, List[str]]:
    categories: Dict[str, List[str]] = {
        "schema": [],
        "grounding": [],
        "domain": [],
        "tool": [],
        "other": [],
    }
    for e in errors:
        e_lower = e.lower()
        if any(kw in e_lower for kw in ("bare array", "non-object", "top-level", "schema", "extra input")):
            categories["schema"].append(e)
        elif any(kw in e_lower for kw in ("flow_id", "app_id", "grounded", "binding", "unresolved")):
            categories["grounding"].append(e)
        elif any(kw in e_lower for kw in ("domain", "qos", "mobility", "tool call", "not permitted")):
            categories["domain"].append(e)
        elif any(kw in e_lower for kw in ("max iteration", "converge", "did not converge")):
            categories["tool"].append(e)
        else:
            categories["other"].append(e)
    return {k: v[:2] for k, v in categories.items() if v}


def build_retry_prompt_replacement(
    *,
    original_prompt: str,
    errors: List[str],
    retry_context: Dict[str, int],
) -> str:
    """Build a retry prompt with a fixed-size error block, replacing any prior
    retry feedback instead of appending to it.

    Args:
        original_prompt: The base prompt (without prior retry feedback).
        errors: Validation or invocation error strings.
        retry_context: dict with keys ``attempt`` and ``max_attempts``.
    """
    attempt = retry_context.get("attempt", 1)
    categorized = _categorize_errors(errors)

    error_block = json.dumps({
        "attempt": attempt,
        "max_attempts": retry_context.get("max_attempts", 3),
        "error_count": len(errors),
        "categories": {k: len(v) for k, v in categorized.items()},
        "sample_errors": [e[:200] for e in errors[:3]],
    }, ensure_ascii=False)

    repair_rules = _repair_rules(categorized, errors)

    cleaned = re.sub(
        r'\n\nRetry feedback \(attempt \d+\).*$',
        '',
        original_prompt,
        flags=re.DOTALL,
    )
    cleaned = re.sub(
        r'\n\nYour previous attempt failed validation.*$',
        '',
        cleaned,
        flags=re.DOTALL,
    )

    return (
        f"{cleaned}\n\n"
        f"Retry feedback (attempt {attempt}):\n"
        f"{error_block}\n\n"
        "Fix the following:\n- " + "\n- ".join(repair_rules)
    )


def _repair_rules(categorized: Dict[str, List[str]], errors: List[str]) -> List[str]:
    rules: List[str] = [
        "Return raw JSON only, with no markdown fence and no prose outside the JSON object.",
        "The top-level JSON value must be an object, never a bare array or bare policy item.",
    ]
    joined = " | ".join(errors)
    if categorized.get("schema"):
        rules.append("Remove every unsupported top-level field or extra key not defined in the output schema.")
    if categorized.get("grounding"):
        rules.append("Ensure every referenced flow_id and app_id is grounded by current evidence.")
    if categorized.get("tool"):
        rules.append("Stop extra tool use unless it adds new evidence. Finalize from current evidence.")
    if "infeasible" in joined:
        rules.append("Preserve infeasibility in planning_status. Do not force executable output.")
    if "bare array" in joined or "non-object" in joined:
        rules.append("Wrap all output arrays inside the top-level output object.")
    return rules


__all__ = ["build_retry_prompt_replacement"]
