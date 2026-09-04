from __future__ import annotations


def replace_contract_clauses(
    original_prompt: str,
    replacements: tuple[tuple[str, str], ...],
) -> str:
    """Replace only explicitly identified data-contract clauses in a prompt."""

    prompt = original_prompt
    for original_clause, replacement_clause in replacements:
        if prompt.count(original_clause) != 1:
            raise ValueError(
                "original prompt contract clause must occur exactly once: "
                f"{original_clause!r}"
            )
        prompt = prompt.replace(original_clause, replacement_clause)
    return prompt


__all__ = ["replace_contract_clauses"]
