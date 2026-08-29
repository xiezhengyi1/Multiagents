from __future__ import annotations

import re


def strip_retry_feedback(base_prompt: str) -> str:
    cleaned = re.sub(
        r"\n\nRetry feedback \(attempt \d+\).*$",
        "",
        base_prompt,
        flags=re.DOTALL,
    )
    cleaned = re.sub(r"\n\nRetry feedback:.*$", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(
        r"\n\nYour previous draft failed validation.*$",
        "",
        cleaned,
        flags=re.DOTALL,
    )
    return re.sub(
        r"\n\nYour previous attempt failed validation.*$",
        "",
        cleaned,
        flags=re.DOTALL,
    )
