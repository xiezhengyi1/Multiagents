from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, Literal

from langchain.tools import tool

logger = logging.getLogger(__name__)


def tool_with_reason(
    *,
    side_effect: Literal["read", "write", "external_action"],
    result_format: Literal["json", "text"],
    output_schema: dict[str, Any] | None = None,
    return_direct: bool = False,
):
    def decorator(fn: Callable):
        original_sig = inspect.signature(fn)
        original_doc = fn.__doc__ or ""

        @functools.wraps(fn)
        def wrapper(reason: str = "", **kwargs):
            if reason:
                logger.info("[ToolReason] %s: %s", fn.__name__, reason)
            return fn(**kwargs)

        wrapper.__annotations__ = {"reason": str, **fn.__annotations__}

        reason_param = inspect.Parameter(
            "reason",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=str,
        )
        new_params = [reason_param] + list(original_sig.parameters.values())
        wrapper.__signature__ = inspect.Signature(
            new_params,
            return_annotation=original_sig.return_annotation,
        )
        wrapper.__doc__ = (
            f"{original_doc}\n\n"
            "Args:\n"
            "    reason: (Required) Explain why you are calling this tool "
            "and what you expect to learn or accomplish."
        )

        structured_tool = tool(wrapper, return_direct=return_direct)
        structured_tool.metadata = {
            **dict(structured_tool.metadata or {}),
            "contract": {
                "side_effect": side_effect,
                "result_format": result_format,
                "output_schema": output_schema,
            },
        }
        return structured_tool

    return decorator


@tool_with_reason(side_effect="read", result_format="text")
def think_tool(message: str) -> str:
    """Return a scratchpad message for internal reasoning traces."""
    return message
