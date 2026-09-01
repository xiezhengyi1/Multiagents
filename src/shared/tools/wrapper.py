from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, get_args

from langchain.tools import tool

from shared.contracts import ToolPurpose


logger = logging.getLogger(__name__)
_TOOL_PURPOSES = frozenset(get_args(ToolPurpose))


def tool_with_purpose(
    *,
    side_effect: str,
    result_format: str,
    output_schema: dict[str, Any] | None = None,
    return_direct: bool = False,
):
    """Expose a tool with a short, typed, trace-visible invocation purpose."""

    def decorator(fn: Callable):
        original_sig = inspect.signature(fn)
        original_doc = fn.__doc__ or ""

        @functools.wraps(fn)
        def wrapper(purpose: ToolPurpose, **kwargs):
            if purpose not in _TOOL_PURPOSES:
                raise ValueError(f"unsupported tool purpose: {purpose}")
            logger.info("[ToolPurpose] %s: %s", fn.__name__, purpose)
            return fn(**kwargs)

        wrapper.__annotations__ = {"purpose": ToolPurpose, **fn.__annotations__}
        purpose_param = inspect.Parameter(
            "purpose",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=ToolPurpose,
        )
        wrapper.__signature__ = inspect.Signature(
            [purpose_param, *original_sig.parameters.values()],
            return_annotation=original_sig.return_annotation,
        )
        wrapper.__doc__ = (
            f"{original_doc}\n\n"
            "Args:\n"
            "    purpose: Required typed purpose for this invocation; it is recorded as audit metadata."
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


__all__ = ["tool_with_purpose"]
