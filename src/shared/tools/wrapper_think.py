from __future__ import annotations

import functools
import inspect
import logging
from typing import Callable

from langchain.tools import tool

logger = logging.getLogger(__name__)


def tool_with_reason(fn: Callable | None = None, *, return_direct: bool = False):
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

        return tool(wrapper, return_direct=return_direct)

    if fn is not None:
        return decorator(fn)
    return decorator


@tool
def think_tool(message: str) -> str:
    """Return a scratchpad message for internal reasoning traces."""
    return message
