"""统一工具包装器 — 为每个 @tool 注入 reason 参数，强制 Agent 先说明调用理由。"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Callable

from langchain.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 核心包装器
# ---------------------------------------------------------------------------

def tool_with_reason(fn: Callable | None = None, *, return_direct: bool = False):
    """
    统一包装器：在 @tool 基础上为每个工具注入 ``reason: str`` 参数，
    强制 Agent 在调用任何工具前先说明调用理由。

    用法与 ``@tool`` 完全一致::

        @tool_with_reason
        def my_tool(x: int) -> str:
            ...

        @tool_with_reason(return_direct=True)
        def my_direct_tool(x: int) -> str:
            ...
    """

    def decorator(fn: Callable):
        original_sig = inspect.signature(fn)
        original_doc = fn.__doc__ or ""

        @functools.wraps(fn)
        def wrapper(reason: str = "", **kwargs):
            # 中文标注：记录调用理由，然后转发到原始工具
            if reason:
                logger.info("[ToolReason] %s: %s", fn.__name__, reason)
            return fn(**kwargs)

        # 中文标注：functools.wraps 会用原始函数的 __annotations__ 覆盖 wrapper 的，
        # 导致 pydantic 找不到 reason 的类型标注，需要手动补回。
        wrapper.__annotations__ = {"reason": str, **fn.__annotations__}

        # 中文标注：构造新签名 — reason 放在第一个位置（无默认值 → schema 中为 required）
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


# ---------------------------------------------------------------------------
# think_tool — 保留向后兼容
# ---------------------------------------------------------------------------

@tool
def think_tool(message: str) -> str:
    """Record visible reasoning before any non-think tool call."""
    return message