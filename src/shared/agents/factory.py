from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from shared.runtime.langchain_trace import AgentRunContext, AtomicTraceMiddleware

ResponseT = TypeVar("ResponseT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class AgentSpec(Generic[ResponseT]):
    name: str
    system_prompt: str
    response_schema: type[ResponseT]
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] = ()


def create_structured_agent(*, model: BaseChatModel, spec: AgentSpec[ResponseT]) -> Any:
    """Create the sole supported model-agent runtime for control and environment graphs."""

    response_tool_name = spec.response_schema.__name__
    system_prompt = (
        f"{spec.system_prompt}\n\n"
        "Structured completion protocol:\n"
        f"- Finish by calling `{response_tool_name}` exactly once.\n"
        "- Observation tools provide evidence; they do not finish the task.\n"
        "- Never call an observation tool again with arguments already used in this invocation.\n"
        "- As soon as the required observations are present, call the structured completion tool."
    )
    return create_agent(
        model=model,
        tools=list(spec.tools),
        system_prompt=system_prompt,
        middleware=[AtomicTraceMiddleware(response_schema=spec.response_schema)],
        response_format=ToolStrategy(spec.response_schema, handle_errors=False),
        context_schema=AgentRunContext,
        name=spec.name,
    )


__all__ = ["AgentSpec", "create_structured_agent"]
