from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, JsonValue, TypeAdapter

from shared.contracts.common import utc_now
from shared.contracts.model_trace import (
    ModelMessage,
    ModelToolCall,
    ModelToolSpec,
    TokenUsage,
)

from .journal import EpisodeJournal

_JSON_VALUE = TypeAdapter(JsonValue)
_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


class ToolObservationSink(Protocol):
    @property
    def event_ids(self) -> tuple[str, ...]: ...

    def record(
        self, *, query_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class AgentRunContext:
    journal: EpisodeJournal
    trigger_event_id: str
    round_index: int
    agent_name: str
    model_name: str
    # Runtime-only service. Keeping it opaque prevents LangChain from trying
    # to publish the service protocol as part of a model-visible tool schema.
    observation_sink: object | None = None


def normalize_langchain_message(message: BaseMessage) -> ModelMessage:
    content = _JSON_VALUE.validate_python(message.content)
    if isinstance(message, SystemMessage):
        return ModelMessage(role="system", content=content)
    if isinstance(message, HumanMessage):
        return ModelMessage(role="user", content=content)
    if isinstance(message, ToolMessage):
        return ModelMessage(
            role="tool",
            content=content,
            tool_call_id=str(message.tool_call_id),
        )
    if isinstance(message, AIMessage):
        calls = [
            ModelToolCall(
                tool_call_id=str(call["id"]) if call.get("id") is not None else None,
                name=str(call["name"]) if call.get("name") is not None else None,
                arguments=_JSON_VALUE.validate_python(call.get("args")),
            )
            for call in [*message.tool_calls, *message.invalid_tool_calls]
        ]
        return ModelMessage(role="assistant", content=content, tool_calls=calls)
    raise TypeError(f"unsupported LangChain message type: {type(message).__name__}")


def snapshot_langchain_tool(tool: BaseTool | dict[str, Any]) -> ModelToolSpec:
    if isinstance(tool, BaseTool):
        return ModelToolSpec(
            name=tool.name,
            description=tool.description or "",
            input_schema=_JSON_OBJECT.validate_python(
                tool.tool_call_schema.model_json_schema()
            ),
        )
    function = tool.get("function") if tool.get("type") == "function" else tool
    if not isinstance(function, dict):
        raise TypeError("LangChain tool dictionary must contain an object definition")
    name = function.get("name")
    description = function.get("description", "")
    input_schema = function.get("parameters", function.get("input_schema"))
    if not isinstance(name, str) or not name.strip():
        raise ValueError("LangChain tool dictionary requires name")
    if not isinstance(description, str):
        raise TypeError(f"tool {name} description must be a string")
    if not isinstance(input_schema, dict):
        raise TypeError(f"tool {name} requires an object input schema")
    return ModelToolSpec(
        name=name,
        description=description,
        input_schema=_JSON_OBJECT.validate_python(input_schema),
    )


def _snapshot_runtime_tools(
    tools: list[BaseTool | dict[str, Any]],
    response_schema: type[BaseModel] | None,
) -> list[ModelToolSpec]:
    response_tool_name = response_schema.__name__ if response_schema is not None else None
    return [
        snapshot
        for tool in tools
        if (snapshot := snapshot_langchain_tool(tool)).name != response_tool_name
    ]


def _response_message(
    response: ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any],
) -> AIMessage:
    if isinstance(response, AIMessage):
        return response
    model_response = (
        response.model_response
        if isinstance(response, ExtendedModelResponse)
        else response
    )
    assistant_messages = [
        message for message in model_response.result if isinstance(message, AIMessage)
    ]
    if len(assistant_messages) != 1:
        raise ValueError("one model request must produce exactly one assistant message")
    return assistant_messages[0]


def _token_usage(message: AIMessage) -> TokenUsage | None:
    usage = message.usage_metadata
    if usage is None:
        return None
    return TokenUsage(
        input_tokens=int(usage["input_tokens"]),
        output_tokens=int(usage["output_tokens"]),
    )


def _normalize_output_message(
    message: AIMessage,
    response_schema: type[BaseModel] | None,
) -> ModelMessage:
    normalized = normalize_langchain_message(message)
    if response_schema is None:
        return normalized
    response_tool_name = response_schema.__name__
    response_calls = [
        call for call in normalized.tool_calls if call.name == response_tool_name
    ]
    if not response_calls:
        return normalized
    if len(response_calls) != 1:
        raise ValueError("structured response must contain one response tool call")
    return ModelMessage(
        role="assistant",
        content=response_calls[0].arguments,
        tool_calls=[
            call for call in normalized.tool_calls if call.name != response_tool_name
        ],
    )


class AtomicTraceMiddleware(AgentMiddleware[Any, AgentRunContext]):
    """Persist the exact input and output of every underlying model request."""

    def __init__(self, *, response_schema: type[BaseModel] | None = None) -> None:
        self.response_schema = response_schema

    def wrap_model_call(
        self,
        request: ModelRequest[AgentRunContext],
        handler: Any,
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        if request.runtime is None or not isinstance(
            request.runtime.context, AgentRunContext
        ):
            raise TypeError("AtomicTraceMiddleware requires AgentRunContext")
        context = request.runtime.context
        messages = [
            *([request.system_message] if request.system_message is not None else []),
            *request.messages,
        ]
        normalized_input = [
            normalize_langchain_message(message) for message in messages
        ]
        tools = _snapshot_runtime_tools(request.tools, self.response_schema)
        started_at = utc_now()
        try:
            response = handler(request)
            assistant_message = _response_message(response)
        except Exception as exc:
            failed_message = getattr(exc, "ai_message", None)
            normalized_output = (
                _normalize_output_message(failed_message, self.response_schema)
                if isinstance(failed_message, AIMessage)
                else None
            )
            context.journal.append_trace(
                trigger_event_id=context.trigger_event_id,
                round_index=context.round_index,
                agent_name=context.agent_name,
                model_name=context.model_name,
                input_messages=normalized_input,
                output_message=normalized_output,
                tools=tools,
                status="error",
                error=str(exc),
                started_at=started_at,
                ended_at=utc_now(),
            )
            raise
        context.journal.append_trace(
            trigger_event_id=context.trigger_event_id,
            round_index=context.round_index,
            agent_name=context.agent_name,
            model_name=context.model_name,
            input_messages=normalized_input,
            output_message=_normalize_output_message(
                assistant_message, self.response_schema
            ),
            tools=tools,
            status="success",
            usage=_token_usage(assistant_message),
            started_at=started_at,
            ended_at=utc_now(),
        )
        return response


__all__ = [
    "AgentRunContext",
    "AtomicTraceMiddleware",
    "ToolObservationSink",
    "normalize_langchain_message",
    "snapshot_langchain_tool",
]
