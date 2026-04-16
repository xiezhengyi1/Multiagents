"""LLM + tool execution loop that ends with a structured response."""

from __future__ import annotations

import inspect
import json
from typing import Any, Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agent_runtime.core.context import AgentRuntimeContext


class ToolLoopExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        output_messages: list[BaseMessage] | None = None,
        structured_response: Any = None,
        failed_tool_call: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.output_messages = list(output_messages or [])
        self.structured_response = structured_response
        self.failed_tool_call = dict(failed_tool_call or {}) if failed_tool_call else None


class StructuredToolLoop:
    """Execute a ReAct-style tool loop until the model returns final structured JSON."""

    def __init__(
        self,
        *,
        llm: ChatOpenAI,
        tools: Iterable[Any],
        system_prompt: str,
        response_model: type[BaseModel],
        max_iterations: int = 8,
    ) -> None:
        self.system_prompt = str(system_prompt or "").strip()
        self.response_model = response_model
        self.max_iterations = max_iterations
        self.tools = list(tools)
        self.tools_by_name: dict[str, Any] = {}
        for tool in self.tools:
            name = str(getattr(tool, "name", "") or getattr(tool, "__name__", "")).strip()
            if not name:
                raise ValueError("tool name is required")
            if name in self.tools_by_name:
                raise ValueError(f"duplicate tool name: {name}")
            self.tools_by_name[name] = tool
        self.llm = llm.bind_tools(self.tools)

    @staticmethod
    def _coerce_message(message: Any) -> BaseMessage:
        if isinstance(message, BaseMessage):
            return message
        if not isinstance(message, dict):
            raise TypeError(f"messages must contain dict or BaseMessage, got {type(message).__name__}")
        role = str(message.get("role") or message.get("type") or "").strip().lower()
        content = message.get("content")
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        if role == "system":
            return SystemMessage(content=text)
        if role in {"assistant", "ai"}:
            return AIMessage(content=text)
        if role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "").strip()
            if not tool_call_id:
                raise ValueError("tool messages require tool_call_id")
            return ToolMessage(
                content=text,
                tool_call_id=tool_call_id,
                name=str(message.get("name") or "").strip() or None,
                status=str(message.get("status") or "success"),
            )
        return HumanMessage(content=text)

    @staticmethod
    def _assistant_text_content(message: AIMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                    continue
                if not isinstance(block, dict):
                    parts.append(str(block))
                    continue
                if block.get("type") in {"text", "output_text"}:
                    parts.append(str(block.get("text") or ""))
            return "".join(parts)
        return str(content)

    @staticmethod
    def _tool_result_to_content(result: Any) -> str:
        if isinstance(result, ToolMessage):
            if isinstance(result.content, str):
                return result.content
            return json.dumps(result.content, ensure_ascii=False)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _build_tool_runtime(context: AgentRuntimeContext | None, tool_call_id: str | None):
        from langchain.tools import ToolRuntime

        return ToolRuntime(
            state={},
            context=context,
            config={},
            stream_writer=lambda *_args, **_kwargs: None,
            tool_call_id=tool_call_id,
            store=None,
        )

    def _invoke_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        context: AgentRuntimeContext | None,
        tool_call_id: str | None,
    ) -> ToolMessage:
        tool = self.tools_by_name.get(tool_name)
        if tool is None:
            raise RuntimeError(f"Model requested unknown tool: {tool_name}")
        func = getattr(tool, "func", None)
        if func is None:
            raise RuntimeError(f"Tool {tool_name} does not expose a callable func")
        kwargs = dict(args)
        if "runtime" in inspect.signature(func).parameters:
            kwargs["runtime"] = self._build_tool_runtime(context, tool_call_id)
        result = func(**kwargs)
        return ToolMessage(
            content=self._tool_result_to_content(result),
            tool_call_id=str(tool_call_id or ""),
            name=tool_name,
            status="success",
        )

    def invoke(
        self,
        payload: dict[str, Any],
        *,
        context: AgentRuntimeContext | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError("structured tool loop requires a non-empty messages list")
        conversation: list[BaseMessage] = [SystemMessage(content=self.system_prompt)]
        conversation.extend(self._coerce_message(message) for message in raw_messages)
        output_messages: list[BaseMessage] = []

        try:
            for _ in range(self.max_iterations):
                ai_message = self.llm.invoke(conversation, **kwargs)
                if not isinstance(ai_message, AIMessage):
                    raise TypeError(f"Expected AIMessage from model, got {type(ai_message).__name__}")
                output_messages.append(ai_message)
                conversation.append(ai_message)
                if ai_message.invalid_tool_calls:
                    raise RuntimeError(f"Model produced invalid tool calls: {ai_message.invalid_tool_calls}")

                tool_calls = getattr(ai_message, "tool_calls", None) or []
                if tool_calls:
                    for tool_call in tool_calls:
                        if not isinstance(tool_call, dict):
                            raise TypeError("tool_call must be a dict")
                        tool_name = str(tool_call.get("name") or "").strip()
                        if not tool_name:
                            raise RuntimeError("tool_call is missing tool name")
                        tool_args = tool_call.get("args")
                        if not isinstance(tool_args, dict):
                            raise RuntimeError(f"tool_call args for {tool_name} must be an object")
                        try:
                            tool_message = self._invoke_tool(
                                tool_name,
                                tool_args,
                                context=context,
                                tool_call_id=str(tool_call.get("id") or ""),
                            )
                        except Exception as exc:
                            raise ToolLoopExecutionError(
                                f"Tool {tool_name} failed: {exc}",
                                output_messages=output_messages,
                                failed_tool_call=tool_call,
                            ) from exc
                        output_messages.append(tool_message)
                        conversation.append(tool_message)
                    continue

                assistant_text = self._assistant_text_content(ai_message).strip()
                if not assistant_text:
                    raise RuntimeError("Model returned empty assistant content without tool calls")
                structured = self.response_model.model_validate_json(assistant_text)
                return {
                    "messages": output_messages,
                    "structured_response": structured,
                }
        except ToolLoopExecutionError:
            raise
        except Exception as exc:
            raise ToolLoopExecutionError(str(exc), output_messages=output_messages) from exc

        raise ToolLoopExecutionError(
            f"Model exceeded max iterations ({self.max_iterations}) without returning final JSON",
            output_messages=output_messages,
        )


__all__ = ["StructuredToolLoop", "ToolLoopExecutionError"]
