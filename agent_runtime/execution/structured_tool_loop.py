"""LLM + tool execution loop that ends with a structured response."""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Iterable, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agent_runtime.core.context import AgentRuntimeContext
from agent_runtime.core.token_budget import TokenCounter, TokenBudget


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
        tool_error_mode: str = "raise",
        max_calls_per_tool: int | None = None,
        forbid_duplicate_tool_calls: bool = False,
        max_tool_result_chars: int = 8000,
        tool_result_limits: dict[str, int] | None = None,
        max_tool_result_tokens: int | None = None,
        tool_result_token_limits: dict[str, int] | None = None,
        token_counter: Optional[TokenCounter] = None,
        token_budget: Optional[TokenBudget] = None,
    ) -> None:
        self.system_prompt = str(system_prompt or "").strip()
        self.response_model = response_model
        self.max_iterations = max_iterations
        normalized_tool_error_mode = str(tool_error_mode or "raise").strip().lower()
        if normalized_tool_error_mode not in {"raise", "return"}:
            raise ValueError("tool_error_mode must be either 'raise' or 'return'")
        self.tool_error_mode = normalized_tool_error_mode
        self.max_calls_per_tool = max_calls_per_tool if max_calls_per_tool is None else int(max_calls_per_tool)
        if self.max_calls_per_tool is not None and self.max_calls_per_tool < 1:
            raise ValueError("max_calls_per_tool must be >= 1")
        self.forbid_duplicate_tool_calls = bool(forbid_duplicate_tool_calls)
        self.max_tool_result_chars = max(1000, int(max_tool_result_chars))
        self._tool_result_limits = dict(tool_result_limits or {})
        self.max_tool_result_tokens = max_tool_result_tokens
        self._tool_result_token_limits = dict(tool_result_token_limits or {})
        self._token_counter = token_counter
        self._token_budget = token_budget
        self.tools = list(tools)
        self.tools_by_name: dict[str, Any] = {}
        for tool in self.tools:
            name = str(getattr(tool, "name", "") or getattr(tool, "__name__", "")).strip()
            if not name:
                raise ValueError("tool name is required")
            if name in self.tools_by_name:
                raise ValueError(f"duplicate tool name: {name}")
            self.tools_by_name[name] = tool
        self.llm = llm.bind_tools(self.tools) if self.tools else llm

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
            return AIMessage(
                content=text,
                additional_kwargs=dict(message.get("additional_kwargs") or {}),
                tool_calls=list(message.get("tool_calls") or []),
                invalid_tool_calls=list(message.get("invalid_tool_calls") or []),
                response_metadata=dict(message.get("response_metadata") or {}),
                id=message.get("id"),
                name=str(message.get("name") or "").strip() or None,
            )
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

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = str(text or "").strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if not lines:
            return stripped
        if lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @staticmethod
    def _wrapper_keys_for_model(response_model: type[BaseModel]) -> list[str]:
        model_name = str(getattr(response_model, "__name__", "") or "").strip()
        snake_name = re.sub(r"(?<!^)(?=[A-Z])", "_", model_name).lower()
        return [
            model_name,
            snake_name,
            "intent_decision",
            "advisor_output",
            "response",
            "result",
            "output",
            "data",
        ]

    def _validate_loaded_payload(self, payload: Any) -> BaseModel:
        try:
            return self.response_model.model_validate(payload)
        except Exception as direct_error:
            if isinstance(payload, dict):
                for key in self._wrapper_keys_for_model(self.response_model):
                    if key not in payload:
                        continue
                    inner = payload.get(key)
                    if not isinstance(inner, (dict, list, str)):
                        continue
                    try:
                        if isinstance(inner, str):
                            return self.response_model.model_validate_json(inner)
                        return self.response_model.model_validate(inner)
                    except Exception:
                        continue
                if len(payload) == 1:
                    inner = next(iter(payload.values()))
                    if isinstance(inner, str):
                        return self.response_model.model_validate_json(inner)
                    if isinstance(inner, (dict, list)):
                        return self.response_model.model_validate(inner)
            raise direct_error

    def _parse_structured_response(self, assistant_text: str) -> BaseModel:
        candidates = [str(assistant_text or "").strip()]
        stripped_fence = self._strip_code_fence(assistant_text)
        if stripped_fence and stripped_fence not in candidates:
            candidates.append(stripped_fence)

        decoder = json.JSONDecoder()
        last_error: Exception | None = None
        for candidate in candidates:
            if not candidate:
                continue
            try:
                return self.response_model.model_validate_json(candidate)
            except Exception as exc:
                last_error = exc
            try:
                return self._validate_loaded_payload(json.loads(candidate))
            except Exception as exc:
                last_error = exc
            for index, ch in enumerate(candidate):
                if ch not in "[{":
                    continue
                try:
                    payload, _end = decoder.raw_decode(candidate[index:])
                    return self._validate_loaded_payload(payload)
                except Exception as exc:
                    last_error = exc
                    continue

        if last_error is not None:
            raise last_error
        raise RuntimeError("Model returned empty structured response")

    def _max_tool_result_chars_for(self, tool_name: str) -> int:
        return self._tool_result_limits.get(tool_name, self.max_tool_result_chars)

    def _max_tool_result_tokens_for(self, tool_name: str) -> int:
        per_tool = self._tool_result_token_limits.get(tool_name)
        if per_tool is not None:
            return per_tool
        if self.max_tool_result_tokens is not None:
            return self.max_tool_result_tokens
        return 0

    def _dynamic_token_limit(self, tool_name: str) -> int:
        static_limit = self._max_tool_result_tokens_for(tool_name)
        if static_limit <= 0 or self._token_budget is None:
            return static_limit
        pressure = self._token_budget.pressure()
        if pressure < 0.5:
            return static_limit
        if pressure < 0.8:
            return max(1000, int(static_limit * 0.75))
        return max(1000, int(static_limit * 0.5))

    @staticmethod
    def _estimate_message_tokens(message: BaseMessage, counter: TokenCounter | None = None) -> int:
        if counter is not None:
            return counter.count_messages([message])
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return len(content) // 4
        return 0

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
        content = self._tool_result_to_content(result)

        if self._token_counter is not None:
            token_limit = self._max_tool_result_tokens_for(tool_name)
            if token_limit <= 0:
                token_limit = self._dynamic_token_limit(tool_name)
            if token_limit > 0 and self._token_counter.count(content) > token_limit:
                content = self._token_counter.truncate_to_tokens(
                    content, token_limit, suffix="\n... [truncated]"
                )
        else:
            limit = self._max_tool_result_chars_for(tool_name)
            if len(content) > limit:
                content = content[:limit] + "\n... [truncated]"

        if self._token_budget is not None:
            estimated = self._estimate_message_tokens(
                ToolMessage(content=content, tool_call_id=str(tool_call_id or ""), name=tool_name),
                self._token_counter,
            )
            self._token_budget.record_tokens("default", estimated)

        return ToolMessage(
            content=content,
            tool_call_id=str(tool_call_id or ""),
            name=tool_name,
            status="success",
        )

    @staticmethod
    def _build_tool_error_message(*, tool_name: str, tool_call_id: str | None, exc: Exception) -> ToolMessage:
        payload = {
            "status": "error",
            "tool": tool_name,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id=str(tool_call_id or ""),
            name=tool_name,
            status="error",
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
        tool_call_counts: dict[str, int] = {}
        seen_tool_calls: set[tuple[str, str]] = set()

        if self._token_budget is None and context is not None and context.token_budget is not None:
            self._token_budget = context.token_budget
        if self._token_counter is None and context is not None and context.token_counter is not None:
            self._token_counter = context.token_counter

        if self._token_budget is not None and self._token_counter is not None:
            sys_tokens = self._token_counter.count(self.system_prompt)
            self._token_budget.system_prompt_tokens = max(
                self._token_budget.system_prompt_tokens, sys_tokens
            )

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
                        normalized_args = json.dumps(tool_args, ensure_ascii=False, sort_keys=True)
                        call_signature = (tool_name, normalized_args)
                        if self.forbid_duplicate_tool_calls and call_signature in seen_tool_calls:
                            raise RuntimeError(f"Duplicate tool call forbidden: {tool_name}({normalized_args})")
                        next_count = tool_call_counts.get(tool_name, 0) + 1
                        if self.max_calls_per_tool is not None and next_count > self.max_calls_per_tool:
                            raise RuntimeError(
                                f"Tool {tool_name} exceeded max_calls_per_tool={self.max_calls_per_tool}"
                            )
                        try:
                            tool_message = self._invoke_tool(
                                tool_name,
                                tool_args,
                                context=context,
                                tool_call_id=str(tool_call.get("id") or ""),
                            )
                        except Exception as exc:
                            if self.tool_error_mode == "return":
                                tool_message = self._build_tool_error_message(
                                    tool_name=tool_name,
                                    tool_call_id=str(tool_call.get("id") or ""),
                                    exc=exc,
                                )
                                output_messages.append(tool_message)
                                conversation.append(tool_message)
                                continue
                            raise ToolLoopExecutionError(
                                f"Tool {tool_name} failed: {exc}",
                                output_messages=output_messages,
                                failed_tool_call=tool_call,
                            ) from exc
                        tool_call_counts[tool_name] = next_count
                        seen_tool_calls.add(call_signature)
                        output_messages.append(tool_message)
                        conversation.append(tool_message)
                    continue

                assistant_text = self._assistant_text_content(ai_message).strip()
                if not assistant_text:
                    raise RuntimeError("Model returned empty assistant content without tool calls")
                structured = self._parse_structured_response(assistant_text)
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
