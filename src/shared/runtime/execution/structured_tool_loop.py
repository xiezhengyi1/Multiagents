"""LLM + tool execution loop that ends with a structured response."""

from __future__ import annotations

import inspect
import json
from typing import Any, Iterable, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from shared.contracts import AgentInvocation, AgentMessage, AgentRunResult
from shared.runtime.core.context import AgentRuntimeContext
from shared.runtime.core.context_policy import ContextPolicy
from shared.runtime.core.token_budget import TokenCounter, TokenBudget
from shared.runtime.messages import contract_to_message, json_friendly, message_to_contract


class ToolLoopExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        output_messages: list[AgentMessage] | None = None,
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
        max_calls_per_tool: int | None = None,
        tool_call_limits: dict[str, int] | None = None,
        forbid_duplicate_tool_calls: bool = False,
        max_tool_result_chars: int = 8000,
        tool_result_limits: dict[str, int] | None = None,
        max_tool_result_tokens: int | None = None,
        tool_result_token_limits: dict[str, int] | None = None,
        token_counter: Optional[TokenCounter] = None,
        token_budget: Optional[TokenBudget] = None,
        context_policy: Optional[ContextPolicy] = None,
    ) -> None:
        self.system_prompt = str(system_prompt or "").strip()
        self.response_model = response_model
        self.max_iterations = max_iterations
        self.max_calls_per_tool = max_calls_per_tool if max_calls_per_tool is None else int(max_calls_per_tool)
        if self.max_calls_per_tool is not None and self.max_calls_per_tool < 1:
            raise ValueError("max_calls_per_tool must be >= 1")
        self._tool_call_limits = {str(name): int(limit) for name, limit in (tool_call_limits or {}).items()}
        if any(limit < 1 for limit in self._tool_call_limits.values()):
            raise ValueError("tool_call_limits values must be >= 1")
        self.forbid_duplicate_tool_calls = bool(forbid_duplicate_tool_calls)
        self.max_tool_result_chars = max(1000, int(max_tool_result_chars))
        self._tool_result_limits = dict(tool_result_limits or {})
        self.max_tool_result_tokens = max_tool_result_tokens
        self._tool_result_token_limits = dict(tool_result_token_limits or {})
        self._token_counter = token_counter
        self._token_budget = token_budget
        self.context_policy = context_policy or ContextPolicy(
            default_tool_result_chars=self.max_tool_result_chars,
            default_tool_result_tokens=max_tool_result_tokens or 2000,
            tool_result_char_limits=self._tool_result_limits,
            tool_result_token_limits=self._tool_result_token_limits,
        )
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

    def _max_calls_for_tool(self, tool_name: str) -> int | None:
        return self._tool_call_limits.get(tool_name, self.max_calls_per_tool)

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

    def _parse_structured_response(self, assistant_text: str) -> BaseModel:
        return self.response_model.model_validate_json(assistant_text)

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

        content = self.context_policy.compact_tool_result(
            tool_name,
            content,
            token_counter=self._token_counter,
            token_budget=self._token_budget,
        )

        return ToolMessage(
            content=content,
            tool_call_id=str(tool_call_id or ""),
            name=tool_name,
            status="success",
        )

    def invoke(
        self,
        invocation: AgentInvocation,
        *,
        context: AgentRuntimeContext | None = None,
        **kwargs: Any,
    ) -> AgentRunResult:
        conversation: list[BaseMessage] = [SystemMessage(content=self.system_prompt)]
        conversation.extend(contract_to_message(message) for message in invocation.messages)
        output_messages: list[AgentMessage] = []
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
                conversation = self.context_policy.compact_tool_history(conversation)
                if self._token_budget is not None:
                    visible_tokens = (
                        self._token_counter.count_messages(conversation)
                        if self._token_counter is not None
                        else sum(self._estimate_message_tokens(message) for message in conversation)
                    )
                    self._token_budget.record_tokens(
                        str(getattr(context, "agent_name", "") or "default"),
                        visible_tokens,
                    )
                ai_message = self.llm.invoke(conversation, **kwargs)
                if not isinstance(ai_message, AIMessage):
                    raise TypeError(f"Expected AIMessage from model, got {type(ai_message).__name__}")
                if ai_message.invalid_tool_calls:
                    raise RuntimeError(f"Model produced invalid tool calls: {ai_message.invalid_tool_calls}")
                canonical_ai_message = message_to_contract(ai_message)
                output_messages.append(canonical_ai_message)
                conversation.append(ai_message)
                tool_calls = list(getattr(ai_message, "tool_calls", None) or [])
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
                        max_calls_for_tool = self._max_calls_for_tool(tool_name)
                        if max_calls_for_tool is not None and next_count > max_calls_for_tool:
                            raise RuntimeError(
                                f"Tool {tool_name} exceeded max_calls_per_tool={max_calls_for_tool}"
                            )
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
                        tool_call_counts[tool_name] = next_count
                        seen_tool_calls.add(call_signature)
                        output_messages.append(message_to_contract(tool_message))
                        conversation.append(tool_message)
                    continue

                assistant_text = self._assistant_text_content(ai_message).strip()
                if not assistant_text:
                    raise RuntimeError("Model returned empty assistant content without tool calls")
                structured = self._parse_structured_response(assistant_text)
                return AgentRunResult(
                    messages=output_messages,
                    structured_response=json_friendly(structured),
                )
        except ToolLoopExecutionError:
            raise
        except Exception as exc:
            raise ToolLoopExecutionError(str(exc), output_messages=output_messages) from exc

        raise ToolLoopExecutionError(
            f"Model exceeded max iterations ({self.max_iterations}) without returning final JSON",
            output_messages=output_messages,
        )


__all__ = ["StructuredToolLoop", "ToolLoopExecutionError"]
