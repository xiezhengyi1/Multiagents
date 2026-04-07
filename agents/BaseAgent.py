from __future__ import annotations

import inspect
import json
import os
from typing import Any, Iterable, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agent_runtime.artifacts import ArtifactEnvelope
from agent_runtime.context import AgentRuntimeContext
from utils.agent_tracing import JsonlTraceWriter, TracedStructuredAgent, build_tool_specs
from utils.logger import setup_logger

# Load environment variables once for all agents.
load_dotenv()

class _JsonToolAgentRunnable:
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
        self.tools_by_name = {}
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
        if isinstance(content, str):
            text = content
        else:
            text = json.dumps(content, ensure_ascii=False)
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

    def _invoke_tool(self, tool_name: str, args: dict[str, Any], *, context: AgentRuntimeContext | None, tool_call_id: str | None) -> ToolMessage:
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

    def invoke(self, payload: dict[str, Any], *, context: AgentRuntimeContext | None = None, **kwargs: Any) -> dict[str, Any]:
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError("json tool agent requires a non-empty messages list")
        conversation: list[BaseMessage] = [SystemMessage(content=self.system_prompt)]
        conversation.extend(self._coerce_message(message) for message in raw_messages)
        output_messages: list[BaseMessage] = []

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
                    tool_message = self._invoke_tool(
                        tool_name,
                        tool_args,
                        context=context,
                        tool_call_id=str(tool_call.get("id") or ""),
                    )
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

        raise RuntimeError(f"Model exceeded max iterations ({self.max_iterations}) without returning final JSON")


class BaseAgent:
    def __init__(self, model_name: str = "qwen-plus", temperature: float = 0):
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        raw_timeout = os.getenv("OPENAI_TIMEOUT_SECONDS", "120")
        raw_max_retries = os.getenv("OPENAI_MAX_RETRIES", "2")

        # 中文标注：API Key 缺失直接抛出，不再静默打印警告
        if not api_key:
            raise RuntimeError("Missing API key: set OPENAI_API_KEY or DASHSCOPE_API_KEY")

        timeout_seconds = float(raw_timeout)
        if timeout_seconds <= 0:
            raise ValueError("OPENAI_TIMEOUT_SECONDS must be positive")

        max_retries = int(raw_max_retries)
        if max_retries < 0:
            raise ValueError("OPENAI_MAX_RETRIES must be >= 0")

        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    def get_llm(self) -> ChatOpenAI:
        return self.llm

    def initialize_agent_runtime(
        self,
        *,
        logger_color: Optional[str] = None,
        logger_name: Optional[str] = None,
        lease_seconds: int = 60,
    ) -> None:
        agent_name = str(getattr(self, "agent_name", "") or "").strip()
        if not agent_name:
            raise RuntimeError("agent_name must be set before initialize_agent_runtime()")
        if not hasattr(self, "init_worker_runtime"):
            raise RuntimeError("initialize_agent_runtime() requires ArtifactWorkerMixin")

        self.init_worker_runtime(lease_seconds=lease_seconds)
        if logger_color is not None:
            self.logger = setup_logger(
                logger_name or self.__class__.__name__,
                default_msg_color=logger_color,
            )

    @staticmethod
    def serialize_artifact_payload(payload: Any) -> dict:
        # 中文标注：统一使用 Pydantic v2 model_dump
        if hasattr(payload, "model_dump"):
            return payload.model_dump(mode="json")
        if isinstance(payload, dict):
            return payload
        raise TypeError(f"Unsupported artifact payload type: {type(payload).__name__}")

    def cache_received_artifact(
        self,
        *,
        artifact_type: str,
        payload: Any,
        source_agent: str = "coordinator",
        session_id: str = "",
        snapshot_id: str = "",
        target_agent: Optional[str] = None,
    ) -> ArtifactEnvelope:
        if not hasattr(self, "cache"):
            raise RuntimeError("cache_received_artifact() requires initialized worker runtime")

        envelope = ArtifactEnvelope(
            artifact_type=str(artifact_type or "").strip(),
            source_agent=str(source_agent or "").strip(),
            target_agent=str(target_agent or getattr(self, "agent_name", "") or "").strip(),
            session_id=str(session_id or "").strip(),
            snapshot_id=str(snapshot_id or "").strip(),
            payload=self.serialize_artifact_payload(payload),
        )
        self.cache.cache_received(envelope)
        return envelope

    def cache_produced_artifact(
        self,
        *,
        artifact_type: str,
        request_envelope: ArtifactEnvelope,
        payload: Any,
        source_agent: Optional[str] = None,
        target_agent: Optional[str] = None,
    ) -> ArtifactEnvelope:
        if not hasattr(self, "cache"):
            raise RuntimeError("cache_produced_artifact() requires initialized worker runtime")

        envelope = ArtifactEnvelope(
            artifact_type=str(artifact_type or "").strip(),
            source_agent=str(source_agent or getattr(self, "agent_name", "") or "").strip(),
            target_agent=str(target_agent or request_envelope.source_agent or "").strip(),
            session_id=request_envelope.session_id,
            snapshot_id=request_envelope.snapshot_id,
            correlation_id=request_envelope.correlation_id,
            upstream_artifact_ids=[request_envelope.artifact_id],
            payload=self.serialize_artifact_payload(payload),
        )
        self.cache.cache_produced(envelope)
        return envelope

    @staticmethod
    def build_runtime_context(
        *,
        agent_name: str,
        session_id: str = "",
        snapshot_id: str = "",
        supi: Optional[str] = None,
        thread_id: str = "",
        allow_user_interaction: bool = False,
    ) -> AgentRuntimeContext:
        normalized_session = str(session_id or "").strip()
        normalized_thread = str(thread_id or normalized_session).strip()
        return AgentRuntimeContext(
            agent_name=str(agent_name or "").strip(),
            session_id=normalized_session,
            snapshot_id=str(snapshot_id or "").strip(),
            supi=str(supi or "").strip() or None,
            thread_id=normalized_thread,
            allow_user_interaction=bool(allow_user_interaction),
        )

    def create_structured_agent(
        self,
        *,
        tools: Iterable[Any],
        system_prompt: str,
        response_format: Any,
    ):
        # 中文标注：延迟导入 create_agent，避免模块级别导入不存在的包
        from langchain.agents import create_agent
        tool_list = list(tools)
        runnable = create_agent(
            model=self.llm,
            tools=tool_list,
            system_prompt=system_prompt,
            response_format=response_format,
            context_schema=AgentRuntimeContext,
        )
        agent_name = str(getattr(self, "agent_name", "") or "").strip()
        if not agent_name:
            raise RuntimeError("agent_name must be set before create_structured_agent()")
        return TracedStructuredAgent(
            agent_name=agent_name,
            model_name=self.model_name,
            system_prompt=system_prompt,
            tool_specs=build_tool_specs(tool_list),
            runnable=runnable,
            writer=JsonlTraceWriter(agent_name),
        )

    def create_json_agent(
        self,
        *,
        tools: Iterable[Any],
        system_prompt: str,
        response_model: type[BaseModel],
        max_iterations: int = 8,
    ):
        tool_list = list(tools)
        runnable = _JsonToolAgentRunnable(
            llm=self.llm,
            tools=tool_list,
            system_prompt=system_prompt,
            response_model=response_model,
            max_iterations=max_iterations,
        )
        agent_name = str(getattr(self, "agent_name", "") or "").strip()
        if not agent_name:
            raise RuntimeError("agent_name must be set before create_json_agent()")
        return TracedStructuredAgent(
            agent_name=agent_name,
            model_name=self.model_name,
            system_prompt=system_prompt,
            tool_specs=build_tool_specs(tool_list),
            runnable=runnable,
            writer=JsonlTraceWriter(agent_name),
        )

    def invoke_json_response(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        runtime_context: AgentRuntimeContext | None,
    ) -> BaseModel:
        if not hasattr(self, "agent"):
            raise RuntimeError("invoke_json_response() requires self.agent to be initialized")
        pending_messages = getattr(self, "_pending_invoke_messages", None)
        if not isinstance(pending_messages, list) or not pending_messages:
            pending_messages = [{"role": "user", "content": user_prompt}]
        result = self.agent.invoke({"messages": pending_messages}, context=runtime_context)
        structured = result.get("structured_response")
        if structured is None:
            raise RuntimeError("Agent returned no structured_response.")
        if isinstance(structured, response_model):
            return structured
        if isinstance(structured, str):
            return response_model.model_validate_json(structured)
        return response_model.model_validate(structured)
