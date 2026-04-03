from __future__ import annotations

import os
from typing import Any, Iterable, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from agent_runtime import AgentRuntimeContext, ArtifactEnvelope
from utils.agent_tracing import JsonlTraceWriter, TracedStructuredAgent, build_tool_specs
from utils.logger import setup_logger

# Load environment variables once for all agents.
load_dotenv()


class BaseAgent:
    def __init__(self, model_name: str = "qwen-plus", temperature: float = 0):
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"

        if not api_key:
            print("警告: 未检测到 API Key (OPENAI_API_KEY 或 DASHSCOPE_API_KEY)")

        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
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
        if hasattr(payload, "model_dump"):
            dumped = payload.model_dump(mode="json")
            if isinstance(dumped, dict):
                return dumped
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
    ) -> AgentRuntimeContext:
        normalized_session = str(session_id or "").strip()
        normalized_thread = str(thread_id or normalized_session).strip()
        return AgentRuntimeContext(
            agent_name=str(agent_name or "").strip(),
            session_id=normalized_session,
            snapshot_id=str(snapshot_id or "").strip(),
            supi=str(supi or "").strip() or None,
            thread_id=normalized_thread,
        )

    def create_structured_agent(
        self,
        *,
        tools: Iterable[Any],
        system_prompt: str,
        response_format: Any,
    ):
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
