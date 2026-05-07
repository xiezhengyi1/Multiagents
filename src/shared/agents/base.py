from __future__ import annotations

import os
from typing import Any, Iterable, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from shared.runtime import RuntimeCache
from shared.runtime import AgentRuntimeContext
from shared.runtime import StructuredToolLoop
from shared.runtime import build_tool_specs, extract_tool_calls
from shared.runtime import ArtifactEnvelope
from shared.runtime import JsonlTraceWriter, TracedStructuredAgent
from shared.logging import setup_logger


load_dotenv()


def extract_grounding_tool_names(result: dict[str, Any], grounding_tools: Iterable[str]) -> list[str]:
    allowed = {
        str(name).strip()
        for name in grounding_tools
        if str(name).strip()
    }
    if not allowed:
        return []
    messages = result.get("messages") or []
    calls = extract_tool_calls(messages)
    names: list[str] = []
    for call in calls:
        name = str(call.get("name") or "").strip()
        if name in allowed:
            names.append(name)
    return names


def coerce_structured_response(
    result: dict[str, Any],
    response_model: type[BaseModel],
    *,
    error_message: str,
) -> BaseModel:
    structured = result.get("structured_response")
    if structured is None:
        raise RuntimeError(error_message)
    if isinstance(structured, response_model):
        return structured
    if isinstance(structured, str):
        return response_model.model_validate_json(structured)
    return response_model.model_validate(structured)


class BaseAgent:
    def __init__(
        self,
        model_name: str = "qwen-plus",
        temperature: float = 0,
        use_local_model: bool = False,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._cache = RuntimeCache()
        if use_local_model:
            base_url = os.getenv("vLLM_URL")
            resolved_model_name = os.getenv("vLLM_MODEL_NAME")
            self.model_name = resolved_model_name
            self.temperature = temperature
            self.llm = ChatOpenAI(
                model=resolved_model_name,
                temperature=temperature,
                api_key=base_url,
                base_url=base_url,
                timeout=120.0,
                max_retries=2,
            )
            return

        resolved_api_key = api_key
        resolved_base_url = base_url
        if resolved_api_key is None:
            resolved_api_key = os.getenv("OPENAI_API_KEY")
        if resolved_base_url is None:
            resolved_base_url = os.getenv("OPENAI_BASE_URL")
        raw_timeout = os.getenv("OPENAI_TIMEOUT_SECONDS", "120")
        raw_max_retries = os.getenv("OPENAI_MAX_RETRIES", "2")

        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            timeout=float(raw_timeout),
            max_retries=int(raw_max_retries),
        )

    def get_llm(self) -> ChatOpenAI:
        return self.llm

    def _sync_cache_agent_name(self) -> None:
        if not hasattr(self, "_cache"):
            self._cache = RuntimeCache()
        agent_name = str(getattr(self, "agent_name", "") or self.__class__.__name__).strip()
        if self._cache.agent_name != agent_name:
            self._cache.agent_name = agent_name

    def get_cached_runtime_value(
        self,
        namespace: str,
        cache_key: Any,
        *,
        snapshot_id: str = "",
        session_id: str = "",
        default: Any = None,
    ) -> Any:
        self._sync_cache_agent_name()
        return self._cache.get(namespace, cache_key, snapshot_id=snapshot_id, session_id=session_id, default=default)

    def cache_runtime_value(
        self,
        namespace: str,
        cache_key: Any,
        value: Any,
        *,
        snapshot_id: str = "",
        session_id: str = "",
    ) -> None:
        self._sync_cache_agent_name()
        self._cache.set(namespace, cache_key, value, snapshot_id=snapshot_id, session_id=session_id)

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
    def serialize_artifact_payload(payload: Any) -> dict[str, Any]:
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

    def create_json_agent(
        self,
        *,
        tools: Iterable[Any],
        system_prompt: str,
        response_model: type[BaseModel],
        max_iterations: int = 8,
        tool_error_mode: str = "raise",
        max_calls_per_tool: int | None = None,
        forbid_duplicate_tool_calls: bool = False,
    ) -> TracedStructuredAgent:
        tool_list = list(tools)
        runnable = StructuredToolLoop(
            llm=self.llm,
            tools=tool_list,
            system_prompt=system_prompt,
            response_model=response_model,
            max_iterations=max_iterations,
            tool_error_mode=tool_error_mode,
            max_calls_per_tool=max_calls_per_tool,
            forbid_duplicate_tool_calls=forbid_duplicate_tool_calls,
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

