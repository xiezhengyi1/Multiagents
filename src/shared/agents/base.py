from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from shared.contracts import AgentRunResult, ExecutionScope
from shared.runtime import ContextPolicy
from shared.runtime import RuntimeCache
from shared.runtime import RuntimePersistence
from shared.runtime import AgentRuntimeContext
from shared.runtime import StructuredToolLoop
from shared.runtime import build_tool_catalog
from shared.runtime import JsonlTraceWriter, TracedStructuredAgent
from shared.logging import setup_logger


load_dotenv()


def resolve_agent_model_name(model_name: str = "", *, use_local_model: bool = False) -> str:
    if use_local_model:
        return str(os.getenv("vLLM_MODEL_NAME") or model_name or "").strip()
    env_model_name = str(os.getenv("MODEL_NAME") or "").strip()
    if env_model_name:
        return env_model_name
    return str(model_name or "qwen-plus").strip()


def extract_grounding_tool_names(result: AgentRunResult, grounding_tools: Iterable[str]) -> list[str]:
    allowed = {
        str(name).strip()
        for name in grounding_tools
        if str(name).strip()
    }
    if not allowed:
        return []
    return [
        call.name
        for message in result.messages
        for call in message.tool_calls
        if call.name in allowed
    ]


def coerce_structured_response(
    result: AgentRunResult,
    response_model: type[BaseModel],
    *,
    error_message: str,
) -> BaseModel:
    return response_model.model_validate(result.structured_response)


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
            resolved_model_name = resolve_agent_model_name(model_name, use_local_model=True)
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

        resolved_model_name = resolve_agent_model_name(model_name, use_local_model=False)
        self.model_name = resolved_model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(
            model=resolved_model_name,
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
        persistence: RuntimePersistence,
    ) -> None:
        agent_name = str(getattr(self, "agent_name", "") or "").strip()
        if not agent_name:
            raise RuntimeError("agent_name must be set before initialize_agent_runtime()")
        if not hasattr(self, "init_worker_runtime"):
            raise RuntimeError("initialize_agent_runtime() requires ArtifactWorkerMixin")

        self.init_worker_runtime(persistence=persistence, lease_seconds=lease_seconds)
        if logger_color is not None:
            self.logger = setup_logger(
                logger_name or self.__class__.__name__,
                default_msg_color=logger_color,
            )

    @staticmethod
    def build_runtime_context(
        *,
        agent_name: str,
        scope: ExecutionScope,
        allow_user_interaction: bool = False,
        token_budget: Any = None,
        token_counter: Any = None,
        trace_attributes: dict[str, Any] | None = None,
    ) -> AgentRuntimeContext:
        return AgentRuntimeContext(
            agent_name=str(agent_name or "").strip(),
            scope=scope,
            allow_user_interaction=bool(allow_user_interaction),
            token_budget=token_budget,
            token_counter=token_counter,
            trace_attributes=dict(trace_attributes or {}),
        )

    def _resolve_token_context(self, token_budget: Any = None, token_counter: Any = None) -> tuple[Any, Any]:
        resolved_budget = token_budget if token_budget is not None else getattr(self, "_token_budget", None)
        resolved_counter = token_counter if token_counter is not None else getattr(self, "_token_counter", None)
        return resolved_budget, resolved_counter

    def create_json_agent(
        self,
        *,
        tools: Iterable[Any],
        system_prompt: str,
        prompt_name: str,
        prompt_version: str,
        response_model: type[BaseModel],
        max_iterations: int = 8,
        max_calls_per_tool: int | None = None,
        tool_call_limits: dict[str, int] | None = None,
        forbid_duplicate_tool_calls: bool = False,
        context_policy: ContextPolicy | None = None,
        capability_aliases: Mapping[str, Iterable[Any]] | None = None,
    ) -> TracedStructuredAgent:
        tool_list = list(tools)
        tool_catalog = build_tool_catalog(tool_list, capability_aliases=capability_aliases)
        runnable = StructuredToolLoop(
            llm=self.llm,
            tools=tool_list,
            tool_catalog=tool_catalog,
            system_prompt=system_prompt,
            response_model=response_model,
            max_iterations=max_iterations,
            max_calls_per_tool=max_calls_per_tool,
            tool_call_limits=tool_call_limits,
            forbid_duplicate_tool_calls=forbid_duplicate_tool_calls,
            context_policy=context_policy,
        )
        agent_name = str(getattr(self, "agent_name", "") or "").strip()
        if not agent_name:
            raise RuntimeError("agent_name must be set before create_json_agent()")
        return TracedStructuredAgent(
            agent_name=agent_name,
            model_name=self.model_name,
            system_prompt=system_prompt,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            tool_catalog=tool_catalog,
            response_model=response_model,
            runnable=runnable,
            writer=JsonlTraceWriter(agent_name),
        )
