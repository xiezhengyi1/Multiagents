from __future__ import annotations

import os
from typing import Any, Iterable, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from agent_runtime import AgentRuntimeContext

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
        return create_agent(
            model=self.llm,
            tools=list(tools),
            system_prompt=system_prompt,
            response_format=response_format,
            context_schema=AgentRuntimeContext,
        )
