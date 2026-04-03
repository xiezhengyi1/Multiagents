from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.memory import InMemoryStore

from agents.BaseAgent import BaseAgent
from database.connection import RAW_DATABASE_URL
from database.langchain_pg import get_embedding_dimensions, get_openai_embeddings
from utils.logger import setup_logger

logger = setup_logger(__name__)


def _build_short_term_graph(checkpointer: Any):
    def _persist_messages(_state: MessagesState) -> Dict[str, Any]:
        return {}

    builder = StateGraph(MessagesState)
    builder.add_node("persist_messages", _persist_messages)
    builder.add_edge(START, "persist_messages")
    builder.add_edge("persist_messages", END)
    return builder.compile(checkpointer=checkpointer)


class MemoryManager:
    """
    LangGraph-backed memory adapter.

    Public interface intentionally stays aligned with the legacy coordinator
    contract: `add_memory(role, content)` and `retrieve(query, top_k)`.
    """

    def __init__(
        self,
        short_term_limit: int = 10,
        long_term_file: str = "",
        similarity_threshold: float = 0.5,
        *,
        database_url: Optional[str] = None,
        embeddings: Any = None,
        checkpointer: Any = None,
        store: Any = None,
    ):
        self.short_term_limit = max(1, int(short_term_limit))
        self.long_term_file = str(long_term_file or "").strip()
        self.similarity_threshold = similarity_threshold
        self.database_url = str(database_url or RAW_DATABASE_URL).strip()
        self.embeddings = embeddings
        if self.embeddings is None and store is None:
            self.embeddings = get_openai_embeddings()
        self.agent = BaseAgent()

        self._checkpointer_manager = None
        self._store_manager = None
        self.checkpointer = checkpointer or self._init_checkpointer()
        self.store = store or self._init_store()
        self._message_graph = _build_short_term_graph(self.checkpointer)

        self._thread_id = ""
        self._supi: Optional[str] = None

        if self.long_term_file:
            logger.info("long_term_file is deprecated and ignored by LangGraph-backed MemoryManager.")

    def __del__(self) -> None:
        for manager in (self._store_manager, self._checkpointer_manager):
            if manager is None:
                continue
            try:
                manager.__exit__(None, None, None)
            except Exception:
                pass

    def _init_checkpointer(self) -> Any:
        from langgraph.checkpoint.postgres import PostgresSaver

        manager = PostgresSaver.from_conn_string(self.database_url)
        checkpointer = manager.__enter__()
        if hasattr(checkpointer, "setup"):
            checkpointer.setup()
        self._checkpointer_manager = manager
        return checkpointer

    def _init_store(self) -> Any:
        from langgraph.store.postgres import PostgresStore

        manager = PostgresStore.from_conn_string(
            self.database_url,
            index={
                "dims": get_embedding_dimensions(),
                "embed": self.embeddings,
                "fields": ["content"],
            },
        )
        store = manager.__enter__()
        if hasattr(store, "setup"):
            store.setup()
        self._store_manager = manager
        return store

    def bind_thread(self, session_id: str) -> None:
        normalized = str(session_id or "").strip()
        if not normalized:
            raise ValueError("session_id is required to bind MemoryManager thread.")
        self._thread_id = normalized

    def bind_supi(self, supi: Optional[str]) -> None:
        normalized = str(supi or "").strip()
        self._supi = normalized or None

    def _require_thread_id(self) -> str:
        if not self._thread_id:
            raise RuntimeError("MemoryManager thread_id is not bound. Call bind_thread(session_id) first.")
        return self._thread_id

    def _short_term_config(self) -> Dict[str, Any]:
        return {"configurable": {"thread_id": self._require_thread_id()}}

    def _namespace(self) -> tuple[str, str]:
        identity = self._supi or self._require_thread_id()
        return (identity, "episodic")

    def _cursor_namespace(self) -> tuple[str, str]:
        return (self._require_thread_id(), "memory_meta")

    @staticmethod
    def _message_to_dict(message: BaseMessage) -> Dict[str, Any]:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = str(message.name or "assistant")
        else:
            role = message.type
        return {
            "role": role,
            "content": str(message.content),
        }

    @staticmethod
    def _build_message(role: str, content: str) -> BaseMessage:
        normalized_role = str(role or "").strip()
        text = str(content or "")
        if normalized_role.lower() == "user":
            return HumanMessage(content=text)
        return AIMessage(content=text, name=normalized_role or "assistant")

    def _load_messages(self) -> List[BaseMessage]:
        state = self._message_graph.get_state(self._short_term_config())
        values = getattr(state, "values", {}) if state is not None else {}
        messages = values.get("messages", []) if isinstance(values, dict) else []
        return [message for message in messages if isinstance(message, BaseMessage)]

    def _get_summary_cursor(self) -> int:
        item = self.store.get(self._cursor_namespace(), "summary_cursor")
        if item is None or not isinstance(getattr(item, "value", None), dict):
            return 0
        try:
            return int(item.value.get("count", 0))
        except (TypeError, ValueError):
            return 0

    def _set_summary_cursor(self, count: int) -> None:
        self.store.put(self._cursor_namespace(), "summary_cursor", {"count": int(count)})

    def _extract_supi_from_content(self, content: str) -> Optional[str]:
        try:
            payload = json.loads(content)
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None

        supi = str(payload.get("supi") or "").strip()
        return supi or None

    def add_memory(self, role: str, content: str):
        self._message_graph.invoke(
            {"messages": [self._build_message(role, content)]},
            config=self._short_term_config(),
        )

        if role == "IEA":
            self.bind_supi(self._extract_supi_from_content(content))

        self.consolidate_memory()

    def consolidate_memory(self):
        messages = self._load_messages()
        if len(messages) <= self.short_term_limit:
            return

        keep_count = max(1, self.short_term_limit // 2)
        summary_cursor = self._get_summary_cursor()
        candidate_messages = messages[summary_cursor : max(summary_cursor, len(messages) - keep_count)]
        if not candidate_messages:
            return

        text_chunk = "\n".join(
            f"{item['role']}: {item['content']}" for item in (self._message_to_dict(message) for message in candidate_messages)
        )
        summary = self._summarize_text(text_chunk)
        if not summary:
            raise RuntimeError("Memory summarization returned empty content.")

        memory_id = f"summary-{uuid.uuid4()}"
        self.store.put(
            self._namespace(),
            memory_id,
            {
                "content": summary,
                "thread_id": self._require_thread_id(),
                "source_count": len(candidate_messages),
            },
        )
        self._set_summary_cursor(summary_cursor + len(candidate_messages))

    def _summarize_text(self, text: str) -> str:
        prompt = (
            "Summarize the following conversation into durable operational memory. "
            "Keep only facts, preferences, task objectives, and corrective feedback.\n\n"
            f"{text}"
        )
        response = self.agent.get_llm().invoke(prompt)
        content = getattr(response, "content", "")
        normalized = str(content or "").strip()
        if not normalized:
            raise RuntimeError("LLM returned empty memory summary.")
        return normalized

    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        short_term = [self._message_to_dict(message) for message in self._load_messages()[-self.short_term_limit :]]
        if self._get_summary_cursor() <= 0:
            return {
                "short_term": short_term,
                "long_term": [],
            }

        long_term_items = self.store.search(
            self._namespace(),
            query=str(query or "").strip(),
            limit=max(1, int(top_k)),
        )
        long_term = [
            str(item.value.get("content"))
            for item in long_term_items
            if isinstance(getattr(item, "value", None), dict) and item.value.get("content")
        ]
        return {
            "short_term": short_term,
            "long_term": long_term,
        }


__all__ = ["MemoryManager", "InMemorySaver", "InMemoryStore"]
