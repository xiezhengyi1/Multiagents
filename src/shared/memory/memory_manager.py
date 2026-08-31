from __future__ import annotations

from typing import Any
from uuid import uuid4

from langgraph.store.memory import InMemoryStore

from shared.contracts import MemoryBundle, MemoryQuery, MemoryRecord


class MemoryManager:
    """Stateless typed memory service; scope is supplied on every operation."""

    def __init__(
        self,
        short_term_limit: int = 10,
        enable_llm_summarization: bool = False,
        *,
        database_url: str | None = None,
        embeddings: Any = None,
        store: Any = None,
        summarizer_llm: Any = None,
    ) -> None:
        if short_term_limit < 1:
            raise ValueError("short_term_limit must be at least 1")
        if enable_llm_summarization and summarizer_llm is None:
            raise ValueError("summarizer_llm is required when LLM summarization is enabled")
        self.short_term_limit = short_term_limit
        self.enable_llm_summarization = enable_llm_summarization
        self._summarizer_llm = summarizer_llm
        self._store_manager = None
        self.store = store if store is not None else self._init_store(database_url, embeddings)

    def _init_store(self, database_url: str | None, embeddings: Any) -> Any:
        from database.langchain_pg import get_embedding_dimensions, get_openai_embeddings
        from langgraph.store.postgres import PostgresStore

        resolved_url = str(database_url or "").strip()
        if not resolved_url:
            from database.connection import RAW_DATABASE_URL

            resolved_url = str(RAW_DATABASE_URL).strip()
        if not resolved_url:
            raise ValueError("database_url is required for persistent memory")
        resolved_embeddings = embeddings if embeddings is not None else get_openai_embeddings()
        manager = PostgresStore.from_conn_string(
            resolved_url,
            index={
                "dims": get_embedding_dimensions(),
                "embed": resolved_embeddings,
                "fields": ["content"],
            },
        )
        store = manager.__enter__()
        store.setup()
        self._store_manager = manager
        return store

    def close(self) -> None:
        if self._store_manager is not None:
            self._store_manager.__exit__(None, None, None)
            self._store_manager = None

    @classmethod
    def _namespace(cls, record_or_query: MemoryRecord | MemoryQuery, kind: str) -> tuple[str, str, str]:
        if kind == "event":
            identity = f"thread:{record_or_query.scope.thread_id}"
        elif kind == "summary":
            identity = (
                f"supi:{record_or_query.scope.supi}"
                if record_or_query.scope.supi is not None
                else f"session:{record_or_query.scope.session_id}"
            )
        else:
            raise ValueError(f"unsupported memory kind: {kind}")
        return (identity, "memory_v2", kind)

    @classmethod
    def _meta_namespace(cls, record_or_query: MemoryRecord | MemoryQuery) -> tuple[str, str, str]:
        return (f"thread:{record_or_query.scope.thread_id}", "memory_v2", "metadata")

    def add(self, record: MemoryRecord) -> None:
        self.store.put(
            self._namespace(record, record.kind),
            record.memory_id,
            record.model_dump(mode="json"),
        )
        if self.enable_llm_summarization and record.kind == "event":
            self._consolidate(record)

    def retrieve(self, query: MemoryQuery) -> MemoryBundle:
        short_term = self._read_records(
            self._namespace(query, "event"),
            query=None,
            limit=self.short_term_limit,
        )
        long_term = self._read_records(
            self._namespace(query, "summary"),
            query=query.text,
            limit=query.top_k,
        )
        return MemoryBundle(
            query=query,
            short_term=short_term[-self.short_term_limit :],
            long_term=long_term,
        )

    def _read_records(
        self,
        namespace: tuple[str, str, str],
        *,
        query: str | None,
        limit: int,
    ) -> list[MemoryRecord]:
        items = self.store.search(namespace, query=query, limit=limit)
        records = [MemoryRecord.model_validate(item.value) for item in items]
        return sorted(records, key=lambda record: (record.created_at, record.memory_id))

    def _summary_cursor(self, record: MemoryRecord) -> int:
        item = self.store.get(self._meta_namespace(record), "summary_cursor")
        if item is None:
            return 0
        value = item.value
        if not isinstance(value, dict) or not isinstance(value.get("count"), int):
            raise TypeError("memory summary cursor must contain an integer count")
        return value["count"]

    def _consolidate(self, trigger: MemoryRecord) -> None:
        events = self._read_records(
            self._namespace(trigger, "event"),
            query=None,
            limit=10_000,
        )
        cursor = self._summary_cursor(trigger)
        keep_count = max(1, self.short_term_limit // 2)
        end_index = len(events) - keep_count
        if end_index <= cursor:
            return
        candidates = events[cursor:end_index]
        source_text = "\n".join(f"{record.role}: {record.content}" for record in candidates)
        response = self._summarizer_llm.invoke(
            "Summarize the following conversation into durable operational memory. "
            "Keep only facts, objectives, decisions, and corrective feedback.\n\n"
            + source_text
        )
        summary = str(response.content).strip()
        if not summary:
            raise RuntimeError("memory summarizer returned empty content")
        self.add(
            MemoryRecord(
                memory_id=f"summary-{uuid4()}",
                scope=trigger.scope,
                kind="summary",
                source_agent="memory_summarizer",
                role="summary",
                content=summary,
                source_trace_ids=list(
                    dict.fromkeys(
                        trace_id
                        for candidate in candidates
                        for trace_id in candidate.source_trace_ids
                    )
                ),
            )
        )
        self.store.put(
            self._meta_namespace(trigger),
            "summary_cursor",
            {"count": end_index},
        )


__all__ = ["InMemoryStore", "MemoryManager"]
