from __future__ import annotations

from langgraph.store.memory import InMemoryStore

from shared.contracts import ExecutionScope, MemoryQuery, MemoryRecord
from shared.memory import MemoryManager


def _scope(session_id: str, thread_id: str) -> ExecutionScope:
    return ExecutionScope(
        session_id=session_id,
        thread_id=thread_id,
        snapshot_id=f"snapshot-{session_id}",
        supi="imsi-208930000000001",
    )


def test_short_term_memory_is_thread_scoped_and_summaries_are_supi_scoped() -> None:
    manager = MemoryManager(store=InMemoryStore())
    first_scope = _scope("session-1", "thread-1")
    second_scope = _scope("session-2", "thread-2")

    manager.add(
        MemoryRecord(
            memory_id="event-1",
            scope=first_scope,
            kind="event",
            source_agent="main_control",
            role="user",
            content="Protect flow-1.",
            source_trace_ids=["trace-1"],
        )
    )
    manager.add(
        MemoryRecord(
            memory_id="summary-1",
            scope=first_scope,
            kind="summary",
            source_agent="memory_summarizer",
            role="summary",
            content="The subscriber requires flow-1 protection.",
            source_trace_ids=["trace-1"],
        )
    )

    first = manager.retrieve(MemoryQuery(scope=first_scope, text="flow-1"))
    second = manager.retrieve(MemoryQuery(scope=second_scope, text="flow-1"))

    assert [record.memory_id for record in first.short_term] == ["event-1"]
    assert [record.memory_id for record in second.short_term] == []
    assert [record.memory_id for record in first.long_term] == ["summary-1"]
    assert [record.memory_id for record in second.long_term] == ["summary-1"]
    assert second.long_term[0].source_trace_ids == ["trace-1"]
