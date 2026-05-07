from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


RunType = Literal["chain", "llm", "tool", "parser"]
RunStatus = Literal["success", "error"]


class RunTreeEvent(BaseModel):
    name: str
    time: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class RunTreeTraceRecord(BaseModel):
    id: str
    trace_id: str
    parent_run_id: Optional[str] = None
    parent_run_ids: List[str] = Field(default_factory=list)
    name: str
    run_type: RunType
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    start_time: str
    end_time: str
    events: List[RunTreeEvent] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: RunStatus
    child_runs: List["RunTreeTraceRecord"] = Field(default_factory=list)
    child_run_ids: List[str] = Field(default_factory=list)
    direct_child_run_ids: List[str] = Field(default_factory=list)
    dotted_order: str


RunTreeTraceRecord.model_rebuild()


def collect_descendant_ids(run: RunTreeTraceRecord) -> List[str]:
    ids: List[str] = []
    for child in run.child_runs:
        ids.append(child.id)
        ids.extend(collect_descendant_ids(child))
    return ids


def dotted_order_key(value: str) -> tuple[int, ...]:
    parts = [segment for segment in str(value or "").split(".") if segment]
    return tuple(int(part) for part in parts) if parts else (0,)


def iter_runs_in_dotted_order(root: RunTreeTraceRecord) -> List[RunTreeTraceRecord]:
    collected: List[RunTreeTraceRecord] = []

    def visit(node: RunTreeTraceRecord) -> None:
        if node is not root:
            collected.append(node)
        for child in node.child_runs:
            visit(child)

    visit(root)
    return sorted(collected, key=lambda item: dotted_order_key(item.dotted_order))


__all__ = [
    "RunTreeEvent",
    "RunTreeTraceRecord",
    "collect_descendant_ids",
    "dotted_order_key",
    "iter_runs_in_dotted_order",
]
