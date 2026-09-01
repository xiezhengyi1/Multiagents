from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from shared.contracts import AgentInvocation, AgentRunResult, AgentTrace, ToolCatalog, TraceTarget, utc_now
from shared.runtime.core.context import AgentRuntimeContext
from shared.runtime.io.paths import (
    current_trace_identity,
    trace_runs_root,
    trace_segment_path,
)
from shared.runtime.trace.builder import build_agent_trace
from shared.runtime.trace.dataset import open_trace_segment, record_trace_append

_FILE_LOCKS: dict[Path, threading.Lock] = {}
_LOCK_GUARD = threading.Lock()


def _file_lock(path: Path) -> threading.Lock:
    with _LOCK_GUARD:
        lock = _FILE_LOCKS.get(path)
        if lock is None:
            lock = threading.Lock()
            _FILE_LOCKS[path] = lock
        return lock


class JsonlTraceWriter:
    def __init__(
        self,
        agent_name: str,
        root: Path | None = None,
        *,
        dataset_run_id: str | None = None,
        worker_id: str | None = None,
    ) -> None:
        normalized_agent = str(agent_name).strip()
        if not normalized_agent:
            raise ValueError("agent_name is required for trace writing")
        current_run_id, current_worker_id = current_trace_identity()
        self.agent_name = normalized_agent
        self.dataset_run_id = dataset_run_id or current_run_id
        self.worker_id = worker_id or current_worker_id
        self.root = Path(root) if root is not None else trace_runs_root()
        self.path = trace_segment_path(
            self.agent_name,
            dataset_run_id=self.dataset_run_id,
            worker_id=self.worker_id,
            root=self.root,
        )
        self._manifest = open_trace_segment(
            segment_path=self.path,
            dataset_run_id=self.dataset_run_id,
            worker_id=self.worker_id,
            agent_name=self.agent_name,
            root=self.root,
        )

    def write(self, trace: AgentTrace) -> Path:
        if trace.agent_name != self.agent_name:
            raise ValueError(
                f"trace writer for {self.agent_name} cannot write agent {trace.agent_name}"
            )
        payload = trace.model_dump_json()
        lock = _file_lock(self.path)
        with lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.write("\n")
            self._manifest = record_trace_append(self.path, self._manifest)
        return self.path


class TracedStructuredAgent:
    def __init__(
        self,
        *,
        agent_name: str,
        model_name: str,
        system_prompt: str,
        prompt_name: str,
        prompt_version: str,
        tool_catalog: ToolCatalog,
        response_model: type[BaseModel],
        runnable: Any,
        writer: JsonlTraceWriter,
    ) -> None:
        self.agent_name = str(agent_name).strip()
        self.model_name = str(model_name).strip()
        self.system_prompt = str(system_prompt).strip()
        self.prompt_name = str(prompt_name).strip()
        self.prompt_version = str(prompt_version).strip()
        if not all(
            [self.agent_name, self.model_name, self.system_prompt, self.prompt_name, self.prompt_version]
        ):
            raise ValueError(
                "agent_name, model_name, system_prompt, prompt_name, and prompt_version are required"
            )
        self.tool_catalog = tool_catalog
        self.response_model = response_model
        self.runnable = runnable
        self.writer = writer

    def __getattr__(self, item: str) -> Any:
        return getattr(self.runnable, item)

    def invoke(
        self,
        invocation: AgentInvocation,
        *,
        context: AgentRuntimeContext,
        defer_trace: bool = False,
        **kwargs: Any,
    ) -> AgentRunResult:
        if invocation.metadata.prompt_name != self.prompt_name:
            raise ValueError(
                f"invocation prompt_name {invocation.metadata.prompt_name} does not match {self.prompt_name}"
            )
        if invocation.metadata.prompt_version != self.prompt_version:
            raise ValueError(
                "invocation prompt_version "
                f"{invocation.metadata.prompt_version} does not match {self.prompt_version}"
            )
        started_at = utc_now()
        try:
            result = self.runnable.invoke(invocation, context=context, **kwargs)
        except Exception as exc:
            self.writer.write(
                self._build_trace(
                    invocation=invocation,
                    context=context,
                    result=None,
                    captured_error=exc,
                    started_at=started_at,
                    ended_at=utc_now(),
                )
            )
            raise
        if not defer_trace:
            self.writer.write(
                self._build_trace(
                    invocation=invocation,
                    context=context,
                    result=result,
                    captured_error=None,
                    started_at=started_at,
                    ended_at=utc_now(),
                )
            )
        return result

    def write_trace(
        self,
        *,
        invocation: AgentInvocation,
        context: AgentRuntimeContext,
        result: AgentRunResult,
        derived_targets: list[TraceTarget] | None = None,
        error: str | None = None,
    ) -> Path:
        now = utc_now()
        return self.writer.write(
            self._build_trace(
                invocation=invocation,
                context=context,
                result=result,
                captured_error=RuntimeError(error) if error is not None else None,
                started_at=now,
                ended_at=now,
                derived_targets=derived_targets,
            )
        )

    def _build_trace(
        self,
        *,
        invocation: AgentInvocation,
        context: AgentRuntimeContext,
        result: AgentRunResult | None,
        captured_error: BaseException | None,
        started_at: datetime,
        ended_at: datetime,
        derived_targets: list[TraceTarget] | None = None,
    ) -> AgentTrace:
        return build_agent_trace(
            agent_name=self.agent_name,
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            tool_catalog=self.tool_catalog,
            response_model=self.response_model,
            invocation=invocation,
            context=context,
            result=result,
            captured_error=captured_error,
            started_at=started_at,
            ended_at=ended_at,
            derived_targets=derived_targets,
        )


__all__ = ["JsonlTraceWriter", "TracedStructuredAgent"]
