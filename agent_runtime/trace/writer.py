from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_runtime.io.paths import trace_root
from agent_runtime.tooling import build_tool_specs
from agent_runtime.trace.builder import TRACE_UNSET, build_run_tree_record, utc_now

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
    def __init__(self, agent_name: str, root: Path | None = None) -> None:
        normalized_agent = str(agent_name or "").strip()
        if not normalized_agent:
            raise ValueError("agent_name is required for trace writing")
        self.agent_name = normalized_agent
        self.root = Path(root) if root is not None else trace_root(normalized_agent)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{self.agent_name}.jsonl"

    def write(self, record: dict[str, Any]) -> Path:
        payload = json.dumps(record, ensure_ascii=False)
        lock = _file_lock(self.path)
        with lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.write("\n")
        return self.path


class TracedStructuredAgent:
    def __init__(
        self,
        agent_name: str,
        model_name: str,
        system_prompt: str,
        tool_specs: list[dict[str, Any]],
        runnable: Any,
        writer: JsonlTraceWriter,
    ) -> None:
        self.agent_name = str(agent_name or "").strip()
        self.model_name = str(model_name or "").strip()
        self.system_prompt = str(system_prompt or "")
        self.tool_specs = list(tool_specs)
        self.runnable = runnable
        self.writer = writer

    def __getattr__(self, item: str) -> Any:
        return getattr(self.runnable, item)

    def invoke(self, payload: dict[str, Any], *, context: Any = None, **kwargs: Any) -> Any:
        if "messages" not in payload:
            raise KeyError("traced agent invoke payload must contain 'messages'")
        run_id = f"run-{uuid4()}"
        start_dt = utc_now()
        end_dt = start_dt
        status = "success"
        result = None
        captured_error: BaseException | None = None
        manual_trace = str(payload.get("trace_write_mode") or "").strip().lower() == "manual"
        try:
            result = self.runnable.invoke(payload, context=context, **kwargs)
            end_dt = utc_now()
            if manual_trace and isinstance(result, dict):
                result["_trace_capture"] = {"payload": payload, "context": context}
            return result
        except Exception as exc:
            end_dt = utc_now()
            status = "error"
            captured_error = exc
            raise
        finally:
            if not (manual_trace and status == "success"):
                self.writer.write(
                    self._build_record(
                        run_id=run_id,
                        payload=payload,
                        context=context,
                        result=result,
                        status=status,
                        captured_error=captured_error,
                        start_dt=start_dt,
                        end_dt=end_dt,
                    )
                )

    async def ainvoke(self, payload: dict[str, Any], *, context: Any = None, **kwargs: Any) -> Any:
        if "messages" not in payload:
            raise KeyError("traced agent ainvoke payload must contain 'messages'")
        run_id = f"run-{uuid4()}"
        start_dt = utc_now()
        end_dt = start_dt
        status = "success"
        result = None
        captured_error: BaseException | None = None
        manual_trace = str(payload.get("trace_write_mode") or "").strip().lower() == "manual"
        try:
            result = await self.runnable.ainvoke(payload, context=context, **kwargs)
            end_dt = utc_now()
            if manual_trace and isinstance(result, dict):
                result["_trace_capture"] = {"payload": payload, "context": context}
            return result
        except Exception as exc:
            end_dt = utc_now()
            status = "error"
            captured_error = exc
            raise
        finally:
            if not (manual_trace and status == "success"):
                self.writer.write(
                    self._build_record(
                        run_id=run_id,
                        payload=payload,
                        context=context,
                        result=result,
                        status=status,
                        captured_error=captured_error,
                        start_dt=start_dt,
                        end_dt=end_dt,
                    )
                )

    def write_trace(
        self,
        *,
        payload: dict[str, Any],
        context: Any = None,
        result: Any = None,
        status: str = "success",
        error: str | None = None,
        structured_response_override: Any = TRACE_UNSET,
    ) -> Path:
        run_id = f"run-{uuid4()}"
        start_dt = utc_now()
        end_dt = utc_now()
        captured_error = RuntimeError(error) if error else None
        return self.writer.write(
            self._build_record(
                run_id=run_id,
                payload=payload,
                context=context,
                result=result,
                status=status,
                captured_error=captured_error,
                start_dt=start_dt,
                end_dt=end_dt,
                structured_response_override=structured_response_override,
            )
        )

    def _build_record(
        self,
        *,
        run_id: str,
        payload: dict[str, Any],
        context: Any,
        result: Any,
        status: str,
        captured_error: BaseException | None,
        start_dt: datetime,
        end_dt: datetime,
        structured_response_override: Any = TRACE_UNSET,
    ) -> dict[str, Any]:
        return build_run_tree_record(
            agent_name=self.agent_name,
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            run_id=run_id,
            payload=payload,
            context=context,
            result=result,
            status=status,
            captured_error=captured_error,
            start_dt=start_dt,
            end_dt=end_dt,
            structured_response_override=structured_response_override,
        )


__all__ = ["JsonlTraceWriter", "TracedStructuredAgent", "build_tool_specs"]
