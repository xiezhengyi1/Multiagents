from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional


@dataclass(frozen=True)
class RegisteredAgent:
    agent_name: str
    input_artifact_types: List[str]
    output_artifact_types: List[str]
    concurrency: int = 1
    blocking: bool = True


class HookRegistry:
    def __init__(self) -> None:
        self.pre_plan_hooks: List[Callable[..., None]] = []
        self.post_plan_hooks: List[Callable[..., None]] = []
        self.post_dispatch_hooks: List[Callable[..., None]] = []
        self.failure_diagnosis_hooks: List[Callable[..., None]] = []


class RuntimeAgentRegistry:
    def __init__(self, *, max_workers: int = 8) -> None:
        self._agents: Dict[str, object] = {}
        self._metadata: Dict[str, RegisteredAgent] = {}
        self.hooks = HookRegistry()
        self.executor = ThreadPoolExecutor(max_workers=max(2, int(max_workers)))

    def register(self, worker: object, *, input_types: List[str], output_types: List[str], concurrency: int = 1, blocking: bool = True) -> None:
        agent_name = str(getattr(worker, "agent_name", "") or "").strip()
        if not agent_name:
            raise ValueError("worker.agent_name is required for registration")
        self._agents[agent_name] = worker
        self._metadata[agent_name] = RegisteredAgent(
            agent_name=agent_name,
            input_artifact_types=list(input_types),
            output_artifact_types=list(output_types),
            concurrency=max(1, int(concurrency)),
            blocking=bool(blocking),
        )

    def get(self, agent_name: str) -> object:
        return self._agents[str(agent_name)]

    def metadata(self, agent_name: str) -> RegisteredAgent:
        return self._metadata[str(agent_name)]

    def submit_artifact(self, agent_name: str, request_path: Path) -> Future:
        worker = self.get(agent_name)
        return self.executor.submit(worker.consume_request_artifact, request_path)
