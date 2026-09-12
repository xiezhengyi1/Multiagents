from __future__ import annotations

from collections.abc import Callable, Mapping

from shared.contracts import JsonValue, utc_now

from .contracts import (
    EnvironmentObservation,
    EnvironmentSnapshot,
    ExecutionAction,
    ExecutionFeedback,
    ObservationRequest,
)

ObservationHandler = Callable[[dict[str, JsonValue]], JsonValue]
ExecutionHandler = Callable[[ExecutionAction], ExecutionFeedback]
SnapshotReader = Callable[[], EnvironmentSnapshot]


class FunctionSimulatorAdapter:
    """Strict adapter for a simulator whose native operations are supplied as callables."""

    def __init__(
        self,
        *,
        snapshot_reader: SnapshotReader,
        observation_handlers: Mapping[str, ObservationHandler],
        execution_handler: ExecutionHandler,
    ) -> None:
        if not observation_handlers:
            raise ValueError("observation_handlers cannot be empty")
        self.snapshot_reader = snapshot_reader
        self.observation_handlers = dict(observation_handlers)
        self.execution_handler = execution_handler
        self._snapshot: EnvironmentSnapshot | None = None

    def capture_snapshot(self) -> EnvironmentSnapshot:
        snapshot = self.snapshot_reader()
        self._snapshot = snapshot
        return snapshot

    def observe(self, request: ObservationRequest) -> EnvironmentObservation:
        if self._snapshot is None:
            raise RuntimeError("capture_snapshot must run before observe")
        handler = self.observation_handlers[request.query_name]
        return EnvironmentObservation(
            base_snapshot_id=self._snapshot.snapshot_id,
            observed_at=utc_now(),
            query=request,
            result=handler(request.arguments),
        )

    def execute(self, action: ExecutionAction) -> ExecutionFeedback:
        feedback = self.execution_handler(action)
        self._snapshot = feedback.snapshot
        return feedback


__all__ = [
    "ExecutionHandler",
    "FunctionSimulatorAdapter",
    "ObservationHandler",
    "SnapshotReader",
]
