from __future__ import annotations

from typing import Protocol

from .contracts import (
    EnvironmentObservation,
    EnvironmentSnapshot,
    ExecutionAction,
    ExecutionFeedback,
    ObservationRequest,
)


class SimulatorAdapter(Protocol):
    """The only boundary the coordination graph has with a simulator or live PCF."""

    def capture_snapshot(self) -> EnvironmentSnapshot: ...

    def observe(self, request: ObservationRequest) -> EnvironmentObservation: ...

    def execute(self, action: ExecutionAction) -> ExecutionFeedback: ...


__all__ = ["SimulatorAdapter"]
