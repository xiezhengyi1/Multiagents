from __future__ import annotations

from .contracts import (
    IntentBinding,
    IntentLifecycle,
    IntentStatus,
    LifecycleTransition,
)

_ALLOWED_TRANSITIONS: dict[IntentStatus, frozenset[IntentStatus]] = {
    "active": frozenset({"degraded", "satisfied", "failed"}),
    "degraded": frozenset({"replanning", "satisfied", "failed"}),
    "replanning": frozenset({"active", "satisfied", "failed"}),
    "satisfied": frozenset(),
    "failed": frozenset(),
}


class IntentLifecycleManager:
    """Episode-local intent state; persistence is deliberately outside this module."""

    def __init__(self) -> None:
        self._items: dict[str, IntentLifecycle] = {}

    def register(
        self,
        *,
        intent_id: str,
        bindings: list[IntentBinding],
        origin: str,
    ) -> IntentLifecycle:
        if intent_id in self._items:
            raise ValueError(f"intent {intent_id} is already registered")
        lifecycle = IntentLifecycle(
            intent_id=intent_id,
            bindings=bindings,
            status="active",
            origin=origin,
        )
        self._items[intent_id] = lifecycle
        return lifecycle

    def transition(
        self,
        *,
        intent_id: str,
        next_status: IntentStatus,
        reason: str,
    ) -> LifecycleTransition:
        current = self._items[intent_id]
        if next_status not in _ALLOWED_TRANSITIONS[current.status]:
            raise ValueError(
                f"intent transition {current.status} -> {next_status} is not allowed"
            )
        transition = LifecycleTransition(
            intent_id=intent_id,
            previous_status=current.status,
            next_status=next_status,
            reason=reason,
        )
        self._items[intent_id] = current.model_copy(update={"status": next_status})
        return transition

    def get(self, intent_id: str) -> IntentLifecycle:
        return self._items[intent_id]


__all__ = ["IntentLifecycleManager"]
