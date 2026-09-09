from __future__ import annotations

from .contracts import (
    IntentBinding,
    IntentLifecycle,
    IntentStatus,
    LifecycleTransition,
)

_ALLOWED_TRANSITIONS: dict[IntentStatus, frozenset[IntentStatus]] = {
    "draft": frozenset({"active", "cancelled", "expired"}),
    "active": frozenset(
        {"suspended", "executing", "degraded", "satisfied", "failed", "cancelled", "expired"}
    ),
    "suspended": frozenset({"active", "cancelled", "expired"}),
    "executing": frozenset({"active", "degraded", "satisfied", "failed", "cancelled", "expired"}),
    "degraded": frozenset({"replanning", "satisfied", "failed", "cancelled", "expired"}),
    "replanning": frozenset({"active", "degraded", "failed", "cancelled", "expired"}),
    "satisfied": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
    "expired": frozenset(),
}


def ensure_transition(current: IntentStatus, next_status: IntentStatus) -> None:
    if next_status not in _ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"intent transition {current} -> {next_status} is not allowed")


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
            version=1,
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
        ensure_transition(current.status, next_status)
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


__all__ = ["IntentLifecycleManager", "ensure_transition"]
