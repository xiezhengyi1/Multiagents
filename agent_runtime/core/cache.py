"""Session-scoped key-value cache for agent runtime state.

Extracted from BaseAgent to give runtime caching a clear single responsibility.
The cache keys are 5-tuples: (agent_name, snapshot_id, session_id, namespace, cache_key).
All values are deep-copied on read/write to prevent accidental mutation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional


class RuntimeCache:
    """Session-scoped key-value cache keyed by
    (agent_name, snapshot_id, session_id, namespace, cache_key)."""

    def __init__(self, agent_name: str = "") -> None:
        self.agent_name = str(agent_name or "").strip()
        self._store: dict[tuple[Any, ...], Any] = {}

    # ── Key Normalization ─────────────────────────────────────

    @classmethod
    def _normalize_key(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, tuple):
            return tuple(cls._normalize_key(item) for item in value)
        if isinstance(value, list):
            return tuple(cls._normalize_key(item) for item in value)
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (str(key), cls._normalize_key(item))
                    for key, item in value.items()
                )
            )
        return str(value)

    def _build_slot(
        self,
        namespace: str,
        cache_key: Any,
        *,
        snapshot_id: str = "",
        session_id: str = "",
    ) -> tuple[Any, ...]:
        return (
            self.agent_name,
            str(snapshot_id or "").strip(),
            str(session_id or "").strip(),
            str(namespace or "").strip(),
            self._normalize_key(cache_key),
        )

    # ── CRUD Operations ─────────────────────────────────────

    def get(
        self,
        namespace: str,
        cache_key: Any,
        *,
        snapshot_id: str = "",
        session_id: str = "",
        default: Any = None,
    ) -> Any:
        slot = self._build_slot(namespace, cache_key, snapshot_id=snapshot_id, session_id=session_id)
        if slot not in self._store:
            return deepcopy(default)
        return deepcopy(self._store[slot])

    def set(
        self,
        namespace: str,
        cache_key: Any,
        value: Any,
        *,
        snapshot_id: str = "",
        session_id: str = "",
    ) -> None:
        slot = self._build_slot(namespace, cache_key, snapshot_id=snapshot_id, session_id=session_id)
        self._store[slot] = deepcopy(value)

    def has(
        self,
        namespace: str,
        cache_key: Any,
        *,
        snapshot_id: str = "",
        session_id: str = "",
    ) -> bool:
        slot = self._build_slot(namespace, cache_key, snapshot_id=snapshot_id, session_id=session_id)
        return slot in self._store

    def clear(
        self,
        *,
        namespace: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        if namespace is None and snapshot_id is None and session_id is None:
            self._store.clear()
            return

        normalized_namespace = None if namespace is None else str(namespace or "").strip()
        normalized_snapshot = None if snapshot_id is None else str(snapshot_id or "").strip()
        normalized_session = None if session_id is None else str(session_id or "").strip()

        remaining: dict[tuple[Any, ...], Any] = {}
        for slot, value in self._store.items():
            _, slot_snapshot, slot_session, slot_namespace, _ = slot
            if normalized_namespace is not None and slot_namespace != normalized_namespace:
                remaining[slot] = value
                continue
            if normalized_snapshot is not None and slot_snapshot != normalized_snapshot:
                remaining[slot] = value
                continue
            if normalized_session is not None and slot_session != normalized_session:
                remaining[slot] = value
                continue
        self._store = remaining
