from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


class TokenBudget:
    """Token-aware context pruning for projected payloads."""

    def __init__(self, max_tokens: int, reserved_for_output: int = 2000) -> None:
        self.limit = max(0, int(max_tokens) - int(reserved_for_output))

    def estimate(self, payload: Any) -> int:
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, default=str)
        return max(1, len(text) // 4)

    def project_with_budget(self, projector: Any, instance: Any) -> dict[str, Any]:
        result = projector.project(instance)
        return self.prune(result)

    def prune(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(payload)
        for key in ("rationale", "explanation", "planning_rationale"):
            if self.estimate(result) <= self.limit:
                return result
            result.pop(key, None)
        if self.estimate(result) <= self.limit:
            return result
        self._drop_prefixed(result, "strictest_")
        if self.estimate(result) <= self.limit:
            return result
        for key in ("flows", "policies", "qos_target_envelopes", "handoff_history"):
            value = result.get(key)
            if isinstance(value, list) and len(value) > 1:
                result[key] = value[-1:]
            if self.estimate(result) <= self.limit:
                return result
        return result

    def _drop_prefixed(self, payload: Any, prefix: str) -> None:
        if isinstance(payload, dict):
            for key in list(payload.keys()):
                if str(key).startswith(prefix):
                    payload.pop(key, None)
                else:
                    self._drop_prefixed(payload[key], prefix)
        elif isinstance(payload, list):
            for item in payload:
                self._drop_prefixed(item, prefix)
