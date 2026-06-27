from __future__ import annotations

from typing import Any

from .base import json_mapping, without_empty_values


class IntentEvidenceProjector:
    @classmethod
    def project(cls, instance: Any) -> dict[str, Any]:
        return without_empty_values(json_mapping(instance))
