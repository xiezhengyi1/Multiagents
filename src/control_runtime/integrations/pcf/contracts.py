from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue

from shared.contracts import ContractModel, NonEmptyStr


PcfReadOperation = Literal[
    "ue_context",
    "sm_ue_context",
    "ue_flow_catalog",
    "sm_ue_flow_catalog",
    "ue_slice_subscription",
    "flow_target_search",
    "sm_flow_target_search",
    "am_policy_context",
    "am_policy_target_search",
]


class PcfReadResult(ContractModel):
    """Canonical result envelope for every read-only PCF grounding tool."""

    operation: PcfReadOperation
    snapshot_id: NonEmptyStr
    supi: NonEmptyStr | None = None
    data: dict[str, JsonValue] = Field(default_factory=dict)


PCF_READ_RESULT_SCHEMA = PcfReadResult.model_json_schema()


__all__ = ["PCF_READ_RESULT_SCHEMA", "PcfReadOperation", "PcfReadResult"]
