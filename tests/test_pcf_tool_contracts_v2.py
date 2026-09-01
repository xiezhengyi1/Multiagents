from __future__ import annotations

from types import SimpleNamespace

import pytest

from control_runtime.integrations.pcf import read_tools
from control_runtime.integrations.pcf.contracts import PcfReadResult
from shared.contracts import ExecutionScope
from shared.runtime import AgentRuntimeContext


def _runtime() -> SimpleNamespace:
    context = AgentRuntimeContext(
        agent_name="intent_encoding",
        scope=ExecutionScope(
            session_id="session-1",
            thread_id="session-1",
            snapshot_id="snapshot-1",
        ),
    )
    return SimpleNamespace(context=context)


def test_pcf_read_tool_returns_one_strict_json_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        read_tools,
        "get_ue_context_by_supi",
        lambda supi, *, snapshot_id: {
            "supi": supi,
            "snapshot_id": snapshot_id,
            "amPolicyContext": {},
        },
    )

    payload = read_tools.get_ue_context.func(
        purpose="retrieve_context",
        supi="imsi-208930000000001",
        runtime=_runtime(),
    )
    result = PcfReadResult.model_validate(payload)

    assert result.operation == "ue_context"
    assert result.snapshot_id == "snapshot-1"
    assert result.supi == "imsi-208930000000001"
    assert result.data["found"] is True
    assert read_tools.get_ue_context.metadata["contract"]["output_schema"]


def test_pcf_read_tool_rejects_missing_identity_instead_of_returning_error_text() -> None:
    with pytest.raises(ValueError, match="requires supi"):
        read_tools.get_ue_context.func(
            purpose="retrieve_context",
            supi="",
            runtime=_runtime(),
        )
