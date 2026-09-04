from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolRuntime

from shared.contracts.coordination import CoordinationEvent
from shared.runtime.journal import EpisodeJournal
from shared.runtime.langchain_trace import AgentRunContext

from .adapter import SimulatorAdapter
from .contracts import EnvironmentObservation, ObservationRequest


class SimulatorObservationSink:
    """Turn every live simulator read into an immutable coordination event."""

    def __init__(
        self,
        *,
        adapter: SimulatorAdapter,
        journal: EpisodeJournal,
        trigger_event: CoordinationEvent[Any],
        target_agent: str,
    ) -> None:
        self.adapter = adapter
        self.journal = journal
        self.trigger_event = trigger_event
        self.target_agent = target_agent
        self._event_ids: list[str] = []

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(self._event_ids)

    def record(self, *, query_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        request = ObservationRequest(query_name=query_name, arguments=arguments)
        observation = self.adapter.observe(request)
        if observation.query != request:
            raise ValueError(
                "simulator observation query does not match the requested query"
            )
        event = self.journal.append_event(
            round_index=self.trigger_event.scope.round_index,
            source_agent="simulator_adapter",
            target_agent=self.target_agent,
            message_type="observation",
            in_reply_to=self.trigger_event.event_id,
            input_artifact_ids=[self.trigger_event.event_id],
            payload=observation,
        )
        self._event_ids.append(event.event_id)
        return {
            "observation_event_id": event.event_id,
            "base_snapshot_id": observation.base_snapshot_id,
            "result": observation.result,
        }


def observation_query_names(
    *,
    journal: EpisodeJournal,
    event_ids: list[str],
) -> set[str]:
    queries: set[str] = set()
    for event_id in event_ids:
        event = journal.get_event(event_id)
        if isinstance(event.payload, EnvironmentObservation):
            queries.add(event.payload.query.query_name)
    return queries


def _sink(runtime: ToolRuntime[AgentRunContext]) -> SimulatorObservationSink:
    sink = runtime.context.observation_sink
    if not isinstance(sink, SimulatorObservationSink):
        raise TypeError("simulator observation tool requires SimulatorObservationSink")
    return sink


@tool
def get_sm_ue_flow_catalog(
    supi: str,
    runtime: ToolRuntime[AgentRunContext],
) -> dict[str, Any]:
    """Read the current SM application and QoS-flow catalog for one exact SUPI."""

    return _sink(runtime).record(
        query_name="get_sm_ue_flow_catalog",
        arguments={"supi": supi},
    )


@tool
def search_sm_flow_targets(
    supi: str,
    query: str,
    runtime: ToolRuntime[AgentRunContext],
) -> dict[str, Any]:
    """Search current SM flow bindings for a named application or flow target."""

    return _sink(runtime).record(
        query_name="search_sm_flow_targets",
        arguments={"supi": supi, "query": query},
    )


@tool
def get_am_policy_context(
    supi: str,
    runtime: ToolRuntime[AgentRunContext],
) -> dict[str, Any]:
    """Read the current access-and-mobility policy context for one exact SUPI."""

    return _sink(runtime).record(
        query_name="get_am_policy_context",
        arguments={"supi": supi},
    )


@tool
def search_am_policy_targets(
    supi: str,
    query: str,
    runtime: ToolRuntime[AgentRunContext],
) -> dict[str, Any]:
    """Search current mobility associations, RFSP, S-NSSAI, and service-area targets."""

    return _sink(runtime).record(
        query_name="search_am_policy_targets",
        arguments={"supi": supi, "query": query},
    )


@tool
def get_ue_slice_subscription(
    supi: str,
    runtime: ToolRuntime[AgentRunContext],
) -> dict[str, Any]:
    """Read authoritative subscribed and allowed S-NSSAI data for one exact SUPI."""

    return _sink(runtime).record(
        query_name="get_ue_slice_subscription",
        arguments={"supi": supi},
    )


@tool
def preview_qos_optimizer(
    supi: str,
    flow_ids: list[str],
    runtime: ToolRuntime[AgentRunContext],
) -> dict[str, Any]:
    """Read a deterministic QoS optimization preview for already grounded flow IDs."""

    return _sink(runtime).record(
        query_name="preview_qos_optimizer",
        arguments={"supi": supi, "flow_ids": flow_ids},
    )


@tool
def inspect_mobility_ue_policies(
    supi: str,
    runtime: ToolRuntime[AgentRunContext],
) -> dict[str, Any]:
    """Read executable mobility-policy bindings for one already grounded SUPI."""

    return _sink(runtime).record(
        query_name="inspect_mobility_ue_policies",
        arguments={"supi": supi},
    )


SBA_OBSERVATION_TOOLS: tuple[BaseTool, ...] = (
    get_sm_ue_flow_catalog,
    search_sm_flow_targets,
    get_am_policy_context,
    search_am_policy_targets,
    get_ue_slice_subscription,
)

PPA_OBSERVATION_TOOLS: tuple[BaseTool, ...] = (
    preview_qos_optimizer,
    inspect_mobility_ue_policies,
)


__all__ = [
    "PPA_OBSERVATION_TOOLS",
    "SBA_OBSERVATION_TOOLS",
    "SimulatorObservationSink",
    "get_am_policy_context",
    "get_sm_ue_flow_catalog",
    "get_ue_slice_subscription",
    "inspect_mobility_ue_policies",
    "observation_query_names",
    "preview_qos_optimizer",
    "search_am_policy_targets",
    "search_sm_flow_targets",
]
