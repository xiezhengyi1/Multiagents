from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class NativeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NativeFlowSla(NativeModel):
    latency: float = Field(gt=0)
    jitter: float = Field(ge=0)
    loss_rate: float = Field(ge=0, le=1)
    bandwidth_ul: float = Field(gt=0)
    bandwidth_dl: float = Field(gt=0)
    guaranteed_bandwidth_ul: float = Field(gt=0)
    guaranteed_bandwidth_dl: float = Field(gt=0)
    priority: int = Field(ge=1, le=255)


class NativeFlowAllocation(NativeModel):
    current_slice_snssai: str = Field(pattern=r"^[0-9A-Fa-f]{8}$")
    allocated_bandwidth_ul: float = Field(gt=0)
    allocated_bandwidth_dl: float = Field(gt=0)
    source_action_id: str | None = None


class NativeFlowTraffic(NativeModel):
    five_tuple: tuple[str, str, int, int, Literal["tcp", "udp"]]
    packet_size: float = Field(gt=0)
    packet_size_unit: Literal["bits"]
    packet_size_bytes: int = Field(gt=0)
    arrival_rate: float = Field(gt=0)
    arrival_rate_unit: Literal["packets_per_second"]
    offered_rate_mbps: float = Field(gt=0)

    @model_validator(mode="after")
    def packet_size_units_match(self) -> NativeFlowTraffic:
        if self.packet_size != self.packet_size_bytes * 8:
            raise ValueError("packet_size must equal packet_size_bytes * 8")
        return self


class NativeFlow(NativeModel):
    id: str
    name: str
    flow_id: str
    supi: str
    app_id: str
    app_name: str
    sla: NativeFlowSla
    allocation: NativeFlowAllocation
    traffic: NativeFlowTraffic
    telemetry: dict[str, Any]
    ran: dict[str, Any]
    session: dict[str, Any]
    path: dict[str, Any]
    data_quality: dict[str, Any]
    ue_policy_route: dict[str, Any] | None = None

    @model_validator(mode="after")
    def identities_match(self) -> NativeFlow:
        if self.id != self.flow_id or self.id != self.name:
            raise ValueError("native flow id, flow_id, and name must match")
        return self


class NativeSliceCapacity(NativeModel):
    total_bandwidth_ul: float = Field(gt=0)
    total_bandwidth_dl: float = Field(gt=0)
    guaranteed_bandwidth_ul: float = Field(ge=0)
    guaranteed_bandwidth_dl: float = Field(ge=0)


class NativeSliceLoad(NativeModel):
    current_bandwidth_ul: float = Field(ge=0)
    current_bandwidth_dl: float = Field(ge=0)


class NativeSliceQos(NativeModel):
    latency: float = Field(gt=0)
    processing_delay: float = Field(ge=0)
    jitter: float = Field(ge=0)
    loss_rate: float = Field(ge=0, le=1)


class NativeSliceTelemetry(NativeModel):
    utilization_ul: float = Field(ge=0)
    utilization_dl: float = Field(ge=0)
    latency: float = Field(ge=0)
    jitter: float = Field(ge=0)
    loss_rate: float = Field(ge=0, le=1)


class NativeSliceState(NativeModel):
    slice_id: str
    snssai: str = Field(pattern=r"^[0-9A-Fa-f]{8}$")
    phase: Literal["running", "stopped", "deleted"]
    sequence: int = Field(ge=0)
    observed_at: datetime
    upf_containers: tuple[str, ...]
    ue_ids: tuple[str, ...]
    flow_ids: tuple[str, ...]
    capacity: NativeSliceCapacity
    capacity_unit: Literal["Mbps"]
    load: NativeSliceLoad
    load_unit: Literal["Mbps"]
    qos: NativeSliceQos
    qos_source: Literal["declared_flow_sla"]
    telemetry: NativeSliceTelemetry
    telemetry_source: Literal["flow_aggregate_worst_case"]


class NativeNodeCapacity(NativeModel):
    cpu: float = Field(gt=0)
    memory: float = Field(gt=0)
    mec: float = Field(ge=0)
    prb: float = Field(ge=0)


class NativeNodeCapacityUnits(NativeModel):
    cpu: Literal["cores"]
    memory: Literal["bytes"]
    mec: Literal["not_applicable"]
    prb: Literal["physical_resource_blocks"]


class NativeNodeTelemetry(NativeModel):
    cpu_utilization: float = Field(ge=0)
    memory_utilization: float = Field(ge=0)
    mec_utilization: float = Field(ge=0)
    prb_utilization: float | None = Field(ge=0)


class NativeRanTelemetry(NativeNodeTelemetry):
    ues: tuple[dict[str, Any], ...]


class NativeRanNode(NativeModel):
    id: str
    name: str
    node_type: Literal["AN"]
    capacity: NativeNodeCapacity
    capacity_units: NativeNodeCapacityUnits
    hosted_slice_snssais: tuple[str, ...]
    telemetry: NativeRanTelemetry


class NativeCoreNode(NativeModel):
    id: str
    name: str
    node_type: Literal["CN"]
    capacity: NativeNodeCapacity
    capacity_units: NativeNodeCapacityUnits
    hosted_slice_snssais: tuple[str, ...]
    telemetry: NativeNodeTelemetry
    dnn: str


class NativeSnapshot(NativeModel):
    snapshot_id: str
    run_id: str
    observed_from: datetime
    observed_to: datetime
    topology: dict[str, Any]
    slice_states: tuple[NativeSliceState, ...]
    sessions: tuple[dict[str, Any], ...]
    flows: tuple[NativeFlow, ...]
    ran_nodes: tuple[NativeRanNode, ...]
    core_nodes: tuple[NativeCoreNode, ...]
    containers: tuple[dict[str, Any], ...]
    mec_sites: tuple[dict[str, Any], ...]
    mobility: tuple[dict[str, Any], ...]
    active_actions: tuple[dict[str, Any], ...]
    trigger_event: str
    ue_policies: tuple[dict[str, Any], ...] = ()


class NativeActionReceipt(NativeModel):
    action_id: str
    run_id: str
    snapshot_before: str
    status: Literal["verified", "rejected", "failed"]
    dispatch: dict[str, Any]
    observed_effect: dict[str, Any]
    snapshot_after: str | None


__all__ = [
    "NativeActionReceipt",
    "NativeFlow",
    "NativeFlowAllocation",
    "NativeFlowSla",
    "NativeFlowTraffic",
    "NativeNodeCapacity",
    "NativeNodeTelemetry",
    "NativeRanNode",
    "NativeSliceCapacity",
    "NativeSliceState",
    "NativeSnapshot",
]
