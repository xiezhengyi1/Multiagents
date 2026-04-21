from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os
import sys

try:
    from utils.logger import setup_logger
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ServiceType:
    id: int
    name: str
    critical_kpis: List[int]


@dataclass
class SLAProfile:
    service_type_id: int
    kpi_thresholds: Dict[int, float]


@dataclass
class Link:
    u: int
    v: int
    capacity: float


@dataclass
class Path:
    id: str
    an_id: int
    links: List[Tuple[int, int]]


@dataclass
class FlowService:
    service_type: str
    service_type_id: int = 1


@dataclass
class FlowTraffic:
    packet_size: float = 0.0
    arrival_rate: float = 0.0
    five_tuple: Optional[Tuple[str, str, int, int, str]] = None


@dataclass
class FlowSLA:
    bandwidth_ul: float
    bandwidth_dl: float
    guaranteed_bandwidth_ul: float
    guaranteed_bandwidth_dl: float
    latency: float
    jitter: float
    loss_rate: float
    priority: int


@dataclass
class FlowAllocation:
    current_slice_snssai: Optional[str] = None
    allocated_bandwidth_ul: Optional[float] = None
    allocated_bandwidth_dl: Optional[float] = None
    optimize_requested: bool = False


@dataclass
class FlowTelemetry:
    throughput_ul: Optional[float] = None
    throughput_dl: Optional[float] = None
    latency: Optional[float] = None
    jitter: Optional[float] = None
    loss_rate: Optional[float] = None
    packet_sent: Optional[int] = None
    packet_received: Optional[int] = None


@dataclass
class Flow:
    id: str
    name: str
    service: FlowService
    sla: FlowSLA
    traffic: FlowTraffic = field(default_factory=FlowTraffic)
    allocation: FlowAllocation = field(default_factory=FlowAllocation)
    telemetry: FlowTelemetry = field(default_factory=FlowTelemetry)

    @property
    def data_rate(self) -> float:
        if self.traffic.packet_size > 0 and self.traffic.arrival_rate > 0:
            return self.traffic.packet_size * self.traffic.arrival_rate
        r_ul = max(0.0, self.sla.bandwidth_ul * 1e6)
        r_dl = max(0.0, self.sla.bandwidth_dl * 1e6)
        if r_ul > 0 and r_dl > 0:
            return (2.0 * r_ul * r_dl) / (r_ul + r_dl)
        return max(r_ul, r_dl)


@dataclass
class AppSummary:
    total_bandwidth_ul: float
    total_bandwidth_dl: float
    min_latency: float
    max_priority: int


@dataclass
class App:
    id: str
    name: str
    flows: List[Flow]
    supi: Optional[str] = None

    @property
    def summary(self) -> AppSummary:
        if not self.flows:
            return AppSummary(
                total_bandwidth_ul=0.0,
                total_bandwidth_dl=0.0,
                min_latency=float("inf"),
                max_priority=0,
            )
        return AppSummary(
            total_bandwidth_ul=sum(flow.sla.bandwidth_ul for flow in self.flows),
            total_bandwidth_dl=sum(flow.sla.bandwidth_dl for flow in self.flows),
            min_latency=min(flow.sla.latency for flow in self.flows),
            max_priority=max(flow.sla.priority for flow in self.flows),
        )


@dataclass
class SliceCapacity:
    total_bandwidth_ul: float = 0.0
    total_bandwidth_dl: float = 0.0
    reserved_bandwidth_ul: float = 0.0
    reserved_bandwidth_dl: float = 0.0


@dataclass
class SliceLoad:
    current_bandwidth_ul: float = 0.0
    current_bandwidth_dl: float = 0.0


@dataclass
class SliceQos:
    latency: float = 0.0
    processing_delay: float = 0.0
    jitter: float = 0.0
    loss_rate: float = 0.0


@dataclass
class SliceTelemetry:
    utilization_ul: Optional[float] = None
    utilization_dl: Optional[float] = None
    latency: Optional[float] = None
    jitter: Optional[float] = None
    loss_rate: Optional[float] = None


@dataclass
class Slice:
    name: str
    sst: int
    sd: str
    capacity: SliceCapacity = field(default_factory=SliceCapacity)
    load: SliceLoad = field(default_factory=SliceLoad)
    qos: SliceQos = field(default_factory=SliceQos)
    telemetry: SliceTelemetry = field(default_factory=SliceTelemetry)
    snssai: str = field(init=False)

    def __post_init__(self):
        self.snssai = f"{self.sst:02X}{self.sd}"

    def can_accommodate(self, flow: Flow) -> bool:
        if flow.service.service_type_id != self.sst:
            return False

        avail_ul = (
            self.capacity.total_bandwidth_ul
            - self.load.current_bandwidth_ul
            - self.capacity.reserved_bandwidth_ul
        )
        avail_dl = (
            self.capacity.total_bandwidth_dl
            - self.load.current_bandwidth_dl
            - self.capacity.reserved_bandwidth_dl
        )
        if avail_ul < flow.sla.bandwidth_ul or avail_dl < flow.sla.bandwidth_dl:
            return False
        return True


@dataclass
class NodeCapacity:
    cpu: float
    memory: float
    mec: float = 0.0
    prb: float = 0.0


@dataclass
class NodeTelemetry:
    cpu_utilization: Optional[float] = 0
    mec_utilization: Optional[float] = 0
    memory_utilization: Optional[float] = 0
    prb_utilization: Optional[float] = 0


@dataclass
class Node:
    id: int
    name: str
    node_type: str
    capacity: NodeCapacity
    hosted_slice_snssais: List[str]
    telemetry: NodeTelemetry = field(default_factory=NodeTelemetry)

    @property
    def is_an(self):
        return self.node_type == "AN"

    @property
    def is_cn(self):
        return self.node_type == "CN"


@dataclass
class AMPolicyState:
    """关键步骤：记录当前 AM 策略的旧状态，供 MILP 计算变更代价。"""
    old_allowed_snssais: List[str] = field(default_factory=list)
    old_target_snssais: List[str] = field(default_factory=list)
    old_rfsp: int = 1
    old_triggers: List[str] = field(default_factory=list)
    old_ue_ambr_ul: float = 0.0
    old_ue_ambr_dl: float = 0.0
    rfsp_max: int = 8
    ambr_headroom: float = 0.2
    trigger_signal_costs: Dict[str, float] = field(default_factory=lambda: {
        "LOC_CH": 1.0,
        "PRA_CH": 0.6,
        "ALLOWED_NSSAI_CH": 0.3,
        "RFSP_CH": 0.2,
        "UE_AMBR_CH": 0.2,
        "NWDAF_DATA_CH": 0.4,
    })
    mandatory_triggers: List[str] = field(default_factory=lambda: ["LOC_CH"])


@dataclass
class OptimizationConfig:
    rho: float = 0.8
    w1: float = 100.0
    w2: float = 50.0
    w3: float = 1000.0
    w4: float = 0.0
    w5: float = 30.0
    w6: float = 10.0
    w7: float = 5.0
    alpha_cn: float = 0.04
    alpha_an: float = 0.01
    beta: float = 0.05
    prb: float = 0.18
    mec_overhead: List[float] = field(default_factory=lambda: [1, 4, 8])
    enable_sla_constraints: bool = True
    enable_am_optimization: bool = False
    am_policy_state: Optional[AMPolicyState] = None
    solver_time_limit: int = 30
