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
class Flow:
    name: str
    flow_id: str
    service_type: str
    bw_ul: float
    bw_dl: float
    gbr_ul: float
    gbr_dl: float
    lat: float
    loss_req: float
    jitter_req: float
    priority: int
    old_slice: Optional[str] = None
    old_allocated_bw_ul: Optional[float] = None
    old_allocated_bw_dl: Optional[float] = None
    optimize_requested: bool = False

    # IBNS-specific defaults
    packet_size: float = 0.0
    arrival_rate: float = 0.0
    service_type_id: int = 1

    # Runtime metrics from simulator
    sim_throughput_ul: Optional[float] = None
    sim_throughput_dl: Optional[float] = None
    sim_latency: Optional[float] = None
    sim_jitter: Optional[float] = None
    sim_loss_rate: Optional[float] = None
    sim_packet_sent: Optional[int] = None
    sim_packet_received: Optional[int] = None
    five_tuple: Optional[Tuple[str, str, int, int, str]] = None

    @property
    def data_rate(self) -> float:
        if self.packet_size > 0 and self.arrival_rate > 0:
            return self.packet_size * self.arrival_rate
        r_ul = max(0.0, self.bw_ul * 1e6)
        r_dl = max(0.0, self.bw_dl * 1e6)
        if r_ul > 0 and r_dl > 0:
            return (2.0 * r_ul * r_dl) / (r_ul + r_dl)
        return max(r_ul, r_dl)


@dataclass
class App:
    name: str
    app_id: str
    flows: List[Flow]
    supi: Optional[str] = None
    total_bw_ul: float = field(init=False)
    total_bw_dl: float = field(init=False)
    min_lat: float = field(init=False)
    max_prio: int = field(init=False)

    def __post_init__(self):
        if not self.flows:
            self.total_bw_ul = 0.0
            self.total_bw_dl = 0.0
            self.min_lat = float("inf")
            self.max_prio = 0
        else:
            self.total_bw_ul = sum(flow.bw_ul for flow in self.flows)
            self.total_bw_dl = sum(flow.bw_dl for flow in self.flows)
            self.min_lat = min(flow.lat for flow in self.flows)
            self.max_prio = max(flow.priority for flow in self.flows)


@dataclass
class Slice:
    name: str
    sst: int
    sd: str
    snssai: str = field(init=False)
    total_bw_ul: float = 0.0
    total_bw_dl: float = 0.0
    current_load_bw_ul: float = 0.0
    current_load_bw_dl: float = 0.0
    latency: float = 0.0
    proc_delay: float = 0.0
    loss: float = 0.0
    jitter: float = 0.0
    reserved_bw: float = 0.0

    sim_utilization_ul: Optional[float] = None
    sim_utilization_dl: Optional[float] = None
    sim_latency: Optional[float] = None
    sim_jitter: Optional[float] = None
    sim_loss_rate: Optional[float] = None

    def __post_init__(self):
        self.snssai = f"{self.sst:02X}{self.sd}"

    def can_accommodate(self, flow: Flow) -> bool:
        if flow.service_type_id != self.sst:
            return False

        avail_ul = self.total_bw_ul - self.current_load_bw_ul - self.reserved_bw
        avail_dl = self.total_bw_dl - self.current_load_bw_dl - self.reserved_bw
        if avail_ul < flow.bw_ul or avail_dl < flow.bw_dl:
            return False
        return True


@dataclass
class Node:
    name: str
    cpu_capacity: float
    memory_capacity: float
    slices_hosted: List[str]
    id: int = -1
    type: str = "Generic"
    mec_capacity: float = 0.0
    prb_capacity: float = 0.0
    sim_cpu_utilization: Optional[float] = 0
    sim_mec_utilization: Optional[float] = 0
    sim_mem_utilization: Optional[float] = 0
    sim_prb_utilization: Optional[float] = 0

    @property
    def is_an(self):
        return self.type == "AN"

    @property
    def is_cn(self):
        return self.type == "CN"


@dataclass
class OptimizationConfig:
    rho: float = 0.8
    w1: float = 100.0
    w2: float = 50.0
    w3: float = 1000.0
    w4: float = 0.0
    alpha_cn: float = 0.04
    alpha_an: float = 0.01
    beta: float = 0.05
    prb: float = 0.18
    mec_overhead: List[float] = field(default_factory=lambda: [1, 4, 8])
    enable_sla_constraints: bool = True
    solver_time_limit: int = 30
