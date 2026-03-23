from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import sys
import os

try:
    from utils.logger import setup_logger
except ImportError:
    # Fallback if running relative
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from utils.logger import setup_logger

logger = setup_logger(__name__)

# --- IBNS Enhancements ---

@dataclass
class ServiceType:
    id: int
    name: str # e.g., HRLLC, MC, IC
    critical_kpis: List[int] # e.g., [1, 3] for HRLLC
    # 1: Latency, 2: Throughput, 3: Reliability, 4: Connection Density

@dataclass
class SLAProfile:
    """SLA Thresholds (tau) for various services"""
    service_type_id: int
    kpi_thresholds: Dict[int, float] # {KPI_ID: Threshold}

@dataclass
class Link:
    u: int
    v: int
    capacity: float # bits/s

@dataclass
class Path:
    id: str
    an_id: int
    links: List[Tuple[int, int]] # List of (u, v)

@dataclass
class Flow:
    """定义应用内的单个数据流需求"""
    name: str # 保留作为描述性名称
    flow_id: str # 流唯一标识
    service_type: str # 业务类型，例如 URLLC, eMBB, mMTC
    bw_ul: float       # 上行带宽 (Mbps)
    bw_dl: float       # 下行带宽 (Mbps)
    gbr_ul: float      # 上行保证比特率 (Mbps)
    gbr_dl: float      # 下行保证比特率 (Mbps)
    lat: float      # 时延 (ms)
    loss_req: float # 丢包率上限 (0~1)
    jitter_req: float # 抖动上限 (ms)
    priority: int   # 优先级 (数值越小越高)
    old_slice: Optional[str] = None # 流的原切片名称 (实为 S-NSSAI)
    old_allocated_bw_ul: Optional[float] = None # 上一次分配的实际上行带宽
    old_allocated_bw_dl: Optional[float] = None # 上一次分配的实际下行带宽

    # IBNS Specific (Defaults provided for compatibility)
    packet_size: float = 0.0 # bits
    arrival_rate: float = 0.0 # packets/sec
    service_type_id: int = 1 # Default to HRLLC or Generic

    # Runtime metrics from simulator (ns-3)
    sim_throughput_ul: Optional[float] = None # 实测上行吞吐 (Mbps)
    sim_throughput_dl: Optional[float] = None # 实测下行吞吐 (Mbps)
    sim_latency: Optional[float] = None # 实测时延 (ms)
    sim_jitter: Optional[float] = None # 实测抖动 (ms)
    sim_loss_rate: Optional[float] = None # 实测丢包率 (0~1)
    sim_packet_sent: Optional[int] = None # 发送包数
    sim_packet_received: Optional[int] = None # 接收包数
    five_tuple: Optional[Tuple[str, str, int, int, str]] = None # 五元组: (源IP, 目的IP, 源端口, 目的端口, 传输协议)

    @property
    def data_rate(self) -> float:
        """Calculate data rate in bits/second"""
        if self.packet_size > 0 and self.arrival_rate > 0:
            return self.packet_size * self.arrival_rate
        # Fallback: 与IBNS引擎保持一致，使用UL/DL双向等效吞吐。
        r_ul = max(0.0, self.bw_ul * 1e6)
        r_dl = max(0.0, self.bw_dl * 1e6)
        if r_ul > 0 and r_dl > 0:
            return (2.0 * r_ul * r_dl) / (r_ul + r_dl)
        return max(r_ul, r_dl)

@dataclass
class App:
    """定义应用及其聚合需求"""
    name: str # 保留作为描述性名称
    app_id: str # 应用唯一标识
    flows: List[Flow]
    supi: Optional[str] = None # 用户标识 (如 imsi-...)
    total_bw_ul: float = field(init=False)
    total_bw_dl: float = field(init=False)
    min_lat: float = field(init=False)
    max_prio: int = field(init=False)

    def __post_init__(self):
        if not self.flows:
            self.total_bw_ul = 0.0
            self.total_bw_dl = 0.0
            self.min_lat = float('inf')
            self.max_prio = 0
        else:
            self.total_bw_ul = sum(f.bw_ul for f in self.flows)
            self.total_bw_dl = sum(f.bw_dl for f in self.flows)
            self.min_lat = min(f.lat for f in self.flows)
            self.max_prio = max(f.priority for f in self.flows)

@dataclass
class Slice:
    """定义网络切片资源与状态"""
    name: str # 描述性名称
    sst: int        # 切片服务类型
    sd: str         # 切片微分器
    snssai: str = field(init=False) # 唯一标识 (SST-SD), 自动生成
    total_bw_ul: float # 总带宽容量
    total_bw_dl: float # 总带宽容量
    current_load_bw_ul: float # 当前基础负载
    current_load_bw_dl: float # 当前基础负载
    latency: float  # 链路传输时延
    proc_delay: float # 处理时延
    loss: float # 切片丢包率 (0~1)
    jitter: float # 切片抖动 (ms)
    reserved_bw: float # 不可抢占的保留带宽

    # Runtime metrics from simulator (ns-3)
    sim_utilization_ul: Optional[float] = None # 实测上行利用率 (0~1)
    sim_utilization_dl: Optional[float] = None # 实测下行利用率 (0~1)
    sim_latency: Optional[float] = None # 实测切片时延 (ms)
    sim_jitter: Optional[float] = None # 实测切片抖动 (ms)
    sim_loss_rate: Optional[float] = None # 实测切片丢包率 (0~1)

    def __post_init__(self):
        # 自动生成 snssai 标识
        self.snssai = f"{self.sst:02X}{self.sd}"

    def can_accommodate(self, flow: Flow) -> bool:
        """检查切片是否有足够的剩余资源容纳 Flow"""
        # 1. 服务类型匹配 (简单逻辑：SST 对应 ServiceType ID)
        # TODO: 更复杂的SST映射逻辑
        if flow.service_type_id != self.sst:
            return False
            
        # 2. 带宽余量检查
        avail_ul = self.total_bw_ul - self.current_load_bw_ul - self.reserved_bw
        avail_dl = self.total_bw_dl - self.current_load_bw_dl - self.reserved_bw
        
        if avail_ul < flow.bw_ul or avail_dl < flow.bw_dl:
            return False
            
        # 3. 如果需要，还可以检查 Latency、Loss 等是否在范围内
        # 这里只做容量准入
        return True

@dataclass
class Node:
    """定义物理节点资源"""
    name: str
    cpu_capacity: float
    memory_capacity: float # 内存容量 (Legacy)
    slices_hosted: List[str] # 节点托管的切片列表 (Legacy)

    # IBNS Fields
    id: int = -1 # Unique ID for graph
    type: str = 'Generic' # 'CN', 'AN', 'FN'
    mec_capacity: float = 0.0
    prb_capacity: float = 0.0 # Only for AN

    # Runtime metrics from simulator (ns-3)
    sim_cpu_utilization: Optional[float] = 0 # CPU 利用率 (0~1)
    sim_mec_utilization: Optional[float] = 0 # MEC 利用率 (0~1)
    sim_mem_utilization: Optional[float] = 0 # 内存利用率 (0~1)
    sim_prb_utilization: Optional[float] = 0 # 无线 PRB 利用率 (0~1, AN 节点)

    @property
    def is_an(self):
        return self.type == 'AN'
    
    @property
    def is_cn(self):
        return self.type == 'CN'

@dataclass
class OptimizationConfig:
    """优化算法参数配置"""
    rho: float = 0.8   # 目标负载率
    w1: float = 100.0  # 负载均衡权重
    w2: float = 50.0   # 信令开销权重
    w3: float = 1000.0 # 体验损失权重
    w4: float = 0.0    # 丢包/抖动软约束权重
    alpha_cn: float = 0.04 # 带宽转CPU消耗系数
    alpha_an: float = 0.01 # 带宽转MEC消耗系数
    beta: float = 0.05 # 带宽转内存消耗系数
    prb: float = 0.18 # 每 prb 带宽 Mbps (仅AN相关)
    mec_overhead: List[float] = field(default_factory=lambda: [1, 4, 8]) # MEC 处理开销占比 cycles/Mbps

    # IBNS Specific
    enable_sla_constraints: bool = True
