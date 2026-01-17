from dataclasses import dataclass, field
from typing import List, Optional
import sys
import os

try:
    from utils.logger import setup_logger
except ImportError:
    # Fallback if running relative
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class Flow:
    """定义应用内的单个数据流需求"""
    name: str # 保留作为描述性名称
    flow_id: str # 流唯一标识
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

@dataclass
class App:
    """定义应用及其聚合需求"""
    name: str # 保留作为描述性名称
    app_id: str # 应用唯一标识
    flows: List[Flow]
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

    def __post_init__(self):
        # 自动生成 snssai 标识
        self.snssai = f"{self.sst:02X}{self.sd}"

@dataclass
class Node:
    """定义物理节点资源"""
    name: str
    cpu_capacity: float
    memory_capacity: float # 内存容量
    slices_hosted: List[str] # 节点托管的切片列表

@dataclass
class OptimizationConfig:
    """优化算法参数配置"""
    rho: float = 0.8   # 目标负载率
    w1: float = 100.0  # 负载均衡权重
    w2: float = 50.0   # 信令开销权重
    w3: float = 1000.0 # 体验损失权重
    w4: float = 0.0    # 丢包/抖动软约束权重
    alpha: float = 0.1 # 带宽转CPU消耗系数
    beta: float = 0.05 # 带宽转内存消耗系数
