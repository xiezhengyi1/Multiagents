"""Optimizer 包对外导出。

关键步骤：统一暴露模型、引擎与接口，便于上层模块稳定导入。
"""

from .models import App, Flow, Slice, Node, OptimizationConfig
from .engine import SliceOptimizationEngine, IBNSOptimizationEngine


def optimize_network_slices(*args, **kwargs):
	# 关键步骤：延迟导入，避免 init_scenario <-> optimizer.interface 循环依赖
	from .interface import optimize_network_slices as _impl
	return _impl(*args, **kwargs)


def optimize_ibns_network(*args, **kwargs):
	# 关键步骤：延迟导入，避免包初始化时触发循环导入
	from .interface import optimize_ibns_network as _impl
	return _impl(*args, **kwargs)

__all__ = [
	"App",
	"Flow",
	"Slice",
	"Node",
	"OptimizationConfig",
	"SliceOptimizationEngine",
	"IBNSOptimizationEngine",
	"optimize_network_slices",
	"optimize_ibns_network",
]
