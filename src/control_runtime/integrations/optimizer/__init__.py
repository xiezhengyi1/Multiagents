from __future__ import annotations

from .contracts import (
    JointOptimizationRequest,
    JointOptimizationResult,
    OptimizerInput,
    OptimizerScope,
)
from .request_factory import build_optimizer_request
from .result_view import summarize_optimizer_result


def run_joint_control_optimizer(*args, **kwargs):
    from .joint_control import (
        run_joint_control_optimizer as _run_joint_control_optimizer,
    )

    return _run_joint_control_optimizer(*args, **kwargs)


__all__ = [
    "JointOptimizationRequest",
    "JointOptimizationResult",
    "OptimizerInput",
    "OptimizerScope",
    "build_optimizer_request",
    "run_joint_control_optimizer",
    "summarize_optimizer_result",
]
