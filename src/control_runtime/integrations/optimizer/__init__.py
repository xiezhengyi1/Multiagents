from __future__ import annotations


def run_joint_control_optimizer(*args, **kwargs):
    from .joint_control import run_joint_control_optimizer as _run_joint_control_optimizer

    return _run_joint_control_optimizer(*args, **kwargs)


__all__ = ["run_joint_control_optimizer"]
