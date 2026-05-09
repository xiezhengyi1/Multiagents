"""Runtime orchestrator entrypoints for the refactored control system."""

from .main_control_orchestrator import MainControlOrchestrator
from .main_control_support import ControlRoundResult, ControlRoundTrace

__all__ = [
    "ControlRoundResult",
    "ControlRoundTrace",
    "MainControlOrchestrator",
    "SingleAgentOrchestrator",
]


def __getattr__(name: str):
    if name == "SingleAgentOrchestrator":
        from .single_agent_orchestrator import SingleAgentOrchestrator

        return SingleAgentOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
