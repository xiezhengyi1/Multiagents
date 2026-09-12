"""Experiment contracts and an injected simulator/model execution boundary."""

from .contracts import ExperimentCase, ExperimentMethod, ExperimentResult
from .runner import ExperimentBackend, PreparedEpisode, run_case, run_cases

__all__ = [
    "ExperimentBackend",
    "ExperimentCase",
    "ExperimentMethod",
    "ExperimentResult",
    "PreparedEpisode",
    "run_case",
    "run_cases",
]
