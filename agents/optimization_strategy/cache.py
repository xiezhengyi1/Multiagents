from __future__ import annotations

from agent_runtime import AgentWorkspace, ArtifactCache


class OptimizationStrategyAgentCache(ArtifactCache):
    def __init__(self) -> None:
        super().__init__(AgentWorkspace.for_agent("optimization_strategy"))
