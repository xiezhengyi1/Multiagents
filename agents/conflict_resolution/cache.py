from __future__ import annotations

from agent_runtime import AgentWorkspace, ArtifactCache


class ConflictResolutionAgentCache(ArtifactCache):
    def __init__(self) -> None:
        super().__init__(AgentWorkspace.for_agent("conflict_resolution"))
