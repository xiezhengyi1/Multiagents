from __future__ import annotations

from agent_runtime import AgentWorkspace, ArtifactCache


class AssuranceDiagnosisAgentCache(ArtifactCache):
    def __init__(self) -> None:
        super().__init__(AgentWorkspace.for_agent("assurance_diagnosis"))
