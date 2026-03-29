from __future__ import annotations

from agent_runtime import AgentWorkspace, ArtifactCache


class IntentEncodingAgentCache(ArtifactCache):
    def __init__(self) -> None:
        super().__init__(AgentWorkspace.for_agent("intent_encoding"))
