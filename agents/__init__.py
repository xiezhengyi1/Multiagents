from __future__ import annotations

from importlib import import_module

__all__ = [
    "AssuranceDiagnosisTool",
    "ConflictResolutionTool",
    "FeedbackReport",
    "FeatureFusionLayer",
    "IntentEncodingAgent",
    "MemoryManager",
    "NaturalLanguageEncoder",
    "OptimizationStrategyAgent",
    "PolicyDispatchAgent",
    "UserDataEncoder",
    "db_tool",
    "knowledge_tool",
    "network_status",
]


_LAZY_EXPORTS = {
    "FeatureFusionLayer": ("agents.Embedding", "FeatureFusionLayer"),
    "NaturalLanguageEncoder": ("agents.Embedding", "NaturalLanguageEncoder"),
    "UserDataEncoder": ("agents.Embedding", "UserDataEncoder"),
    "MemoryManager": ("agents.MemoryManager", "MemoryManager"),
    "AssuranceDiagnosisTool": ("agents.assurance_diagnosis", "AssuranceDiagnosisTool"),
    "ConflictResolutionTool": ("agents.conflict_resolution", "ConflictResolutionTool"),
    "IntentEncodingAgent": ("agents.intent_encoding", "IntentEncodingAgent"),
    "OptimizationStrategyAgent": ("agents.optimization_strategy", "OptimizationStrategyAgent"),
    "FeedbackReport": ("agents.policy_dispatch", "FeedbackReport"),
    "PolicyDispatchAgent": ("agents.policy_dispatch", "PolicyDispatchAgent"),
    "db_tool": ("agents.tools", "db_tool"),
    "knowledge_tool": ("agents.tools", "knowledge_tool"),
    "network_status": ("agents.tools", "network_status"),
}


def __getattr__(name):
    module_name, attr_name = _LAZY_EXPORTS.get(name, (None, None))
    if module_name is None:
        raise AttributeError(f"module 'agents' has no attribute '{name}'")
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
