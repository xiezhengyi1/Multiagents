from __future__ import annotations

from importlib import import_module

from .logger import setup_logger

__all__ = [
	"setup_logger",
	"ToolUsageSemanticJudge",
	"build_intent_encoding_semantic_judge_messages",
	"evaluate_intent_encoding_semantic_tool_usage",
	"evaluate_tool_usage",
	"evaluate_intent_encoding_tool_usage",
	"validate_tool_usage",
	"validate_intent_encoding_tool_usage",
]


_LAZY_EXPORTS = {
	"ToolUsageSemanticJudge": ("utils.tool_usage_judge", "ToolUsageSemanticJudge"),
	"build_intent_encoding_semantic_judge_messages": (
		"utils.tool_usage_judge",
		"build_intent_encoding_semantic_judge_messages",
	),
	"evaluate_intent_encoding_semantic_tool_usage": (
		"utils.tool_usage_judge",
		"evaluate_intent_encoding_semantic_tool_usage",
	),
	"evaluate_intent_encoding_tool_usage": ("utils.tool_usage_validation", "evaluate_intent_encoding_tool_usage"),
	"evaluate_tool_usage": ("utils.tool_usage_validation", "evaluate_tool_usage"),
	"validate_intent_encoding_tool_usage": ("utils.tool_usage_validation", "validate_intent_encoding_tool_usage"),
	"validate_tool_usage": ("utils.tool_usage_validation", "validate_tool_usage"),
}


def __getattr__(name):
	module_name, attr_name = _LAZY_EXPORTS.get(name, (None, None))
	if module_name is None:
		raise AttributeError(f"module 'utils' has no attribute '{name}'")
	module = import_module(module_name)
	value = getattr(module, attr_name)
	globals()[name] = value
	return value
