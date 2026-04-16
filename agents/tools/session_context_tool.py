import json
from typing import Optional

from langchain.tools import ToolRuntime, tool

from agents.tools.wrapper_think import tool_with_reason

from agent_runtime import AgentRuntimeContext
from agents.tools.db_tool import get_latest_session_context


@tool_with_reason
def load_latest_session_context(
    status: Optional[str] = None,
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """
    Load the latest session context from the session_context table.

    Args:
        status: Optional session status filter such as active, completed, or failed.

    Returns:
        The latest session context as formatted JSON text.
    """
    latest = get_latest_session_context(status=status)
    if latest is None:
        if status:
            return f"No session_context row found for status={status}."
        return "No session_context row found."

    prefix = ""
    if runtime is not None:
        ctx = runtime.context
        prefix = f"[agent={ctx.agent_name}][session={ctx.session_id}][snapshot={ctx.snapshot_id}] "

    return f"{prefix}Latest session context loaded from DB:\n{json.dumps(latest, ensure_ascii=False, indent=2)}"
