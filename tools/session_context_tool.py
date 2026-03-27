import json
from typing import Optional

from langchain_core.tools import tool

from tools.db_tool import get_latest_session_context


@tool
def load_latest_session_context(status: Optional[str] = None) -> str:
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

    return (
        "Latest session context loaded from DB:\n"
        f"{json.dumps(latest, ensure_ascii=False, indent=2)}"
    )
