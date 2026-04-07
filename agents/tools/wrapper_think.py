from langchain_classic.agents import tool

@tool
def think_tool(message: str) -> str:
    """Record visible reasoning before any non-think tool call."""
    return message