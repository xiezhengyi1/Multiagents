from .contracts import FeedbackReport

__all__ = [
    "FeedbackReport",
    "PolicyDispatchAgent",
]


def __getattr__(name: str):
    if name == "PolicyDispatchAgent":
        from .agent import PolicyDispatchAgent

        return PolicyDispatchAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
