from .runtime import (
    get_current_scenario,
    get_initial_scenario,
    init_main,
    initialize_scenario,
)

__all__ = [
    "get_current_scenario",
    "get_initial_scenario",
    "init_main",
    "initialize_scenario",
]


if __name__ == "__main__":
    init_main()
