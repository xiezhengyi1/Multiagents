"""Canonical namespace for code shared across project runtimes.

Import these modules as ``shared.*``.  Loading the source-layout spelling
``src.shared.*`` would create a second module graph with split global state.
"""

if __name__ != "shared":
    raise ImportError("Use the canonical 'shared' package name; 'src.shared' is unsupported")

