from __future__ import annotations

import logging


def setup_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


__all__ = ["setup_logger"]
