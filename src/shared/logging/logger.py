from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator


def _reconfigure_stream_utf8(stream: Any) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


_reconfigure_stream_utf8(sys.stdout)
_reconfigure_stream_utf8(sys.stderr)


class LogColors:
    RESET = "\033[0m"
    DEBUG = "\033[36m"
    INFO = "\033[92m"
    WARNING = "\033[93m"
    ERROR = "\033[91m"
    CRITICAL = "\033[1;91m"


class ColoredFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: LogColors.DEBUG,
        logging.INFO: LogColors.INFO,
        logging.WARNING: LogColors.WARNING,
        logging.ERROR: LogColors.ERROR,
        logging.CRITICAL: LogColors.CRITICAL,
    }

    def format(self, record: logging.LogRecord) -> str:
        level_color = self.LEVEL_COLORS.get(record.levelno, LogColors.RESET)
        msg_color = getattr(record, "msg_color", LogColors.RESET)
        prefix = f"{level_color}[%(asctime)s] [{record.levelname}]: {LogColors.RESET}"
        formatter = logging.Formatter(prefix + "%(message)s" + LogColors.RESET, datefmt="%Y-%m-%d %H:%M:%S")

        original_msg = record.msg
        original_args = record.args
        try:
            record.msg = f"{msg_color}{record.getMessage()}{LogColors.RESET}"
            record.args = ()
            return formatter.format(record)
        finally:
            record.msg = original_msg
            record.args = original_args


def setup_logger(name: str = "MultiAgents", level: int = logging.INFO, default_msg_color: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter())

    if default_msg_color:
        class _DefaultMsgColorFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                if not hasattr(record, "msg_color"):
                    record.msg_color = default_msg_color
                return True

        logger.addFilter(_DefaultMsgColorFilter())

    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


def _format_fields(fields: Dict[str, Any]) -> str:
    if not fields:
        return ""
    return " | " + ", ".join(f"{key}={value}" for key, value in fields.items())


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, f"[EVENT] {event}{_format_fields(fields)}")


def log_timing(logger: logging.Logger, metric: str, duration_s: float, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, f"[METRIC] {metric} | duration_ms={duration_s * 1000:.2f}{_format_fields(fields)}")


@contextmanager
def observe_time(logger: logging.Logger, metric: str, level: int = logging.INFO, **fields: Any) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        log_timing(logger, metric, time.perf_counter() - start, level=level, **fields)
