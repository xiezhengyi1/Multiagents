import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict

# ASCII 颜色代码
class LogColors:
    RESET = "\033[0m"
    DEBUG = "\033[36m"      # 青色
    INFO = "\033[92m"       # 绿色
    WARNING = "\033[93m"    # 黄色
    ERROR = "\033[91m"      # 红色
    CRITICAL = "\033[1;91m" # 粗体红色

class ColoredFormatter(logging.Formatter):
    """
    自定义日志格式化器，为不同级别的日志添加颜色
    """

    LEVEL_COLORS = {
        logging.DEBUG: LogColors.DEBUG,
        logging.INFO: LogColors.INFO,
        logging.WARNING: LogColors.WARNING,
        logging.ERROR: LogColors.ERROR,
        logging.CRITICAL: LogColors.CRITICAL,
    }

    def format(self, record):
        level_color = self.LEVEL_COLORS.get(record.levelno, LogColors.RESET)
        msg_color = getattr(record, "msg_color", LogColors.RESET)

        prefix = f"{level_color}[%(asctime)s] [{record.levelname}]: {LogColors.RESET}"
        formatter = logging.Formatter(prefix + "%(message)s" + LogColors.RESET, datefmt="%Y-%m-%d %H:%M:%S")

        original_msg = record.msg
        try:
            record.msg = f"{msg_color}{record.getMessage()}{LogColors.RESET}"
            return formatter.format(record)
        finally:
            record.msg = original_msg

def setup_logger(name="MultiAgents", level=logging.INFO, default_msg_color: str = None):
    """
    初始化并返回一个带有颜色输出的 Logger
    
    :param name: Logger 名称
    :param level: 日志级别 (默认 logging.INFO)
    :param default_msg_color: 日志内容默认颜色 (可选)
    :return: 配置好的 logger 对象
    """
    # 获取 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 Handler (防止日志重复打印)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建控制台 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # 设置自定义的颜色 Formatter
    console_handler.setFormatter(ColoredFormatter())

    # 设置内容默认颜色 (仅在未显式传入 msg_color 时生效)
    if default_msg_color:
        class _DefaultMsgColorFilter(logging.Filter):
            def filter(self, record):
                if not hasattr(record, "msg_color"):
                    record.msg_color = default_msg_color
                return True

        logger.addFilter(_DefaultMsgColorFilter())

    # 添加 Handler 到 Logger
    logger.addHandler(console_handler)
    
    # 还可以防止日志向上传播到根记录器（如果需要完全独立控制）
    logger.propagate = False

    return logger


def _format_fields(fields: Dict[str, Any]) -> str:
    """关键步骤: 将结构化字段统一格式化为 k=v 片段。"""
    if not fields:
        return ""
    parts = []
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    return " | " + ", ".join(parts)


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """统一事件日志输出。"""
    logger.log(level, f"[EVENT] {event}{_format_fields(fields)}")


def log_timing(logger: logging.Logger, metric: str, duration_s: float, level: int = logging.INFO, **fields: Any) -> None:
    """统一耗时日志输出，单位为毫秒。"""
    logger.log(level, f"[METRIC] {metric} | duration_ms={duration_s * 1000:.2f}{_format_fields(fields)}")


@contextmanager
def observe_time(logger: logging.Logger, metric: str, level: int = logging.INFO, **fields: Any):
    """关键步骤: 上下文计时器，自动记录代码块耗时。"""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log_timing(logger, metric, elapsed, level=level, **fields)
