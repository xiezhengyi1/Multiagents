import logging
import sys

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
    
    # 定义每种日志级别的格式
    # fmt 可以根据需要调整，例如："[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s"
    FORMATS = {
        logging.DEBUG:    LogColors.DEBUG + "[%(asctime)s] [DEBUG]: %(message)s" + LogColors.RESET,
        logging.INFO:     LogColors.INFO + "[%(asctime)s] [INFO]: %(message)s" + LogColors.RESET,
        logging.WARNING:  LogColors.WARNING + "[%(asctime)s] [WARNING]: %(message)s" + LogColors.RESET,
        logging.ERROR:    LogColors.ERROR + "[%(asctime)s] [ERROR]: %(message)s" + LogColors.RESET,
        logging.CRITICAL: LogColors.CRITICAL + "[%(asctime)s] [CRITICAL]: %(message)s" + LogColors.RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

def setup_logger(name="MultiAgents", level=logging.INFO):
    """
    初始化并返回一个带有颜色输出的 Logger
    
    :param name: Logger 名称
    :param level: 日志级别 (默认 logging.INFO)
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

    # 添加 Handler 到 Logger
    logger.addHandler(console_handler)
    
    # 还可以防止日志向上传播到根记录器（如果需要完全独立控制）
    logger.propagate = False

    return logger
