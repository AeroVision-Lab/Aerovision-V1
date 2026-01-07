"""
日志配置

结构化日志输出
"""

import sys
import logging
from typing import Optional

from .config import settings


def setup_logging(
    level: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    配置应用日志

    Args:
        level: 日志级别
        log_format: 日志格式

    Returns:
        根日志记录器
    """
    level = level or settings.log_level
    log_format = log_format or settings.log_format

    # 配置根日志记录器
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # 获取应用日志记录器
    logger = logging.getLogger("aerovision")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 减少第三方库日志噪音
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("paddleocr").setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取命名日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器
    """
    return logging.getLogger(f"aerovision.{name}")


# 初始化日志
logger = setup_logging()
