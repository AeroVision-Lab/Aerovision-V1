"""
统一的日志配置模块

提供统一的日志设置功能,支持从配置文件加载日志配置.
所有训练脚本应使用此模块而不是自己实现 logging 设置.

Usage:
    from configs.logger import setup_logger

    logger = setup_logger(
        name="AircraftClassifier",
        log_dir="logs/classify",
        config=config_obj  # 可选,从配置对象读取设置
    )
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config_loader import Config


def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    config: Optional[Config] = None,
    console_level: Optional[str] = None,
    file_level: Optional[str] = None,
) -> logging.Logger:
    """
    设置统一的日志记录器

    Args:
        name: Logger 名称
        log_dir: 日志文件保存目录 (如果为 None 则不保存文件)
        config: 配置对象 (可选,用于读取日志配置)
        console_level: 控制台日志级别 (可选,覆盖配置)
        file_level: 文件日志级别 (可选,覆盖配置)

    Returns:
        配置好的 Logger 实例
    """
    # 从配置读取日志设置
    if config:
        log_level = config.get("logging.level", "INFO")
        log_format = config.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        date_format = config.get("logging.date_format", "%Y-%m-%d %H:%M:%S")
        console_level = console_level or config.get("logging.console_level", "INFO")
        file_level = file_level or config.get("logging.level", "DEBUG")
        save_to_file = config.get("logging.save_to_file", True)
    else:
        log_level = "INFO"
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        console_level = console_level or "INFO"
        file_level = file_level or "DEBUG"
        save_to_file = True

    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 设置为最低级别,由 handler 控制实际输出
    logger.handlers.clear()  # 清除已有 handlers

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper()))
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 文件 handler
    if save_to_file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, file_level.upper()))
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info(f"日志文件: {log_file}")

    return logger
