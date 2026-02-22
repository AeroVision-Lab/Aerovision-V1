# training/configs/__init__.py
"""配置模块初始化文件"""

from .config_loader import Config, load_config, save_config, get_default_config
from .logger import setup_logger

__all__ = ['Config', 'load_config', 'save_config', 'get_default_config', 'setup_logger']

