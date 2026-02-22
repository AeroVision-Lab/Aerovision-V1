# training/config/config_loader.py
"""配置文件加载器 - 支持模块化配置结构"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List
import os


class Config:
    """配置类，用于加载和访问YAML配置"""

    def __init__(self, config_dict: Dict[str, Any], config_base_path: Optional[Path] = None):
        """
        初始化配置对象

        Args:
            config_dict: 配置字典
            config_base_path: 配置文件基准路径（用于解析相对路径，默认为 /training/configs）
        """
        self._config = config_dict
        # ⚠️ 重要：所有相对路径都相对于 /training/configs 目录
        # 如果没有指定config_base_path，则使用项目的 training/configs 目录
        if config_base_path is None:
            # 从当前文件位置推断 training/configs 路径
            # 当前文件在 training/config/config_loader.py
            # 目标路径是 training/configs
            current_file = Path(__file__).resolve()
            training_dir = current_file.parent.parent  # training/
            self._config_base_path = training_dir / "configs"
        else:
            self._config_base_path = config_base_path

    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self._config[key]

    def __getattr__(self, key: str) -> Any:
        """支持属性式访问"""
        if key.startswith('_'):
            return object.__getattribute__(self, key)
        return self._config.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        Args:
            key: 配置键，支持点号分隔的嵌套键（如 'paths.data_root'）
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_path(self, key: str, create: bool = False) -> Path:
        """
        获取路径配置并转换为Path对象
        相对路径会相对于yaml配置文件所在目录进行解析

        Args:
            key: 配置键
            create: 是否创建目录（如果不存在）

        Returns:
            Path对象（绝对路径）
        """
        path_str = self.get(key)
        if path_str is None:
            raise ValueError(f"配置键 '{key}' 未找到")

        path = Path(path_str)

        # 如果是相对路径，则相对于配置文件所在目录
        if not path.is_absolute():
            path = (self._config_base_path / path).resolve()

        if create and not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        return path

    def update(self, updates: Dict[str, Any]):
        """
        更新配置

        Args:
            updates: 要更新的配置字典
        """
        self._deep_update(self._config, updates)

    @staticmethod
    def _deep_update(base_dict: Dict, update_dict: Dict):
        """递归更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                Config._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（深拷贝）"""
        import copy
        return copy.deepcopy(self._config)

    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(
    config_path: Optional[str] = None,
    modules: Optional[List[str]] = None,
    load_all_modules: bool = True,
    **kwargs
) -> Config:
    """
    加载配置文件（支持新的模块化配置结构）

    Args:
        config_path: 配置文件路径。如果为None则加载 configs/base.yaml
                    如果指定路径，则加载该路径的配置（支持旧的default.yaml）
        modules: 要加载的模块列表。如 ['yolo', 'crop']
                如果为None且load_all_modules=True，则加载所有模块
        load_all_modules: 是否加载所有模块配置（默认True）
        **kwargs: 额外的配置覆盖项

    Returns:
        Config对象

    Example:
        >>> # 加载base配置和所有模块
        >>> config = load_config()

        >>> # 只加载base和特定模块
        >>> config = load_config(modules=['yolo', 'crop'])

        >>> # 加载旧的配置文件（向后兼容）
        >>> config = load_config('config/default.yaml')

        >>> # 运行时覆盖配置
        >>> config = load_config(device={'default': 'cpu'})
    """
    # 确定项目根路径和configs路径
    current_dir = Path(__file__).parent.resolve()  # training/config/
    training_dir = current_dir.parent  # training/
    configs_dir = training_dir / "configs"  # training/configs/

    # 确定配置文件路径
    if config_path is None:
        # 使用新的base.yaml
        base_config_path = configs_dir / "base.yaml"
        use_modular = True
    else:
        # 使用指定的配置文件
        base_config_path = Path(config_path)
        # 如果是相对路径，相对于当前工作目录解析
        if not base_config_path.is_absolute():
            base_config_path = Path.cwd() / base_config_path
        use_modular = False

    # 检查文件是否存在
    if not base_config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {base_config_path}")

    # 加载基础配置
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # 如果使用模块化配置，加载模块配置
    if use_modular:
        # 确定要加载的模块
        if modules is None and load_all_modules:
            # 动态扫描config目录下的所有yaml文件
            config_module_dir = configs_dir / "config"
            if config_module_dir.exists():
                modules_to_load = [f.stem for f in config_module_dir.glob("*.yaml")]
            else:
                modules_to_load = []
        elif modules is not None:
            modules_to_load = modules
        else:
            modules_to_load = []

        # 加载每个模块配置并合并
        for module_name in modules_to_load:
            module_path = configs_dir / "config" / f"{module_name}.yaml"
            if module_path.exists():
                with open(module_path, 'r', encoding='utf-8') as f:
                    module_config = yaml.safe_load(f)
                    # 合并模块配置到主配置
                    if module_config:
                        config_dict = _deep_merge(config_dict, module_config)

    # 创建Config对象（使用configs_dir作为基准路径）
    config = Config(config_dict, configs_dir)

    # 应用额外的配置覆盖
    if kwargs:
        config.update(kwargs)

    return config


def _deep_merge(base_dict: Dict, update_dict: Dict) -> Dict:
    """
    深度合并两个字典

    Args:
        base_dict: 基础字典
        update_dict: 更新字典

    Returns:
        合并后的字典
    """
    result = base_dict.copy()
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config: Config, save_path: str):
    """
    保存配置到文件

    Args:
        config: Config对象
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)

    print(f"配置已保存到: {save_path}")


# 便捷函数：获取默认配置
def get_default_config() -> Config:
    """
    获取默认配置（加载base.yaml和所有模块）

    Returns:
        Config对象
    """
    return load_config()
