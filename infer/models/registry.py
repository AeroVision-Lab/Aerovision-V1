"""
模型注册表

管理已加载模型的生命周期，提供单例访问
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from threading import Lock

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    模型注册表

    单例模式，管理所有已加载的模型
    """

    _instance: Optional["ModelRegistry"] = None
    _lock: Lock = Lock()

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._models: Dict[str, Any] = {}
        self._model_info: Dict[str, Dict[str, Any]] = {}
        self._loader = None
        self._initialized = True

        logger.info("模型注册表初始化")

    @property
    def loader(self):
        """获取模型加载器"""
        if self._loader is None:
            from .loader import create_loader
            self._loader = create_loader()
        return self._loader

    def register(self, name: str, model: Any, info: Optional[Dict] = None) -> None:
        """
        注册模型

        Args:
            name: 模型名称
            model: 模型实例
            info: 模型信息
        """
        self._models[name] = model
        self._model_info[name] = info or {}
        logger.info(f"模型已注册: {name}")

    def get(self, name: str) -> Optional[Any]:
        """
        获取已注册的模型

        Args:
            name: 模型名称

        Returns:
            模型实例或 None
        """
        return self._models.get(name)

    def get_info(self, name: str) -> Dict[str, Any]:
        """
        获取模型信息

        Args:
            name: 模型名称

        Returns:
            模型信息字典
        """
        return self._model_info.get(name, {})

    def is_loaded(self, name: str) -> bool:
        """检查模型是否已加载"""
        return name in self._models

    def unload(self, name: str) -> bool:
        """
        卸载模型

        Args:
            name: 模型名称

        Returns:
            是否成功卸载
        """
        if name in self._models:
            del self._models[name]
            self._model_info.pop(name, None)
            logger.info(f"模型已卸载: {name}")
            return True
        return False

    def unload_all(self) -> None:
        """卸载所有模型"""
        self._models.clear()
        self._model_info.clear()
        logger.info("所有模型已卸载")

    def list_models(self) -> Dict[str, str]:
        """
        列出所有已加载的模型

        Returns:
            模型名称和状态的字典
        """
        return {name: "loaded" for name in self._models}

    def load_classifier(
        self,
        name: str,
        model_path: Path,
        force_reload: bool = False
    ) -> Any:
        """
        加载分类模型

        Args:
            name: 模型注册名称
            model_path: 模型文件路径
            force_reload: 是否强制重新加载

        Returns:
            模型实例
        """
        if not force_reload and self.is_loaded(name):
            logger.debug(f"使用缓存的模型: {name}")
            return self.get(name)

        model = self.loader.load_yolo_classifier(model_path)
        info = self.loader.get_model_info(model)
        self.register(name, model, info)

        return model

    def load_detector(
        self,
        name: str,
        model_path: Path,
        force_reload: bool = False
    ) -> Any:
        """
        加载检测模型

        Args:
            name: 模型注册名称
            model_path: 模型文件路径
            force_reload: 是否强制重新加载

        Returns:
            模型实例
        """
        if not force_reload and self.is_loaded(name):
            logger.debug(f"使用缓存的模型: {name}")
            return self.get(name)

        model = self.loader.load_yolo_detector(model_path)
        info = self.loader.get_model_info(model)
        self.register(name, model, info)

        return model


# 全局注册表实例
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """获取全局模型注册表"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
