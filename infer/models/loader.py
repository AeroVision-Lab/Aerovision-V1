"""
模型加载器

负责加载和管理深度学习模型
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any

import torch

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    模型加载器

    支持加载 YOLOv8 分类和检测模型
    """

    def __init__(self, device: str = "cuda:0", half: bool = True):
        """
        初始化模型加载器

        Args:
            device: 推理设备 (cuda:0, cpu, mps)
            half: 是否使用 FP16 推理
        """
        self.device = device
        self.half = half and torch.cuda.is_available()

        # 验证设备
        if "cuda" in device and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，回退到 CPU")
            self.device = "cpu"
            self.half = False

        logger.info(f"模型加载器初始化: device={self.device}, half={self.half}")

    def load_yolo_classifier(
        self,
        model_path: Union[str, Path],
        task: str = "classify"
    ) -> Any:
        """
        加载 YOLOv8 分类模型

        Args:
            model_path: 模型文件路径
            task: 任务类型 (classify)

        Returns:
            YOLO 模型实例
        """
        from ultralytics import YOLO

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        logger.info(f"加载分类模型: {model_path}")
        model = YOLO(str(model_path), task=task)

        # 设置设备
        model.to(self.device)

        # 预热模型
        self._warmup_model(model, imgsz=224)

        logger.info(f"分类模型加载完成: {model_path.name}")
        return model

    def load_yolo_detector(
        self,
        model_path: Union[str, Path],
        task: str = "detect"
    ) -> Any:
        """
        加载 YOLOv8 检测模型

        Args:
            model_path: 模型文件路径
            task: 任务类型 (detect)

        Returns:
            YOLO 模型实例
        """
        from ultralytics import YOLO

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        logger.info(f"加载检测模型: {model_path}")
        model = YOLO(str(model_path), task=task)

        # 设置设备
        model.to(self.device)

        # 预热模型
        self._warmup_model(model, imgsz=640)

        logger.info(f"检测模型加载完成: {model_path.name}")
        return model

    def _warmup_model(self, model: Any, imgsz: int = 224) -> None:
        """
        预热模型（进行一次空推理以优化性能）

        Args:
            model: YOLO 模型实例
            imgsz: 输入图像尺寸
        """
        try:
            import numpy as np
            dummy_input = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            model(dummy_input, verbose=False)
            logger.debug("模型预热完成")
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")

    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """
        获取模型信息

        Args:
            model: YOLO 模型实例

        Returns:
            模型信息字典
        """
        info = {
            "task": getattr(model, "task", "unknown"),
            "device": str(self.device),
            "half": self.half,
        }

        # 获取类别名称
        if hasattr(model, "names"):
            info["names"] = model.names
            info["num_classes"] = len(model.names)

        return info


def create_loader(
    device: Optional[str] = None,
    half: Optional[bool] = None
) -> ModelLoader:
    """
    创建模型加载器实例

    Args:
        device: 推理设备
        half: 是否使用 FP16

    Returns:
        ModelLoader 实例
    """
    from ..config import get_config

    config = get_config()
    device = device or config.device
    half = half if half is not None else config.half

    return ModelLoader(device=device, half=half)
