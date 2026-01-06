"""
分类推理模块

支持飞机机型分类和航空公司识别
"""

import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

import numpy as np
from PIL import Image

from .config import get_config, InferConfig
from .models import get_registry

logger = logging.getLogger(__name__)


class AircraftClassifier:
    """
    飞机机型分类器

    基于 YOLOv8-cls 的机型分类推理
    """

    MODEL_NAME = "aircraft_classifier"

    def __init__(self, config: Optional[InferConfig] = None):
        """
        初始化分类器

        Args:
            config: 推理配置，为 None 则使用全局配置
        """
        self.config = config or get_config()
        self.registry = get_registry()
        self._model = None

    @property
    def model(self):
        """懒加载模型"""
        if self._model is None:
            model_path = self.config.get_model_path("aircraft_classifier")
            self._model = self.registry.load_classifier(
                self.MODEL_NAME,
                model_path
            )
        return self._model

    @property
    def class_names(self) -> Dict[int, str]:
        """获取类别名称映射"""
        info = self.registry.get_info(self.MODEL_NAME)
        return info.get("names", {})

    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        预测图片机型

        Args:
            image: 输入图片（路径、numpy数组或PIL Image）
            top_k: 返回 top-k 个预测结果

        Returns:
            预测结果字典:
            {
                "success": True,
                "top1": {"class_id": 0, "class_name": "A320", "confidence": 0.95},
                "top_k": [...],
                "all_probs": {...}
            }
        """
        try:
            # 执行推理
            results = self.model(
                image,
                imgsz=self.config.model.classifier_imgsz,
                verbose=False
            )

            if not results or len(results) == 0:
                return {"success": False, "error": "推理无结果"}

            result = results[0]
            probs = result.probs

            if probs is None:
                return {"success": False, "error": "无法获取概率分布"}

            # 获取 top-k 结果
            top_k_indices = probs.top5 if hasattr(probs, "top5") else []
            top_k_confs = probs.top5conf if hasattr(probs, "top5conf") else []

            # 构建返回结果
            names = self.class_names
            top_k_results = []

            for idx, conf in zip(top_k_indices[:top_k], top_k_confs[:top_k]):
                idx = int(idx)
                conf = float(conf)
                top_k_results.append({
                    "class_id": idx,
                    "class_name": names.get(idx, f"class_{idx}"),
                    "confidence": conf
                })

            # Top-1 结果
            top1 = top_k_results[0] if top_k_results else None

            return {
                "success": True,
                "top1": top1,
                "top_k": top_k_results,
                "is_confident": top1["confidence"] >= self.config.model.classifier_conf if top1 else False
            }

        except Exception as e:
            logger.error(f"机型分类推理失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def classify(
        self,
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Dict[str, Any]:
        """
        分类图片（简化接口）

        Args:
            image: 输入图片

        Returns:
            {
                "pass": True/False,
                "aircraft_type": "A320",
                "confidence": 0.95
            }
        """
        result = self.predict(image, top_k=1)

        if not result["success"]:
            return {
                "pass": False,
                "aircraft_type": None,
                "confidence": 0.0,
                "error": result.get("error")
            }

        top1 = result["top1"]
        is_confident = result["is_confident"]

        return {
            "pass": is_confident,
            "aircraft_type": top1["class_name"] if top1 else None,
            "confidence": top1["confidence"] if top1 else 0.0
        }


class AirlineClassifier:
    """
    航空公司识别器

    基于 YOLOv8-cls 的航司涂装识别
    """

    MODEL_NAME = "airline_classifier"

    def __init__(self, config: Optional[InferConfig] = None):
        """
        初始化分类器

        Args:
            config: 推理配置
        """
        self.config = config or get_config()
        self.registry = get_registry()
        self._model = None

    @property
    def model(self):
        """懒加载模型"""
        if self._model is None:
            model_path = self.config.get_model_path("airline_classifier")
            self._model = self.registry.load_classifier(
                self.MODEL_NAME,
                model_path
            )
        return self._model

    @property
    def class_names(self) -> Dict[int, str]:
        """获取类别名称映射"""
        info = self.registry.get_info(self.MODEL_NAME)
        return info.get("names", {})

    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        预测图片航司

        Args:
            image: 输入图片
            top_k: 返回 top-k 个预测结果

        Returns:
            预测结果字典
        """
        try:
            results = self.model(
                image,
                imgsz=self.config.model.classifier_imgsz,
                verbose=False
            )

            if not results or len(results) == 0:
                return {"success": False, "error": "推理无结果"}

            result = results[0]
            probs = result.probs

            if probs is None:
                return {"success": False, "error": "无法获取概率分布"}

            top_k_indices = probs.top5 if hasattr(probs, "top5") else []
            top_k_confs = probs.top5conf if hasattr(probs, "top5conf") else []

            names = self.class_names
            top_k_results = []

            for idx, conf in zip(top_k_indices[:top_k], top_k_confs[:top_k]):
                idx = int(idx)
                conf = float(conf)
                top_k_results.append({
                    "class_id": idx,
                    "class_name": names.get(idx, f"class_{idx}"),
                    "confidence": conf
                })

            top1 = top_k_results[0] if top_k_results else None

            return {
                "success": True,
                "top1": top1,
                "top_k": top_k_results,
                "is_confident": top1["confidence"] >= self.config.model.classifier_conf if top1 else False
            }

        except Exception as e:
            logger.error(f"航司识别推理失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def classify(
        self,
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Dict[str, Any]:
        """
        识别航司（简化接口）

        Args:
            image: 输入图片

        Returns:
            {
                "pass": True/False,
                "airline": "China Eastern",
                "confidence": 0.95
            }
        """
        result = self.predict(image, top_k=1)

        if not result["success"]:
            return {
                "pass": False,
                "airline": None,
                "confidence": 0.0,
                "error": result.get("error")
            }

        top1 = result["top1"]
        is_confident = result["is_confident"]

        return {
            "pass": is_confident,
            "airline": top1["class_name"] if top1 else None,
            "confidence": top1["confidence"] if top1 else 0.0
        }


# 全局实例
_aircraft_classifier: Optional[AircraftClassifier] = None
_airline_classifier: Optional[AirlineClassifier] = None


def get_aircraft_classifier() -> AircraftClassifier:
    """获取全局机型分类器"""
    global _aircraft_classifier
    if _aircraft_classifier is None:
        _aircraft_classifier = AircraftClassifier()
    return _aircraft_classifier


def get_airline_classifier() -> AirlineClassifier:
    """获取全局航司分类器"""
    global _airline_classifier
    if _airline_classifier is None:
        _airline_classifier = AirlineClassifier()
    return _airline_classifier
