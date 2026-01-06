"""
检测推理模块

支持注册号区域检测
"""

import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image
import cv2

from .config import get_config, InferConfig
from .models import get_registry

logger = logging.getLogger(__name__)


class RegistrationDetector:
    """
    注册号区域检测器

    基于 YOLOv8 的目标检测模型
    """

    MODEL_NAME = "registration_detector"

    def __init__(self, config: Optional[InferConfig] = None):
        """
        初始化检测器

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
            model_path = self.config.get_model_path("registration_detector")
            self._model = self.registry.load_detector(
                self.MODEL_NAME,
                model_path
            )
        return self._model

    def detect(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        检测图片中的注册号区域

        Args:
            image: 输入图片
            conf_threshold: 置信度阈值（默认使用配置值）
            iou_threshold: NMS IoU 阈值（默认使用配置值）

        Returns:
            检测结果字典:
            {
                "success": True,
                "detected": True,
                "count": 1,
                "boxes": [
                    {
                        "xyxy": [x1, y1, x2, y2],
                        "xywh": [x, y, w, h],
                        "xywhn": [x_center, y_center, w, h],  # 归一化
                        "confidence": 0.95,
                        "class_id": 0,
                        "class_name": "registration"
                    }
                ],
                "image_size": [height, width]
            }
        """
        conf = conf_threshold or self.config.model.detector_conf
        iou = iou_threshold or self.config.model.detector_iou

        try:
            # 加载图片获取尺寸
            img_array = self._load_image(image)
            img_h, img_w = img_array.shape[:2]

            # 执行推理
            results = self.model(
                image,
                imgsz=self.config.model.detector_imgsz,
                conf=conf,
                iou=iou,
                verbose=False
            )

            if not results or len(results) == 0:
                return {
                    "success": True,
                    "detected": False,
                    "count": 0,
                    "boxes": [],
                    "image_size": [img_h, img_w]
                }

            result = results[0]
            boxes_data = result.boxes

            if boxes_data is None or len(boxes_data) == 0:
                return {
                    "success": True,
                    "detected": False,
                    "count": 0,
                    "boxes": [],
                    "image_size": [img_h, img_w]
                }

            # 解析检测框
            boxes = []
            names = result.names or {}

            for i in range(len(boxes_data)):
                # 获取边界框坐标
                xyxy = boxes_data.xyxy[i].cpu().numpy().tolist()
                xywh = boxes_data.xywh[i].cpu().numpy().tolist()
                xywhn = boxes_data.xywhn[i].cpu().numpy().tolist()
                conf_score = float(boxes_data.conf[i].cpu().numpy())
                cls_id = int(boxes_data.cls[i].cpu().numpy())

                boxes.append({
                    "xyxy": xyxy,
                    "xywh": xywh,
                    "xywhn": xywhn,
                    "confidence": conf_score,
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, f"class_{cls_id}")
                })

            return {
                "success": True,
                "detected": len(boxes) > 0,
                "count": len(boxes),
                "boxes": boxes,
                "image_size": [img_h, img_w]
            }

        except Exception as e:
            logger.error(f"注册号检测推理失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def detect_and_crop(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        padding: float = 0.1,
        conf_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        检测并裁剪注册号区域

        Args:
            image: 输入图片
            padding: 边界框扩展比例
            conf_threshold: 置信度阈值

        Returns:
            {
                "success": True,
                "detected": True,
                "crops": [
                    {
                        "image": np.ndarray,  # 裁剪后的图片
                        "bbox": [x1, y1, x2, y2],
                        "confidence": 0.95
                    }
                ]
            }
        """
        # 先检测
        det_result = self.detect(image, conf_threshold=conf_threshold)

        if not det_result["success"]:
            return det_result

        if not det_result["detected"]:
            return {
                "success": True,
                "detected": False,
                "crops": []
            }

        # 加载图片
        img_array = self._load_image(image)
        img_h, img_w = img_array.shape[:2]

        # 裁剪每个检测框
        crops = []
        for box in det_result["boxes"]:
            x1, y1, x2, y2 = box["xyxy"]

            # 添加 padding
            if padding > 0:
                box_w = x2 - x1
                box_h = y2 - y1
                pad_w = int(box_w * padding)
                pad_h = int(box_h * padding)
                x1 = max(0, int(x1) - pad_w)
                y1 = max(0, int(y1) - pad_h)
                x2 = min(img_w, int(x2) + pad_w)
                y2 = min(img_h, int(y2) + pad_h)
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 裁剪
            crop_img = img_array[y1:y2, x1:x2]

            crops.append({
                "image": crop_img,
                "bbox": [x1, y1, x2, y2],
                "confidence": box["confidence"]
            })

        return {
            "success": True,
            "detected": len(crops) > 0,
            "crops": crops
        }

    def _load_image(
        self,
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> np.ndarray:
        """加载图片为 numpy 数组"""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"无法读取图片: {image}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            return image
        else:
            raise ValueError(f"不支持的图片类型: {type(image)}")

    def get_best_detection(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        conf_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        获取置信度最高的检测结果

        Args:
            image: 输入图片
            conf_threshold: 置信度阈值

        Returns:
            最佳检测结果或空字典
        """
        result = self.detect(image, conf_threshold=conf_threshold)

        if not result["success"] or not result["detected"]:
            return {
                "detected": False,
                "bbox": None,
                "confidence": 0.0
            }

        # 按置信度排序
        boxes = sorted(result["boxes"], key=lambda x: x["confidence"], reverse=True)
        best = boxes[0]

        return {
            "detected": True,
            "bbox": best["xyxy"],
            "bbox_normalized": best["xywhn"],
            "confidence": best["confidence"]
        }


# 全局实例
_detector: Optional[RegistrationDetector] = None


def get_registration_detector() -> RegistrationDetector:
    """获取全局注册号检测器"""
    global _detector
    if _detector is None:
        _detector = RegistrationDetector()
    return _detector
