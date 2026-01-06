"""
OCR 推理模块

基于 PaddleOCR 的注册号文字识别
"""

import os
import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image
import cv2

from .config import get_config, InferConfig, OCRConfig

logger = logging.getLogger(__name__)

# 设置 PaddleOCR 环境变量（必须在导入前设置）
os.environ.setdefault('DISABLE_MODEL_SOURCE_CHECK', 'True')
os.environ.setdefault('PADDLEX_LOG_LEVEL', 'ERROR')


class RegistrationOCR:
    """
    注册号 OCR 识别器

    使用 PaddleOCR 进行文字识别，专门针对飞机注册号优化
    """

    def __init__(self, config: Optional[InferConfig] = None):
        """
        初始化 OCR

        Args:
            config: 推理配置
        """
        self.config = config or get_config()
        self.ocr_config = self.config.ocr
        self._ocr = None

    @property
    def ocr(self):
        """懒加载 OCR 引擎"""
        if self._ocr is None:
            self._ocr = self._init_ocr()
        return self._ocr

    def _init_ocr(self):
        """初始化 PaddleOCR"""
        # Monkey patch for compatibility
        try:
            import importlib.util
            _original_find_spec = importlib.util.find_spec

            def _patched_find_spec(name, *args, **kwargs):
                if name == 'torch':
                    return None
                return _original_find_spec(name, *args, **kwargs)

            importlib.util.find_spec = _patched_find_spec
        except Exception:
            pass

        # Paddle compatibility patch
        try:
            import paddle
            if hasattr(paddle, 'base') and hasattr(paddle.base, 'libpaddle'):
                AnalysisConfig = paddle.base.libpaddle.AnalysisConfig
                if not hasattr(AnalysisConfig, 'set_optimization_level'):
                    AnalysisConfig.set_optimization_level = lambda self, level: None
        except Exception:
            pass

        from paddleocr import PaddleOCR

        logger.info(f"初始化 PaddleOCR: lang={self.ocr_config.lang}")

        ocr = PaddleOCR(
            lang=self.ocr_config.lang,
            text_recognition_model_name=self.ocr_config.rec_model_name,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        logger.info("PaddleOCR 初始化完成")
        return ocr

    def _load_image(
        self,
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> np.ndarray:
        """加载图片为 numpy 数组 (RGB)"""
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
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            return image
        else:
            raise ValueError(f"不支持的图片类型: {type(image)}")

    def _postprocess(self, text: str, confidence: float) -> Tuple[str, bool]:
        """
        后处理识别结果

        Args:
            text: 原始识别文本
            confidence: 置信度

        Returns:
            (处理后文本, 是否有效)
        """
        # 清理文本
        text = text.upper().replace(' ', '').replace('.', '').replace(',', '')
        text = ''.join(c for c in text if c in self.ocr_config.whitelist)

        # 验证长度
        if not (self.ocr_config.min_chars <= len(text) <= self.ocr_config.max_chars):
            return text, False

        # 验证置信度
        if confidence < self.ocr_config.min_confidence:
            return text, False

        return text, True

    def _parse_ocr_result(self, result) -> Tuple[str, float]:
        """解析 PaddleOCR 结果"""
        if not result or not result[0]:
            return '', 0.0

        texts = []
        confs = []

        for line in result[0]:
            if line and len(line) >= 2:
                text_info = line[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    texts.append(str(text_info[0]))
                    confs.append(float(text_info[1]))
                elif isinstance(text_info, str):
                    texts.append(text_info)
                    confs.append(1.0)

        if not texts:
            return '', 0.0

        return ''.join(texts), sum(confs) / len(confs) if confs else 0.0

    def recognize(
        self,
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Dict[str, Any]:
        """
        识别图片中的注册号

        Args:
            image: 输入图片（应该是裁剪后的注册号区域）

        Returns:
            {
                "success": True,
                "text": "B-1234",
                "raw_text": "B-1234",
                "confidence": 0.95,
                "valid": True
            }
        """
        try:
            img = self._load_image(image)

            # 执行 OCR
            result = self.ocr.ocr(img)

            # 解析结果
            raw_text, confidence = self._parse_ocr_result(result)
            processed_text, is_valid = self._postprocess(raw_text, confidence)

            return {
                "success": True,
                "text": processed_text,
                "raw_text": raw_text,
                "confidence": confidence,
                "valid": is_valid
            }

        except Exception as e:
            logger.error(f"OCR 识别失败: {e}", exc_info=True)
            return {
                "success": False,
                "text": "",
                "confidence": 0.0,
                "valid": False,
                "error": str(e)
            }

    def recognize_from_bbox(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        bbox: List[float],
        padding: float = 0.1
    ) -> Dict[str, Any]:
        """
        从边界框区域识别注册号

        Args:
            image: 输入图片
            bbox: YOLO 格式边界框 [x_center, y_center, w, h] (归一化)
            padding: 边界框扩展比例

        Returns:
            识别结果字典
        """
        try:
            img = self._load_image(image)
            h, w = img.shape[:2]

            # 转换 YOLO 格式为像素坐标
            x_center, y_center, box_w, box_h = bbox
            x1 = int((x_center - box_w / 2) * w)
            y1 = int((y_center - box_h / 2) * h)
            x2 = int((x_center + box_w / 2) * w)
            y2 = int((y_center + box_h / 2) * h)

            # 添加 padding
            if padding > 0:
                pad_w = int((x2 - x1) * padding)
                pad_h = int((y2 - y1) * padding)
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(w, x2 + pad_w)
                y2 = min(h, y2 + pad_h)

            # 裁剪
            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                return {
                    "success": False,
                    "text": "",
                    "confidence": 0.0,
                    "valid": False,
                    "error": "裁剪区域为空"
                }

            # 识别
            result = self.recognize(crop)
            result["bbox"] = [x1, y1, x2, y2]

            return result

        except Exception as e:
            logger.error(f"从边界框识别失败: {e}", exc_info=True)
            return {
                "success": False,
                "text": "",
                "confidence": 0.0,
                "valid": False,
                "error": str(e)
            }

    def recognize_from_crops(
        self,
        crops: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        批量识别裁剪区域

        Args:
            crops: 裁剪结果列表，每个元素包含 "image" 键

        Returns:
            识别结果列表
        """
        results = []
        for crop in crops:
            if "image" not in crop:
                results.append({
                    "success": False,
                    "error": "缺少 image 键"
                })
                continue

            result = self.recognize(crop["image"])
            result["bbox"] = crop.get("bbox")
            result["detection_confidence"] = crop.get("confidence", 0.0)
            results.append(result)

        return results


# 全局实例
_ocr: Optional[RegistrationOCR] = None


def get_registration_ocr() -> RegistrationOCR:
    """获取全局 OCR 实例"""
    global _ocr
    if _ocr is None:
        _ocr = RegistrationOCR()
    return _ocr
