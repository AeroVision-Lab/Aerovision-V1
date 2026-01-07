"""
依赖注入模块

提供 FastAPI 依赖项
"""

import base64
import io
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional, Generator
from contextlib import contextmanager

import numpy as np
from PIL import Image
import httpx

from ..config import InferConfig, get_config
from ..classifier import AircraftClassifier, AirlineClassifier
from ..detector import RegistrationDetector
from ..ocr import RegistrationOCR
from ..quality import ImageQualityAssessor
from ..models import get_registry

logger = logging.getLogger(__name__)


# ============ 配置依赖 ============

@lru_cache()
def get_settings() -> InferConfig:
    """获取配置（缓存）"""
    return get_config()


# ============ 推理服务依赖 ============

class InferenceService:
    """
    推理服务聚合类

    提供所有推理功能的统一入口
    """

    def __init__(self, config: Optional[InferConfig] = None):
        self.config = config or get_config()
        self._aircraft_classifier: Optional[AircraftClassifier] = None
        self._airline_classifier: Optional[AirlineClassifier] = None
        self._detector: Optional[RegistrationDetector] = None
        self._ocr: Optional[RegistrationOCR] = None
        self._quality_assessor: Optional[ImageQualityAssessor] = None

    @property
    def aircraft_classifier(self) -> AircraftClassifier:
        if self._aircraft_classifier is None:
            self._aircraft_classifier = AircraftClassifier(self.config)
        return self._aircraft_classifier

    @property
    def airline_classifier(self) -> AirlineClassifier:
        if self._airline_classifier is None:
            self._airline_classifier = AirlineClassifier(self.config)
        return self._airline_classifier

    @property
    def detector(self) -> RegistrationDetector:
        if self._detector is None:
            self._detector = RegistrationDetector(self.config)
        return self._detector

    @property
    def ocr(self) -> RegistrationOCR:
        if self._ocr is None:
            self._ocr = RegistrationOCR(self.config)
        return self._ocr

    @property
    def quality_assessor(self) -> ImageQualityAssessor:
        if self._quality_assessor is None:
            self._quality_assessor = ImageQualityAssessor(self.config)
        return self._quality_assessor


# 全局推理服务实例
_inference_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """获取推理服务实例"""
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service


# ============ 图片处理依赖 ============

async def load_image_from_url(url: str, timeout: float = 30.0) -> np.ndarray:
    """
    从 URL 加载图片

    Args:
        url: 图片 URL
        timeout: 超时时间

    Returns:
        numpy 数组 (RGB)
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content))
        return np.array(image.convert("RGB"))


def load_image_from_base64(base64_str: str) -> np.ndarray:
    """
    从 Base64 加载图片

    Args:
        base64_str: Base64 编码字符串

    Returns:
        numpy 数组 (RGB)
    """
    # 移除可能的 data URI 前缀
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]

    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return np.array(image.convert("RGB"))


def encode_image_to_base64(image: np.ndarray, format: str = "JPEG") -> str:
    """
    将图片编码为 Base64

    Args:
        image: numpy 数组 (RGB)
        format: 图片格式

    Returns:
        Base64 字符串
    """
    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


async def resolve_image_input(
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None
) -> np.ndarray:
    """
    解析图片输入

    Args:
        image_url: 图片 URL
        image_base64: Base64 编码

    Returns:
        numpy 数组 (RGB)
    """
    if image_base64:
        return load_image_from_base64(image_base64)
    elif image_url:
        return await load_image_from_url(str(image_url))
    else:
        raise ValueError("必须提供 image_url 或 image_base64")


# ============ 计时依赖 ============

@contextmanager
def timing() -> Generator[dict, None, None]:
    """计时上下文管理器"""
    result = {"start": time.time(), "end": 0, "elapsed_ms": 0}
    try:
        yield result
    finally:
        result["end"] = time.time()
        result["elapsed_ms"] = (result["end"] - result["start"]) * 1000


# ============ 请求 ID 依赖 ============

import uuid


def generate_request_id() -> str:
    """生成请求 ID"""
    return str(uuid.uuid4())
