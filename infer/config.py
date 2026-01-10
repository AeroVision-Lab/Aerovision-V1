"""
Infer 模块配置

集中管理推理服务的配置参数
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """模型配置"""
    # 模型文件路径（可通过环境变量覆盖）
    aircraft_classifier: str = field(
        default_factory=lambda: os.getenv("AIRCRAFT_CLASSIFIER_MODEL", "aircraft_classifier.pt")
    )
    airline_classifier: str = field(
        default_factory=lambda: os.getenv("AIRLINE_CLASSIFIER_MODEL", "airline_classifier.pt")
    )
    registration_detector: str = field(
        default_factory=lambda: os.getenv("REGISTRATION_DETECTOR_MODEL", "registration_detector.pt")
    )

    # 推理参数
    classifier_imgsz: int = 224
    detector_imgsz: int = 640
    classifier_conf: float = field(
        default_factory=lambda: float(os.getenv("CLASSIFIER_CONFIDENCE_THRESHOLD", "0.5"))
    )
    detector_conf: float = field(
        default_factory=lambda: float(os.getenv("DETECTOR_CONF", "0.25"))
    )
    detector_iou: float = field(
        default_factory=lambda: float(os.getenv("DETECTOR_IOU", "0.45"))
    )


@dataclass
class OCRConfig:
    """OCR 配置"""
    lang: str = field(default_factory=lambda: os.getenv("OCR_LANG", "en"))
    rec_model_name: str = "PP-OCRv4_server_rec_doc"
    min_confidence: float = field(
        default_factory=lambda: float(os.getenv("OCR_MIN_CONFIDENCE", "0.5"))
    )
    min_chars: int = 4
    max_chars: int = 10
    whitelist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"


@dataclass
class QualityConfig:
    """质量评估配置"""
    # 各指标权重
    sharpness_weight: float = 0.30
    exposure_weight: float = 0.25
    composition_weight: float = 0.20
    noise_weight: float = 0.15
    color_weight: float = 0.10

    # 阈值
    pass_threshold: float = 0.70


@dataclass
class InferConfig:
    """推理服务总配置"""
    # 模型目录
    model_dir: Path = field(default_factory=lambda: Path("models"))

    # 设备配置
    device: str = "cuda:0"
    half: bool = True  # FP16 推理

    # 子配置
    model: ModelConfig = field(default_factory=ModelConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)

    def __post_init__(self):
        # 从环境变量读取配置
        if os.getenv("MODEL_DIR"):
            self.model_dir = Path(os.getenv("MODEL_DIR"))
        if os.getenv("DEVICE"):
            self.device = os.getenv("DEVICE")
        if os.getenv("QUALITY_THRESHOLD"):
            self.quality.pass_threshold = float(os.getenv("QUALITY_THRESHOLD"))

    def get_model_path(self, model_name: str) -> Path:
        """获取模型完整路径"""
        model_file = getattr(self.model, model_name, None)
        if model_file:
            return self.model_dir / model_file
        return self.model_dir / model_name


# 全局配置实例
_config: Optional[InferConfig] = None


def get_config() -> InferConfig:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = InferConfig()
    return _config


def set_config(config: InferConfig) -> None:
    """设置全局配置实例"""
    global _config
    _config = config
