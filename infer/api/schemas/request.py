"""
请求模型定义
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field, HttpUrl, field_validator
import base64


class ImageInput(BaseModel):
    """图片输入基类"""
    image_url: Optional[HttpUrl] = Field(None, description="图片 URL")
    image_base64: Optional[str] = Field(None, description="Base64 编码的图片")

    @field_validator("image_base64")
    @classmethod
    def validate_base64(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        # 移除可能的 data URI 前缀
        if "," in v:
            v = v.split(",", 1)[1]
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError("无效的 Base64 编码")
        return v

    def model_post_init(self, __context) -> None:
        if not self.image_url and not self.image_base64:
            raise ValueError("必须提供 image_url 或 image_base64")


class ReviewRequest(ImageInput):
    """
    综合审核请求

    支持选择性启用各审核模块
    """
    review_types: List[Literal["quality", "aircraft", "airline", "registration"]] = Field(
        default=["quality", "aircraft", "airline", "registration"],
        description="要执行的审核类型"
    )
    # 可选参数覆盖
    quality_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="质量评估通过阈值"
    )
    classifier_conf: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="分类置信度阈值"
    )
    detector_conf: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="检测置信度阈值"
    )


class ClassifyRequest(ImageInput):
    """分类请求"""
    task: Literal["aircraft", "airline"] = Field(
        "aircraft",
        description="分类任务类型"
    )
    top_k: int = Field(5, ge=1, le=10, description="返回 top-k 结果")
    conf_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="置信度阈值"
    )


class DetectRequest(ImageInput):
    """检测请求"""
    conf_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="置信度阈值"
    )
    iou_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="NMS IoU 阈值"
    )
    return_crops: bool = Field(
        False,
        description="是否返回裁剪图片（Base64）"
    )
    crop_padding: float = Field(
        0.1,
        ge=0.0,
        le=0.5,
        description="裁剪边界扩展比例"
    )


class OCRRequest(ImageInput):
    """OCR 请求"""
    bbox: Optional[List[float]] = Field(
        None,
        description="YOLO 格式边界框 [x_center, y_center, w, h]（归一化）"
    )
    padding: float = Field(0.1, ge=0.0, le=0.5, description="边界框扩展比例")

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return v
        if len(v) != 4:
            raise ValueError("bbox 必须包含 4 个值")
        if not all(0 <= x <= 1 for x in v):
            raise ValueError("bbox 值必须在 0-1 之间（归一化坐标）")
        return v


class QualityRequest(ImageInput):
    """质量评估请求"""
    quick_mode: bool = Field(
        False,
        description="快速模式（仅评估清晰度）"
    )
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="通过阈值"
    )


class BatchReviewRequest(BaseModel):
    """批量审核请求"""
    images: List[ImageInput] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="图片列表"
    )
    review_types: List[Literal["quality", "aircraft", "airline", "registration"]] = Field(
        default=["quality", "aircraft"],
        description="要执行的审核类型"
    )
    async_mode: bool = Field(
        False,
        description="异步模式（返回任务 ID）"
    )
