"""
API Schemas - Pydantic 请求/响应模型
"""

from .request import (
    ReviewRequest,
    ClassifyRequest,
    DetectRequest,
    OCRRequest,
    QualityRequest,
    BatchReviewRequest,
)
from .response import (
    BaseResponse,
    ReviewResponse,
    ClassifyResponse,
    DetectResponse,
    OCRResponse,
    QualityResponse,
    BatchReviewResponse,
    HealthResponse,
    StatsResponse,
    ModelInfoResponse,
)

__all__ = [
    # Request
    "ReviewRequest",
    "ClassifyRequest",
    "DetectRequest",
    "OCRRequest",
    "QualityRequest",
    "BatchReviewRequest",
    # Response
    "BaseResponse",
    "ReviewResponse",
    "ClassifyResponse",
    "DetectResponse",
    "OCRResponse",
    "QualityResponse",
    "BatchReviewResponse",
    "HealthResponse",
    "StatsResponse",
    "ModelInfoResponse",
]
