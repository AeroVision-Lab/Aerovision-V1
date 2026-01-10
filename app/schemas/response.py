"""
响应模型定义
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from pydantic import BaseModel, Field


class QualityDetails(BaseModel):
    """质量评估详情"""
    sharpness: float = Field(ge=0, le=1, description="清晰度评分")
    exposure: float = Field(ge=0, le=1, description="曝光评分")
    composition: float = Field(ge=0, le=1, description="构图评分")
    noise: float = Field(ge=0, le=1, description="噪点评分（越高越好）")
    color: float = Field(ge=0, le=1, description="色彩评分")


class QualityResult(BaseModel):
    """质量评估结果"""
    passed: bool = Field(description="是否通过质量审核")
    score: float = Field(ge=0, le=1, description="综合质量评分")
    details: QualityDetails = Field(description="各维度评分详情")
    reason: Optional[str] = Field(default=None, description="未通过原因")


class ClassPrediction(BaseModel):
    """分类预测结果"""
    class_id: int = Field(description="类别 ID")
    class_name: str = Field(description="类别名称")
    confidence: float = Field(ge=0, le=1, description="置信度")


class AircraftResult(BaseModel):
    """飞机识别结果"""
    passed: bool = Field(description="是否通过飞机识别审核")
    is_aircraft: bool = Field(description="是否包含飞机")
    confidence: float = Field(ge=0, le=1, description="识别置信度")
    aircraft_type: Optional[str] = Field(default=None, description="飞机型号 (Top-1)")
    aircraft_type_confidence: Optional[float] = Field(
        default=None, ge=0, le=1, description="机型识别置信度"
    )
    aircraft_type_top3: Optional[List[ClassPrediction]] = Field(
        default=None, description="机型识别 Top-3 预测"
    )
    airline: Optional[str] = Field(default=None, description="航空公司 (Top-1)")
    airline_confidence: Optional[float] = Field(
        default=None, ge=0, le=1, description="航司识别置信度"
    )
    airline_top3: Optional[List[ClassPrediction]] = Field(
        default=None, description="航司识别 Top-3 预测"
    )
    reason: Optional[str] = Field(default=None, description="未通过原因")


class RegistrationResult(BaseModel):
    """注册号识别结果"""
    passed: bool = Field(description="是否通过注册号审核")
    detected: bool = Field(description="是否检测到注册号")
    value: Optional[str] = Field(default=None, description="识别到的注册号")
    confidence: float = Field(ge=0, le=1, description="识别置信度")
    clarity_score: float = Field(ge=0, le=1, description="清晰度评分")
    bbox: Optional[List[float]] = Field(
        default=None, description="注册号边界框 [x1, y1, x2, y2]"
    )
    reason: Optional[str] = Field(default=None, description="未通过原因")


class OcclusionResult(BaseModel):
    """遮挡检测结果"""
    passed: bool = Field(description="是否通过遮挡检测")
    occlusion_percentage: float = Field(
        ge=0, le=1, description="遮挡比例"
    )
    details: Optional[str] = Field(default=None, description="遮挡详情描述")
    reason: Optional[str] = Field(default=None, description="未通过原因")


class ViolationResult(BaseModel):
    """违规检测结果"""
    passed: bool = Field(description="是否通过违规检测")
    has_watermark: bool = Field(default=False, description="是否包含水印")
    has_sensitive_content: bool = Field(
        default=False, description="是否包含敏感内容"
    )
    has_other_logo: bool = Field(
        default=False, description="是否包含其他网站 logo"
    )
    violations: List[str] = Field(
        default_factory=list, description="违规项列表"
    )
    reason: Optional[str] = Field(default=None, description="未通过原因")


class ReviewResults(BaseModel):
    """审核结果汇总"""
    overall_pass: bool = Field(description="是否通过所有审核")
    quality: Optional[QualityResult] = Field(default=None, description="质量评估结果")
    aircraft: Optional[AircraftResult] = Field(default=None, description="飞机识别结果")
    registration: Optional[RegistrationResult] = Field(
        default=None, description="注册号识别结果"
    )
    occlusion: Optional[OcclusionResult] = Field(default=None, description="遮挡检测结果")
    violation: Optional[ViolationResult] = Field(default=None, description="违规检测结果")


class ReviewResponse(BaseModel):
    """图片审核响应"""
    success: bool = Field(description="请求是否成功")
    review_id: str = Field(description="审核任务 ID")
    results: Optional[ReviewResults] = Field(default=None, description="审核结果")
    fail_reasons: List[str] = Field(
        default_factory=list, description="未通过的原因列表"
    )
    processing_time_ms: float = Field(description="处理耗时（毫秒）")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="响应时间戳"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="请求中的附加元数据"
    )
    error: Optional[str] = Field(default=None, description="错误信息")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "review_id": "550e8400-e29b-41d4-a716-446655440000",
                    "results": {
                        "overall_pass": False,
                        "quality": {
                            "passed": True,
                            "score": 0.85,
                            "details": {
                                "sharpness": 0.90,
                                "exposure": 0.80,
                                "composition": 0.85,
                                "noise": 0.88,
                                "color": 0.82
                            }
                        },
                        "aircraft": {
                            "passed": True,
                            "is_aircraft": True,
                            "confidence": 0.95,
                            "aircraft_type": "Boeing 737-800",
                            "aircraft_type_confidence": 0.95,
                            "aircraft_type_top3": [
                                {"class_id": 12, "class_name": "Boeing 737-800", "confidence": 0.95},
                                {"class_id": 8, "class_name": "Boeing 737-700", "confidence": 0.03},
                                {"class_id": 15, "class_name": "Boeing 737 MAX 8", "confidence": 0.01}
                            ],
                            "airline": "China Eastern",
                            "airline_confidence": 0.92,
                            "airline_top3": [
                                {"class_id": 5, "class_name": "China Eastern", "confidence": 0.92},
                                {"class_id": 3, "class_name": "Shanghai Airlines", "confidence": 0.05},
                                {"class_id": 8, "class_name": "China Southern", "confidence": 0.02}
                            ]
                        },
                        "registration": {
                            "passed": False,
                            "detected": True,
                            "value": "B-1234",
                            "confidence": 0.75,
                            "clarity_score": 0.60,
                            "reason": "注册号清晰度不足"
                        }
                    },
                    "fail_reasons": ["注册号清晰度不足"],
                    "processing_time_ms": 1523.5,
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(description="服务状态")
    version: str = Field(description="服务版本")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="响应时间戳"
    )
    models_loaded: Dict[str, bool] = Field(
        default_factory=dict, description="模型加载状态"
    )
    gpu_available: bool = Field(default=False, description="GPU 是否可用")
    gpu_memory_used: Optional[float] = Field(
        default=None, description="GPU 显存使用量 (GB)"
    )
    gpu_memory_total: Optional[float] = Field(
        default=None, description="GPU 显存总量 (GB)"
    )
