"""
响应模型定义
"""

from typing import Optional, List, Dict, Any, Generic, TypeVar
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


T = TypeVar("T")


class StatusEnum(str, Enum):
    """状态枚举"""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    PROCESSING = "processing"


class BaseResponse(BaseModel, Generic[T]):
    """基础响应模型"""
    success: bool = Field(..., description="请求是否成功")
    message: str = Field("OK", description="响应消息")
    data: Optional[T] = Field(None, description="响应数据")
    error: Optional[str] = Field(None, description="错误信息")
    request_id: Optional[str] = Field(None, description="请求 ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")


# ============ 质量评估 ============

class QualityDetails(BaseModel):
    """质量评估详情"""
    sharpness: float = Field(..., ge=0, le=1, description="清晰度")
    exposure: float = Field(..., ge=0, le=1, description="曝光")
    composition: float = Field(..., ge=0, le=1, description="构图")
    noise: float = Field(..., ge=0, le=1, description="噪点")
    color: float = Field(..., ge=0, le=1, description="色彩")


class QualityResult(BaseModel):
    """质量评估结果"""
    passed: bool = Field(..., description="是否通过")
    score: float = Field(..., ge=0, le=1, description="综合分数")
    details: Optional[QualityDetails] = Field(None, description="各项评分")


class QualityResponse(BaseResponse[QualityResult]):
    """质量评估响应"""
    pass


# ============ 分类 ============

class ClassPrediction(BaseModel):
    """分类预测结果"""
    class_id: int = Field(..., description="类别 ID")
    class_name: str = Field(..., description="类别名称")
    confidence: float = Field(..., ge=0, le=1, description="置信度")


class ClassifyResult(BaseModel):
    """分类结果"""
    passed: bool = Field(..., description="是否通过置信度阈值")
    task: str = Field(..., description="分类任务类型")
    prediction: Optional[ClassPrediction] = Field(None, description="Top-1 预测")
    top_k: List[ClassPrediction] = Field(default=[], description="Top-K 预测")


class ClassifyResponse(BaseResponse[ClassifyResult]):
    """分类响应"""
    pass


# ============ 检测 ============

class BoundingBox(BaseModel):
    """边界框"""
    xyxy: List[float] = Field(..., description="像素坐标 [x1, y1, x2, y2]")
    xywh: List[float] = Field(..., description="中心坐标 [x, y, w, h]")
    xywhn: List[float] = Field(..., description="归一化坐标 [x, y, w, h]")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    class_id: int = Field(..., description="类别 ID")
    class_name: str = Field(..., description="类别名称")


class DetectResult(BaseModel):
    """检测结果"""
    detected: bool = Field(..., description="是否检测到目标")
    count: int = Field(..., ge=0, description="检测数量")
    boxes: List[BoundingBox] = Field(default=[], description="边界框列表")
    image_size: List[int] = Field(..., description="图片尺寸 [height, width]")
    crops_base64: Optional[List[str]] = Field(None, description="裁剪图片 Base64")


class DetectResponse(BaseResponse[DetectResult]):
    """检测响应"""
    pass


# ============ OCR ============

class OCRResult(BaseModel):
    """OCR 结果"""
    text: str = Field(..., description="识别文本")
    raw_text: str = Field(..., description="原始识别文本")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    valid: bool = Field(..., description="是否有效")
    bbox: Optional[List[int]] = Field(None, description="边界框像素坐标")


class OCRResponse(BaseResponse[OCRResult]):
    """OCR 响应"""
    pass


# ============ 综合审核 ============

class RegistrationResult(BaseModel):
    """注册号识别结果"""
    detected: bool = Field(..., description="是否检测到注册号区域")
    text: Optional[str] = Field(None, description="识别文本")
    confidence: float = Field(0.0, ge=0, le=1, description="置信度")
    valid: bool = Field(False, description="是否有效")
    bbox: Optional[List[float]] = Field(None, description="边界框")


class ReviewResult(BaseModel):
    """综合审核结果"""
    overall_pass: bool = Field(..., description="总体是否通过")
    fail_reasons: List[str] = Field(default=[], description="失败原因")
    quality: Optional[QualityResult] = Field(None, description="质量评估结果")
    aircraft: Optional[ClassifyResult] = Field(None, description="机型分类结果")
    airline: Optional[ClassifyResult] = Field(None, description="航司识别结果")
    registration: Optional[RegistrationResult] = Field(None, description="注册号识别结果")
    processing_time_ms: float = Field(..., description="处理耗时（毫秒）")


class ReviewResponse(BaseResponse[ReviewResult]):
    """综合审核响应"""
    pass


# ============ 批量审核 ============

class BatchReviewItem(BaseModel):
    """批量审核单项结果"""
    index: int = Field(..., description="图片索引")
    success: bool = Field(..., description="处理是否成功")
    result: Optional[ReviewResult] = Field(None, description="审核结果")
    error: Optional[str] = Field(None, description="错误信息")


class BatchReviewResult(BaseModel):
    """批量审核结果"""
    total: int = Field(..., description="总数")
    success_count: int = Field(..., description="成功数")
    failed_count: int = Field(..., description="失败数")
    items: List[BatchReviewItem] = Field(..., description="各项结果")
    task_id: Optional[str] = Field(None, description="异步任务 ID")


class BatchReviewResponse(BaseResponse[BatchReviewResult]):
    """批量审核响应"""
    pass


# ============ 健康检查 ============

class ModelStatus(BaseModel):
    """模型状态"""
    name: str = Field(..., description="模型名称")
    loaded: bool = Field(..., description="是否已加载")
    device: Optional[str] = Field(None, description="运行设备")
    num_classes: Optional[int] = Field(None, description="类别数")


class HealthResult(BaseModel):
    """健康检查结果"""
    status: StatusEnum = Field(..., description="服务状态")
    version: str = Field(..., description="服务版本")
    uptime_seconds: float = Field(..., description="运行时间（秒）")
    models: List[ModelStatus] = Field(default=[], description="模型状态")
    gpu_available: bool = Field(..., description="GPU 是否可用")
    gpu_memory_used: Optional[float] = Field(None, description="GPU 显存使用（GB）")
    gpu_memory_total: Optional[float] = Field(None, description="GPU 显存总量（GB）")


class HealthResponse(BaseResponse[HealthResult]):
    """健康检查响应"""
    pass


# ============ 统计 ============

class RequestStats(BaseModel):
    """请求统计"""
    total_requests: int = Field(..., description="总请求数")
    success_count: int = Field(..., description="成功数")
    failed_count: int = Field(..., description="失败数")
    avg_latency_ms: float = Field(..., description="平均延迟（毫秒）")
    p50_latency_ms: float = Field(..., description="P50 延迟（毫秒）")
    p95_latency_ms: float = Field(..., description="P95 延迟（毫秒）")
    p99_latency_ms: float = Field(..., description="P99 延迟（毫秒）")


class EndpointStats(BaseModel):
    """端点统计"""
    endpoint: str = Field(..., description="端点路径")
    method: str = Field(..., description="HTTP 方法")
    stats: RequestStats = Field(..., description="请求统计")


class StatsResult(BaseModel):
    """统计结果"""
    period: str = Field(..., description="统计周期")
    start_time: datetime = Field(..., description="开始时间")
    end_time: datetime = Field(..., description="结束时间")
    overall: RequestStats = Field(..., description="总体统计")
    by_endpoint: List[EndpointStats] = Field(default=[], description="按端点统计")
    by_review_type: Dict[str, RequestStats] = Field(default={}, description="按审核类型统计")


class StatsResponse(BaseResponse[StatsResult]):
    """统计响应"""
    pass


# ============ 模型信息 ============

class ModelInfo(BaseModel):
    """模型详细信息"""
    name: str = Field(..., description="模型名称")
    task: str = Field(..., description="任务类型")
    loaded: bool = Field(..., description="是否已加载")
    device: str = Field(..., description="运行设备")
    half: bool = Field(..., description="是否使用 FP16")
    num_classes: Optional[int] = Field(None, description="类别数")
    class_names: Optional[Dict[int, str]] = Field(None, description="类别名称映射")
    input_size: Optional[int] = Field(None, description="输入尺寸")


class ModelInfoResponse(BaseResponse[List[ModelInfo]]):
    """模型信息响应"""
    pass
