"""
健康检查接口
"""

import torch
from fastapi import APIRouter

from app.core import settings
from app.schemas.response import HealthResponse
from app.services import get_review_service
from app.api.deps import request_counter

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    健康检查

    返回服务状态、版本信息和模型加载状态
    """
    # GPU 状态
    gpu_available = torch.cuda.is_available()
    gpu_memory_used = None
    gpu_memory_total = None

    if gpu_available:
        try:
            gpu_memory_used = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            pass

    # 模型状态
    review_service = get_review_service()
    model_status = review_service.get_model_status()

    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        models_loaded=model_status,
        gpu_available=gpu_available,
        gpu_memory_used=gpu_memory_used,
        gpu_memory_total=gpu_memory_total,
    )


@router.get("/stats")
async def get_stats() -> dict:
    """
    获取请求统计

    返回请求计数、成功/失败数、运行时间等
    """
    return request_counter.get_stats()
