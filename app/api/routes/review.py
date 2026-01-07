"""
图片审核接口
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
import httpx

from app.core import settings, get_logger
from app.schemas.request import ReviewRequest
from app.schemas.response import ReviewResponse
from app.services import get_review_service, ReviewService
from app.api.deps import request_counter

router = APIRouter()
logger = get_logger("api.review")


async def send_callback(callback_url: str, response: ReviewResponse):
    """发送回调通知"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(
                callback_url,
                json=response.model_dump(mode="json"),
            )
            logger.info(f"回调发送成功: {callback_url}")
    except Exception as e:
        logger.error(f"回调发送失败: {callback_url}, 错误: {e}")


@router.post("", response_model=ReviewResponse)
async def review_image(
    request: ReviewRequest,
    background_tasks: BackgroundTasks,
    review_service: ReviewService = Depends(get_review_service),
) -> ReviewResponse:
    """
    图片审核接口

    对上传的航空照片进行自动化审核，包括：
    - 图片质量评估
    - 飞机识别与机型分类
    - 注册号识别与清晰度检测
    - 主体遮挡检测
    - 违规内容检测

    **请求参数**:
    - `image_url`: 图片 URL（与 image_base64 二选一）
    - `image_base64`: Base64 编码的图片（与 image_url 二选一）
    - `review_types`: 需要执行的审核类型列表
    - `callback_url`: 审核完成后的回调 URL（可选）
    - `metadata`: 附加元数据（可选）

    **返回**:
    - 审核结果，包含各项审核的详细信息
    """
    logger.info(f"收到审核请求，审核类型: {request.review_types}")

    # 执行审核
    response = await review_service.review(request)

    # 计数
    request_counter.increment(success=response.success)

    # 发送回调
    callback_url = request.callback_url or settings.backend_callback_url
    if callback_url:
        background_tasks.add_task(send_callback, callback_url, response)

    return response


@router.post("/batch", response_model=list[ReviewResponse])
async def review_images_batch(
    requests: list[ReviewRequest],
    background_tasks: BackgroundTasks,
    review_service: ReviewService = Depends(get_review_service),
) -> list[ReviewResponse]:
    """
    批量图片审核接口

    同时审核多张图片，适用于批量上传场景。

    **限制**:
    - 单次最多 10 张图片
    """
    if len(requests) > 10:
        raise HTTPException(
            status_code=400,
            detail="单次批量审核最多支持 10 张图片"
        )

    logger.info(f"收到批量审核请求，数量: {len(requests)}")

    responses = []
    for request in requests:
        response = await review_service.review(request)
        responses.append(response)

        # 计数
        request_counter.increment(success=response.success)

        # 发送回调
        callback_url = request.callback_url or settings.backend_callback_url
        if callback_url:
            background_tasks.add_task(send_callback, callback_url, response)

    return responses
