"""
审核 API 路由

提供图片审核相关的 API 端点
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from ..schemas import (
    ReviewRequest,
    ReviewResponse,
    ClassifyRequest,
    ClassifyResponse,
    DetectRequest,
    DetectResponse,
    OCRRequest,
    OCRResponse,
    QualityRequest,
    QualityResponse,
    BatchReviewRequest,
    BatchReviewResponse,
)
from ..schemas.response import (
    ReviewResult,
    ClassifyResult,
    ClassPrediction,
    DetectResult,
    BoundingBox,
    OCRResult,
    QualityResult,
    QualityDetails,
    RegistrationResult,
    BatchReviewResult,
    BatchReviewItem,
)
from ..deps import (
    get_inference_service,
    InferenceService,
    resolve_image_input,
    encode_image_to_base64,
    timing,
    generate_request_id,
)
from .stats import stats_collector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/review", tags=["审核"])


# ============ 综合审核 ============

@router.post(
    "",
    response_model=ReviewResponse,
    summary="综合图片审核",
    description="对图片进行综合审核，包括质量评估、机型分类、航司识别、注册号识别"
)
async def review_image(
    request: ReviewRequest,
    service: InferenceService = Depends(get_inference_service)
) -> ReviewResponse:
    """综合图片审核"""
    request_id = generate_request_id()

    with timing() as t:
        try:
            # 加载图片
            image = await resolve_image_input(
                image_url=str(request.image_url) if request.image_url else None,
                image_base64=request.image_base64
            )

            result = ReviewResult(
                overall_pass=True,
                fail_reasons=[],
                processing_time_ms=0
            )

            # 质量评估
            if "quality" in request.review_types:
                quality_result = service.quality_assessor.assess(image)
                if quality_result["success"]:
                    threshold = request.quality_threshold or service.config.quality.pass_threshold
                    passed = quality_result["score"] >= threshold
                    result.quality = QualityResult(
                        passed=passed,
                        score=quality_result["score"],
                        details=QualityDetails(**quality_result["details"])
                    )
                    if not passed:
                        result.overall_pass = False
                        result.fail_reasons.append("图片质量不达标")

            # 机型分类
            if "aircraft" in request.review_types:
                clf_result = service.aircraft_classifier.predict(image)
                if clf_result["success"]:
                    top1 = clf_result.get("top1")
                    passed = clf_result.get("is_confident", False)
                    result.aircraft = ClassifyResult(
                        passed=passed,
                        task="aircraft",
                        prediction=ClassPrediction(**top1) if top1 else None,
                        top_k=[ClassPrediction(**p) for p in clf_result.get("top_k", [])]
                    )
                    if not passed:
                        result.overall_pass = False
                        result.fail_reasons.append("机型识别置信度不足")

            # 航司识别
            if "airline" in request.review_types:
                clf_result = service.airline_classifier.predict(image)
                if clf_result["success"]:
                    top1 = clf_result.get("top1")
                    passed = clf_result.get("is_confident", False)
                    result.airline = ClassifyResult(
                        passed=passed,
                        task="airline",
                        prediction=ClassPrediction(**top1) if top1 else None,
                        top_k=[ClassPrediction(**p) for p in clf_result.get("top_k", [])]
                    )

            # 注册号识别
            if "registration" in request.review_types:
                det_result = service.detector.detect_and_crop(image)
                if det_result["success"] and det_result["detected"]:
                    ocr_results = service.ocr.recognize_from_crops(det_result["crops"])
                    valid_results = [r for r in ocr_results if r.get("valid")]

                    if valid_results:
                        best = max(valid_results, key=lambda x: x["confidence"])
                        result.registration = RegistrationResult(
                            detected=True,
                            text=best["text"],
                            confidence=best["confidence"],
                            valid=True,
                            bbox=best.get("bbox")
                        )
                    else:
                        first = ocr_results[0] if ocr_results else {}
                        result.registration = RegistrationResult(
                            detected=True,
                            text=first.get("text", ""),
                            confidence=first.get("confidence", 0),
                            valid=False,
                            bbox=first.get("bbox")
                        )
                        result.overall_pass = False
                        result.fail_reasons.append("注册号识别无效")
                else:
                    result.registration = RegistrationResult(detected=False)
                    result.overall_pass = False
                    result.fail_reasons.append("未检测到注册号")

            result.processing_time_ms = t["elapsed_ms"]

            # 记录统计
            stats_collector.record_request(
                endpoint="/review",
                method="POST",
                success=True,
                latency_ms=t["elapsed_ms"],
                review_types=request.review_types
            )

            return ReviewResponse(
                success=True,
                message="审核完成",
                data=result,
                request_id=request_id
            )

        except Exception as e:
            logger.error(f"审核失败: {e}", exc_info=True)
            stats_collector.record_request(
                endpoint="/review",
                method="POST",
                success=False,
                latency_ms=t["elapsed_ms"]
            )
            raise HTTPException(status_code=500, detail=str(e))


# ============ 分类 ============

@router.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="图片分类",
    description="对图片进行分类（机型或航司）"
)
async def classify_image(
    request: ClassifyRequest,
    service: InferenceService = Depends(get_inference_service)
) -> ClassifyResponse:
    """图片分类"""
    request_id = generate_request_id()

    with timing() as t:
        try:
            image = await resolve_image_input(
                image_url=str(request.image_url) if request.image_url else None,
                image_base64=request.image_base64
            )

            if request.task == "aircraft":
                clf_result = service.aircraft_classifier.predict(image, top_k=request.top_k)
            else:
                clf_result = service.airline_classifier.predict(image, top_k=request.top_k)

            if not clf_result["success"]:
                raise HTTPException(status_code=500, detail=clf_result.get("error", "分类失败"))

            top1 = clf_result.get("top1")
            result = ClassifyResult(
                passed=clf_result.get("is_confident", False),
                task=request.task,
                prediction=ClassPrediction(**top1) if top1 else None,
                top_k=[ClassPrediction(**p) for p in clf_result.get("top_k", [])]
            )

            stats_collector.record_request(
                endpoint="/review/classify",
                method="POST",
                success=True,
                latency_ms=t["elapsed_ms"],
                review_types=[request.task]
            )

            return ClassifyResponse(
                success=True,
                message="分类完成",
                data=result,
                request_id=request_id
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"分类失败: {e}", exc_info=True)
            stats_collector.record_request(
                endpoint="/review/classify",
                method="POST",
                success=False,
                latency_ms=t["elapsed_ms"]
            )
            raise HTTPException(status_code=500, detail=str(e))


# ============ 检测 ============

@router.post(
    "/detect",
    response_model=DetectResponse,
    summary="注册号区域检测",
    description="检测图片中的注册号区域"
)
async def detect_registration(
    request: DetectRequest,
    service: InferenceService = Depends(get_inference_service)
) -> DetectResponse:
    """注册号区域检测"""
    request_id = generate_request_id()

    with timing() as t:
        try:
            image = await resolve_image_input(
                image_url=str(request.image_url) if request.image_url else None,
                image_base64=request.image_base64
            )

            if request.return_crops:
                det_result = service.detector.detect_and_crop(
                    image,
                    padding=request.crop_padding,
                    conf_threshold=request.conf_threshold
                )
            else:
                det_result = service.detector.detect(
                    image,
                    conf_threshold=request.conf_threshold,
                    iou_threshold=request.iou_threshold
                )

            if not det_result["success"]:
                raise HTTPException(status_code=500, detail=det_result.get("error", "检测失败"))

            # 构建结果
            boxes = []
            crops_base64 = None

            if request.return_crops and det_result.get("crops"):
                crops_base64 = []
                for crop in det_result["crops"]:
                    crops_base64.append(encode_image_to_base64(crop["image"]))
                    boxes.append(BoundingBox(
                        xyxy=crop["bbox"],
                        xywh=[0, 0, 0, 0],  # 简化
                        xywhn=[0, 0, 0, 0],
                        confidence=crop["confidence"],
                        class_id=0,
                        class_name="registration"
                    ))
            elif det_result.get("boxes"):
                for box in det_result["boxes"]:
                    boxes.append(BoundingBox(**box))

            result = DetectResult(
                detected=det_result.get("detected", False),
                count=len(boxes),
                boxes=boxes,
                image_size=det_result.get("image_size", [0, 0]),
                crops_base64=crops_base64
            )

            stats_collector.record_request(
                endpoint="/review/detect",
                method="POST",
                success=True,
                latency_ms=t["elapsed_ms"],
                review_types=["registration"]
            )

            return DetectResponse(
                success=True,
                message="检测完成",
                data=result,
                request_id=request_id
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"检测失败: {e}", exc_info=True)
            stats_collector.record_request(
                endpoint="/review/detect",
                method="POST",
                success=False,
                latency_ms=t["elapsed_ms"]
            )
            raise HTTPException(status_code=500, detail=str(e))


# ============ OCR ============

@router.post(
    "/ocr",
    response_model=OCRResponse,
    summary="注册号 OCR",
    description="识别图片中的注册号文字"
)
async def recognize_registration(
    request: OCRRequest,
    service: InferenceService = Depends(get_inference_service)
) -> OCRResponse:
    """注册号 OCR"""
    request_id = generate_request_id()

    with timing() as t:
        try:
            image = await resolve_image_input(
                image_url=str(request.image_url) if request.image_url else None,
                image_base64=request.image_base64
            )

            if request.bbox:
                ocr_result = service.ocr.recognize_from_bbox(
                    image,
                    bbox=request.bbox,
                    padding=request.padding
                )
            else:
                ocr_result = service.ocr.recognize(image)

            if not ocr_result["success"]:
                raise HTTPException(status_code=500, detail=ocr_result.get("error", "OCR 失败"))

            result = OCRResult(
                text=ocr_result["text"],
                raw_text=ocr_result.get("raw_text", ocr_result["text"]),
                confidence=ocr_result["confidence"],
                valid=ocr_result["valid"],
                bbox=ocr_result.get("bbox")
            )

            stats_collector.record_request(
                endpoint="/review/ocr",
                method="POST",
                success=True,
                latency_ms=t["elapsed_ms"],
                review_types=["registration"]
            )

            return OCRResponse(
                success=True,
                message="识别完成",
                data=result,
                request_id=request_id
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"OCR 失败: {e}", exc_info=True)
            stats_collector.record_request(
                endpoint="/review/ocr",
                method="POST",
                success=False,
                latency_ms=t["elapsed_ms"]
            )
            raise HTTPException(status_code=500, detail=str(e))


# ============ 质量评估 ============

@router.post(
    "/quality",
    response_model=QualityResponse,
    summary="图片质量评估",
    description="评估图片质量（清晰度、曝光、构图等）"
)
async def assess_quality(
    request: QualityRequest,
    service: InferenceService = Depends(get_inference_service)
) -> QualityResponse:
    """图片质量评估"""
    request_id = generate_request_id()

    with timing() as t:
        try:
            image = await resolve_image_input(
                image_url=str(request.image_url) if request.image_url else None,
                image_base64=request.image_base64
            )

            if request.quick_mode:
                quality_result = service.quality_assessor.quick_assess(image)
                result = QualityResult(
                    passed=quality_result["pass"],
                    score=quality_result["sharpness"],
                    details=None
                )
            else:
                quality_result = service.quality_assessor.assess(image)
                if not quality_result["success"]:
                    raise HTTPException(
                        status_code=500,
                        detail=quality_result.get("error", "质量评估失败")
                    )

                threshold = request.threshold or service.config.quality.pass_threshold
                passed = quality_result["score"] >= threshold

                result = QualityResult(
                    passed=passed,
                    score=quality_result["score"],
                    details=QualityDetails(**quality_result["details"])
                )

            stats_collector.record_request(
                endpoint="/review/quality",
                method="POST",
                success=True,
                latency_ms=t["elapsed_ms"],
                review_types=["quality"]
            )

            return QualityResponse(
                success=True,
                message="评估完成",
                data=result,
                request_id=request_id
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"质量评估失败: {e}", exc_info=True)
            stats_collector.record_request(
                endpoint="/review/quality",
                method="POST",
                success=False,
                latency_ms=t["elapsed_ms"]
            )
            raise HTTPException(status_code=500, detail=str(e))


# ============ 批量审核 ============

@router.post(
    "/batch",
    response_model=BatchReviewResponse,
    summary="批量图片审核",
    description="批量审核多张图片"
)
async def batch_review(
    request: BatchReviewRequest,
    background_tasks: BackgroundTasks,
    service: InferenceService = Depends(get_inference_service)
) -> BatchReviewResponse:
    """批量图片审核"""
    request_id = generate_request_id()

    with timing() as t:
        items = []
        success_count = 0
        failed_count = 0

        for idx, img_input in enumerate(request.images):
            try:
                image = await resolve_image_input(
                    image_url=str(img_input.image_url) if img_input.image_url else None,
                    image_base64=img_input.image_base64
                )

                # 简化的审核逻辑
                item_result = ReviewResult(
                    overall_pass=True,
                    fail_reasons=[],
                    processing_time_ms=0
                )

                if "quality" in request.review_types:
                    qr = service.quality_assessor.assess(image)
                    if qr["success"]:
                        item_result.quality = QualityResult(
                            passed=qr["pass"],
                            score=qr["score"],
                            details=QualityDetails(**qr["details"])
                        )

                if "aircraft" in request.review_types:
                    cr = service.aircraft_classifier.predict(image)
                    if cr["success"]:
                        top1 = cr.get("top1")
                        item_result.aircraft = ClassifyResult(
                            passed=cr.get("is_confident", False),
                            task="aircraft",
                            prediction=ClassPrediction(**top1) if top1 else None,
                            top_k=[]
                        )

                items.append(BatchReviewItem(
                    index=idx,
                    success=True,
                    result=item_result
                ))
                success_count += 1

            except Exception as e:
                logger.error(f"批量审核第 {idx} 张失败: {e}")
                items.append(BatchReviewItem(
                    index=idx,
                    success=False,
                    error=str(e)
                ))
                failed_count += 1

        result = BatchReviewResult(
            total=len(request.images),
            success_count=success_count,
            failed_count=failed_count,
            items=items
        )

        stats_collector.record_request(
            endpoint="/review/batch",
            method="POST",
            success=True,
            latency_ms=t["elapsed_ms"]
        )

        return BatchReviewResponse(
            success=True,
            message=f"批量审核完成: {success_count}/{len(request.images)} 成功",
            data=result,
            request_id=request_id
        )
