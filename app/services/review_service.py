"""
图片审核主服务

整合所有审核模块，提供统一的审核入口
"""

import uuid
import time
import base64
import asyncio
from io import BytesIO
from typing import Optional, List, Tuple
from functools import lru_cache

import httpx
import numpy as np
from PIL import Image

from app.core import settings, get_logger
from app.schemas.request import ReviewRequest, ReviewType
from app.schemas.response import (
    ReviewResponse,
    ReviewResults,
    QualityResult,
    QualityDetails,
    AircraftResult,
    ClassPrediction,
    RegistrationResult,
    OcclusionResult,
    ViolationResult,
)

logger = get_logger("review_service")


class ReviewService:
    """图片审核服务"""

    def __init__(self):
        self._infer_loaded = False
        self._classifier = None
        self._airline_classifier = None
        self._detector = None
        self._ocr = None
        self._quality_assessor = None

    def _lazy_load_infer(self):
        """延迟加载推理模块"""
        if self._infer_loaded:
            return

        try:
            from infer import (
                get_aircraft_classifier,
                get_airline_classifier,
                RegistrationDetector,
                RegistrationOCR,
                ImageQualityAssessor,
            )

            self._classifier = get_aircraft_classifier()
            self._airline_classifier = get_airline_classifier()
            self._detector = RegistrationDetector()
            self._ocr = RegistrationOCR()
            self._quality_assessor = ImageQualityAssessor()
            self._infer_loaded = True
            logger.info("推理模块加载完成")
        except ImportError as e:
            logger.warning(f"推理模块加载失败: {e}，将使用占位实现")
        except Exception as e:
            logger.error(f"推理模块初始化错误: {e}")

    async def _load_image(self, request: ReviewRequest) -> Image.Image:
        """从请求中加载图片"""
        if request.image_base64:
            image_data = base64.b64decode(request.image_base64)
            return Image.open(BytesIO(image_data)).convert("RGB")

        if request.image_url:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(request.image_url)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")

        raise ValueError("未提供有效的图片来源")

    def _assess_quality(self, image: Image.Image) -> QualityResult:
        """评估图片质量"""
        if self._quality_assessor is None:
            # 占位实现
            return QualityResult(
                passed=True,
                score=0.8,
                details=QualityDetails(
                    sharpness=0.8,
                    exposure=0.8,
                    composition=0.8,
                    noise=0.8,
                    color=0.8,
                ),
            )

        img_array = np.array(image)
        result = self._quality_assessor.assess(img_array)

        details = QualityDetails(
            sharpness=result.get("sharpness", 0.0),
            exposure=result.get("exposure", 0.0),
            composition=result.get("composition", 0.0),
            noise=result.get("noise", 0.0),
            color=result.get("color", 0.0),
        )

        score = result.get("overall", 0.0)
        passed = score >= settings.quality_threshold

        return QualityResult(
            passed=passed,
            score=score,
            details=details,
            reason=None if passed else f"图片质量评分 {score:.2f} 低于阈值 {settings.quality_threshold}",
        )

    def _identify_aircraft(self, image: Image.Image) -> AircraftResult:
        """识别飞机类型和航司"""
        if self._classifier is None:
            # 占位实现
            return AircraftResult(
                passed=True,
                is_aircraft=True,
                confidence=0.95,
                aircraft_type="Unknown",
                airline="Unknown",
            )

        img_array = np.array(image)

        # 机型分类 (Top-3)
        type_result = self._classifier.predict(img_array, top_k=3)
        top1_type = type_result.get("top1")
        is_aircraft = type_result.get("is_confident", False)

        # 构建机型 Top-3
        type_top3 = None
        if type_result.get("success") and type_result.get("top_k"):
            type_top3 = [
                ClassPrediction(
                    class_id=p["class_id"],
                    class_name=p["class_name"],
                    confidence=p["confidence"]
                )
                for p in type_result["top_k"][:3]
            ]

        # 航司分类 (Top-3)
        airline_result = None
        airline_top3 = None
        if self._airline_classifier and is_aircraft:
            airline_result = self._airline_classifier.predict(img_array, top_k=3)
            if airline_result.get("success") and airline_result.get("top_k"):
                airline_top3 = [
                    ClassPrediction(
                        class_id=p["class_id"],
                        class_name=p["class_name"],
                        confidence=p["confidence"]
                    )
                    for p in airline_result["top_k"][:3]
                ]

        top1_airline = airline_result.get("top1") if airline_result else None

        return AircraftResult(
            passed=is_aircraft,
            is_aircraft=is_aircraft,
            confidence=top1_type["confidence"] if top1_type else 0.0,
            aircraft_type=top1_type.get("class_name") if top1_type and is_aircraft else None,
            aircraft_type_confidence=top1_type["confidence"] if top1_type else None,
            aircraft_type_top3=type_top3,
            airline=top1_airline.get("class_name") if top1_airline else None,
            airline_confidence=top1_airline.get("confidence") if top1_airline else None,
            airline_top3=airline_top3,
            reason=None if is_aircraft else "未检测到飞机或置信度过低",
        )

    def _detect_registration(self, image: Image.Image) -> RegistrationResult:
        """检测并识别注册号"""
        if self._detector is None or self._ocr is None:
            # 占位实现
            return RegistrationResult(
                passed=True,
                detected=False,
                value=None,
                confidence=0.0,
                clarity_score=0.0,
            )

        img_array = np.array(image)

        # 检测注册号区域
        detections = self._detector.detect(img_array)
        if not detections:
            return RegistrationResult(
                passed=False,
                detected=False,
                value=None,
                confidence=0.0,
                clarity_score=0.0,
                reason="未检测到注册号区域",
            )

        # 获取最佳检测结果
        best = self._detector.get_best_detection(detections)
        bbox = best["bbox"]

        # OCR 识别
        crops = self._detector.detect_and_crop(img_array)
        if crops:
            ocr_results = self._ocr.recognize_from_crops(crops)
            if ocr_results:
                reg_value = ocr_results[0]["text"]
                confidence = ocr_results[0]["confidence"]

                # 计算清晰度（基于检测置信度和 OCR 置信度）
                clarity_score = (best["confidence"] + confidence) / 2
                passed = clarity_score >= settings.registration_clarity_threshold

                return RegistrationResult(
                    passed=passed,
                    detected=True,
                    value=reg_value,
                    confidence=confidence,
                    clarity_score=clarity_score,
                    bbox=bbox,
                    reason=None if passed else f"注册号清晰度 {clarity_score:.2f} 低于阈值",
                )

        return RegistrationResult(
            passed=False,
            detected=True,
            value=None,
            confidence=best["confidence"],
            clarity_score=best["confidence"],
            bbox=bbox,
            reason="检测到注册号区域但 OCR 识别失败",
        )

    def _detect_occlusion(self, image: Image.Image) -> OcclusionResult:
        """检测飞机主体遮挡"""
        # TODO: 实现遮挡检测逻辑
        # 当前为占位实现
        return OcclusionResult(
            passed=True,
            occlusion_percentage=0.0,
            details="未检测到明显遮挡",
        )

    def _detect_violation(self, image: Image.Image) -> ViolationResult:
        """检测违规内容"""
        # TODO: 实现违规检测逻辑
        # 当前为占位实现
        return ViolationResult(
            passed=True,
            has_watermark=False,
            has_sensitive_content=False,
            has_other_logo=False,
            violations=[],
        )

    async def review(self, request: ReviewRequest) -> ReviewResponse:
        """
        执行图片审核

        Args:
            request: 审核请求

        Returns:
            审核响应
        """
        start_time = time.perf_counter()
        review_id = str(uuid.uuid4())

        try:
            # 延迟加载推理模块
            self._lazy_load_infer()

            # 加载图片
            image = await self._load_image(request)
            logger.info(f"[{review_id}] 图片加载完成，尺寸: {image.size}")

            # 执行各项审核
            results = ReviewResults(overall_pass=True)
            fail_reasons: List[str] = []

            if ReviewType.QUALITY in request.review_types:
                results.quality = self._assess_quality(image)
                if not results.quality.passed:
                    results.overall_pass = False
                    if results.quality.reason:
                        fail_reasons.append(results.quality.reason)

            if ReviewType.AIRCRAFT in request.review_types:
                results.aircraft = self._identify_aircraft(image)
                if not results.aircraft.passed:
                    results.overall_pass = False
                    if results.aircraft.reason:
                        fail_reasons.append(results.aircraft.reason)

            if ReviewType.REGISTRATION in request.review_types:
                results.registration = self._detect_registration(image)
                if not results.registration.passed:
                    results.overall_pass = False
                    if results.registration.reason:
                        fail_reasons.append(results.registration.reason)

            if ReviewType.OCCLUSION in request.review_types:
                results.occlusion = self._detect_occlusion(image)
                if not results.occlusion.passed:
                    results.overall_pass = False
                    if results.occlusion.reason:
                        fail_reasons.append(results.occlusion.reason)

            if ReviewType.VIOLATION in request.review_types:
                results.violation = self._detect_violation(image)
                if not results.violation.passed:
                    results.overall_pass = False
                    if results.violation.reason:
                        fail_reasons.append(results.violation.reason)

            processing_time = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"[{review_id}] 审核完成，耗时: {processing_time:.2f}ms，"
                f"结果: {'通过' if results.overall_pass else '未通过'}"
            )

            return ReviewResponse(
                success=True,
                review_id=review_id,
                results=results,
                fail_reasons=fail_reasons,
                processing_time_ms=processing_time,
                metadata=request.metadata,
            )

        except httpx.HTTPError as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"[{review_id}] 图片下载失败: {e}")
            return ReviewResponse(
                success=False,
                review_id=review_id,
                fail_reasons=["图片下载失败"],
                processing_time_ms=processing_time,
                error=str(e),
                metadata=request.metadata,
            )

        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.exception(f"[{review_id}] 审核过程发生错误: {e}")
            return ReviewResponse(
                success=False,
                review_id=review_id,
                fail_reasons=["内部服务错误"],
                processing_time_ms=processing_time,
                error=str(e),
                metadata=request.metadata,
            )

    def get_model_status(self) -> dict:
        """获取模型加载状态"""
        return {
            "infer_loaded": self._infer_loaded,
            "classifier": self._classifier is not None,
            "airline_classifier": self._airline_classifier is not None,
            "detector": self._detector is not None,
            "ocr": self._ocr is not None,
            "quality_assessor": self._quality_assessor is not None,
        }


# 单例实例
_review_service: Optional[ReviewService] = None


@lru_cache()
def get_review_service() -> ReviewService:
    """获取审核服务单例"""
    global _review_service
    if _review_service is None:
        _review_service = ReviewService()
    return _review_service
