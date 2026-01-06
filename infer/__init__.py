"""
Infer 推理服务模块

提供飞机机型分类、航司识别、注册号检测与 OCR、图片质量评估等功能。

使用示例:

    from infer import (
        get_aircraft_classifier,
        get_airline_classifier,
        get_registration_detector,
        get_registration_ocr,
        get_quality_assessor
    )

    # 机型分类
    classifier = get_aircraft_classifier()
    result = classifier.classify("aircraft.jpg")
    print(result)  # {"pass": True, "aircraft_type": "A320", "confidence": 0.95}

    # 航司识别
    airline_clf = get_airline_classifier()
    result = airline_clf.classify("aircraft.jpg")
    print(result)  # {"pass": True, "airline": "China Eastern", "confidence": 0.92}

    # 注册号检测 + OCR
    detector = get_registration_detector()
    ocr = get_registration_ocr()

    det_result = detector.detect_and_crop("aircraft.jpg")
    if det_result["detected"]:
        ocr_results = ocr.recognize_from_crops(det_result["crops"])
        print(ocr_results)  # [{"text": "B-1234", "confidence": 0.95, "valid": True}]

    # 图片质量评估
    assessor = get_quality_assessor()
    result = assessor.assess("aircraft.jpg")
    print(result)  # {"pass": True, "score": 0.85, "details": {...}}
"""

from .config import InferConfig, get_config, set_config
from .classifier import (
    AircraftClassifier,
    AirlineClassifier,
    get_aircraft_classifier,
    get_airline_classifier
)
from .detector import (
    RegistrationDetector,
    get_registration_detector
)
from .ocr import (
    RegistrationOCR,
    get_registration_ocr
)
from .quality import (
    ImageQualityAssessor,
    get_quality_assessor
)
from .models import ModelLoader, ModelRegistry, get_registry

__all__ = [
    # Config
    "InferConfig",
    "get_config",
    "set_config",
    # Classifier
    "AircraftClassifier",
    "AirlineClassifier",
    "get_aircraft_classifier",
    "get_airline_classifier",
    # Detector
    "RegistrationDetector",
    "get_registration_detector",
    # OCR
    "RegistrationOCR",
    "get_registration_ocr",
    # Quality
    "ImageQualityAssessor",
    "get_quality_assessor",
    # Models
    "ModelLoader",
    "ModelRegistry",
    "get_registry",
]

__version__ = "0.1.0"
