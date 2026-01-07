# Infer 推理服务模块

AeroVision 航空摄影图片审核推理模块，提供飞机机型分类、航空公司识别、注册号检测与 OCR、图片质量评估等功能。

## 目录结构

```
infer/
├── __init__.py         # 模块入口，导出公共 API
├── config.py           # 配置管理
├── classifier.py       # 分类推理（机型 + 航司）
├── detector.py         # 注册号区域检测
├── ocr.py              # 注册号 OCR 识别
├── quality.py          # 图片质量评估
├── models/             # 模型管理子模块
│   ├── __init__.py
│   ├── loader.py       # 模型加载器
│   └── registry.py     # 模型注册表（单例）
└── README.md           # 本文件
```

## 快速开始

### 安装依赖

```bash
pip install torch ultralytics paddleocr opencv-python pillow numpy
```

### 基础用法

```python
from infer import (
    get_aircraft_classifier,
    get_airline_classifier,
    get_registration_detector,
    get_registration_ocr,
    get_quality_assessor
)

# 1. 机型分类
classifier = get_aircraft_classifier()
result = classifier.classify("aircraft.jpg")
print(result)
# {"pass": True, "aircraft_type": "A320", "confidence": 0.95}

# 2. 航司识别
airline_clf = get_airline_classifier()
result = airline_clf.classify("aircraft.jpg")
print(result)
# {"pass": True, "airline": "China Eastern", "confidence": 0.92}

# 3. 注册号检测 + OCR（Pipeline）
detector = get_registration_detector()
ocr = get_registration_ocr()

det_result = detector.detect_and_crop("aircraft.jpg")
if det_result["detected"]:
    ocr_results = ocr.recognize_from_crops(det_result["crops"])
    print(ocr_results)
    # [{"text": "B-1234", "confidence": 0.95, "valid": True}]

# 4. 图片质量评估
assessor = get_quality_assessor()
result = assessor.assess("aircraft.jpg")
print(result)
# {"pass": True, "score": 0.85, "details": {...}}
```

## 模块详细说明

### 1. 配置模块 (`config.py`)

集中管理推理服务的配置参数。

#### 配置类

| 类 | 说明 |
|---|---|
| `InferConfig` | 推理服务总配置 |
| `ModelConfig` | 模型文件路径和推理参数 |
| `OCRConfig` | OCR 相关配置 |
| `QualityConfig` | 质量评估配置 |

#### 配置项

```python
from infer import InferConfig, get_config, set_config

# 获取全局配置
config = get_config()

# 自定义配置
custom_config = InferConfig(
    model_dir=Path("./checkpoints"),
    device="cuda:0",
    half=True
)
set_config(custom_config)
```

#### 环境变量支持

| 环境变量 | 说明 | 默认值 |
|---|---|---|
| `MODEL_DIR` | 模型文件目录 | `models` |
| `DEVICE` | 推理设备 | `cuda:0` |
| `QUALITY_THRESHOLD` | 质量评估通过阈值 | `0.70` |

#### ModelConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `aircraft_classifier` | str | `aircraft_classifier.pt` | 机型分类模型文件名 |
| `airline_classifier` | str | `airline_classifier.pt` | 航司分类模型文件名 |
| `registration_detector` | str | `registration_detector.pt` | 注册号检测模型文件名 |
| `classifier_imgsz` | int | 224 | 分类模型输入尺寸 |
| `detector_imgsz` | int | 640 | 检测模型输入尺寸 |
| `classifier_conf` | float | 0.5 | 分类置信度阈值 |
| `detector_conf` | float | 0.25 | 检测置信度阈值 |
| `detector_iou` | float | 0.45 | 检测 NMS IoU 阈值 |

#### OCRConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `lang` | str | `en` | OCR 语言 |
| `rec_model_name` | str | `PP-OCRv4_server_rec_doc` | 识别模型名称 |
| `min_confidence` | float | 0.5 | 最低置信度 |
| `min_chars` | int | 4 | 最小字符数 |
| `max_chars` | int | 10 | 最大字符数 |
| `whitelist` | str | `A-Z0-9-` | 允许的字符 |

#### QualityConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `sharpness_weight` | float | 0.30 | 清晰度权重 |
| `exposure_weight` | float | 0.25 | 曝光权重 |
| `composition_weight` | float | 0.20 | 构图权重 |
| `noise_weight` | float | 0.15 | 噪点权重 |
| `color_weight` | float | 0.10 | 色彩权重 |
| `pass_threshold` | float | 0.70 | 通过阈值 |

---

### 2. 分类模块 (`classifier.py`)

基于 YOLOv8-cls 的图像分类推理。

#### AircraftClassifier - 机型分类器

```python
from infer import AircraftClassifier, get_aircraft_classifier

# 方式一：使用全局单例
classifier = get_aircraft_classifier()

# 方式二：自定义实例
classifier = AircraftClassifier(config=custom_config)
```

**简化接口 `classify()`**

```python
result = classifier.classify("aircraft.jpg")
# 返回:
# {
#     "pass": True,           # 是否通过置信度阈值
#     "aircraft_type": "A320", # 预测机型
#     "confidence": 0.95       # 置信度
# }
```

**详细接口 `predict()`**

```python
result = classifier.predict("aircraft.jpg", top_k=5)
# 返回:
# {
#     "success": True,
#     "top1": {"class_id": 0, "class_name": "A320", "confidence": 0.95},
#     "top_k": [
#         {"class_id": 0, "class_name": "A320", "confidence": 0.95},
#         {"class_id": 1, "class_name": "A321", "confidence": 0.03},
#         ...
#     ],
#     "is_confident": True
# }
```

#### AirlineClassifier - 航司识别器

```python
from infer import AirlineClassifier, get_airline_classifier

classifier = get_airline_classifier()
result = classifier.classify("aircraft.jpg")
# 返回:
# {
#     "pass": True,
#     "airline": "China Eastern",
#     "confidence": 0.92
# }
```

**支持的输入类型**

- 文件路径 (`str` / `Path`)
- NumPy 数组 (`np.ndarray`)
- PIL Image (`Image.Image`)

---

### 3. 检测模块 (`detector.py`)

基于 YOLOv8 的注册号区域检测。

#### RegistrationDetector - 注册号检测器

```python
from infer import RegistrationDetector, get_registration_detector

detector = get_registration_detector()
```

**检测接口 `detect()`**

```python
result = detector.detect("aircraft.jpg")
# 返回:
# {
#     "success": True,
#     "detected": True,
#     "count": 1,
#     "boxes": [
#         {
#             "xyxy": [x1, y1, x2, y2],       # 像素坐标
#             "xywh": [x, y, w, h],            # 中心点 + 宽高
#             "xywhn": [x, y, w, h],           # 归一化坐标
#             "confidence": 0.95,
#             "class_id": 0,
#             "class_name": "registration"
#         }
#     ],
#     "image_size": [height, width]
# }
```

**检测并裁剪 `detect_and_crop()`**

```python
result = detector.detect_and_crop("aircraft.jpg", padding=0.1)
# 返回:
# {
#     "success": True,
#     "detected": True,
#     "crops": [
#         {
#             "image": np.ndarray,  # 裁剪后的图片（RGB）
#             "bbox": [x1, y1, x2, y2],
#             "confidence": 0.95
#         }
#     ]
# }
```

**获取最佳检测 `get_best_detection()`**

```python
result = detector.get_best_detection("aircraft.jpg")
# 返回置信度最高的检测结果
# {
#     "detected": True,
#     "bbox": [x1, y1, x2, y2],
#     "bbox_normalized": [x, y, w, h],
#     "confidence": 0.95
# }
```

---

### 4. OCR 模块 (`ocr.py`)

基于 PaddleOCR 的注册号文字识别。

#### RegistrationOCR - 注册号 OCR

```python
from infer import RegistrationOCR, get_registration_ocr

ocr = get_registration_ocr()
```

**单图识别 `recognize()`**

```python
# 输入应为裁剪后的注册号区域
result = ocr.recognize(crop_image)
# 返回:
# {
#     "success": True,
#     "text": "B-1234",          # 处理后的文本
#     "raw_text": "B-1234",      # 原始识别文本
#     "confidence": 0.95,
#     "valid": True               # 是否通过验证
# }
```

**从边界框识别 `recognize_from_bbox()`**

```python
# YOLO 格式归一化坐标
bbox = [0.85, 0.65, 0.12, 0.04]  # [x_center, y_center, w, h]
result = ocr.recognize_from_bbox("aircraft.jpg", bbox, padding=0.1)
```

**批量识别 `recognize_from_crops()`**

配合 `detect_and_crop()` 使用：

```python
det_result = detector.detect_and_crop("aircraft.jpg")
if det_result["detected"]:
    ocr_results = ocr.recognize_from_crops(det_result["crops"])
    # 返回列表，每个元素包含:
    # {
    #     "success": True,
    #     "text": "B-1234",
    #     "confidence": 0.95,
    #     "valid": True,
    #     "bbox": [x1, y1, x2, y2],
    #     "detection_confidence": 0.95
    # }
```

**后处理规则**

1. 转大写，移除空格、点、逗号
2. 仅保留白名单字符（`A-Z0-9-`）
3. 验证长度（4-10 字符）
4. 验证置信度（>= 0.5）

---

### 5. 质量评估模块 (`quality.py`)

基于传统图像处理算法的质量评估。

#### ImageQualityAssessor - 图片质量评估器

```python
from infer import ImageQualityAssessor, get_quality_assessor

assessor = get_quality_assessor()
```

**综合评估 `assess()`**

```python
result = assessor.assess("aircraft.jpg")
# 返回:
# {
#     "success": True,
#     "pass": True,              # 是否通过阈值
#     "score": 0.85,             # 综合分数 (0-1)
#     "details": {
#         "sharpness": 0.90,     # 清晰度
#         "exposure": 0.80,      # 曝光
#         "composition": 0.85,   # 构图
#         "noise": 0.88,         # 噪点（越高越好）
#         "color": 0.82          # 色彩
#     }
# }
```

**快速评估 `quick_assess()`**

仅评估清晰度，速度更快：

```python
result = assessor.quick_assess("aircraft.jpg")
# 返回:
# {
#     "pass": True,
#     "sharpness": 0.90
# }
```

#### 评估算法说明

| 指标 | 算法 | 说明 |
|---|---|---|
| **清晰度** | Laplacian 方差 | 方差越大越清晰，归一化到 0-1 |
| **曝光** | LAB 亮度分析 | 检查亮度均值和过曝/欠曝比例 |
| **构图** | 边缘质心 + 三分法则 | 主体越接近三分点分数越高 |
| **噪点** | 高斯滤波差分 | 估计噪声标准差，越低越好 |
| **色彩** | HSV 饱和度 + 白平衡 | 检查饱和度和 RGB 通道均衡 |

---

### 6. 模型管理模块 (`models/`)

#### ModelLoader - 模型加载器

负责加载 YOLOv8 模型并进行预热。

```python
from infer.models import ModelLoader

loader = ModelLoader(device="cuda:0", half=True)

# 加载分类模型
model = loader.load_yolo_classifier("path/to/model.pt")

# 加载检测模型
model = loader.load_yolo_detector("path/to/model.pt")

# 获取模型信息
info = loader.get_model_info(model)
# {"task": "classify", "device": "cuda:0", "half": True, "names": {...}, "num_classes": 100}
```

#### ModelRegistry - 模型注册表

单例模式，管理所有已加载模型的生命周期。

```python
from infer.models import ModelRegistry, get_registry

registry = get_registry()

# 检查模型是否已加载
if not registry.is_loaded("aircraft_classifier"):
    model = registry.load_classifier(
        "aircraft_classifier",
        Path("models/aircraft_classifier.pt")
    )

# 获取已加载模型
model = registry.get("aircraft_classifier")

# 获取模型信息
info = registry.get_info("aircraft_classifier")

# 列出所有模型
models = registry.list_models()
# {"aircraft_classifier": "loaded", "airline_classifier": "loaded"}

# 卸载模型
registry.unload("aircraft_classifier")

# 卸载所有模型
registry.unload_all()
```

**特性**

- 线程安全的单例模式
- 懒加载（首次使用时加载）
- 模型缓存（避免重复加载）
- 自动预热（加载后执行一次空推理）

---

## 完整使用示例

### 审核 Pipeline

```python
from pathlib import Path
from infer import (
    get_config,
    get_aircraft_classifier,
    get_airline_classifier,
    get_registration_detector,
    get_registration_ocr,
    get_quality_assessor
)

def review_image(image_path: str) -> dict:
    """完整的图片审核流程"""

    results = {
        "image": image_path,
        "overall_pass": True,
        "fail_reasons": []
    }

    # 1. 质量评估
    assessor = get_quality_assessor()
    quality_result = assessor.assess(image_path)
    results["quality"] = quality_result

    if not quality_result["pass"]:
        results["overall_pass"] = False
        results["fail_reasons"].append("图片质量不达标")

    # 2. 机型分类
    aircraft_clf = get_aircraft_classifier()
    aircraft_result = aircraft_clf.classify(image_path)
    results["aircraft"] = aircraft_result

    if not aircraft_result["pass"]:
        results["overall_pass"] = False
        results["fail_reasons"].append("机型识别置信度不足")

    # 3. 航司识别
    airline_clf = get_airline_classifier()
    airline_result = airline_clf.classify(image_path)
    results["airline"] = airline_result

    # 4. 注册号检测 + OCR
    detector = get_registration_detector()
    ocr = get_registration_ocr()

    det_result = detector.detect_and_crop(image_path)
    if det_result["detected"]:
        ocr_results = ocr.recognize_from_crops(det_result["crops"])
        # 取置信度最高的结果
        valid_results = [r for r in ocr_results if r.get("valid")]
        if valid_results:
            best = max(valid_results, key=lambda x: x["confidence"])
            results["registration"] = {
                "detected": True,
                "text": best["text"],
                "confidence": best["confidence"],
                "valid": True
            }
        else:
            results["registration"] = {
                "detected": True,
                "text": ocr_results[0]["text"] if ocr_results else "",
                "confidence": ocr_results[0]["confidence"] if ocr_results else 0,
                "valid": False
            }
            results["fail_reasons"].append("注册号识别无效")
    else:
        results["registration"] = {"detected": False}
        results["fail_reasons"].append("未检测到注册号")

    if results["fail_reasons"]:
        results["overall_pass"] = False

    return results


# 使用示例
if __name__ == "__main__":
    result = review_image("test_aircraft.jpg")
    print(result)
```

### 批量处理

```python
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from infer import get_aircraft_classifier

def batch_classify(image_dir: str, max_workers: int = 4):
    """批量分类图片"""
    classifier = get_aircraft_classifier()
    image_files = list(Path(image_dir).glob("*.jpg"))

    def process(img_path):
        return {
            "file": str(img_path),
            "result": classifier.classify(img_path)
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process, image_files))

    return results
```

---

## API 参考

### 公共导出

```python
from infer import (
    # 配置
    InferConfig,
    get_config,
    set_config,

    # 分类器
    AircraftClassifier,
    AirlineClassifier,
    get_aircraft_classifier,
    get_airline_classifier,

    # 检测器
    RegistrationDetector,
    get_registration_detector,

    # OCR
    RegistrationOCR,
    get_registration_ocr,

    # 质量评估
    ImageQualityAssessor,
    get_quality_assessor,

    # 模型管理
    ModelLoader,
    ModelRegistry,
    get_registry,
)
```

---

## 模型文件

推理需要以下模型文件（放置于 `MODEL_DIR` 目录）：

| 文件名 | 说明 | 来源 |
|---|---|---|
| `aircraft_classifier.pt` | 机型分类模型 | YOLOv8-cls 微调 |
| `airline_classifier.pt` | 航司识别模型 | YOLOv8-cls 微调 |
| `registration_detector.pt` | 注册号检测模型 | YOLOv8 微调 |

模型训练流程请参考 `training/` 目录和 `conductor.md`。

---

## 日志配置

模块使用 Python 标准 logging：

```python
import logging

# 设置日志级别
logging.getLogger("infer").setLevel(logging.DEBUG)

# 或配置处理器
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logging.getLogger("infer").addHandler(handler)
```

---

## 性能优化

1. **GPU 加速**: 默认使用 CUDA，自动回退到 CPU
2. **FP16 推理**: 默认开启半精度推理
3. **模型缓存**: 通过 ModelRegistry 避免重复加载
4. **懒加载**: 首次使用时才加载模型
5. **模型预热**: 加载后执行一次推理优化

---

## 依赖版本

```
torch>=2.0.0
ultralytics>=8.0.0
paddleocr>=2.7.0
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
```

---

## 版本

当前版本: `0.1.0`
