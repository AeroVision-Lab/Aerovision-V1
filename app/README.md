# AeroVision App - FastAPI 审核服务

AeroVision AI 审核微服务的核心应用模块，基于 FastAPI 构建，提供航空照片自动化审核 API。

## 目录结构

```
app/
├── __init__.py                 # 包元信息 (version 0.1.0)
├── main.py                     # FastAPI 应用入口
├── api/                        # API 层
│   ├── __init__.py             # 路由聚合
│   ├── deps.py                 # 依赖注入 & 请求计数器
│   └── routes/                 # 路由处理器
│       ├── health.py           # 健康检查 & 统计接口
│       └── review.py           # 图片审核接口
├── core/                       # 核心配置
│   ├── config.py               # 配置管理 (Pydantic Settings)
│   └── logging.py              # 日志配置
├── schemas/                    # Pydantic 数据模型
│   ├── request.py              # 请求模型
│   └── response.py             # 响应模型
└── services/                   # 业务逻辑
    └── review_service.py       # 审核主服务
```

## 快速开始

### 启动服务

```bash
# 开发模式
python -m app.main

# 或使用 uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 访问文档

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 接口

### 审核接口

#### 单图审核

```http
POST /api/v1/review
Content-Type: application/json

{
  "image_url": "https://example.com/image.jpg",
  "review_types": ["quality", "aircraft", "registration"],
  "callback_url": "https://your-backend.com/callback",
  "metadata": {"user_id": "123"}
}
```

或使用 Base64：

```json
{
  "image_base64": "/9j/4AAQSkZJRg...",
  "review_types": ["quality", "aircraft"]
}
```

**响应示例**:

```json
{
  "success": true,
  "review_id": "550e8400-e29b-41d4-a716-446655440000",
  "results": {
    "overall_pass": true,
    "quality": {
      "passed": true,
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
      "passed": true,
      "is_aircraft": true,
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
      "passed": true,
      "detected": true,
      "value": "B-1234",
      "confidence": 0.89,
      "clarity_score": 0.85,
      "bbox": [0.75, 0.60, 0.95, 0.68]
    }
  },
  "fail_reasons": [],
  "processing_time_ms": 1523.5,
  "metadata": {"user_id": "123"}
}
```

#### 批量审核

```http
POST /api/v1/review/batch
Content-Type: application/json

{
  "images": [
    {"image_url": "https://example.com/image1.jpg"},
    {"image_url": "https://example.com/image2.jpg"}
  ],
  "review_types": ["quality", "aircraft"],
  "callback_url": "https://your-backend.com/callback"
}
```

**限制**: 单次最多 10 张图片

### 健康检查接口

#### 服务健康状态

```http
GET /api/v1/health
```

**响应**:

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "gpu_available": true,
  "gpu_memory": {
    "total": 8192,
    "used": 2048,
    "free": 6144
  },
  "models_loaded": {
    "aircraft_classifier": true,
    "airline_classifier": true,
    "registration_detector": true
  }
}
```

#### 请求统计

```http
GET /api/v1/stats
```

**响应**:

```json
{
  "total_requests": 1000,
  "successful_requests": 980,
  "failed_requests": 20,
  "uptime_seconds": 3600.5,
  "requests_per_minute": 16.7
}
```

## 审核类型

| 类型 | 枚举值 | 说明 |
|------|--------|------|
| 图片质量 | `quality` | 清晰度、曝光、构图、噪点、色彩评估 |
| 飞机识别 | `aircraft` | 机型分类、航司涂装识别，返回 Top-3 预测结果 |
| 注册号 | `registration` | OCR 识别注册号及清晰度评分 |
| 遮挡检测 | `occlusion` | 主体遮挡比例检测 (待实现) |
| 违规检测 | `violation` | 水印/敏感内容检测 (待实现) |

## 配置说明

通过环境变量或 `.env` 文件配置：

### 基础配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `APP_NAME` | AeroVision AI | 应用名称 |
| `VERSION` | 0.1.0 | 版本号 |
| `DEBUG` | false | 调试模式 |

### 服务配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HOST` | 0.0.0.0 | 监听地址 |
| `PORT` | 8000 | 监听端口 |
| `WORKERS` | 1 | 工作进程数 |

### 模型配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_DIR` | models | 模型目录 |
| `DEVICE` | cuda | 推理设备 (cuda/cpu) |

### 审核阈值

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `QUALITY_THRESHOLD` | 0.70 | 质量评分通过阈值 |
| `REGISTRATION_CLARITY_THRESHOLD` | 0.80 | 注册号清晰度阈值 |
| `OCCLUSION_THRESHOLD` | 0.20 | 遮挡比例上限 |
| `CLASSIFIER_CONFIDENCE_THRESHOLD` | 0.50 | 分类器置信度阈值 |

### 请求限制

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MAX_IMAGE_SIZE` | 10485760 | 最大图片大小 (10MB) |
| `REQUEST_TIMEOUT` | 60 | 请求超时时间 (秒) |

### CORS 配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CORS_ORIGINS` | ["*"] | 允许的来源 |
| `CORS_ALLOW_CREDENTIALS` | true | 允许凭证 |

### 外部服务

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BACKEND_CALLBACK_URL` | - | 后端回调地址 |

## 服务架构

### ReviewService

主审核服务类，采用单例模式，通过 `get_review_service()` 获取实例。

**特性**:
- 懒加载推理模块，首次请求时加载模型
- 支持生产模式下的模型预热
- 集成 `infer` 模块进行实际推理

**依赖模块**:
- `AircraftClassifier` - 机型分类器
- `AirlineClassifier` - 航司分类器
- `RegistrationDetector` - 注册号检测器
- `RegistrationOCR` - 注册号 OCR
- `QualityAssessor` - 质量评估器

### 请求计数器

`RequestCounter` 类提供线程安全的请求统计：

```python
from app.api.deps import request_counter

# 记录请求
request_counter.increment()
request_counter.increment_success()
request_counter.increment_failed()

# 获取统计
stats = request_counter.get_stats()
```

## 日志配置

日志使用结构化格式输出，根日志器为 `aerovision`：

```python
from app.core.logging import get_logger

logger = get_logger(__name__)
logger.info("Processing image", extra={"image_id": "123"})
```

默认抑制以下第三方日志：
- `uvicorn.access`
- `ultralytics`
- `paddleocr`

## 错误处理

服务提供统一的错误响应格式：

```json
{
  "success": false,
  "review_id": "550e8400-e29b-41d4-a716-446655440000",
  "error": "Image download failed",
  "fail_reasons": ["无法下载图片"]
}
```

调试模式下 (`DEBUG=true`) 会返回详细的错误堆栈信息。

## 开发指南

### 添加新的审核模块

1. 在 `schemas/response.py` 中定义结果模型
2. 在 `services/review_service.py` 中实现审核逻辑
3. 在 `schemas/request.py` 的 `ReviewType` 枚举中添加类型
4. 更新 `ReviewResults` 模型

### 代码规范

- 遵循 PEP 8
- 使用 type hints 类型注解
- 异步优先 (async/await)
- 使用 Black 格式化代码

### 测试

```bash
# 运行测试
pytest tests/

# 测试覆盖率
pytest --cov=app tests/
```

## 依赖项

核心依赖：
- `fastapi` - Web 框架
- `uvicorn` - ASGI 服务器
- `pydantic` - 数据验证
- `pydantic-settings` - 配置管理
- `httpx` - HTTP 客户端
- `pillow` - 图像处理

推理依赖 (来自 `infer` 模块):
- `torch` - 深度学习框架
- `timm` - 预训练模型
- `ultralytics` - YOLOv8
- `paddleocr` - OCR 引擎

## 待实现功能

- [ ] `occlusion` - 遮挡检测模块
- [ ] `violation` - 违规检测模块
- [ ] `app/utils/` - 通用工具函数
- [ ] 更细粒度的服务模块拆分

## 相关文档

- [项目主文档](../CLAUDE.md) - 项目概述与规范
- [模型训练路线图](../conductor.md) - 训练流程详情
- [推理模块](../infer/README.md) - 推理模块文档
