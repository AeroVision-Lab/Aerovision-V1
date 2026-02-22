# AeroVision Training Module

航空摄影智能审核系统 - 模型训练模块

---

## 📋 目录

- [概述](#概述)
- [快速开始](#快速开始)
- [目录结构](#目录结构)
- [配置系统](#配置系统)
- [数据流程](#数据流程)
- [训练脚本](#训练脚本)
- [验证与测试](#验证与测试)
- [常见问题](#常见问题)

---

## 🎯 概述

本模块用于训练 AeroVision 审核系统所需的深度学习模型。

### 支持的任务

| 任务 | 模型 | 用途 |
|------|------|------|
| 飞机机型分类 | YOLOv8-cls | 识别飞机型号 (A320, B737-800 等) |
| 航空公司识别 | YOLOv8-cls | 识别航司涂装 (国航, 东航等) |
| 注册号检测 | YOLOv8 | 检测注册号区域位置 |
| 注册号识别 | PaddleOCR | OCR 识别注册号文字 |

### 主要特性

- ✅ **模块化配置系统** - 统一的 YAML 配置管理
- ✅ **统一日志系统** - 集中的日志配置和输出
- ✅ **完整数据流程** - 从原始图片到训练数据的完整流程
- ✅ **灵活的训练脚本** - 支持命令行参数和配置文件
- ✅ **自动验证** - 配置和环境自动验证

---

## 🚀 快速开始

### 1. 环境配置

**系统要求**:
- Python 3.11+
- CUDA 11.8+ (GPU 训练)
- 8GB+ GPU 显存

**安装依赖**:
**安装依赖**:
```bash
pip install -r requirements.txt

cd training

**验证环境**:
```bash
# 验证配置系统
python verify_configs.py

# 验证 GPU 环境
python tools/check_gpu.py
```

### 2. 数据准备

```bash
cd scripts/data_prep

# 步骤1: 准备数据集 (验证、清洗)
python prepare_dataset.py

# 步骤2: 划分数据集 (train/val/test)
python split_dataset.py
```

### 3. 训练模型

```bash
cd scripts/train

# 机型分类
python train_classify.py --epochs 100 --batch-size 32

# 航司识别
python train_airline.py --epochs 100 --batch-size 32

# 注册号检测
python train_detection.py --epochs 100 --batch-size 16
```

---

## 📁 目录结构

```
training/
├── configs/                    # 配置系统
│   ├── base.yaml              # 基础配置 (项目、设备、种子)
│   ├── config/                # 模块配置
│   │   ├── paths.yaml         # 路径配置
│   │   ├── training.yaml      # 训练参数
│   │   ├── airline.yaml       # 航司训练配置
│   │   ├── yolo.yaml          # YOLO 检测配置
│   │   ├── crop.yaml          # 裁剪配置
│   │   ├── augmentation.yaml  # 数据增强
│   │   ├── review.yaml        # 审查配置
│   │   ├── vlm.yaml           # VLM 配置
│   │   └── logging.yaml       # 日志配置
│   ├── config_loader.py       # 配置加载器
│   ├── logger.py              # 统一日志系统
│   └── __init__.py
│
├── scripts/                    # 训练脚本
│   ├── data_prep/             # 数据准备
│   │   ├── crop_airplane.py   # 飞机检测与裁剪
│   │   ├── prepare_dataset.py # 数据验证清洗
│   │   ├── split_dataset.py   # 数据集划分
│   │   └── verify_data.py     # 数据验证
│   ├── train/                 # 训练脚本
│   │   ├── train_classify.py  # 机型分类训练
│   │   ├── train_airline.py   # 航司识别训练
│   │   ├── train_detection.py # 注册号检测训练
│   │   └── training_utils.py  # 训练工具
│   ├── eval/                  # 评估工具
│   │   ├── evaluate_classify.py
│   │   ├── review_crops.py    # 裁剪结果审查
│   │   └── ocr_pipeline.py    # OCR 流程
│   └── fgvc/                  # FGVC 数据集工具
│
├── data/                       # 数据目录
│   ├── raw/                   # 原始图片
│   ├── processed/             # 处理后数据
│   │   ├── aircraft_crop/     # 裁剪后飞机图
│   │   └── labeled/           # 标注数据
│   ├── prepared/              # 准备好的数据
│   └── splits/                # 划分后数据
│
├── model/                      # 预训练模型
├── ckpt/                       # 训练检查点
│   ├── classify/              # 分类模型检查点
│   ├── airline/               # 航司模型检查点
│   └── detection/             # 检测模型检查点
├── logs/                       # 训练日志
│   ├── classify/              # 分类训练日志
│   ├── airline/               # 航司训练日志
│   └── detection/             # 检测训练日志
├── output/                     # 训练输出 (YOLO)
│
├── tools/                      # 工具脚本
├── tests/                      # 测试
├── verify_configs.py          # 配置验证脚本
└── README.md
```

---

## ⚙️ 配置系统

### 模块化配置

配置采用模块化设计，所有相对路径相对于 `training/configs/` 目录。

**加载配置**:
```python
from configs import load_config

# 加载所有配置模块
config = load_config()

# 只加载特定模块
config = load_config(modules=['training', 'paths'], load_all_modules=False)

# 访问配置 (支持点号分隔的嵌套键)
epochs = config.get('training.epochs')
lr = config.get('training.optimizer.lr0')

# 获取路径 (自动转为绝对路径)
data_path = config.get_path('data.prepared.root')

# 运行时覆盖
config = load_config(device={'default': 'cpu'})
```

### 配置文件说明

**base.yaml** - 全局基础配置:
```yaml
project:
  name: "AeroVision-V1"
  version: "0.1.0"

device:
  default: "cuda"
  gpu_ids: [0]

seed:
  random: 42

# 输出路径
path_output:
  classify: "../output/classify"
  airline: "../output/airline"
  detection: "../output/detection"

# 检查点路径
checkpoints:
  classify: "../ckpt/classify"
  airline: "../ckpt/airline"
  detection: "../ckpt/detection"

# 日志路径
logs:
  classify: "../logs/classify"
  airline: "../logs/airline"
  detection: "../logs/detection"
```

**training.yaml** - 训练参数:
```yaml
training:
  epochs: 200
  batch_size: 16
  image_size: 640
  workers: 8
  amp: true

  optimizer:
    type: "AdamW"
    lr0: 0.001
    momentum: 0.937
    weight_decay: 0.0005

  scheduler:
    cosine: true
    lrf: 0.01

  early_stopping:
    patience: 50

  warmup:
    epochs: 3.0

  output:
    project: "../output/classify"
    name: "aircraft_classifier"
```

**paths.yaml** - 路径配置:
```yaml
data:
  raw: "../data/raw"
  prepared:
    root: "../data/prepared"
  processed:
    aircraft_crop:
      unsorted: "../data/processed/aircraft_crop/unsorted"
    labeled:
      images: "../data/processed/labeled/images"

labels:
  main: "../data/processed/labeled/labels.csv"

path_models:
  pretrained:
    yolov8m: "../model/yolov8m.pt"

logs:
  root: "../logs"
```

### 统一日志系统

所有训练脚本使用统一的日志配置:

```python
from configs import setup_logger

logger = setup_logger(
    name="AircraftClassifier",
    log_dir=log_dir,
    config=config  # 从 logging.yaml 读取配置
)

logger.info("训练开始")
logger.debug("调试信息")
```

**logging.yaml** 配置:
```yaml
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  date_format: "%Y-%m-%d %H:%M:%S"
  console_level: "INFO"
  save_to_file: true
```

---

## 📊 数据流程

### 完整训练流程

```
原始图片 (data/raw/)
        ↓
   [crop_airplane.py]  ← YOLO 检测裁剪飞机
        ↓
裁剪后图片 (data/processed/aircraft_crop/)
        ↓
   [人工标注]  ← 标注机型、航司、质量
        ↓
标注文件 (labels.csv)
        ↓
   [prepare_dataset.py]  ← 验证、清洗、去重
        ↓
清洗后数据 (data/prepared/<timestamp>/)
        ↓
   [split_dataset.py]  ← 划分 train/val/test
        ↓
划分后数据 (data/splits/<timestamp>/)
    ├── aerovision/
    │   ├── aircraft/   ← 机型分类
    │   │   ├── train/<class>/
    │   │   ├── val/<class>/
    │   │   └── test/<class>/
    │   └── airline/    ← 航司分类
    └── detection/      ← 检测 (YOLO 格式)
        ├── images/
        ├── labels/
        └── dataset.yaml
```

### 标注文件格式

**labels.csv**:
```csv
filename,typename,airlinename,registration,clarity,block
IMG_0001.jpg,A320,China Eastern,B-1234,0.95,0
IMG_0002.jpg,B737-800,Air China,B-5678,0.85,0
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| filename | string | ✓ | 图片文件名 |
| typename | string | ✓ | 机型名称 |
| airlinename | string | | 航司名称 |
| registration | string | | 注册号 |
| clarity | float | ✓ | 清晰度 (0-1) |
| block | int | ✓ | 遮挡标记 (0/1) |

---

## 🎓 训练脚本

### train_classify.py - 机型分类

基于 YOLOv8-cls 的机型分类模型。

```bash
# 使用配置文件默认参数
python train_classify.py

# 自定义参数
python train_classify.py \
    --epochs 100 \
    --batch-size 32 \
    --imgsz 224 \
    --lr0 0.001 \
    --optimizer AdamW \
    --device 0

# 使用自定义配置文件
python train_classify.py --config my_config.yaml

# 从检查点恢复
python train_classify.py --resume ckpt/classify/last.pt
```

**关键参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | yolov8n-cls.pt | 预训练模型 |
| `--epochs` | 200 | 训练轮数 |
| `--batch-size` | 16 | 批次大小 |
| `--imgsz` | 640 | 输入图片尺寸 |
| `--lr0` | 0.001 | 初始学习率 |
| `--optimizer` | AdamW | 优化器 |
| `--patience` | 50 | 早停耐心值 |
| `--device` | 0 | GPU ID |

**高级功能**:
```bash
# 使用 Focal Loss
python train_classify.py --focal-loss --focal-alpha 0.25 --focal-gamma 2.0

# 使用 Mixup 增强
python train_classify.py --mixup --mixup-alpha 0.4

# 使用 ElasticFace Loss
python train_classify.py --elastic-face --elastic-type arc

# 梯度累积
python train_classify.py --accumulate 4
```

### train_airline.py - 航司识别

与机型分类类似，针对航空公司涂装识别。

```bash
python train_airline.py \
    --epochs 100 \
    --batch-size 32 \
    --dropout 0.1
```

### train_detection.py - 注册号检测

基于 YOLOv8 的目标检测模型。

```bash
python train_detection.py \
    --model-size m \
    --epochs 100 \
    --batch-size 16 \
    --imgsz 640
```

**模型大小选择**:

| 模型 | 参数量 | 推荐场景 |
|------|--------|----------|
| yolov8n | 3.2M | 边缘设备 |
| yolov8s | 11.2M | 轻量部署 |
| yolov8m | 25.9M | 平衡选择 ⭐ |
| yolov8l | 43.7M | 高精度 |

### 训练输出

训练完成后，模型保存在:

```
output/
├── classify/
│   └── aircraft_classifier_<timestamp>/
│       ├── weights/
│       │   ├── best.pt   # 最佳模型
│       │   └── last.pt   # 最后模型
│       ├── results.png   # 训练曲线
│       └── confusion_matrix.png
├── airline/
│   └── airline_classifier_<timestamp>/
└── detection/
    └── registration_detector_<timestamp>/
```

检查点保存在:
```
ckpt/
├── classify/<timestamp>/
│   ├── best.pt
│   └── last.pt
├── airline/<timestamp>/
└── detection/<timestamp>/
```

日志保存在:
```
logs/
├── classify/<timestamp>/
│   └── train_<timestamp>.log
├── airline/<timestamp>/
└── detection/<timestamp>/
```

---

## ✅ 验证与测试

### 配置验证

运行配置验证脚本:
```bash
python verify_configs.py
```

验证内容:
- ✓ 配置文件加载
- ✓ 配置键存在性
- ✓ 路径配置正确性
- ✓ Logger 功能
- ✓ 训练脚本配置使用

### 环境验证

```bash
# GPU 环境
python tools/check_gpu.py

# 完整环境
python tools/verify_env.py
```

### 数据验证

```bash
# 验证标注数据
python scripts/data_prep/verify_data.py

# 审查裁剪结果
python scripts/eval/review_crops.py --n-samples 20
```

---

## 🔧 常见问题

### Q: CUDA out of memory

**解决方案**: 减小批次大小或图片尺寸
```bash
python train_classify.py --batch-size 16 --imgsz 160
```

或启用梯度累积:
```bash
python train_classify.py --batch-size 8 --accumulate 4  # 等效 batch=32
```

### Q: 找不到数据集

**解决方案**: 确保按顺序执行数据准备流程
```bash
cd scripts/data_prep
python prepare_dataset.py  # 1. 准备
python split_dataset.py    # 2. 划分

cd ../train
python train_classify.py   # 3. 训练
```

### Q: 配置加载失败

**解决方案**: 运行配置验证
```bash
python verify_configs.py
```

检查配置文件路径是否正确 (相对于 `training/configs/`)。

### Q: 模型下载失败

**解决方案**: 手动下载模型到 `training/model/` 目录
```bash
# YOLOv8 分类模型
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-cls.pt \
     -O model/yolov8m-cls.pt

# YOLOv8 检测模型
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt \
     -O model/yolov8m.pt
```

### Q: PaddleOCR 初始化慢

**原因**: 首次运行会下载模型

**解决方案**: 预先下载模型
```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(lang='en', rec_model_dir='PP-OCRv4_server_rec')
```

### Q: 训练中断如何恢复

**解决方案**: 使用 `--resume` 参数
```bash
python train_classify.py --resume ckpt/classify/<timestamp>/last.pt
```

### Q: 如何查看训练日志

**方案1**: 查看日志文件
```bash
tail -f logs/classify/<timestamp>/train_<timestamp>.log
```

**方案2**: 使用 TensorBoard
```bash
tensorboard --logdir logs/tensorboard
```

---

## 📚 参考资源

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/)
- [配置系统文档](configs/EVALUATION_REPORT.md)
- [配置使用检查](configs/CONFIG_USAGE_CHECK.md)

---

## 📝 更新日志

### 2026-02-22
- ✅ 重构配置系统,实现模块化配置
- ✅ 统一日志系统,集中配置管理
- ✅ 修复所有配置键冲突和路径问题
- ✅ 添加配置验证脚本
- ✅ 优化训练脚本,统一配置使用
- ✅ 更新文档,反映最新架构

---

**License**: MIT
**Author**: AeroVision Team
**Contact**: [项目仓库](https://github.com/your-repo/aerovision)
