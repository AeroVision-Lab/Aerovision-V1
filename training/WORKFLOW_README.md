# Aerovision-V1 Workflow 使用说明

## 概述

本文档说明如何使用Aerovision-V1项目的workflow来训练飞机检测模型（基于YOLOv8x）。

## 数据格式

项目使用以下数据格式：

1. **labels.csv**: 包含图片的标签信息
   - `filename`: 图片文件名
   - `typeid`: 机型ID
   - `typename`: 机型名称（如A320、B737-800等）
   - `airlineid`: 航空公司ID
   - `airlinename`: 航空公司名称
   - `clarity`: 清晰度评分（0-1）
   - `block`: 遮挡程度评分（0-1）
   - `registration`: 注册号

2. **data/labeled/**: 原始图片目录
   - 包含所有标注的图片文件

3. **data/*.txt**: 注册号区域标注（YOLO格式）
   - 格式：`class_id center_x center_y width height`
   - 坐标为归一化坐标（0-1）

## 配置文件

所有训练参数统一配置在 [`training/configs/config/training_params.yaml`](training/configs/config/training_params.yaml) 文件中。

主要配置项：

- **数据路径配置**: 原始数据、处理后数据、标签文件等路径
- **数据集划分配置**: train/val/test划分比例
- **检测模型配置**: YOLOv8x模型、训练参数、优化器、数据增强等
- **OCR配置**: PaddleOCR相关配置
- **质量评估配置**: 清晰度、遮挡阈值等

## Workflow步骤

### 步骤1: 准备数据集

使用 [`training/scripts/prepare_detection_dataset.py`](training/scripts/prepare_detection_dataset.py) 脚本准备数据集：

```bash
python training/scripts/prepare_detection_dataset.py \
    --labels-csv data/labels.csv \
    --images-dir data/labeled \
    --output-dir data \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --random-seed 42
```

该脚本会：
1. 读取labels.csv获取机型信息
2. 按比例划分train/val/test数据集
3. 创建YOLOv8检测格式的目录结构
4. 复制图片和标签文件
5. 生成YOLOv8配置文件（dataset.yaml）

输出目录：
- `data/processed/detection/train/`: 训练集
- `data/processed/detection/val/`: 验证集
- `data/processed/detection/test/`: 测试集
- `data/processed/detection/dataset.yaml`: YOLOv8配置文件

### 步骤2: 训练检测模型

使用 [`training/scripts/train_detection_model.py`](training/scripts/train_detection.py) 脚本训练模型：

```bash
python training/scripts/train_detection.py \
    --data data/processed/detection/dataset.yaml \
    --model x \
    --epochs 50 \
    --batch 8 \
    --imgsz 640 \
    --device cpu
```

主要参数：
- `--data`: YOLO数据集配置文件路径
- `--model`: 模型大小（n, s, m, l, x），默认使用x（YOLOv8x）
- `--epochs`: 训练轮数
- `--batch`: 批次大小
- `--imgsz`: 输入图像大小
- `--device`: 设备（0=GPU, cpu=CPU）

输出目录：
- `training/checkpoints/registration_detection/`: 训练结果
  - `weights/best.pt`: 最佳模型权重
  - `weights/last.pt`: 最后一个epoch的模型权重
  - `results.png`: 训练曲线图
  - `confusion_matrix.png`: 混淆矩阵

### 步骤3: 推理测试

训练完成后，可以使用训练好的模型进行推理：

```bash
python app/test_inference.py \
    --model training/checkpoints/registration_detection/weights/best.pt \
    --image data/labeled/example.jpg \
    --config training/configs/config/inference.yaml
```

## 使用主workflow脚本

也可以使用 [`run_workflow.py`](run_workflow.py) 脚本一次性运行整个workflow：

```bash
python run_workflow.py --config training/configs/config/training_params.yaml
```

可选参数：
- `--skip-prepare`: 跳过数据集准备步骤
- `--skip-train`: 跳过训练步骤
- `--skip-evaluate`: 跳过评估步骤

## 调参

所有训练参数都在 [`training/configs/config/training_params.yaml`](training/configs/config/training_params.yaml) 文件中，修改该文件即可调整训练参数。

主要调参项：

### 检测模型 (detection)

- `model`: 模型类型（yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt），推荐使用yolov8x
- `epochs`: 训练轮数
- `batch_size`: 批次大小
- `lr0`: 初始学习率
- `optimizer`: 优化器（AdamW, SGD, Adam等）
- `imgsz`: 输入图像尺寸
- `device`: 设备（0=GPU, cpu=CPU）

### 数据增强参数

- `hsv_h`: HSV色调增强
- `hsv_s`: HSV饱和度增强
- `hsv_v`: HSV明度增强
- `degrees`: 旋转角度
- `translate`: 平移
- `scale`: 缩放
- `fliplr`: 左右翻转概率

## 注意事项

1. **数据集大小**: 确保有足够的训练数据，建议每个类别至少50个样本。

2. **类别平衡**: 确保每个类别有足够的样本。

3. **GPU使用**: 如果有GPU，将`device`设置为GPU编号（如0, 1）。

4. **模型选择**: 
   - yolov8n.pt: 最小模型，速度快，精度较低
   - yolov8s.pt: 小模型
   - yolov8m.pt: 中等模型
   - yolov8l.pt: 大模型
   - yolov8x.pt: 最大模型，精度最高，速度较慢（推荐）

5. **batch_size**: 根据GPU内存调整，显存不足时减小batch_size。

## 文件结构

```
Aerovision-V1/
├── configs/
│   └── training_params.yaml      # 统一训练参数配置
├── data/
│   ├── labels.csv                # 标签文件
│   ├── labeled/                  # 原始图片
│   └── processed/
│       └── detection/            # 处理后的数据集
│           ├── train/           # 训练集
│           ├── val/             # 验证集
│           ├── test/            # 测试集
│           └── dataset.yaml      # YOLOv8配置文件
├── training/
│   └── scripts/
│       ├── prepare_detection_dataset.py  # 数据集准备脚本
│       └── train_detection_model.py     # 训练脚本
├── app/
│   └── services/
│       └── inference_service.py   # 推理服务
├── run_workflow.py              # 主workflow脚本
└── yolov8x.pt                 # 预训练模型
```

## 故障排除

### 问题1: 验证集没有图片

**原因**: 数据集划分时，某些类别的样本太少，导致验证集为空。

**解决**: 修改 `prepare_detection_dataset.py` 中的划分逻辑，确保每个数据集都有图片。

### 问题2: 训练时显存不足

**原因**: batch_size太大或模型太大。

**解决**: 减小batch_size或使用更小的模型（如yolov8n.pt）。

### 问题3: CSV文件读取失败

**原因**: CSV文件有BOM（字节顺序标记）。

**解决**: 使用 `utf-8-sig` 编码读取，或手动去除BOM。

## 下一步

1. 准备更大的数据集
2. 调整训练参数以获得更好的性能
3. 优化OCR识别功能
4. 添加航司分类功能
