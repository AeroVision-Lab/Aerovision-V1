# AeroVision 训练流程技术指南

深入技术细节的训练流程文档

---

## 📋 目录

- [架构概览](#架构概览)
- [配置系统架构](#配置系统架构)
- [数据处理流程](#数据处理流程)
- [训练流程详解](#训练流程详解)
- [模型架构](#模型架构)
- [日志系统](#日志系统)
- [性能优化](#性能优化)
- [扩展开发](#扩展开发)

---

## 🏗️ 架构概览

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     AeroVision Training                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   配置系统    │───▶│   日志系统    │───▶│   训练脚本    │  │
│  │ Config Loader│    │    Logger     │    │   Trainers    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                    │          │
│         ▼                    ▼                    ▼          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              数据处理流程 (Data Pipeline)              │  │
│  │  crop → prepare → split → augment → dataloader       │  │
│  └──────────────────────────────────────────────────────┘  │
│         │                                         │          │
│         ▼                                         ▼          │
│  ┌──────────────┐                        ┌──────────────┐  │
│  │   YOLO 模型   │                        │  PyTorch 模型 │  │
│  │  Detection    │                        │ Classification│  │
│  │ Classification│                        │   (timm)      │  │
│  └──────────────┘                        └──────────────┘  │
│         │                                         │          │
│         └─────────────────┬───────────────────────┘          │
│                           ▼                                  │
│                  ┌──────────────┐                           │
│                  │  模型输出     │                           │
│                  │ Checkpoints  │                           │
│                  │    Logs      │                           │
│                  └──────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| **深度学习框架** | PyTorch 2.0+ | 模型训练基础 |
| **模型库** | Ultralytics (YOLOv8) | 检测和分类 |
| **模型库** | timm | 预训练模型 |
| **数据增强** | albumentations | 图像增强 |
| **OCR** | PaddleOCR | 文字识别 |
| **配置管理** | PyYAML | 配置文件解析 |
| **日志** | Python logging | 日志记录 |
| **实验跟踪** | TensorBoard | 训练监控 |

---

## ⚙️ 配置系统架构

### 配置加载流程

```python
# 配置加载流程
load_config()
    │
    ├─ 1. 确定配置文件路径
    │   └─ training/configs/base.yaml
    │
    ├─ 2. 加载基础配置
    │   └─ yaml.safe_load(base.yaml)
    │
    ├─ 3. 动态扫描模块配置
    │   └─ glob("configs/config/*.yaml")
    │       ├─ paths.yaml
    │       ├─ training.yaml
    │       ├─ airline.yaml
    │       ├─ yolo.yaml
    │       ├─ crop.yaml
    │       ├─ augmentation.yaml
    │       ├─ review.yaml
    │       ├─ vlm.yaml
    │       └─ logging.yaml
    │
    ├─ 4. 深度合并配置
    │   └─ _deep_merge(base_dict, module_dict)
    │
    ├─ 5. 创建 Config 对象
    │   └─ Config(config_dict, config_base_path)
    │
    └─ 6. 应用运行时覆盖
        └─ config.update(kwargs)
```

### Config 类实现

```python
class Config:
    """配置类核心实现"""

    def __init__(self, config_dict: Dict[str, Any], config_base_path: Path):
        self._config = config_dict
        self._config_base_path = config_base_path  # training/configs/

    def get(self, key: str, default: Any = None) -> Optional[Any]:
        """
        支持点号分隔的嵌套键访问

        Example:
            config.get('training.optimizer.lr0')
            # 等价于 config['training']['optimizer']['lr0']
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_path(self, key: str, create: bool = False) -> Path:
        """
        获取路径配置并转换为绝对路径

        相对路径解析规则:
        - ../data → training/data
        - ../model → training/model

        Example:
            config.get_path('data.prepared.root')
            # '../data/prepared' → '/absolute/path/to/training/data/prepared'
        """
        path_str = self.get(key)
        if path_str is None:
            raise ValueError(f"配置键 '{key}' 未找到")

        path = Path(path_str)

        # 相对路径相对于 training/configs/ 解析
        if not path.is_absolute():
            path = (self._config_base_path / path).resolve()

        if create and not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        return path
```

### 配置命名空间设计

为避免配置键冲突,采用模块前缀命名:

```yaml
# ✅ 正确: 使用模块前缀
yolo_detection:
  conf_threshold: 0.5

crop_output:
  quality: 95

review_output:
  save_path: "../logs/review.png"

# ❌ 错误: 无前缀会冲突
output:  # 多个模块都定义 output 会冲突
  quality: 95
```

---

## 📊 数据处理流程

### 数据流水线架构

```
原始图片 (raw/)
    │
    ├─ [crop_airplane.py]
    │   ├─ YOLO 检测飞机
    │   ├─ 裁剪边界框 (padding=0.1)
    │   ├─ 尺寸过滤 (min_size=224)
    │   └─ 保存裁剪图
    │
    ▼
裁剪图片 (processed/aircraft_crop/)
    │
    ├─ [人工标注]
    │   └─ labels.csv
    │
    ▼
标注数据 (processed/labeled/)
    │
    ├─ [prepare_dataset.py]
    │   ├─ 验证文件存在性
    │   ├─ 去重 (基于文件名)
    │   ├─ 清洗无效数据
    │   ├─ 生成类别映射
    │   │   ├─ type_classes.json
    │   │   └─ airline_classes.json
    │   └─ 保存清洗后数据
    │
    ▼
准备数据 (prepared/<timestamp>/)
    │
    ├─ [split_dataset.py]
    │   ├─ 过滤低频类别 (min_samples)
    │   ├─ 分层划分 (stratified split)
    │   │   ├─ train: 70%
    │   │   ├─ val: 15%
    │   │   └─ test: 15%
    │   ├─ 创建分类数据集结构
    │   │   └─ aerovision/
    │   │       ├─ aircraft/
    │   │       │   ├─ train/<class>/
    │   │       │   ├─ val/<class>/
    │   │       │   └─ test/<class>/
    │   │       └─ airline/
    │   └─ 创建检测数据集 (YOLO 格式)
    │       └─ detection/
    │           ├─ images/
    │           ├─ labels/
    │           └─ dataset.yaml
    │
    ▼
划分数据 (splits/<timestamp>/)
    │
    └─ [训练脚本读取]
```

### 数据增强策略

**分类任务增强** (augmentation.yaml):
```yaml
geometric:
  horizontal_flip: 0.5
  rotation: 15°
  scale: [0.8, 1.2]
  shift: 0.1

color:
  brightness: 0.2
  contrast: 0.2
  saturation: 0.3
  hue: 0.1

quality:
  blur: 0.3
  gaussian_noise: 0.3
  jpeg_compression: 0.3

advanced:
  mixup: 0.2
  cutmix: 1.0
  random_erasing: 0.25
```

**检测任务增强** (YOLO 内置):
```yaml
hsv_h: 0.015  # 色调
hsv_s: 0.7    # 饱和度
hsv_v: 0.4    # 明度
degrees: 0.0  # 旋转 (文字敏感,禁用)
translate: 0.1
scale: 0.5
flipud: 0.0   # 上下翻转 (文字敏感,禁用)
fliplr: 0.5   # 左右翻转
mosaic: 1.0
```

---

## 🎓 训练流程详解

### 训练脚本架构

```python
# train_classify.py 核心流程

def main():
    # 1. 环境设置
    setup_environment()

    # 2. 解析参数
    args = parse_arguments()

    # 3. 加载配置
    config_obj = load_config(modules=['training', 'paths'])

    # 4. 构建扁平配置字典
    config = build_flat_config(config_obj, args)

    # 5. 设置日志
    logger = setup_logger(
        name="AircraftClassifier",
        log_dir=config['log_dir'],
        config=config_obj
    )

    # 6. 创建训练器
    trainer = AircraftClassifierTrainer(config, args, logger)

    # 7. 执行训练
    trainer.train()


class AircraftClassifierTrainer:
    """训练器类"""

    def __init__(self, config, args, logger):
        self.config = config
        self.args = args
        self.logger = logger

        # 初始化高级功能
        self._init_advanced_features()

        # 初始化模型
        self._init_model()

        # 设置 TensorBoard
        self._setup_tensorboard()

    def _init_advanced_features(self):
        """初始化高级训练功能"""
        # 梯度累积
        self.accumulate = self.config.get("accumulate", 1)

        # Focal Loss
        self.use_focal_loss = self.config.get("focal_loss", False)
        if self.use_focal_loss:
            self.focal_loss = FocalLoss(
                alpha=self.config.get("focal_alpha", 0.25),
                gamma=self.config.get("focal_gamma", 2.0)
            )

        # Mixup
        self.use_mixup = self.config.get("mixup", False)
        if self.use_mixup:
            self.mixup = Mixup(alpha=self.config.get("mixup_alpha", 0.4))

        # ElasticFace Loss
        self.use_elastic_face = self.config.get("elastic_face", False)
        if self.use_elastic_face:
            self.elastic_face = ElasticFaceArcFace(
                s=self.config.get("elastic_s", 30.0),
                m=self.config.get("elastic_m", 0.30),
                std=self.config.get("elastic_std", 0.01)
            )

    def train(self):
        """执行训练"""
        # 构建 YOLO 训练参数
        train_args = {
            'data': self.config['data'],
            'epochs': self.config['epochs'],
            'batch': self.config['batch_size'],
            'imgsz': self.config['imgsz'],
            'lr0': self.config['lr0'],
            'optimizer': self.config['optimizer'],
            'device': self.config['device'],
            'workers': self.config['workers'],
            'amp': self.config['amp'],
            'seed': self.config['seed'],
            'project': self.config['project'],
            'name': self.config['name'],
            'patience': self.config['patience'],
            'plots': self.config['plots'],
        }

        # 执行训练
        results = self.model.train(**train_args)

        # 保存检查点
        self._save_final_checkpoint()
```

### 配置展平逻辑

训练脚本需要将嵌套配置展平为 YOLO 期望的格式:

```python
def build_flat_config(config_obj, args):
    """
    将嵌套配置展平

    优先级: 命令行参数 > 配置文件 > 默认值
    """
    config = {
        # 从嵌套配置提取
        "epochs": config_obj.get("training.epochs") or args.epochs or 100,
        "batch_size": config_obj.get("training.batch_size") or args.batch_size or 32,
        "lr0": config_obj.get("training.optimizer.lr0") or args.lr0 or 0.001,
        "optimizer": config_obj.get("training.optimizer.type") or args.optimizer or "AdamW",

        # 从 base.yaml 提取
        "device": config_obj.get("device.default") or args.device or "0",
        "seed": config_obj.get("seed.random") or args.seed or 42,

        # 路径配置
        "project": config_obj.get("training.output.project") or "../output/classify",
        "checkpoint_dir": config_obj.get("checkpoints.classify") or "../ckpt/classify",
        "log_dir": config_obj.get("logs.classify") or "../logs/classify",
    }

    return config
```

### 训练循环钩子

```python
# YOLO 训练循环中的自定义逻辑

class CustomTrainer(YOLO):
    def on_train_batch_end(self, batch_idx, loss):
        """批次结束回调"""
        # 梯度累积
        if self.accumulate > 1:
            if (batch_idx + 1) % self.accumulate == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        # TensorBoard 记录
        if self.tb_writer and batch_idx % 10 == 0:
            self.tb_writer.add_scalar('train/loss', loss, batch_idx)

    def on_val_end(self, metrics):
        """验证结束回调"""
        # 保存最佳模型
        if metrics['accuracy'] > self.best_accuracy:
            self.best_accuracy = metrics['accuracy']
            self.save_checkpoint('best.pt')

        # 记录到 TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar('val/accuracy', metrics['accuracy'])
```

---

## 🧠 模型架构

### YOLOv8 Classification

```
Input (640x640x3)
    │
    ├─ Backbone (CSPDarknet)
    │   ├─ Conv + BN + SiLU
    │   ├─ C2f Block × N
    │   └─ SPPF
    │
    ├─ Neck (省略,分类任务不需要)
    │
    └─ Head (Classification)
        ├─ Global Average Pooling
        ├─ Dropout (可选)
        └─ Linear(num_classes)
            └─ Softmax
                └─ Output (num_classes)
```

**模型变体**:

| 模型 | Depth | Width | 参数量 | FLOPs |
|------|-------|-------|--------|-------|
| yolov8n-cls | 0.33 | 0.25 | 2.7M | 4.3G |
| yolov8s-cls | 0.33 | 0.50 | 6.4M | 11.4G |
| yolov8m-cls | 0.67 | 0.75 | 12.9M | 35.7G |
| yolov8l-cls | 1.00 | 1.00 | 37.5M | 99.2G |
| yolov8x-cls | 1.00 | 1.25 | 57.4M | 154.8G |

### 高级损失函数

**Focal Loss** (解决类别不平衡):
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

**ElasticFace Loss** (基于角度的度量学习):
```python
class ElasticFaceArcFace(nn.Module):
    def __init__(self, s=30.0, m=0.30, std=0.01):
        super().__init__()
        self.s = s      # 缩放因子
        self.m = m      # 基础 margin
        self.std = std  # margin 采样标准差

    def forward(self, embeddings, labels):
        # L2 归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 计算余弦相似度
        cosine = F.linear(embeddings, self.weight)

        # 添加动态 margin
        margin = torch.normal(self.m, self.std, size=(1,))
        theta = torch.acos(cosine.clamp(-1, 1))
        theta_m = theta + margin

        # 缩放
        logits = self.s * torch.cos(theta_m)

        return logits
```

**Mixup** (数据增强):
```python
class Mixup:
    def __init__(self, alpha=0.4):
        self.alpha = alpha

    def __call__(self, images, labels):
        # 采样 lambda
        lam = np.random.beta(self.alpha, self.alpha)

        # 随机打乱
        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        # 混合图像
        mixed_images = lam * images + (1 - lam) * images[index]

        # 混合标签
        labels_a, labels_b = labels, labels[index]

        return mixed_images, labels_a, labels_b, lam
```

---

## 📝 日志系统

### 统一日志架构

```python
# configs/logger.py

def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    config: Optional[Config] = None,
    console_level: Optional[str] = None,
    file_level: Optional[str] = None,
) -> logging.Logger:
    """
    统一日志设置

    特性:
    - 从 logging.yaml 读取配置
    - 双输出: 控制台 + 文件
    - 不同级别: console (INFO), file (DEBUG)
    - 自动创建日志目录
    - 时间戳文件名
    """
    # 从配置读取
    if config:
        log_level = config.get("logging.level", "INFO")
        log_format = config.get("logging.format")
        console_level = console_level or config.get("logging.console_level", "INFO")
        file_level = file_level or config.get("logging.level", "DEBUG")

    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # 控制台 handler (INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper()))
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 文件 handler (DEBUG)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, file_level.upper()))
        file_formatter = logging.Formatter(
            log_format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
```

### 日志级别使用规范

```python
# DEBUG: 详细的调试信息
logger.debug(f"Batch {batch_idx}: loss={loss:.4f}, lr={lr:.6f}")

# INFO: 关键进度信息
logger.info(f"Epoch {epoch}/{total_epochs} completed")
logger.info(f"Validation accuracy: {acc:.2%}")

# WARNING: 警告信息
logger.warning(f"Low sample count for class {class_name}: {count}")

# ERROR: 错误信息
logger.error(f"Failed to load checkpoint: {e}", exc_info=True)

# CRITICAL: 严重错误
logger.critical(f"CUDA out of memory, training aborted")
```

---

## ⚡ 性能优化

### 训练加速技术

**1. 混合精度训练 (AMP)**:
```yaml
training:
  amp: true  # 启用自动混合精度
```

效果:
- 训练速度提升 2-3x
- 显存占用减少 ~50%
- 精度损失 < 0.1%

**2. 梯度累积**:
```bash
# 等效 batch_size=128
python train_classify.py --batch-size 32 --accumulate 4
```

效果:
- 突破显存限制
- 等效更大 batch size
- 训练更稳定

**3. 多 GPU 训练**:
```yaml
device:
  gpu_ids: [0, 1, 2, 3]
  distributed: true
```

**4. 数据加载优化**:
```yaml
training:
  workers: 8  # 数据加载线程数
  pin_memory: true  # 固定内存
  persistent_workers: true  # 持久化 workers
```

### 显存优化

| 技术 | 显存节省 | 速度影响 |
|------|---------|---------|
| AMP | ~50% | +100% |
| 梯度累积 | 按比例 | -10% |
| 梯度检查点 | ~30% | -20% |
| 小 batch size | 按比例 | -20% |

---

## 🔧 扩展开发

### 添加新的训练脚本

```python
# scripts/train/train_custom.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config, setup_logger

def main():
    # 1. 加载配置
    config_obj = load_config(modules=['training', 'paths'])

    # 2. 设置日志
    logger = setup_logger(
        name="CustomTrainer",
        log_dir=config_obj.get_path('logs.root') / 'custom',
        config=config_obj
    )

    # 3. 构建配置
    config = {
        'epochs': config_obj.get('training.epochs'),
        'batch_size': config_obj.get('training.batch_size'),
        # ... 更多配置
    }

    # 4. 训练逻辑
    logger.info("开始训练")
    # your training code here

if __name__ == '__main__':
    main()
```

### 添加新的配置模块

```yaml
# configs/config/custom.yaml

custom_training:
  # 自定义训练参数
  learning_rate: 0.001
  scheduler: "cosine"

custom_output:
  # 自定义输出路径
  save_path: "../output/custom"
```

然后在代码中加载:
```python
config = load_config(modules=['custom', 'paths'])
lr = config.get('custom_training.learning_rate')
```

### 添加自定义损失函数

```python
# scripts/train/training_utils.py

class CustomLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred, target):
        # 自定义损失计算
        loss = ...
        return loss * self.weight
```

在训练脚本中使用:
```python
if config.get('training.advanced.custom_loss.enabled'):
    custom_loss = CustomLoss(
        weight=config.get('training.advanced.custom_loss.weight')
    )
```

### 添加自定义回调

```python
class CustomCallback:
    def on_epoch_end(self, epoch, metrics):
        """Epoch 结束回调"""
        # 自定义逻辑
        pass

    def on_train_end(self, results):
        """训练结束回调"""
        # 自定义逻辑
        pass
```

---

## 📚 参考资料

### 核心文件

- `configs/config_loader.py` - 配置加载器实现
- `configs/logger.py` - 统一日志系统
- `scripts/train/train_classify.py` - 分类训练参考实现
- `scripts/train/training_utils.py` - 训练工具函数

### 配置文档

- `configs/EVALUATION_REPORT.md` - 配置系统评估报告
- `configs/CONFIG_USAGE_CHECK.md` - 配置使用检查报告
- `configs/UNUSED_CONFIGS.md` - 未使用配置报告
- `configs/NEW_ISSUES_REPORT.md` - 配置问题报告

### 外部资源

- [YOLOv8 源码](https://github.com/ultralytics/ultralytics)
- [PyTorch 文档](https://pytorch.org/docs/)
- [timm 文档](https://huggingface.co/docs/timm/)
- [albumentations 文档](https://albumentations.ai/docs/)

---

## 📝 更新日志

### 2026-02-22
- ✅ 创建技术指南文档
- ✅ 详细说明配置系统架构
- ✅ 记录训练流程技术细节
- ✅ 添加性能优化建议
- ✅ 提供扩展开发指南

---

**维护者**: AeroVision Team
**最后更新**: 2026-02-22
