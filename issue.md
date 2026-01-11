# AeroVision 项目问题清单

> 审核时间：2026-01-06
> 结论：**这个项目的管理是一团糟**

---

## 致命问题

### 1. requirements.txt 完全废掉了

```
# 实际内容长这样：
��a b s l - p y = = 2 .3 .1
a i s t u d i o - s d k = = 0 . 3 . 8
```

UTF-16 编码，每个字符之间夹着空字符。`pip install -r requirements.txt` 直接报错。

**这不是疏忽，这是根本没测试过。**

---

### 2. OpenCV 依赖打架

同时装了三个互斥的包：

```
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
opencv-contrib-python==4.10.0.84
```

- `opencv-python` 和 `opencv-python-headless` 不能共存
- 版本号还不一致（4.12 vs 4.10）

**基本的依赖管理都不会。**

---

### 3. 文档是假的

#### conductor.md 画的饼：

```
training/
├── src/                    # 源代码
│   ├── data/               # Dataset 类
│   ├── models/             # 模型定义
│   ├── trainers/           # 训练器
│   └── utils/              # 工具函数
├── data/
│   ├── raw/                # 原始图片
│   ├── processed/          # 处理后数据
│   └── labels/             # 标注文件
```

#### 实际情况：

```bash
$ ls training/src/
ls: training/src/: No such file or directory

$ ls training/data/
ls: training/data/: No such file or directory
```

**整个 src/ 目录压根不存在。data/ 目录也不存在。文档写的全是幻想。**

---

### 4. 技术路线撒谎

#### conductor.md 说的：

> 使用 timm + ConvNeXt/Swin Transformer
> 多 Head 自定义模型架构
> 分阶段训练（Stage 2-7）

#### 代码实际用的：

```python
# train_classify.py 第 45 行
from ultralytics import YOLO
```

**全是 YOLOv8，没有一行 ConvNeXt 或 Swin 的代码。**

但配置文件里还留着这种废话：

```yaml
# training.yaml 第 103-118 行
custom_model:
  backbone: "convnext"
  convnext:
    model_name: "convnext_base"
  swin:
    model_name: "swin_base_patch4_window7_224"
```

**写了一堆从来没用过的配置，增加维护成本和理解成本。**

---

### 5. 阶段文档缺失 50%

conductor.md 承诺的文档：

| 文档 | 状态 |
|------|------|
| stage0_environment.md | ✅ 存在 |
| stage1_data_preparation.md | ✅ 存在 |
| stage2_single_task.md | ❌ **不存在** |
| stage3_multi_head.md | ❌ **不存在** |
| stage4_quality_block.md | ✅ 存在 |
| stage5_hybrid.md | ❌ **不存在** |
| stage6_ocr.md | ✅ 存在 |
| stage7_integration.md | ❌ **不存在** |

**承诺 8 个文档，实际只写了 4 个。**

而且 stage2/3/5 是核心训练阶段的文档，全部缺失。

---

## 配置管理混乱

### 6. 同一个参数，N 个文件定义，N 个不同的值

| 参数 | training.yaml | training_params.yaml | airline.yaml | yolo.yaml |
|------|---------------|----------------------|--------------|-----------|
| epochs | 100 | 50 | 100 | 100 |
| batch_size | 32 | 8 | 32 | 16 |
| image_size | 224 | 640 | 224 | 640 |
| device | (未定义) | "cpu" | (未定义) | (未定义) |

**到底用哪个？没人知道。**

---

### 7. 配置文件命名也是假的

conductor.md 说配置是：
```
configs/
├── stage2_type.yaml
├── stage3_multi.yaml
└── stage5_hybrid.yaml
```

实际是：
```
configs/
├── base.yaml
└── config/
    ├── paths.yaml
    ├── training.yaml
    ├── airline.yaml
    └── ...
```

**文档和代码完全对不上。**

---

### 8. 环境变量硬编码在代码里

```python
# ocr/paddle_ocr.py
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['PADDLEX_LOG_LEVEL'] = 'ERROR'

# train_classify.py
os.environ['YOLO_CONFIG_DIR'] = str(training_root / 'model')
```

- 没有 `.env` 文件
- 没有 `.env.example` 模板
- 环境变量散落在各个脚本里

**配置管理 101 都不会。**

---

## 代码质量问题

### 9. 训练脚本复制粘贴

| 文件 | 行数 |
|------|------|
| train_classify.py | 918 |
| train_airline.py | 911 |
| train_detection.py | 504 |

三个脚本的 `parse_arguments()`、`setup_logging()` 函数几乎一模一样。

**2333 行代码，至少 30% 是重复的。**

---

### 10. sys.path 滥用

9 个文件都有这种操作：

```python
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config
```

**不会用 `setup.py` 或 `pyproject.toml` 做成可安装包吗？**

---

### 11. 测试覆盖率约等于零

测试文件总数：**0**

只有 3 个验证脚本：
- `check_gpu.py` - 检查 GPU
- `verify_env.py` - 检查环境
- `verify_data.py` - 检查数据

**没有单元测试，没有集成测试，什么都没有。**

---

## 数据流程问题

### 12. 文档说的数据流程 vs 实际脚本

**conductor.md 说的：**
```
原始图片 → crop_airplane.py → aircraft_crop/unsorted → 标注 → train/val/test
```

**实际脚本做的：**
```
原始图片 → prepare_dataset.py → prepared/<timestamp>/ → split_dataset.py → splits/<timestamp>/
```

- 多了 `prepare_dataset.py` 这一步（文档没提）
- 输出目录带时间戳（文档没说）
- 中间多了 `prepared/` 和 `splits/` 两层目录

**按照文档操作会一脸懵逼。**

---

### 13. 路径引用三种风格混用

**风格 1：配置文件相对路径**
```yaml
# paths.yaml
data.splits.root: "../data/splits"
```

**风格 2：脚本里硬编码**
```python
# train_classify.py
data_path = "../data/prepared/20260102_221524/aerovision/aircraft"
```

**风格 3：配置加载器**
```python
config.get_path('data.splits.root')
```

**三种风格混着用，出问题了都不知道该查哪里。**

---

## 项目管理问题

### 14. 技术方案大转弯但没告诉任何人

证据链：

1. conductor.md 描述的是自研 ConvNeXt/Swin 多 Head 模型
2. training/README.md 说用 YOLOv8
3. 代码实际用的是 YOLOv8
4. conductor.md 从来没更新过

**结论：项目中途放弃了原计划，改用 YOLOv8 简化实现，但没人更新文档。**

这导致：
- 新人看 conductor.md 会完全走偏
- 配置文件里留着一堆废弃配置
- 阶段文档缺失是因为压根就没按原计划做

---

### 15. training/docs/workflow.md 更新日志自曝

```markdown
| v2.0 | 2026-01-02 | 重构为专注于检测任务，删除分类相关内容 |
```

**承认了技术方案变更，但 conductor.md 依然是旧版描述。**

---

## 总结

| 问题类别 | 数量 | 严重程度 |
|----------|------|----------|
| 致命问题 | 5 | 🔴 项目无法运行 |
| 配置混乱 | 3 | 🟠 维护噩梦 |
| 代码质量 | 3 | 🟡 技术债 |
| 数据流程 | 2 | 🟠 新人陷阱 |
| 项目管理 | 2 | 🔴 根本原因 |

**根本问题：这个项目经历了一次未经记录的技术方案大转弯，从"自研复杂模型"变成了"直接用 YOLOv8"，但所有文档、配置、目录结构都停留在原计划阶段。**

---

## 如果要修的话

### 必须立即修复

1. 用正确编码重新生成 `requirements.txt`
2. 删掉冲突的 OpenCV 依赖
3. 要么更新 conductor.md 反映实际技术栈，要么删掉它

### 应该尽快修复

4. 删掉配置文件里从来没用过的 ConvNeXt/Swin 配置
5. 统一配置参数，消除重复定义
6. 创建 `.env.example` 模板

### 有空再说

7. 重构训练脚本，提取公共代码
8. 把项目做成可安装包
9. 写测试（虽然估计永远不会写）

---

**评价：这不是一个成熟的项目，这是一个原型验证阶段的代码被错误地当成了正式项目。**

---

## 补充问题（2026-01-11 更新）

### 16. 配置系统浅拷贝漏洞

**文件**: `training/configs/config_loader.py:113-115`

**问题描述**: `to_dict()` 方法使用浅拷贝，外部代码可以修改返回的字典从而影响原始配置。

```python
def to_dict(self) -> Dict[str, Any]:
    return self._config.copy()  # 浅拷贝！
```

**详见**: `training/configs/ISSUES.md`

---

### 17. 配置模块加载不完整

**文件**: `training/configs/config_loader.py:185`

**问题描述**: 硬编码的模块列表缺少 3 个实际存在的配置文件。

**缺失的模块**: `airline`, `inference`, `training_params`

**详见**: `training/configs/ISSUES.md`

---

### 18. 配置键冲突

**问题描述**: 多个配置文件定义了相同的顶层键，后加载的会覆盖先加载的。

**详见**: `training/configs/ISSUES.md`

---

### 19. 训练脚本 args 默认值应从 config 读取

**问题描述**: 训练脚本的 argparse 默认值硬编码在代码中，应该从配置文件读取。

**当前代码**:
```python
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch', type=int, default=16)
```

**建议修复**:
```python
config = load_config()
parser.add_argument('--epochs', type=int,
    default=config.get('training.epochs', 100))
```

**详见**: `training/configs/ISSUES.md`

---

## 更新后的问题统计

| 问题类别 | 数量 | 严重程度 |
|----------|------|----------|
| 致命问题 | 5 | 🔴 项目无法运行 |
| 配置混乱 | 7 (+4) | 🟠 维护噩梦 |
| 代码质量 | 3 | 🟡 技术债 |
| 数据流程 | 2 | 🟠 新人陷阱 |
| 项目管理 | 2 | 🔴 根本原因 |

**总计**: 19 个问题（原 15 个 + 新增 4 个）

---

## 相关文档

- `training/configs/ISSUES.md` - 配置模块专项问题清单（19 个问题）
