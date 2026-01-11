# Training Config 模块问题清单

> 生成时间: 2026-01-07
> 检查范围: `training/configs/` 目录

---

## 严重问题 (Critical)

### 1. 示例代码使用了不存在的配置键

**文件**: `config_usage_examples.py`

**问题描述**: 示例代码中使用的多个配置键在实际配置文件中不存在，导致示例无法正常运行。

**不存在的键**:
- `basic.batch_size` - 实际应为 `training.batch_size`
- `basic.learning_rate` - 实际应为 `training.optimizer.lr0`
- `basic.num_epochs` - 实际应为 `training.epochs`
- `optimizer.type` - 实际应为 `training.optimizer.type`
- `paths.data_root` - 实际应为 `data.local_data_root`
- `paths.yolo_model` - 实际应为 `model.weights`

**影响**: 用户按照示例使用配置系统会得到 `None` 值，可能导致运行时错误。

**建议修复**:
```python
# 修改 example_basic_usage() 中的代码
print(f"训练批次大小: {config.get('training.batch_size')}")  # 而非 basic.batch_size
```

---

### 2. `to_dict()` 返回浅拷贝导致配置被意外修改

**文件**: `config_loader.py:113-115`

**问题描述**:
```python
def to_dict(self) -> Dict[str, Any]:
    """转换为字典"""
    return self._config.copy()  # 浅拷贝！
```

`.copy()` 只做浅拷贝，嵌套的字典仍然是同一个引用。

**复现代码**:
```python
config = load_config()
d = config.to_dict()
d['project']['name'] = 'HACKED'
# 现在 config.get('project.name') 也变成了 'HACKED'
```

**影响**: 外部代码修改返回的字典会影响原始配置对象，可能导致难以追踪的 bug。

**建议修复**:
```python
import copy

def to_dict(self) -> Dict[str, Any]:
    """转换为字典（深拷贝）"""
    return copy.deepcopy(self._config)
```

---

### 3. 模块加载列表不完整

**文件**: `config_loader.py:185`

**问题描述**: 硬编码的模块列表缺少实际存在的配置文件。

**当前代码**:
```python
modules_to_load = ['paths', 'yolo', 'crop', 'review', 'training', 'augmentation', 'ocr', 'logging']
```

**缺失的模块**:
- `airline` - 航司识别配置
- `inference` - 推理配置
- `training_params` - 训练参数配置

**影响**: 调用 `load_config()` 不会加载这三个配置文件的内容。

**建议修复**:
```python
# 方案1: 添加缺失的模块
modules_to_load = ['paths', 'yolo', 'crop', 'review', 'training', 'augmentation', 'ocr', 'logging', 'airline', 'inference', 'training_params']

# 方案2: 动态扫描目录（推荐）
modules_to_load = [f.stem for f in (configs_dir / "config").glob("*.yaml")]
```

---

## 高优先级问题 (High)

### 4. 配置键冲突导致值被覆盖

**问题描述**: 多个配置文件定义了相同的顶层键，后加载的会覆盖先加载的。

**冲突列表**:
| 键名 | 定义位置 |
|------|----------|
| `batch` | crop.yaml, ocr.yaml |
| `detection` | training_params.yaml, yolo.yaml |
| `image` | base.yaml, inference.yaml |
| `logging` | logging.yaml, training_params.yaml |
| `models` | inference.yaml, paths.yaml |
| `ocr` | inference.yaml, ocr.yaml, training_params.yaml |
| `output` | crop.yaml, inference.yaml, ocr.yaml, paths.yaml, review.yaml |
| `performance` | logging.yaml, ocr.yaml |
| `quality` | augmentation.yaml, training_params.yaml |
| `training` | logging.yaml, training.yaml |

**影响**: 配置行为依赖于加载顺序，难以预测最终值。

**建议修复**:
1. 为每个模块添加命名空间前缀，如 `crop_output`, `ocr_output`
2. 或者重构配置结构，确保键名唯一

---

### 5. 路径格式不一致

**问题描述**: 不同配置文件使用了不同的路径表示方式。

| 文件 | 路径示例 | 说明 |
|------|----------|------|
| paths.yaml | `../data/raw` | 相对于 configs 目录 |
| inference.yaml | `training/checkpoints/...` | 相对于项目根目录 |
| training_params.yaml | `data` | 相对于项目根目录 |

**影响**: 使用 `get_path()` 解析路径时，不同文件的路径会被错误解析。

**建议修复**:
统一所有配置文件使用相同的路径约定（推荐使用 `../xxx` 格式，相对于 configs 目录）。

---

### 6. 文件头注释路径错误

**文件**: `config_loader.py:1`

**问题描述**:
```python
# training/config/config_loader.py  # 错误！
```

实际路径是 `training/configs/config_loader.py`（注意是 configs 不是 config）。

**影响**: 误导开发者。

**建议修复**: 更正为实际路径。

---

## 中优先级问题 (Medium)

### 7. `__getattr__` 不抛出 `AttributeError`

**文件**: `config_loader.py:38-42`

**问题描述**:
```python
def __getattr__(self, key: str) -> Any:
    if key.startswith('_'):
        return object.__getattribute__(self, key)
    return self._config.get(key)  # 返回 None 而非抛出异常
```

访问不存在的属性返回 `None`，违反 Python 惯例。

**影响**:
- 无法使用 `hasattr()` 正确检查属性是否存在
- 拼写错误不会被发现
- IDE 自动补全失效

**建议修复**:
```python
def __getattr__(self, key: str) -> Any:
    if key.startswith('_'):
        return object.__getattribute__(self, key)
    if key in self._config:
        return self._config[key]
    raise AttributeError(f"Config has no attribute '{key}'")
```

---

### 8. `base.yaml` 中的 `modules` 字段未被使用

**文件**: `base.yaml:58-69`

**问题描述**:
```yaml
modules:
  paths: "config/paths.yaml"
  yolo: "config/yolo.yaml"
  # ...
```

这个字段看起来像是定义模块路径的，但实际代码中是硬编码的模块列表。

**影响**:
- 维护两处配置（yaml 和代码），容易不一致
- 误导开发者以为修改 yaml 可以控制加载行为

**建议修复**:
方案1: 删除 `base.yaml` 中的 `modules` 字段，或添加注释说明仅供参考
方案2: 修改 `load_config()` 读取 `base.yaml` 的 `modules` 字段来决定加载哪些模块

---

### 9. `_deep_merge` 与 `_deep_update` 行为不一致

**文件**: `config_loader.py:104-111, 211-228`

**问题描述**:
- `_deep_merge`: 返回新字典（不修改原字典）
- `_deep_update`: 原地修改字典（无返回值）

**影响**: 两个功能相似的函数行为不一致，容易导致 bug。

**建议修复**: 统一使用一种模式，或重命名以明确区分。

---

### 10. `save_config` 包含 `print` 语句

**文件**: `config_loader.py:245`

**问题描述**:
```python
def save_config(config: Config, save_path: str):
    # ...
    print(f"配置已保存到: {save_path}")  # 库代码不应该直接打印
```

**影响**: 作为库函数，输出应该通过日志系统或返回值传递，而非直接打印。

**建议修复**:
```python
import logging
logger = logging.getLogger(__name__)

def save_config(config: Config, save_path: str) -> Path:
    # ...
    logger.info(f"配置已保存到: {save_path}")
    return save_path
```

---

## 低优先级问题 (Low)

### 11. 类型注解不准确

**文件**: `config_loader.py`

**问题描述**:
```python
def get(self, key: str, default: Any = None) -> Any:  # 应该是 Optional[Any]
def __getattr__(self, key: str) -> Any:  # 应该是 Optional[Any]
```

**建议修复**:
```python
from typing import Optional

def get(self, key: str, default: T = None) -> Optional[T]: ...
```

---

### 12. 配置文件中大量 `null` 值

**问题描述**: 配置文件中有 26 处 `null` 值，虽然可能是有意的（表示使用默认值），但应该文档化。

**示例**:
- `inference.yaml: image.resize = null`
- `logging.yaml: wandb.entity = null`
- `training.yaml: stage_configs.stage2.num_classes = null`

**建议**: 添加注释说明 `null` 的含义。

---

### 13. 线程安全问题

**文件**: `config_loader.py:95-111`

**问题描述**: `Config.update()` 和 `_deep_update()` 不是线程安全的。

**影响**: 多线程环境下同时更新配置可能导致数据竞争。

**建议修复**: 如果需要线程安全，添加锁机制或使用不可变配置。

---

### 14. `get` 方法不支持带点号的键名

**文件**: `config_loader.py:44-66`

**问题描述**: 使用 `.split('.')` 解析键名，无法支持键名本身包含点号的情况。

**示例**:
```yaml
# 如果配置是这样的
"model.v2": "some_value"  # 键名包含点号
```
```python
config.get("model.v2")  # 会被解析为 config["model"]["v2"]，而非 config["model.v2"]
```

**建议**: 文档化此限制，或提供原始键访问方式。

---

## 建议的重构方向

1. **统一路径约定**: 所有配置文件使用相同的相对路径基准
2. **消除键冲突**: 重构配置结构，使用命名空间或嵌套结构
3. **动态模块发现**: 自动扫描 config 目录而非硬编码模块列表
4. **添加配置验证**: 使用 Pydantic 或 JSON Schema 验证配置结构
5. **更新示例代码**: 确保示例代码使用的键与实际配置匹配
6. **完善文档**: 为每个配置项添加类型和默认值说明

---

## 快速修复清单

- [ ] 修复 `config_usage_examples.py` 中的配置键
- [ ] 将 `to_dict()` 改为深拷贝
- [ ] 添加缺失的模块到加载列表
- [ ] 修正文件头注释路径
- [ ] 移除 `save_config` 中的 `print` 语句
- [x] 训练脚本 args 默认值改为从 config 读取 ✅ (2026-01-11)
- [x] 修复 `or` 运算符处理 0 值的问题 ✅ (2026-01-11)

---

## 训练脚本与配置集成问题

> 以下问题涉及训练脚本如何使用配置系统

### 15. `or` 运算符无法正确处理 0 值

**问题描述**: 使用 `or` 运算符设置默认值时，如果配置值为 `0`，会错误地返回默认值而非 `0`。

**错误代码**:
```python
patience = config.get('training.patience') or 50
# 如果 config 中 patience = 0，结果是 50 而不是 0！
```

**原因**: Python 中 `0` 是 falsy 值，`0 or 50` 返回 `50`。

**建议修复**:
```python
# 方案1: 使用 get 的 default 参数
patience = config.get('training.patience', 50)

# 方案2: 显式检查 None
patience_val = config.get('training.patience')
patience = patience_val if patience_val is not None else 50

# 方案3: 使用三元表达式
patience = config.get('training.patience') ?? 50  # Python 没有 ??，需用方案1或2
```

**受影响的参数类型**:
- `patience` - 早停耐心值，0 表示禁用早停
- `warmup_epochs` - 预热轮数，0 表示不预热
- `weight_decay` - 权重衰减，0 表示不使用
- 其他可能为 0 的数值参数

**影响的文件**:
- `train_classify.py`
- `train_detection.py`
- `train_airline.py`

---

### 16. 训练脚本 args 默认值应从 config 读取

**问题描述**: 训练脚本的 argparse 默认值硬编码在代码中，应该从配置文件读取。

**当前代码** (`train_classify.py`):
```python
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--imgsz', type=int, default=640)
parser.add_argument('--lr0', type=float, default=0.01)
```

**建议修复**:
```python
from configs import load_config

config = load_config()

parser.add_argument('--epochs', type=int,
    default=config.get('training.epochs', 100))
parser.add_argument('--batch', type=int,
    default=config.get('training.batch_size', 16))
parser.add_argument('--imgsz', type=int,
    default=config.get('training.imgsz', 640))
parser.add_argument('--lr0', type=float,
    default=config.get('training.optimizer.lr0', 0.01))
```

**影响的文件**:
- `train_classify.py`
- `train_detection.py`
- `train_airline.py`

**好处**:
- 统一配置来源，避免代码和配置不一致
- 修改默认值只需改配置文件，无需改代码
- 命令行参数仍可覆盖配置文件的值

---

### 17. 配置路径解析函数不统一

**问题描述**: 不同脚本使用不同的方式解析配置路径。

| 脚本 | 使用的函数 |
|------|-----------|
| `train_classify.py` | `_resolve_training_path()` |
| `train_detection.py` | `resolve_config_path()` |
| `train_airline.py` | `_resolve_training_path()` |

**影响**: 维护困难，行为可能不一致。

**建议修复**: 统一使用 `config_loader.py` 中的 `get_path()` 方法。
