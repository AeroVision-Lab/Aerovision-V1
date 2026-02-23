# AeroVision 训练流程 - TUI 交互式界面

统一的训练流程执行器，支持命令行和交互式 TUI 两种模式。

---

## 🎨 TUI 模式特性

- **交互式配置向导** - 无需记忆复杂的命令行参数
- **美观的终端界面** - 使用 Rich 库美化输出
- **实时进度显示** - 清晰的执行状态反馈
- **配置摘要确认** - 执行前预览所有配置
- **智能默认值** - 常用配置自动填充

---

## 📦 安装依赖

TUI 模式需要额外的依赖库：

```bash
pip install questionary rich
```

或者安装完整依赖：

```bash
pip install -r requirements.txt
```

---

## 🚀 使用方法

**重要**: 必须在 `training/` 目录下运行脚本！

### 方式 1: 交互式 TUI 模式（推荐）

直接运行脚本，自动进入交互式界面：

```bash
# Windows
cd training
python run_workflow.py --tui

# 或使用快捷脚本
cd training
run_workflow.bat

# Linux/Mac
cd training
python3 run_workflow.py --tui

# 或使用快捷脚本
cd training
./run_workflow.sh
```

或者不带任何参数（如果安装了 questionary）：

```bash
cd training
python run_workflow.py
```

### 方式 2: 命令行模式

传统的命令行参数模式：

```bash
cd training

# 完整流程
python run_workflow.py --workflow full

# 只验证环境
python run_workflow.py --workflow verify

# 只准备数据
python run_workflow.py --workflow prepare

# 训练单个模型
python run_workflow.py --workflow train --task classify --epochs 100

# 训练所有模型
python run_workflow.py --workflow train-all
```

---

## 🎯 TUI 交互流程

### 1. 选择工作流类型

```
? 选择工作流类型:
  ❯ 完整流程 (验证 + 准备数据 + 训练所有模型)
    仅验证环境
    仅准备数据
    训练单个模型
    训练所有模型
```

### 2. 选择训练任务（如果选择"训练单个模型"）

```
? 选择训练任务:
  ❯ 飞机分类 (classify)
    航司识别 (airline)
    注册号检测 (detection)
```

### 3. 数据准备选项（如果涉及数据准备）

```
? 跳过数据准备步骤? (y/N)
? 跳过数据划分步骤? (y/N)
```

### 4. 训练参数配置（如果涉及训练）

```
? 配置训练参数? (否则使用默认值) (y/N)

# 如果选择 Yes:
? 训练轮数 (epochs): 100
? 批次大小 (batch-size): 32
? 训练设备:
  ❯ GPU 0
    GPU 1
    CPU
```

### 5. 其他选项

```
? 出错时继续执行? (y/N)
? 显示详细输出? (y/N)
```

### 6. 配置摘要确认

```
┌─────────────── 配置摘要 ───────────────┐
│ 参数       │ 值                        │
├────────────┼───────────────────────────┤
│ 工作流     │ train                     │
│ 任务       │ classify                  │
│ 训练轮数   │ 100                       │
│ 批次大小   │ 32                        │
│ 设备       │ 0                         │
│ 详细输出   │ 否                        │
└────────────┴───────────────────────────┘

? 开始执行? (Y/n)
```

---

## 🎨 TUI 界面示例

### 执行过程

```
╭─────────────────────────────────────╮
│         验证环境                     │
╰─────────────────────────────────────╯

Python 版本: 3.11.5
▶ 验证配置系统
✓ 验证配置系统

╭─────────────────────────────────────╮
│         数据准备流程                 │
╰─────────────────────────────────────╯

▶ 步骤 1/2: 准备数据集 (验证、清洗)
✓ 步骤 1/2: 准备数据集 (验证、清洗)

▶ 步骤 2/2: 划分数据集 (train/val/test)
✓ 步骤 2/2: 划分数据集 (train/val/test)
```

### 成功完成

```
╭─────────────────────────────────────╮
│     ✓ 工作流执行成功                 │
╰─────────────────────────────────────╯
```

### 执行失败

```
╭─────────────────────────────────────╮
│     ✗ 工作流执行失败                 │
╰─────────────────────────────────────╯
```

---

## 📋 支持的工作流

| 工作流 | 说明 | TUI 支持 |
|--------|------|----------|
| `full` | 完整流程：验证 → 准备数据 → 训练所有模型 | ✅ |
| `verify` | 仅验证环境和配置 | ✅ |
| `prepare` | 仅准备和划分数据 | ✅ |
| `train` | 训练单个指定模型 | ✅ |
| `train-all` | 训练所有模型（classify + airline + detection） | ✅ |

---

## ⚙️ 命令行参数参考

### 工作流参数

```bash
--workflow {full,verify,prepare,train,train-all}
                        工作流类型
--task {classify,airline,detection}
                        训练任务 (用于 train workflow)
```

### 数据准备参数

```bash
--labels LABELS         标注文件路径
--images IMAGES         图片目录路径
--prepare-dir PREPARE_DIR
                        准备好的数据目录
--skip-prepare          跳过数据准备步骤
--skip-split            跳过数据划分步骤
```

### 训练参数

```bash
--epochs EPOCHS         训练轮数
--batch-size BATCH_SIZE 批次大小
--imgsz IMGSZ           图片尺寸
--lr0 LR0               初始学习率
--device DEVICE         训练设备 (0, 1, cpu)
--model MODEL           预训练模型路径
--resume RESUME         恢复训练的检查点路径
```

### 其他选项

```bash
--verbose               显示详细输出
--continue-on-error     出错时继续执行 (用于 full/train-all workflow)
--tui                   使用交互式 TUI 界面
```

---

## 💡 使用技巧

### 1. 快速开始（推荐新手）

```bash
cd training
python run_workflow.py --tui
```

跟随交互式向导，无需记忆任何参数。

### 2. 自动化脚本（推荐 CI/CD）

```bash
cd training
python run_workflow.py \
    --workflow train \
    --task classify \
    --epochs 100 \
    --batch-size 32 \
    --device 0 \
    --verbose
```

适合写入脚本或 CI/CD 流程。

### 3. 调试模式

```bash
cd training
python run_workflow.py --workflow verify --verbose
```

详细输出所有执行信息，便于排查问题。

### 4. 批量训练

```bash
cd training
python run_workflow.py \
    --workflow train-all \
    --continue-on-error \
    --epochs 50
```

训练所有模型，即使某个失败也继续执行。

---

## 🔧 故障排除

### 问题 1: 在错误的目录运行脚本

**错误信息:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'training/configs/...'
```

**解决方法:**
```bash
# 必须在 training/ 目录下运行！
cd training
python run_workflow.py --tui
```

### 问题 2: 缺少 TUI 依赖

**错误信息:**
```
错误: 需要安装 questionary 库
运行: pip install questionary
```

**解决方法:**
```bash
pip install questionary rich
```

### 问题 3: 命令行模式缺少 --workflow 参数

**错误信息:**
```
错误: 需要指定 --workflow 参数或使用 --tui 进入交互式模式
```

**解决方法:**
```bash
# 方式 1: 使用 TUI
cd training
python run_workflow.py --tui

# 方式 2: 指定 workflow
cd training
python run_workflow.py --workflow verify
```

### 问题 4: Windows 终端显示异常

**解决方法:**

使用 Windows Terminal 或启用 UTF-8 支持：

```bash
# PowerShell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 或使用 Windows Terminal (推荐)
```

---

## 📚 相关文档

- `TECHNICAL_GUIDE.md` - 技术实现细节
- `verify_configs.py` - 配置验证脚本
- `configs/` - 配置文件目录

---

**维护者**: AeroVision Team
**最后更新**: 2026-02-22
