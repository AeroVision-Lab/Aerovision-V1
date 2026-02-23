#!/usr/bin/env python3
"""
AeroVision Training Workflow Runner

统一的训练流程执行脚本，支持 Windows 和 Linux

Usage:
    # 完整流程
    python run_workflow.py --workflow full

    # 只准备数据
    python run_workflow.py --workflow prepare

    # 只训练
    python run_workflow.py --workflow train --task classify

    # 自定义参数
    python run_workflow.py --workflow train --task classify --epochs 100 --batch-size 32
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    import questionary
    from questionary import Style
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class WorkflowRunner:
    """训练流程执行器"""

    def __init__(self, verbose: bool = False, use_tui: bool = False):
        self.verbose = verbose
        self.use_tui = use_tui and HAS_RICH
        self.training_root = Path(__file__).parent
        self.scripts_dir = self.training_root / "scripts"
        self.logger = self._setup_logger()

        # 检测操作系统
        self.is_windows = sys.platform.startswith('win')
        self.python_cmd = sys.executable

        # Rich console
        if self.use_tui:
            self.console = Console()
        else:
            self.console = None

    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("WorkflowRunner")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        logger.handlers.clear()

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def run_command(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None,
        description: str = ""
    ) -> bool:
        """
        执行命令

        Args:
            cmd: 命令列表
            cwd: 工作目录
            description: 命令描述

        Returns:
            是否成功
        """
        if description:
            if self.use_tui and self.console:
                self.console.print(f"[bold cyan]▶[/bold cyan] {description}")
            else:
                self.logger.info(f"[执行] {description}")

        if self.verbose:
            self.logger.debug(f"命令: {' '.join(cmd)}")
            self.logger.debug(f"工作目录: {cwd or Path.cwd()}")

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.training_root,
                check=True,
                capture_output=not self.verbose,
                text=True
            )

            if self.verbose and result.stdout:
                self.logger.debug(f"输出:\n{result.stdout}")

            if self.use_tui and self.console:
                self.console.print(f"[bold green]✓[/bold green] {description or '命令执行成功'}")
            else:
                self.logger.info(f"[成功] {description or '命令执行成功'}")
            return True

        except subprocess.CalledProcessError as e:
            if self.use_tui and self.console:
                self.console.print(f"[bold red]✗[/bold red] {description or '命令执行失败'}")
            else:
                self.logger.error(f"[失败] {description or '命令执行失败'}")
            if e.stdout:
                self.logger.error(f"标准输出:\n{e.stdout}")
            if e.stderr:
                self.logger.error(f"错误输出:\n{e.stderr}")
            return False

        except Exception as e:
            if self.use_tui and self.console:
                self.console.print(f"[bold red]✗[/bold red] {description}: {e}")
            else:
                self.logger.error(f"[异常] {description}: {e}")
            return False

    def verify_environment(self) -> bool:
        """验证环境"""
        if self.use_tui and self.console:
            self.console.print(Panel.fit(
                "[bold cyan]验证环境[/bold cyan]",
                border_style="cyan"
            ))
        else:
            self.logger.info("=" * 70)
            self.logger.info("验证环境")
            self.logger.info("=" * 70)

        all_passed = True

        # 1. 检查 Python 版本
        python_version = sys.version_info
        version_str = f"{python_version.major}.{python_version.minor}.{python_version.micro}"

        if self.use_tui and self.console:
            if python_version >= (3, 8):
                self.console.print(f"[green]✓[/green] Python 版本: {version_str}")
            else:
                self.console.print(f"[red]✗[/red] Python 版本: {version_str} (需要 >= 3.8)")
                all_passed = False
        else:
            self.logger.info(f"Python 版本: {version_str}")

        if python_version < (3, 11):
            warning = "建议使用 Python 3.11+"
            if self.use_tui and self.console:
                self.console.print(f"[yellow]⚠[/yellow] {warning}")
            else:
                self.logger.warning(warning)

        # 2. 检查 PyTorch 和 CUDA
        try:
            import torch

            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()

            if self.use_tui and self.console:
                self.console.print(f"[green]✓[/green] PyTorch 版本: {torch_version}")
                self.console.print(f"[green]✓[/green] CUDA 可用: {cuda_available}")
            else:
                self.logger.info(f"PyTorch 版本: {torch_version}")
                self.logger.info(f"CUDA 可用: {cuda_available}")

            if cuda_available:
                cuda_version = torch.version.cuda
                gpu_count = torch.cuda.device_count()

                if self.use_tui and self.console:
                    self.console.print(f"[green]✓[/green] CUDA 版本: {cuda_version}")
                    self.console.print(f"[green]✓[/green] GPU 数量: {gpu_count}")
                else:
                    self.logger.info(f"CUDA 版本: {cuda_version}")
                    self.logger.info(f"GPU 数量: {gpu_count}")

                # 显示每个 GPU 的信息
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    gpu_memory = props.total_memory / (1024 ** 3)

                    if self.use_tui and self.console:
                        self.console.print(
                            f"[cyan]  GPU {i}:[/cyan] {gpu_name} "
                            f"({gpu_memory:.1f}GB, 计算能力 {props.major}.{props.minor})"
                        )
                    else:
                        self.logger.info(
                            f"  GPU {i}: {gpu_name} "
                            f"({gpu_memory:.1f}GB, 计算能力 {props.major}.{props.minor})"
                        )

                # 测试 GPU 运算
                try:
                    x = torch.randn(100, 100).cuda()
                    y = torch.randn(100, 100).cuda()
                    z = torch.matmul(x, y)

                    if self.use_tui and self.console:
                        self.console.print(f"[green]✓[/green] GPU 运算测试通过")
                    else:
                        self.logger.info("GPU 运算测试通过")
                except Exception as e:
                    if self.use_tui and self.console:
                        self.console.print(f"[red]✗[/red] GPU 运算测试失败: {e}")
                    else:
                        self.logger.error(f"GPU 运算测试失败: {e}")
                    all_passed = False
            else:
                warning = "CUDA 不可用，将使用 CPU 训练（速度较慢）"
                if self.use_tui and self.console:
                    self.console.print(f"[yellow]⚠[/yellow] {warning}")
                else:
                    self.logger.warning(warning)

        except ImportError:
            error = "PyTorch 未安装"
            if self.use_tui and self.console:
                self.console.print(f"[red]✗[/red] {error}")
            else:
                self.logger.error(error)
            all_passed = False

        # 3. 检查关键依赖包
        packages = [
            ("ultralytics", "ultralytics"),
            ("paddleocr", "paddleocr"),
            ("albumentations", "albumentations"),
            ("pandas", "pandas"),
            ("pyyaml", "yaml"),
        ]

        missing_packages = []
        for package_name, import_name in packages:
            try:
                module = __import__(import_name)
                version = getattr(module, '__version__', 'unknown')

                if self.use_tui and self.console:
                    self.console.print(f"[green]✓[/green] {package_name}: {version}")
                else:
                    self.logger.info(f"{package_name}: {version}")
            except ImportError:
                missing_packages.append(package_name)
                if self.use_tui and self.console:
                    self.console.print(f"[red]✗[/red] {package_name}: 未安装")
                else:
                    self.logger.error(f"{package_name}: 未安装")
                all_passed = False

        if missing_packages:
            install_cmd = f"pip install {' '.join(missing_packages)}"
            if self.use_tui and self.console:
                self.console.print(f"[yellow]提示:[/yellow] {install_cmd}")
            else:
                self.logger.warning(f"安装缺失的包: {install_cmd}")

        # 4. 验证配置系统
        verify_script = self.training_root / "verify_configs.py"
        if verify_script.exists():
            success = self.run_command(
                [self.python_cmd, str(verify_script)],
                description="验证配置系统"
            )
            if not success:
                self.logger.error("配置验证失败，请检查配置文件")
                all_passed = False
        else:
            warning = "未找到配置验证脚本"
            if self.use_tui and self.console:
                self.console.print(f"[yellow]⚠[/yellow] {warning}")
            else:
                self.logger.warning(warning)

        # 5. 检查目录结构
        required_dirs = [
            "data/processed",
            "data/processed/labeled",
            "ckpt",
            "logs",
            "configs",
            "scripts",
        ]

        missing_dirs = []
        for dir_path in required_dirs:
            full_path = self.training_root / dir_path
            if full_path.exists():
                if self.use_tui and self.console:
                    self.console.print(f"[green]✓[/green] {dir_path}/")
                else:
                    self.logger.debug(f"{dir_path}/ 存在")
            else:
                missing_dirs.append(dir_path)
                if self.use_tui and self.console:
                    self.console.print(f"[yellow]⚠[/yellow] {dir_path}/ (不存在)")
                else:
                    self.logger.warning(f"{dir_path}/ 不存在")

        # 6. 显示摘要
        if self.use_tui and self.console:
            self.console.print()
            if all_passed and not missing_dirs:
                self.console.print(Panel.fit(
                    "[bold green]✓ 环境验证通过！可以开始训练。[/bold green]",
                    border_style="green"
                ))
            elif all_passed:
                self.console.print(Panel.fit(
                    "[bold yellow]⚠ 环境验证通过，但有些目录不存在（不影响使用）[/bold yellow]",
                    border_style="yellow"
                ))
            else:
                self.console.print(Panel.fit(
                    "[bold red]✗ 环境验证失败，请修复上述问题[/bold red]",
                    border_style="red"
                ))
        else:
            if all_passed:
                self.logger.info("环境验证通过")
            else:
                self.logger.error("环境验证失败")
            self.logger.info("")

        return all_passed

    def prepare_data(self, args: argparse.Namespace) -> bool:
        """准备数据"""
        self.logger.info("=" * 70)
        self.logger.info("数据准备流程")
        self.logger.info("=" * 70)

        data_prep_dir = self.scripts_dir / "data_prep"

        # 步骤 1: 准备数据集
        if not args.skip_prepare:
            prepare_cmd = [self.python_cmd, "prepare_dataset.py"]

            if args.labels:
                prepare_cmd.extend(["--labels", args.labels])
            if args.images:
                prepare_cmd.extend(["--images", args.images])

            success = self.run_command(
                prepare_cmd,
                cwd=data_prep_dir,
                description="步骤 1/2: 准备数据集 (验证、清洗)"
            )

            if not success:
                self.logger.error("数据准备失败")
                return False
        else:
            self.logger.info("跳过数据准备步骤")

        # 步骤 2: 划分数据集
        if not args.skip_split:
            split_cmd = [self.python_cmd, "split_dataset.py"]

            if args.prepare_dir:
                split_cmd.extend(["--prepare-dir", args.prepare_dir])

            success = self.run_command(
                split_cmd,
                cwd=data_prep_dir,
                description="步骤 2/2: 划分数据集 (train/val/test)"
            )

            if not success:
                self.logger.error("数据划分失败")
                return False
        else:
            self.logger.info("跳过数据划分步骤")

        self.logger.info("")
        return True

    def train_model(self, args: argparse.Namespace) -> bool:
        """训练模型"""
        self.logger.info("=" * 70)
        self.logger.info(f"训练模型: {args.task}")
        self.logger.info("=" * 70)

        train_dir = self.scripts_dir / "train"

        # 确定训练脚本
        script_map = {
            "classify": "train_classify.py",
            "airline": "train_airline.py",
            "detection": "train_detection.py",
        }

        if args.task not in script_map:
            self.logger.error(f"未知任务: {args.task}")
            self.logger.error(f"支持的任务: {', '.join(script_map.keys())}")
            return False

        script_name = script_map[args.task]
        train_cmd = [self.python_cmd, script_name]

        # 添加训练参数
        if args.epochs:
            train_cmd.extend(["--epochs", str(args.epochs)])
        if args.batch_size:
            train_cmd.extend(["--batch-size", str(args.batch_size)])
        if args.imgsz:
            train_cmd.extend(["--imgsz", str(args.imgsz)])
        if args.lr0:
            train_cmd.extend(["--lr0", str(args.lr0)])
        if args.device:
            train_cmd.extend(["--device", args.device])
        if args.model:
            train_cmd.extend(["--model", args.model])
        if args.resume:
            train_cmd.extend(["--resume", args.resume])

        # 执行训练
        success = self.run_command(
            train_cmd,
            cwd=train_dir,
            description=f"训练 {args.task} 模型"
        )

        if not success:
            self.logger.error("训练失败")
            return False

        self.logger.info("")
        return True

    def run_full_workflow(self, args: argparse.Namespace) -> bool:
        """运行完整流程"""
        self.logger.info("=" * 70)
        self.logger.info("AeroVision 完整训练流程")
        self.logger.info("=" * 70)
        self.logger.info("")

        # 1. 验证环境
        if not self.verify_environment():
            return False

        # 2. 准备数据
        if not self.prepare_data(args):
            return False

        # 3. 训练所有模型
        tasks = ["classify", "airline", "detection"]

        for task in tasks:
            self.logger.info(f"开始训练: {task}")
            args.task = task

            if not self.train_model(args):
                self.logger.error(f"训练 {task} 失败")
                if not args.continue_on_error:
                    return False
                else:
                    self.logger.warning(f"跳过 {task}，继续下一个任务")

        self.logger.info("=" * 70)
        self.logger.info("完整流程执行完成")
        self.logger.info("=" * 70)

        return True


def interactive_workflow() -> argparse.Namespace:
    """交互式 TUI 界面"""
    if not HAS_QUESTIONARY:
        print("错误: 需要安装 questionary 库")
        print("运行: pip install questionary")
        sys.exit(1)

    console = Console() if HAS_RICH else None

    # 自定义样式
    custom_style = Style([
        ('qmark', 'fg:#673ab7 bold'),
        ('question', 'bold'),
        ('answer', 'fg:#2196f3 bold'),
        ('pointer', 'fg:#673ab7 bold'),
        ('highlighted', 'fg:#673ab7 bold'),
        ('selected', 'fg:#4caf50'),
        ('separator', 'fg:#cc5454'),
        ('instruction', ''),
        ('text', ''),
    ])

    if console:
        console.print(Panel.fit(
            "[bold cyan]AeroVision 训练流程[/bold cyan]\n"
            "[dim]交互式配置向导[/dim]",
            border_style="cyan",
            box=box.DOUBLE
        ))
    else:
        print("=" * 70)
        print("AeroVision 训练流程 - 交互式配置向导")
        print("=" * 70)
        print()

    # 1. 选择工作流
    workflow = questionary.select(
        "选择工作流类型:",
        choices=[
            questionary.Choice("完整流程 (验证 + 准备数据 + 训练所有模型)", value="full"),
            questionary.Choice("仅验证环境", value="verify"),
            questionary.Choice("仅准备数据", value="prepare"),
            questionary.Choice("训练单个模型", value="train"),
            questionary.Choice("训练所有模型", value="train-all"),
        ],
        style=custom_style
    ).ask()

    if workflow is None:
        sys.exit(0)

    args = argparse.Namespace(
        workflow=workflow,
        task=None,
        labels=None,
        images=None,
        prepare_dir=None,
        skip_prepare=False,
        skip_split=False,
        epochs=None,
        batch_size=None,
        imgsz=None,
        lr0=None,
        device=None,
        model=None,
        resume=None,
        verbose=False,
        continue_on_error=False,
    )

    # 2. 如果是 train workflow，选择任务
    if workflow == "train":
        task = questionary.select(
            "选择训练任务:",
            choices=[
                questionary.Choice("飞机分类 (classify)", value="classify"),
                questionary.Choice("航司识别 (airline)", value="airline"),
                questionary.Choice("注册号检测 (detection)", value="detection"),
            ],
            style=custom_style
        ).ask()

        if task is None:
            sys.exit(0)

        args.task = task

    # 3. 数据准备选项 (如果是 prepare 或 full)
    if workflow in ["prepare", "full"]:
        skip_prepare = questionary.confirm(
            "跳过数据准备步骤?",
            default=False,
            style=custom_style
        ).ask()

        skip_split = questionary.confirm(
            "跳过数据划分步骤?",
            default=False,
            style=custom_style
        ).ask()

        args.skip_prepare = skip_prepare
        args.skip_split = skip_split

    # 4. 训练参数 (如果涉及训练)
    if workflow in ["train", "train-all", "full"]:
        configure_training = questionary.confirm(
            "配置训练参数? (否则使用默认值)",
            default=False,
            style=custom_style
        ).ask()

        if configure_training:
            # 模型选择
            if task == "classify":
                model_choices = [
                    questionary.Choice("YOLO26x-cls (最大，精度优先) [默认]", value="yolo26x-cls.pt"),
                    questionary.Choice("YOLO26l-cls (大型)", value="yolo26l-cls.pt"),
                    questionary.Choice("YOLO26m-cls (中型)", value="yolo26m-cls.pt"),
                    questionary.Choice("YOLO26s-cls (小型)", value="yolo26s-cls.pt"),
                    questionary.Choice("YOLO26n-cls (最小，速度优先)", value="yolo26n-cls.pt"),
                    questionary.Choice("使用配置文件默认值", value=None),
                ]
                model_prompt = "选择机型分类模型:"
            elif task == "airline":
                model_choices = [
                    questionary.Choice("YOLO26m-cls (中型，平衡) [默认]", value="yolo26m-cls.pt"),
                    questionary.Choice("YOLO26x-cls (最大，精度优先)", value="yolo26x-cls.pt"),
                    questionary.Choice("YOLO26l-cls (大型)", value="yolo26l-cls.pt"),
                    questionary.Choice("YOLO26s-cls (小型)", value="yolo26s-cls.pt"),
                    questionary.Choice("YOLO26n-cls (最小，速度优先)", value="yolo26n-cls.pt"),
                    questionary.Choice("使用配置文件默认值", value=None),
                ]
                model_prompt = "选择航司识别模型:"
            elif task == "detection":
                model_choices = [
                    questionary.Choice("YOLO26x (最大，精度优先) [默认]", value="yolo26x.pt"),
                    questionary.Choice("YOLO26l (大型)", value="yolo26l.pt"),
                    questionary.Choice("YOLO26m (中型)", value="yolo26m.pt"),
                    questionary.Choice("YOLO26s (小型)", value="yolo26s.pt"),
                    questionary.Choice("YOLO26n (最小，速度优先)", value="yolo26n.pt"),
                    questionary.Choice("使用配置文件默认值", value=None),
                ]
                model_prompt = "选择注册号检测模型:"
            else:
                model_choices = [questionary.Choice("使用配置文件默认值", value=None)]
                model_prompt = "选择模型:"

            model = questionary.select(
                model_prompt,
                choices=model_choices,
                style=custom_style
            ).ask()

            epochs = questionary.text(
                "训练轮数 (epochs):",
                default="100",
                validate=lambda x: x.isdigit() and int(x) > 0,
                style=custom_style
            ).ask()

            batch_size = questionary.text(
                "批次大小 (batch-size):",
                default="32",
                validate=lambda x: x.isdigit() and int(x) > 0,
                style=custom_style
            ).ask()

            device = questionary.select(
                "训练设备:",
                choices=[
                    questionary.Choice("GPU 0", value="0"),
                    questionary.Choice("GPU 1", value="1"),
                    questionary.Choice("CPU", value="cpu"),
                ],
                style=custom_style
            ).ask()

            args.model = model
            args.epochs = int(epochs) if epochs else None
            args.batch_size = int(batch_size) if batch_size else None
            args.device = device

    # 5. 其他选项
    if workflow in ["train-all", "full"]:
        continue_on_error = questionary.confirm(
            "出错时继续执行?",
            default=False,
            style=custom_style
        ).ask()

        args.continue_on_error = continue_on_error

    verbose = questionary.confirm(
        "显示详细输出?",
        default=False,
        style=custom_style
    ).ask()

    args.verbose = verbose

    # 显示配置摘要
    if console:
        table = Table(title="配置摘要", box=box.ROUNDED, border_style="cyan")
        table.add_column("参数", style="cyan", no_wrap=True)
        table.add_column("值", style="green")

        table.add_row("工作流", workflow)
        if args.task:
            table.add_row("任务", args.task)
        if hasattr(args, 'model') and args.model:
            table.add_row("模型", args.model)
        elif hasattr(args, 'model') and args.model is None:
            table.add_row("模型", "配置文件默认值")
        if args.epochs:
            table.add_row("训练轮数", str(args.epochs))
        if args.batch_size:
            table.add_row("批次大小", str(args.batch_size))
        if args.device:
            table.add_row("设备", args.device)
        table.add_row("详细输出", "是" if args.verbose else "否")

        console.print()
        console.print(table)
        console.print()

    # 确认执行
    confirm = questionary.confirm(
        "开始执行?",
        default=True,
        style=custom_style
    ).ask()

    if not confirm:
        if console:
            console.print("[yellow]已取消[/yellow]")
        else:
            print("已取消")
        sys.exit(0)

    return args


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="AeroVision 训练流程执行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流程
  python run_workflow.py --workflow full

  # 只准备数据
  python run_workflow.py --workflow prepare

  # 只训练分类模型
  python run_workflow.py --workflow train --task classify --epochs 100

  # 训练所有模型
  python run_workflow.py --workflow train-all

  # 自定义参数
  python run_workflow.py --workflow train --task classify \\
      --epochs 100 --batch-size 32 --device 0
        """
    )

    # 工作流选择
    parser.add_argument(
        "--workflow",
        type=str,
        required=False,
        choices=["full", "verify", "prepare", "train", "train-all"],
        help="工作流类型"
    )

    # TUI 模式
    parser.add_argument(
        "--tui",
        action="store_true",
        help="使用交互式 TUI 界面"
    )

    # 任务选择 (用于 train workflow)
    parser.add_argument(
        "--task",
        type=str,
        choices=["classify", "airline", "detection"],
        help="训练任务 (用于 train workflow)"
    )

    # 数据准备参数
    parser.add_argument(
        "--labels",
        type=str,
        help="标注文件路径"
    )
    parser.add_argument(
        "--images",
        type=str,
        help="图片目录路径"
    )
    parser.add_argument(
        "--prepare-dir",
        type=str,
        help="准备好的数据目录"
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="跳过数据准备步骤"
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="跳过数据划分步骤"
    )

    # 训练参数
    parser.add_argument(
        "--epochs",
        type=int,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="批次大小"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        help="图片尺寸"
    )
    parser.add_argument(
        "--lr0",
        type=float,
        help="初始学习率"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="训练设备 (0, 1, cpu)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="预训练模型路径"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="恢复训练的检查点路径"
    )

    # 其他选项
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细输出"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="出错时继续执行 (用于 full/train-all workflow)"
    )

    return parser.parse_args()


def main():
    """主函数"""
    # 检查是否使用 TUI 模式
    if "--tui" in sys.argv or (len(sys.argv) == 1 and HAS_QUESTIONARY):
        # 交互式 TUI 模式
        args = interactive_workflow()
        use_tui = True
    else:
        # 命令行模式
        args = parse_arguments()
        use_tui = False

        # 检查 workflow 参数
        if not args.workflow:
            print("错误: 需要指定 --workflow 参数或使用 --tui 进入交互式模式")
            print("运行 'python run_workflow.py --help' 查看帮助")
            sys.exit(1)

    # 创建工作流执行器
    runner = WorkflowRunner(verbose=args.verbose, use_tui=use_tui)

    # 执行工作流
    success = False

    if args.workflow == "verify":
        success = runner.verify_environment()

    elif args.workflow == "prepare":
        success = runner.prepare_data(args)

    elif args.workflow == "train":
        if not args.task:
            runner.logger.error("train workflow 需要指定 --task 参数")
            sys.exit(1)
        success = runner.train_model(args)

    elif args.workflow == "train-all":
        # 训练所有模型
        tasks = ["classify", "airline", "detection"]
        success = True

        for task in tasks:
            runner.logger.info(f"开始训练: {task}")
            args.task = task

            if not runner.train_model(args):
                runner.logger.error(f"训练 {task} 失败")
                if not args.continue_on_error:
                    success = False
                    break
                else:
                    runner.logger.warning(f"跳过 {task}，继续下一个任务")

    elif args.workflow == "full":
        success = runner.run_full_workflow(args)

    # 退出
    if success:
        if runner.use_tui and runner.console:
            runner.console.print(Panel.fit(
                "[bold green]✓ 工作流执行成功[/bold green]",
                border_style="green"
            ))
        else:
            runner.logger.info("工作流执行成功")
        sys.exit(0)
    else:
        if runner.use_tui and runner.console:
            runner.console.print(Panel.fit(
                "[bold red]✗ 工作流执行失败[/bold red]",
                border_style="red"
            ))
        else:
            runner.logger.error("工作流执行失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
