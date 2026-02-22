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


class WorkflowRunner:
    """训练流程执行器"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.training_root = Path(__file__).parent
        self.scripts_dir = self.training_root / "scripts"
        self.logger = self._setup_logger()

        # 检测操作系统
        self.is_windows = sys.platform.startswith('win')
        self.python_cmd = "python" if self.is_windows else "python3"

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

            self.logger.info(f"[成功] {description or '命令执行成功'}")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"[失败] {description or '命令执行失败'}")
            if e.stdout:
                self.logger.error(f"标准输出:\n{e.stdout}")
            if e.stderr:
                self.logger.error(f"错误输出:\n{e.stderr}")
            return False

        except Exception as e:
            self.logger.error(f"[异常] {description}: {e}")
            return False

    def verify_environment(self) -> bool:
        """验证环境"""
        self.logger.info("=" * 70)
        self.logger.info("验证环境")
        self.logger.info("=" * 70)

        # 检查 Python 版本
        python_version = sys.version_info
        self.logger.info(f"Python 版本: {python_version.major}.{python_version.minor}.{python_version.micro}")

        if python_version < (3, 11):
            self.logger.warning("建议使用 Python 3.11+")

        # 验证配置系统
        verify_script = self.training_root / "verify_configs.py"
        if verify_script.exists():
            success = self.run_command(
                [self.python_cmd, str(verify_script)],
                description="验证配置系统"
            )
            if not success:
                self.logger.error("配置验证失败，请检查配置文件")
                return False
        else:
            self.logger.warning("未找到配置验证脚本")

        self.logger.info("")
        return True

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
        required=True,
        choices=["full", "verify", "prepare", "train", "train-all"],
        help="工作流类型"
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
    args = parse_arguments()

    # 创建工作流执行器
    runner = WorkflowRunner(verbose=args.verbose)

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
        runner.logger.info("工作流执行成功")
        sys.exit(0)
    else:
        runner.logger.error("工作流执行失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
