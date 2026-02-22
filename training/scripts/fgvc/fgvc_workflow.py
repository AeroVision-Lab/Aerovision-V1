#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FGVC_Aircraft 数据集一键准备脚本

本脚本自动化 FGVC_Aircraft 数据集的准备流程，包括：
1. 转换 FGVC 数据集格式
2. 准备数据集（裁剪、缩放等预处理）
3. 划分数据集为训练/验证/测试集

训练步骤由 train_classify.py 完成
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_step(step_name: str, cmd: List[str], timeout: int = 300) -> Tuple[bool, str]:
    """
    运行单个步骤

    Args:
        step_name: 步骤名称
        cmd: 命令列表
        timeout: 超时时间（秒）

    Returns:
        (是否成功, 输出/错误信息)
    """
    logger.info(f"步骤: {step_name}")
    logger.info(f"命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=timeout
        )

        logger.info(f"✓ {step_name} 完成")
        return True, result.stdout

    except subprocess.TimeoutExpired:
        logger.error(f"✗ {step_name} 超时")
        return False, "Timeout"
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {step_name} 失败")
        logger.error(f"错误: {e.stderr}")
        return False, e.stderr
    except Exception as e:
        logger.error(f"✗ {step_name} 异常: {str(e)}")
        return False, str(e)


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="FGVC_Aircraft 数据集一键准备脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行完整数据准备流程
  python fgvc_workflow.py --fgvc-dir path/to/FGVC_Aircraft/raw --output-base data/fgvc

  # 只跳过转换步骤
  python fgvc_workflow.py --fgvc-dir path/to/FGVC_Aircraft/raw --output-base data/fgvc --skip-convert

  # 准备完成后，使用以下命令进行训练:
  #   python train_classify.py --data data/fgvc/splits/latest/aerovision/aircraft --epochs 100
        """,
    )

    parser.add_argument(
        "--fgvc-dir", type=str, required=True, help="FGVC_Aircraft数据目录 (raw)"
    )
    parser.add_argument("--output-base", type=str, required=True, help="输出基础目录")
    parser.add_argument("--skip-convert", action="store_true", help="跳过FGVC数据转换")
    parser.add_argument("--skip-prepare", action="store_true", help="跳过数据准备阶段")
    parser.add_argument("--skip-split", action="store_true", help="跳过数据划分阶段")

    # 数据过滤参数
    parser.add_argument(
        "--types",
        type=str,
        default="all",
        help="包含的机型列表，逗号分隔，如 '737-800,A320' (默认: all 表示包含所有机型)"
    )
    parser.add_argument(
        "--airlines",
        type=str,
        default="all",
        help="包含的航司/制造商列表，逗号分隔，如 'Boeing,Airbus' (默认: all 表示包含所有航司)"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("FGVC_Aircraft 数据集准备")
    logger.info("=" * 60)
    logger.info(f"FGVC目录: {args.fgvc_dir}")
    logger.info(f"输出基础: {args.output_base}")
    logger.info(f"包含机型: {args.types}")
    logger.info(f"包含航司: {args.airlines}")
    logger.info("=" * 60)
    logger.info("")

    # 脚本目录
    scripts_dir = Path(__file__).parent

    # 定义路径
    converted_base = Path(args.output_base) / "fgvc_converted"
    prepared_base = Path(args.output_base) / "fgvc_prepared"
    splits_base = Path(args.output_base) / "fgvc_splits"

    # 阶段 1: 转换 FGVC 数据集
    if not args.skip_convert:
        logger.info("")
        logger.info("=" * 60)
        logger.info("阶段 1: 转换 FGVC 数据集")
        logger.info("=" * 60)

        # 转换三个分割集
        for split in ["train", "val", "test"]:
            output_dir = str(converted_base / split)
            cmd = [
                sys.executable,
                str(scripts_dir / "convert_fgvc_aerocraft.py"),
                "--fgvc-dir",
                args.fgvc_dir,
                "--output",
                output_dir,
                "--split",
                split,
                "--types",
                args.types,
                "--airlines",
                args.airlines,
            ]

            success, output = run_step(f"转换 {split} 集", cmd, timeout=300)

            if not success:
                logger.error(f"转换失败，停止工作流")
                return

        # 合并分割集
        cmd = [
            sys.executable,
            str(scripts_dir / "merge_fgvc_splits.py"),
            "--train",
            str(converted_base / "train"),
            "--val",
            str(converted_base / "val"),
            "--test",
            str(converted_base / "test"),
            "--output",
            str(converted_base / "combined"),
        ]

        success, output = run_step("合并分割集", cmd, timeout=300)

        if not success:
            logger.error(f"合并失败，停止工作流")
            return

    # 阶段 2: 准备数据集
    if not args.skip_prepare:
        logger.info("")
        logger.info("=" * 60)
        logger.info("阶段 2: 准备数据集 (prepare_dataset.py)")
        logger.info("=" * 60)

        cmd = [
            sys.executable,
            str(scripts_dir.parent / "data_prep" / "prepare_dataset.py"),
            "--labels",
            str(converted_base / "combined" / "labels.csv"),
            "--images",
            str(converted_base / "combined" / "images"),
            "--output",
            str(prepared_base),
        ]

        success, output = run_step("准备数据集", cmd, timeout=600)

        if not success:
            logger.error(f"准备失败，停止工作流")
            return

    # 阶段 3: 划分数据集
    latest_splits = None

    if not args.skip_split:
        logger.info("")
        logger.info("=" * 60)
        logger.info("阶段 3: 划分数据集 (split_dataset.py)")
        logger.info("=" * 60)

        # 读取最新的prepared目录
        latest_txt = prepared_base / "latest.txt"
        if latest_txt.exists():
            with open(latest_txt, "r", encoding="utf-8") as f:
                prepare_dir = Path(f.read().strip())
        else:
            logger.error(f"找不到latest.txt: {latest_txt}")
            return

        cmd = [
            sys.executable,
            str(scripts_dir.parent / "data_prep" / "split_dataset.py"),
            "--prepare-dir",
            str(prepare_dir),
            "--output",
            str(splits_base),
            "--mode",
            "all",
        ]

        success, output = run_step("划分数据集", cmd, timeout=600)

        if not success:
            logger.error(f"划分失败，停止工作流")
            return

    # 读取最新的splits目录（无论是否跳过划分，都需要找到 splits 目录）
    if not splits_base.exists():
        logger.error(f"找不到splits目录: {splits_base}")
        return

    timestamp_dirs = [d for d in splits_base.iterdir() if d.is_dir()]
    if not timestamp_dirs:
        logger.error(f"splits目录为空: {splits_base}")
        return

    latest_splits = sorted(
        timestamp_dirs, key=lambda x: x.stat().st_mtime, reverse=True
    )[0]
    logger.info(f"使用splits目录: {latest_splits}")

    # 阶段 4: 输出训练命令
    logger.info("")
    logger.info("=" * 60)
    logger.info("数据集准备完成!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("现在可以使用 train_classify.py 进行训练")
    logger.info("")
    logger.info("训练机型分类:")
    logger.info(
        f"  python train_classify.py --data {latest_splits}/aerovision/aircraft --epochs 100"
    )
    logger.info("")
    logger.info("训练航司分类:")
    logger.info(
        f"  python train_classify.py --data {latest_splits}/aerovision/airline --epochs 100"
    )
    logger.info("")


if __name__ == "__main__":
    sys.exit(main())
