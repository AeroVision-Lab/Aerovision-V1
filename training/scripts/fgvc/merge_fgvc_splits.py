#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 FGVC_Aircraft 转换后的数据集（train/val/test）
"""

import argparse
import logging
import shutil
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def merge_fgvc_splits(
    train_dir: str, val_dir: str, test_dir: str, output_dir: str
) -> None:
    """
    合并三个分割集为一个完整数据集

    Args:
        train_dir: 训练集目录
        val_dir: 验证集目录
        test_dir: 测试集目录
        output_dir: 输出目录
    """
    logger.info("=" * 60)
    logger.info("合并 FGVC_Aircraft 数据集")
    logger.info("=" * 60)

    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # 合并CSV
    logger.info("合并CSV文件...")
    train_csv = train_dir / "labels.csv"
    val_csv = val_dir / "labels.csv"
    test_csv = test_dir / "labels.csv"

    # 读取CSV
    train_df = pd.read_csv(train_csv, encoding="utf-8-sig")
    val_df = pd.read_csv(val_csv, encoding="utf-8-sig")
    test_df = pd.read_csv(test_csv, encoding="utf-8-sig")

    logger.info(f"  训练集: {len(train_df)} 条")
    logger.info(f"  验证集: {len(val_df)} 条")
    logger.info(f"  测试集: {len(test_df)} 条")

    # 合并
    merged_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    logger.info(f"  合并后: {len(merged_df)} 条")

    # 检查是否为空
    if len(merged_df) == 0:
        logger.error("合并后的数据集为空！")
        logger.error("可能的原因:")
        logger.error("  1. 机型过滤参数不正确（使用了错误的机型名称）")
        logger.error("  2. 航司过滤参数不正确")
        logger.error("  3. FGVC数据集中没有符合条件的数据")
        logger.error("")
        logger.error("提示: FGVC数据集使用简单名称，如 '747-100', '747-300'（无 'Boeing_' 前缀）")
        raise ValueError("合并后的数据集为空")

    # 统计信息
    num_variants = merged_df["typename"].nunique()
    num_airlines = merged_df["airline"].nunique()

    logger.info(f"  机型变体数: {num_variants}")
    logger.info(f"  制造商数: {num_airlines}")

    # 保存合并后的CSV
    output_csv = output_dir / "labels.csv"
    merged_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info(f"已保存合并后的CSV: {output_csv}")

    # 合并图片
    logger.info("合并图片...")
    copied = 0

    for split_dir in [train_dir, val_dir, test_dir]:
        split_images_dir = split_dir / "images"
        if not split_images_dir.exists():
            continue

        for image_file in split_images_dir.glob("*.jpg"):
            dst_path = output_images_dir / image_file.name
            if not dst_path.exists():
                shutil.copy2(image_file, dst_path)
                copied += 1

    logger.info(f"复制了 {copied} 张图片到输出目录")

    # 打印摘要
    logger.info("=" * 60)
    logger.info("合并完成!")
    logger.info("=" * 60)
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"总图片数: {copied}")
    logger.info(f"总标注数: {len(merged_df)}")
    logger.info(f"机型变体: {num_variants}")
    logger.info(f"制造商: {num_airlines}")
    logger.info("=" * 60)

    print("")
    print("下一步:")
    print(
        f"  python prepare_dataset.py --labels {output_csv} --images {output_images_dir}"
    )
    print(f"  python split_dataset.py --prepare-dir {output_dir}")


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="合并 FGVC_Aircraft 转换后的数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 合并三个分割集
  python merge_fgvc_splits.py --train path/to/train --val path/to/val --test path/to/test --output path/to/combined
        """,
    )

    parser.add_argument("--train", type=str, required=True, help="训练集目录")
    parser.add_argument("--val", type=str, required=True, help="验证集目录")
    parser.add_argument("--test", type=str, required=True, help="测试集目录")
    parser.add_argument("--output", type=str, required=True, help="输出目录")

    args = parser.parse_args()

    merge_fgvc_splits(args.train, args.val, args.test, args.output)


if __name__ == "__main__":
    main()
