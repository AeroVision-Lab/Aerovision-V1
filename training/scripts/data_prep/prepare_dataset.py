#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据准备脚本 - 验证和预处理原始标注数据

设计理念：
-----------
本脚本是数据处理流程的第一步，遵循"先准备再划分"的原则：
1. 先运行本脚本，验证数据完整性并进行预处理
2. 再运行 split_dataset.py，读取准备好的数据进行划分

配置说明：
-----------
本脚本使用模块化配置系统，自动加载以下配置模块：
- paths.yaml: 路径配置 (labels.*, data.*)
- base.yaml: 基础配置 (seed.*)

功能：
------
1. 从 aircraft_labels.csv 读取原始标注
2. 验证数据完整性
   - 检查图片文件是否存在
   - 检查必填字段是否完整
   - 检查标注格式是否正确
3. 预处理
   - 过滤无效记录（图片缺失、标注不完整）
   - 去重（相同 filename）
4. 输出干净的数据集
   - aircraft_labels_clean.csv
   - 复制有效图片
   - 复制有效的注册号标注（如有）
   - 生成验证报告

使用方法：
-----------
  # 使用默认配置
  python prepare_dataset.py

  # 指定标注文件和图片目录
  python prepare_dataset.py --labels path/to/labels.csv --images path/to/images

  # 使用自定义配置
  python prepare_dataset.py --config my_config.yaml
"""

import argparse
import json
import logging
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# 添加 configs 模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetPreparer:
    """数据集准备类 - 验证和预处理原始标注数据"""

    # 必填字段
    REQUIRED_COLUMNS = ['filename', 'typename', 'clarity', 'block']
    # 支持的图片格式
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(
        self,
        labels_file: str,
        images_dir: str,
        registration_dir: Optional[str] = None,
        output_root: Optional[str] = None
    ) -> None:
        """
        初始化数据准备器

        Args:
            labels_file: 原始标注文件路径
            images_dir: 图片目录路径
            registration_dir: 注册号标注目录（YOLO格式txt文件）
            output_root: 输出根目录
        """
        self.labels_file = Path(labels_file)
        self.images_dir = Path(images_dir)
        self.registration_dir = Path(registration_dir) if registration_dir else None
        self.output_root = Path(output_root) if output_root else self.labels_file.parent / 'prepared'

        # 创建带时间戳的输出目录
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = self.output_root / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 输出路径
        self.output_csv = self.output_dir / 'aircraft_labels_clean.csv'
        self.output_images_dir = self.output_dir / 'images'
        self.output_registration_dir = self.output_dir / 'registration'
        self.output_report = self.output_dir / 'prepare_report.json'

        # 统计信息
        self.stats = {
            'total': 0,
            'valid': 0,
            'invalid': 0,
            'duplicates_removed': 0,
            'skip_reasons': defaultdict(int),
            'class_distribution': defaultdict(int),
            'airline_distribution': defaultdict(int),
            'has_airline_count': 0
        }

        logger.info(f"输出目录: {self.output_dir}")

    def load_labels(self) -> pd.DataFrame:
        """
        加载原始标注文件

        Returns:
            标注数据 DataFrame
        """
        logger.info(f"加载标注文件: {self.labels_file}")

        if not self.labels_file.exists():
            raise FileNotFoundError(f"标注文件不存在: {self.labels_file}")

        # 读取 CSV 文件（使用 utf-8-sig 自动处理 BOM）
        df = pd.read_csv(self.labels_file, encoding='utf-8-sig')

        # 清理列名（去除可能的空格）
        df.columns = df.columns.str.strip()

        self.stats['total'] = len(df)
        logger.info(f"加载了 {len(df)} 条标注记录")

        if len(df) == 0:
            logger.error("标注文件为空，无法准备数据集！")
            logger.error("可能的原因:")
            logger.error("  1. 机型过滤参数不正确（使用了错误的机型名称）")
            logger.error("  2. 航司过滤参数不正确")
            logger.error("  3. FGVC数据集中没有符合条件的数据")
            logger.error("")
            logger.error("提示: FGVC数据集使用简单名称，如 '747-100', '747-300'（无 'Boeing_' 前缀）")
            raise ValueError("标注文件为空，无法准备数据集")

        return df

    def validate_columns(self, df: pd.DataFrame) -> bool:
        """
        验证 DataFrame 是否包含所有必填列

        Args:
            df: 标注数据

        Returns:
            是否验证通过
        """
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(f"标注文件缺少必填列: {missing_columns}")

        logger.info(f"列验证通过，包含列: {list(df.columns)}")
        return True

    def validate_and_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        验证每条记录并过滤无效数据

        Args:
            df: 原始标注数据

        Returns:
            过滤后的有效数据
        """
        logger.info("开始验证数据...")

        valid_records = []
        invalid_indices = []

        for idx, row in df.iterrows():
            is_valid, reason = self._validate_record(row)
            if is_valid:
                valid_records.append(row)
            else:
                invalid_indices.append(idx)
                self.stats['skip_reasons'][reason] += 1

        self.stats['invalid'] = len(invalid_indices)

        if invalid_indices:
            logger.warning(f"发现 {len(invalid_indices)} 条无效记录")

        valid_df = pd.DataFrame(valid_records)
        logger.info(f"验证完成: {len(valid_df)} 条有效记录")

        return valid_df

    def _validate_record(self, row: pd.Series) -> Tuple[bool, str]:
        """
        验证单条记录

        Args:
            row: 单条标注记录

        Returns:
            (是否有效, 无效原因)
        """
        filename = row.get('filename')

        # 检查 filename 是否存在
        if pd.isna(filename) or not filename:
            return False, 'filename_missing'

        # 检查图片文件是否存在
        image_path = self.images_dir / filename
        if not image_path.exists():
            return False, 'image_not_found'

        # 检查文件扩展名
        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return False, 'unsupported_format'

        # 检查 typename 是否存在
        typename = row.get('typename')
        if pd.isna(typename) or not typename:
            return False, 'typename_missing'

        # 检查 clarity 是否有效
        clarity = row.get('clarity')
        if pd.isna(clarity):
            return False, 'clarity_missing'

        # 检查 block 是否有效
        block = row.get('block')
        if pd.isna(block):
            return False, 'block_missing'

        return True, ''

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        移除重复记录（基于 filename）

        Args:
            df: 标注数据

        Returns:
            去重后的数据
        """
        original_count = len(df)
        df_dedup = df.drop_duplicates(subset=['filename'], keep='first')
        duplicates = original_count - len(df_dedup)

        self.stats['duplicates_removed'] = duplicates

        if duplicates > 0:
            logger.info(f"移除 {duplicates} 条重复记录")

        return df_dedup

    def copy_valid_files(self, df: pd.DataFrame) -> None:
        """
        复制有效的图片和标注文件

        Args:
            df: 有效的标注数据
        """
        logger.info("复制有效文件...")

        # 创建输出目录
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        if self.registration_dir:
            self.output_registration_dir.mkdir(parents=True, exist_ok=True)

        copied_images = 0
        copied_labels = 0

        for _, row in df.iterrows():
            filename = row['filename']
            typename = row['typename']

            # 复制图片
            src_image = self.images_dir / filename
            dst_image = self.output_images_dir / filename

            if src_image.exists() and not dst_image.exists():
                shutil.copy2(src_image, dst_image)
                copied_images += 1

            # 统计类别分布
            self.stats['class_distribution'][typename] += 1

            # 统计航司分布（支持 'airline' 或 'airlinename' 列名）
            airline_column = None
            if 'airline' in row.index:
                airline_column = 'airline'
            elif 'airlinename' in row.index:
                airline_column = 'airlinename'

            if airline_column:
                airline = row.get(airline_column)
                if pd.notna(airline) and airline and str(airline).strip():
                    airline_name = str(airline).strip()
                    self.stats['airline_distribution'][airline_name] += 1
                    self.stats['has_airline_count'] += 1

            # 复制注册号标注（如果存在）
            if self.registration_dir:
                label_filename = Path(filename).stem + '.txt'
                src_label = self.registration_dir / label_filename
                dst_label = self.output_registration_dir / label_filename

                if src_label.exists() and not dst_label.exists():
                    shutil.copy2(src_label, dst_label)
                    copied_labels += 1

        logger.info(f"复制完成: {copied_images} 张图片, {copied_labels} 个标注文件")

    def save_clean_csv(self, df: pd.DataFrame) -> None:
        """
        保存清洗后的标注文件

        Args:
            df: 清洗后的标注数据
        """
        df.to_csv(self.output_csv, index=False, encoding='utf-8-sig')
        self.stats['valid'] = len(df)
        logger.info(f"已保存清洗后的标注: {self.output_csv} ({len(df)} 条)")

    def save_report(self) -> None:
        """保存验证报告"""
        report = {
            'timestamp': self.timestamp,
            'input': {
                'labels_file': str(self.labels_file),
                'images_dir': str(self.images_dir),
                'registration_dir': str(self.registration_dir) if self.registration_dir else None
            },
            'output': {
                'output_dir': str(self.output_dir),
                'clean_csv': str(self.output_csv),
                'images_dir': str(self.output_images_dir)
            },
            'statistics': {
                'total_records': self.stats['total'],
                'valid_records': self.stats['valid'],
                'invalid_records': self.stats['invalid'],
                'duplicates_removed': self.stats['duplicates_removed'],
                'skip_reasons': dict(self.stats['skip_reasons']),
                'num_classes': len(self.stats['class_distribution']),
                'class_distribution': dict(self.stats['class_distribution']),
                'num_airlines': len(self.stats['airline_distribution']),
                'has_airline_count': self.stats['has_airline_count'],
                'airline_distribution': dict(self.stats['airline_distribution'])
            }
        }

        with open(self.output_report, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"验证报告已保存: {self.output_report}")

    def print_summary(self) -> None:
        """打印处理摘要"""
        logger.info("=" * 60)
        logger.info("数据准备摘要")
        logger.info("=" * 60)
        logger.info(f"总记录数: {self.stats['total']}")
        logger.info(f"有效记录: {self.stats['valid']}")
        logger.info(f"无效记录: {self.stats['invalid']}")
        logger.info(f"移除重复: {self.stats['duplicates_removed']}")
        logger.info(f"机型类别数量: {len(self.stats['class_distribution'])}")
        logger.info(f"航司类别数量: {len(self.stats['airline_distribution'])}")
        logger.info(f"含航司标注: {self.stats['has_airline_count']}")

        if self.stats['skip_reasons']:
            logger.info("\n跳过原因统计:")
            for reason, count in sorted(self.stats['skip_reasons'].items()):
                logger.info(f"  {reason}: {count}")

        logger.info("=" * 60)
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("=" * 60)

    def prepare(self) -> Path:
        """
        执行完整的数据准备流程

        Returns:
            输出目录路径
        """
        logger.info("=" * 60)
        logger.info("开始数据准备")
        logger.info("=" * 60)

        # 1. 加载标注文件
        df = self.load_labels()

        # 2. 验证列
        self.validate_columns(df)

        # 3. 验证并过滤无效记录
        df = self.validate_and_filter(df)

        # 4. 去重
        df = self.remove_duplicates(df)

        # 5. 复制有效文件
        self.copy_valid_files(df)

        # 6. 保存清洗后的 CSV
        self.save_clean_csv(df)

        # 7. 保存验证报告
        self.save_report()

        # 8. 创建 latest 链接
        self._create_latest_link()

        # 9. 打印摘要
        self.print_summary()

        return self.output_dir

    def _create_latest_link(self) -> None:
        """创建 latest 软链接/目录引用"""
        latest_path = self.output_root / 'latest'

        # 在 Windows 上，如果没有管理员权限，软链接可能失败
        # 所以我们创建一个 latest.txt 文件来存储最新目录路径
        latest_txt = self.output_root / 'latest.txt'
        with open(latest_txt, 'w', encoding='utf-8') as f:
            f.write(str(self.output_dir))

        logger.info(f"最新目录已记录: {latest_txt}")


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="数据准备脚本 - 验证和预处理原始标注数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python prepare_dataset.py

  # 指定标注文件和图片目录
  python prepare_dataset.py --labels data/labels/aircraft_labels.csv --images data/processed/images

  # 使用自定义配置文件
  python prepare_dataset.py --config my_config.yaml

注意:
  本脚本是数据处理流程的第一步
  运行后请使用 split_dataset.py --prepare-dir <输出目录> 进行数据集划分
        """
    )

    parser.add_argument(
        '--labels',
        type=str,
        default=None,
        help='标注文件路径（默认从配置文件读取）'
    )
    parser.add_argument(
        '--images',
        type=str,
        default=None,
        help='图片目录路径（默认从配置文件读取）'
    )
    parser.add_argument(
        '--registration',
        type=str,
        default=None,
        help='注册号区域标注目录（默认从配置文件读取）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出根目录（默认为标注文件同目录下的 prepared/）'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='自定义配置文件路径'
    )

    args = parser.parse_args()

    # 加载配置
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config(modules=['paths'], load_all_modules=False)

    # 从配置文件获取路径（优先使用命令行参数）
    labels_file = args.labels or config.get('labels.main') or config.get('paths.main_labels')
    images_dir = args.images or config.get('data.processed.labeled.images') or config.get('paths.aircraft_crop_unsorted')
    registration_dir = args.registration or config.get('data.registration.area') or config.get('paths.registration_area')
    output_root = args.output or config.get('data.prepared.root')

    # 如果配置中的路径是相对路径，转换为绝对路径
    if labels_file and not Path(labels_file).is_absolute():
        labels_file = config.get_path('labels.main') or config.get_path('paths.main_labels')
    if images_dir and not Path(images_dir).is_absolute():
        images_dir = config.get_path('data.processed.labeled.images') or config.get_path('paths.aircraft_crop_unsorted')
    if registration_dir and not Path(registration_dir).is_absolute():
        registration_dir = config.get_path('data.registration.area') or config.get_path('paths.registration_area')
    if output_root and not Path(output_root).is_absolute():
        output_root = config.get_path('data.prepared.root')

    logger.info("=" * 60)
    logger.info("配置信息:")
    logger.info(f"  标注文件: {labels_file}")
    logger.info(f"  图片目录: {images_dir}")
    logger.info(f"  注册号标注目录: {registration_dir}")
    logger.info(f"  输出根目录: {output_root or '(默认)'}")
    logger.info("=" * 60)

    # 创建数据准备器
    preparer = DatasetPreparer(
        labels_file=str(labels_file),
        images_dir=str(images_dir),
        registration_dir=str(registration_dir) if registration_dir else None,
        output_root=str(output_root) if output_root else None
    )

    # 执行数据准备
    output_dir = preparer.prepare()

    logger.info("")
    logger.info("下一步:")
    logger.info(f"  python split_dataset.py --prepare-dir {output_dir}")


if __name__ == '__main__':
    main()
