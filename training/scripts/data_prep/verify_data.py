#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集验证脚本 - 验证数据集质量

功能：
1. 检查标注文件格式是否正确
2. 检查图片文件是否存在
3. 检查清晰度和遮挡评分是否在有效范围内
4. 统计每个类别的样本数量
5. 检查数据集划分是否合理
6. 生成数据质量报告

Usage:
    # 基本用法
    python verify_data.py

    # 验证特定数据集
    python verify_data.py --labels training/data/labels/aircraft_labels.csv --images training/data/processed/aircraft_crop
"""

import argparse
import logging
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataVerifier:
    """数据集验证类，用于检查数据集质量"""

    def __init__(
        self,
        labels_file: str,
        images_dir: str,
        check_splits: bool = True
    ) -> None:
        """
        初始化数据集验证器

        Args:
            labels_file: 标注文件路径
            images_dir: 图片目录路径
            check_splits: 是否检查数据集划分
        """
        self.labels_file = Path(labels_file)
        self.images_dir = Path(images_dir)
        self.check_splits = check_splits

        # 验证结果
        self.issues = []
        self.warnings = []
        self.info = []

        # 数据质量报告
        self.quality_report = {
            'timestamp': datetime.now().isoformat(),
            'labels_file': str(self.labels_file),
            'images_dir': str(self.images_dir),
            'total_samples': 0,
            'valid_samples': 0,
            'invalid_samples': 0,
            'missing_images': 0,
            'format_errors': 0,
            'score_errors': 0,
            'class_distribution': {},
            'quality_scores': {
                'clarity': {'valid': 0, 'invalid': 0, 'distribution': {}},
                'block': {'valid': 0, 'invalid': 0, 'distribution': {}}
            },
            'issues': [],
            'warnings': [],
            'split_info': {}
        }

    def log_issue(self, message: str) -> None:
        """记录问题"""
        self.issues.append(message)
        self.quality_report['issues'].append(message)
        logger.error(f"[问题] {message}")

    def log_warning(self, message: str) -> None:
        """记录警告"""
        self.warnings.append(message)
        self.quality_report['warnings'].append(message)
        logger.warning(f"[警告] {message}")

    def log_info(self, message: str) -> None:
        """记录信息"""
        self.info.append(message)
        logger.info(f"[信息] {message}")

    def load_labels(self) -> Optional[pd.DataFrame]:
        """
        加载标注文件

        Returns:
            标注数据 DataFrame，如果加载失败则返回 None
        """
        logger.info(f"加载标注文件: {self.labels_file}")

        if not self.labels_file.exists():
            self.log_issue(f"标注文件不存在: {self.labels_file}")
            return None

        try:
            df = pd.read_csv(self.labels_file)
            self.quality_report['total_samples'] = len(df)
            logger.info(f"成功加载 {len(df)} 条标注记录")
            return df

        except Exception as e:
            self.log_issue(f"加载标注文件失败: {e}")
            return None

    def check_format(self, df: pd.DataFrame) -> bool:
        """
        检查标注文件格式

        Args:
            df: 标注数据

        Returns:
            格式是否正确
        """
        logger.info("检查标注文件格式...")

        # 检查必需的列
        required_columns = ['filename', 'typename', 'clarity', 'block']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            self.log_issue(f"标注文件缺少必需的列: {missing_columns}")
            self.quality_report['format_errors'] += len(missing_columns)
            return False

        # 检查空值
        for col in required_columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                self.log_issue(f"列 '{col}' 有 {null_count} 个空值")
                self.quality_report['format_errors'] += null_count

        # 检查文件名格式
        invalid_filenames = df[~df['filename'].str.match(r'^.+\.(jpg|jpeg|png|bmp|tiff|webp)$', case=False, na=False)]
        if len(invalid_filenames) > 0:
            self.log_issue(f"发现 {len(invalid_filenames)} 个无效的文件名")
            self.quality_report['format_errors'] += len(invalid_filenames)

        self.log_info(f"格式检查完成: 共 {len(df)} 条记录")
        return True

    def check_images_exist(self, df: pd.DataFrame) -> None:
        """
        检查图片文件是否存在

        Args:
            df: 标注数据
        """
        logger.info("检查图片文件是否存在...")

        missing_images = []

        for idx, row in df.iterrows():
            filename = row['filename']
            image_path = self.images_dir / filename

            if not image_path.exists():
                missing_images.append(filename)

        if missing_images:
            self.log_issue(f"发现 {len(missing_images)} 个缺失的图片文件")
            self.quality_report['missing_images'] = len(missing_images)

            # 只显示前10个
            for filename in missing_images[:10]:
                self.log_warning(f"缺失图片: {filename}")
            if len(missing_images) > 10:
                self.log_warning(f"... 还有 {len(missing_images) - 10} 个缺失图片")
        else:
            self.log_info("所有图片文件都存在")

    def check_scores(self, df: pd.DataFrame) -> None:
        """
        检查清晰度和遮挡评分是否在有效范围内

        Args:
            df: 标注数据
        """
        logger.info("检查评分有效性...")

        # 检查清晰度评分 (0.0-1.0)
        invalid_clarity = df[(df['clarity'] < 0.0) | (df['clarity'] > 1.0)]
        if len(invalid_clarity) > 0:
            self.log_issue(f"发现 {len(invalid_clarity)} 个无效的清晰度评分")
            self.quality_report['score_errors'] += len(invalid_clarity)
        else:
            self.quality_report['quality_scores']['clarity']['valid'] = len(df)

        # 检查遮挡评分 (0.0-1.0)
        invalid_block = df[(df['block'] < 0.0) | (df['block'] > 1.0)]
        if len(invalid_block) > 0:
            self.log_issue(f"发现 {len(invalid_block)} 个无效的遮挡评分")
            self.quality_report['score_errors'] += len(invalid_block)
        else:
            self.quality_report['quality_scores']['block']['valid'] = len(df)

        # 统计评分分布
        self.quality_report['quality_scores']['clarity']['distribution'] = self._get_score_distribution(df['clarity'])
        self.quality_report['quality_scores']['block']['distribution'] = self._get_score_distribution(df['block'])

        self.log_info("评分检查完成")

    def _get_score_distribution(self, series: pd.Series) -> Dict[str, int]:
        """
        获取评分分布

        Args:
            series: 评分序列

        Returns:
            评分分布字典
        """
        distribution = {
            '0.0-0.2': 0,
            '0.2-0.4': 0,
            '0.4-0.6': 0,
            '0.6-0.8': 0,
            '0.8-1.0': 0
        }

        for score in series:
            if 0.0 <= score < 0.2:
                distribution['0.0-0.2'] += 1
            elif 0.2 <= score < 0.4:
                distribution['0.2-0.4'] += 1
            elif 0.4 <= score < 0.6:
                distribution['0.4-0.6'] += 1
            elif 0.6 <= score < 0.8:
                distribution['0.6-0.8'] += 1
            elif 0.8 <= score <= 1.0:
                distribution['0.8-1.0'] += 1

        return distribution

    def check_class_distribution(self, df: pd.DataFrame) -> None:
        """
        统计每个类别的样本数量

        Args:
            df: 标注数据
        """
        logger.info("统计类别分布...")

        # 按类别统计
        class_counts = df['typename'].value_counts().to_dict()

        # 计算统计信息
        total_samples = len(df)
        num_classes = len(class_counts)
        avg_samples = total_samples / num_classes if num_classes > 0 else 0

        self.quality_report['class_distribution'] = {
            'num_classes': num_classes,
            'total_samples': total_samples,
            'avg_samples_per_class': avg_samples,
            'min_samples': min(class_counts.values()) if class_counts else 0,
            'max_samples': max(class_counts.values()) if class_counts else 0,
            'classes': class_counts
        }

        self.log_info(f"共 {num_classes} 个类别")
        self.log_info(f"平均每个类别 {avg_samples:.1f} 个样本")

        # 检查类别不平衡
        if num_classes > 0:
            min_count = min(class_counts.values())
            max_count = max(class_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

            if imbalance_ratio > 10:
                self.log_warning(f"类别不平衡严重: 最大/最小样本比 = {imbalance_ratio:.1f}")
            elif imbalance_ratio > 5:
                self.log_warning(f"类别存在不平衡: 最大/最小样本比 = {imbalance_ratio:.1f}")

            # 检查样本数过少的类别
            min_samples_threshold = 10
            rare_classes = {cls: count for cls, count in class_counts.items() if count < min_samples_threshold}
            if rare_classes:
                self.log_warning(f"发现 {len(rare_classes)} 个样本数少于 {min_samples_threshold} 的类别")
                for cls, count in rare_classes.items():
                    self.log_warning(f"  {cls}: {count} 个样本")

        # 打印每个类别的样本数
        logger.info("\n各类别样本数:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_samples * 100
            logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")

    def check_splits(self) -> None:
        """检查数据集划分是否合理"""
        if not self.check_splits:
            return

        logger.info("检查数据集划分...")

        labels_dir = self.labels_file.parent

        # 检查划分后的 CSV 文件
        split_files = {
            'train': labels_dir / 'train.csv',
            'val': labels_dir / 'val.csv',
            'test': labels_dir / 'test.csv'
        }

        split_data = {}
        for split_name, file_path in split_files.items():
            if file_path.exists():
                split_data[split_name] = pd.read_csv(file_path)
                self.log_info(f"找到 {split_name} 数据集: {len(split_data[split_name])} 条记录")
            else:
                self.log_warning(f"未找到 {split_name} 数据集: {file_path}")

        # 检查划分比例
        if len(split_data) == 3:
            total = sum(len(df) for df in split_data.values())
            train_ratio = len(split_data['train']) / total
            val_ratio = len(split_data['val']) / total
            test_ratio = len(split_data['test']) / total

            self.quality_report['split_info'] = {
                'train': {'count': len(split_data['train']), 'ratio': train_ratio},
                'val': {'count': len(split_data['val']), 'ratio': val_ratio},
                'test': {'count': len(split_data['test']), 'ratio': test_ratio},
                'total': total
            }

            self.log_info(f"划分比例: 训练集 {train_ratio*100:.1f}%, "
                         f"验证集 {val_ratio*100:.1f}%, "
                         f"测试集 {test_ratio*100:.1f}%")

            # 检查每个类别在各个子集中是否有样本
            self._check_split_class_coverage(split_data)

    def _check_split_class_coverage(self, split_data: Dict[str, pd.DataFrame]) -> None:
        """
        检查每个类别在各个子集中是否有样本

        Args:
            split_data: 各子集数据
        """
        logger.info("检查类别覆盖情况...")

        # 获取所有类别
        all_classes = set()
        for df in split_data.values():
            all_classes.update(df['typename'].unique())

        # 检查每个类别在各个子集中的覆盖情况
        missing_classes = defaultdict(list)

        for class_name in all_classes:
            for split_name, df in split_data.items():
                if class_name not in df['typename'].values:
                    missing_classes[split_name].append(class_name)

        # 报告缺失的类别
        for split_name, classes in missing_classes.items():
            if classes:
                self.log_warning(f"{split_name} 数据集缺少 {len(classes)} 个类别")
                for class_name in classes[:5]:
                    self.log_warning(f"  缺失类别: {class_name}")
                if len(classes) > 5:
                    self.log_warning(f"  ... 还有 {len(classes) - 5} 个类别")

    def generate_report(self) -> None:
        """生成数据质量报告"""
        logger.info("生成数据质量报告...")

        # 计算有效样本数
        self.quality_report['valid_samples'] = (
            self.quality_report['total_samples'] -
            self.quality_report['missing_images'] -
            self.quality_report['format_errors'] -
            self.quality_report['score_errors']
        )
        self.quality_report['invalid_samples'] = (
            self.quality_report['total_samples'] -
            self.quality_report['valid_samples']
        )

        # 保存报告
        report_file = self.labels_file.parent / f'data_quality_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.quality_report, f, ensure_ascii=False, indent=2)

        logger.info(f"数据质量报告已保存: {report_file}")

    def print_summary(self) -> None:
        """打印验证摘要"""
        logger.info("=" * 60)
        logger.info("数据集验证摘要")
        logger.info("=" * 60)

        logger.info(f"总样本数: {self.quality_report['total_samples']}")
        logger.info(f"有效样本数: {self.quality_report['valid_samples']}")
        logger.info(f"无效样本数: {self.quality_report['invalid_samples']}")

        if self.quality_report['missing_images'] > 0:
            logger.warning(f"缺失图片: {self.quality_report['missing_images']}")
        if self.quality_report['format_errors'] > 0:
            logger.warning(f"格式错误: {self.quality_report['format_errors']}")
        if self.quality_report['score_errors'] > 0:
            logger.warning(f"评分错误: {self.quality_report['score_errors']}")

        logger.info(f"问题数: {len(self.issues)}")
        logger.info(f"警告数: {len(self.warnings)}")

        if self.quality_report['split_info']:
            split_info = self.quality_report['split_info']
            logger.info(f"\n数据集划分:")
            logger.info(f"  训练集: {split_info['train']['count']} ({split_info['train']['ratio']*100:.1f}%)")
            logger.info(f"  验证集: {split_info['val']['count']} ({split_info['val']['ratio']*100:.1f}%)")
            logger.info(f"  测试集: {split_info['test']['count']} ({split_info['test']['ratio']*100:.1f}%)")

        logger.info("=" * 60)

        # 最终结论
        if len(self.issues) == 0:
            logger.info("✓ 数据集验证通过，没有发现严重问题")
        else:
            logger.error(f"✗ 数据集验证失败，发现 {len(self.issues)} 个问题")

    def verify(self) -> bool:
        """
        执行完整的数据集验证流程

        Returns:
            验证是否通过
        """
        logger.info("=" * 60)
        logger.info("开始数据集验证")
        logger.info("=" * 60)

        # 1. 加载标注文件
        df = self.load_labels()
        if df is None:
            return False

        # 2. 检查格式
        if not self.check_format(df):
            return False

        # 3. 检查图片是否存在
        self.check_images_exist(df)

        # 4. 检查评分
        self.check_scores(df)

        # 5. 检查类别分布
        self.check_class_distribution(df)

        # 6. 检查数据集划分
        self.check_splits()

        # 7. 生成报告
        self.generate_report()

        # 8. 打印摘要
        self.print_summary()

        logger.info("=" * 60)
        logger.info("数据集验证完成!")
        logger.info("=" * 60)

        return len(self.issues) == 0


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="验证数据集质量",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--labels',
        type=str,
        default='training/data/labels/aircraft_labels.csv',
        help='标注文件路径'
    )
    parser.add_argument(
        '--images',
        type=str,
        default='training/data/processed/aircraft_crop',
        help='图片目录路径'
    )
    parser.add_argument(
        '--no-splits',
        action='store_true',
        help='不检查数据集划分'
    )

    args = parser.parse_args()

    # 创建数据集验证器
    verifier = DataVerifier(
        labels_file=args.labels,
        images_dir=args.images,
        check_splits=not args.no_splits
    )

    # 执行验证
    success = verifier.verify()

    # 返回退出码
    import sys
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
