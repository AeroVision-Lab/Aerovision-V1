#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集划分脚本 - 将准备好的数据划分为 train/val/test 并组织目录结构

设计理念：
-----------
本脚本是数据处理流程的第二步，遵循"先准备再划分"的原则：
1. 先运行 prepare_dataset.py，验证数据完整性并进行预处理
2. 再运行本脚本，读取准备好的数据进行划分和目录组织

配置说明：
-----------
本脚本使用模块化配置系统，自动加载以下配置模块：
- paths.yaml: 路径配置
- base.yaml: 基础配置 (seed.*)

功能：
------
1. 从 prepare_dataset.py 输出读取干净的标注数据
2. 按比例划分数据集（70%/15%/15%）
3. 确保每个类别在各个子集中都有样本
4. 组织成训练所需的目录结构
   - Aerovision 分类格式（按类别分目录）
   - Detection 检测格式（YOLO格式）
5. 生成类别映射文件

使用方法：
-----------
  # 使用 prepare_dataset.py 的输出
  python split_dataset.py --prepare-dir ../data/prepared/20250104_120000

  # 指定划分比例
  python split_dataset.py --prepare-dir path/to/prepared --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

  # 只生成分类数据集
  python split_dataset.py --prepare-dir path/to/prepared --mode aerovision

  # 只生成检测数据集
  python split_dataset.py --prepare-dir path/to/prepared --mode detection
"""

import argparse
import json
import logging
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import yaml

# 添加 configs 模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetSplitter:
    """数据集划分类 - 划分数据并组织目录结构"""

    def __init__(
        self,
        prepare_dir: str,
        output_root: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        mode: str = "all",
        random_seed: int = 42,
        min_type_samples: int = 10,
        min_airline_samples: int = 10
    ) -> None:
        """
        初始化数据集划分器

        Args:
            prepare_dir: prepare_dataset.py 输出的目录
            output_root: 输出根目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            mode: 输出模式 ("all", "aerovision", "airline", "detection")
            random_seed: 随机种子
            min_type_samples: 机型最小样本数（低于此数量的机型将被过滤）
            min_airline_samples: 航司最小样本数（低于此数量的航司将被过滤）
        """
        self.prepare_dir = Path(prepare_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.mode = mode
        self.random_seed = random_seed
        self.min_type_samples = min_type_samples
        self.min_airline_samples = min_airline_samples

        # 验证比例总和
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError(
                f"训练集、验证集、测试集比例之和必须为1.0，"
                f"当前为{train_ratio + val_ratio + test_ratio}"
            )

        # 验证 prepare_dir
        self._validate_prepare_dir()

        # 输入路径
        self.labels_file = self.prepare_dir / 'aircraft_labels_clean.csv'
        self.images_dir = self.prepare_dir / 'images'
        self.registration_dir = self.prepare_dir / 'registration'

        # 创建带时间戳的输出目录
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if output_root:
            self.output_root = Path(output_root)
        else:
            self.output_root = self.prepare_dir.parent.parent / 'splits'
        self.output_dir = self.output_root / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Aerovision 分类数据集路径
        self.aerovision_root = self.output_dir / "aerovision"
        self.aircraft_dir = self.aerovision_root / "aircraft"
        self.airline_dir = self.aerovision_root / "airline"
        self.labels_output_dir = self.aerovision_root / "labels"

        # Detection 检测数据集路径
        self.detection_root = self.output_dir / "detection"

        # 类别映射（机型）
        self.class_to_id: Dict[str, int] = {}
        self.id_to_class: Dict[int, str] = {}

        # 航司类别映射
        self.airline_to_id: Dict[str, int] = {}
        self.id_to_airline: Dict[int, str] = {}

        # 设置随机种子
        random.seed(random_seed)

        logger.info(f"输出目录: {self.output_dir}")

    def _validate_prepare_dir(self) -> None:
        """验证 prepare_dir 是否有效"""
        if not self.prepare_dir.exists():
            raise FileNotFoundError(
                f"Prepare 目录不存在: {self.prepare_dir}\n"
                f"请先运行 prepare_dataset.py"
            )

        required_files = ['aircraft_labels_clean.csv']
        required_dirs = ['images']

        for filename in required_files:
            if not (self.prepare_dir / filename).exists():
                raise FileNotFoundError(
                    f"Prepare 目录缺少必需的文件: {filename}\n"
                    f"目录: {self.prepare_dir}\n"
                    f"请先运行 prepare_dataset.py"
                )

        for dirname in required_dirs:
            if not (self.prepare_dir / dirname).exists():
                raise FileNotFoundError(
                    f"Prepare 目录缺少必需的目录: {dirname}\n"
                    f"目录: {self.prepare_dir}\n"
                    f"请先运行 prepare_dataset.py"
                )

        logger.info(f"Prepare 目录验证通过: {self.prepare_dir}")

    def load_labels(self) -> pd.DataFrame:
        """
        加载标注文件

        Returns:
            标注数据 DataFrame
        """
        logger.info(f"加载标注文件: {self.labels_file}")

        df = pd.read_csv(self.labels_file, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()

        logger.info(f"加载了 {len(df)} 条标注记录")

        if len(df) == 0:
            logger.error("标注文件为空，无法进行数据集划分！")
            logger.error("可能的原因:")
            logger.error("  1. 机型过滤参数不正确（使用了错误的机型名称）")
            logger.error("  2. 航司过滤参数不正确")
            logger.error("  3. FGVC数据集中没有符合条件的数据")
            raise ValueError("标注文件为空，无法进行数据集划分")

        return df

    def build_class_mapping(self, df: pd.DataFrame, class_column: str = 'typename') -> pd.DataFrame:
        """
        构建类别映射，过滤样本数不足的机型

        Args:
            df: 标注数据
            class_column: 类别列名

        Returns:
            过滤后的 DataFrame（只包含有足够样本的机型）
        """
        logger.info("构建机型类别映射...")

        # 统计每个机型的样本数
        type_counts = df[class_column].value_counts()
        logger.info(f"原始机型数量: {len(type_counts)}")

        # 过滤样本数不足的机型
        valid_types = type_counts[type_counts >= self.min_type_samples].index.tolist()
        filtered_count = len(type_counts) - len(valid_types)

        if filtered_count > 0:
            logger.info(f"过滤掉 {filtered_count} 个样本数少于 {self.min_type_samples} 的机型")

        # 只保留有效机型的记录
        df_filtered = df[df[class_column].isin(valid_types)].copy()

        # 构建类别映射
        classes = sorted(df_filtered[class_column].unique())
        self.class_to_id = {name: idx for idx, name in enumerate(classes)}
        self.id_to_class = {idx: name for idx, name in enumerate(classes)}

        logger.info(f"有效机型数量: {len(self.class_to_id)}")
        logger.info(f"机型数据集样本数: {len(df_filtered)}")

        return df_filtered

    def build_airline_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建航司类别映射，过滤样本数不足的航司

        Args:
            df: 标注数据

        Returns:
            过滤后的 DataFrame（只包含有足够样本的航司）
        """
        logger.info("构建航司类别映射...")

        # 检查是否有 airline 列（支持 'airline' 或 'airlinename' 列名）
        airline_column = None
        if 'airline' in df.columns:
            airline_column = 'airline'
        elif 'airlinename' in df.columns:
            airline_column = 'airlinename'
        else:
            logger.warning("标注数据中没有 'airline' 或 'airlinename' 列，跳过航司数据集准备")
            return pd.DataFrame()

        # 过滤掉空的航司
        df_airline = df[df[airline_column].notna() & (df[airline_column] != '')].copy()
        df_airline['airline'] = df_airline[airline_column].astype(str).str.strip()

        if len(df_airline) == 0:
            logger.warning("没有有效的航司标注数据")
            return pd.DataFrame()

        # 统计每个航司的样本数
        airline_counts = df_airline['airline'].value_counts()
        logger.info(f"原始航司数量: {len(airline_counts)}")

        # 过滤样本数不足的航司
        valid_airlines = airline_counts[airline_counts >= self.min_airline_samples].index.tolist()
        filtered_count = len(airline_counts) - len(valid_airlines)

        if filtered_count > 0:
            logger.info(f"过滤掉 {filtered_count} 个样本数少于 {self.min_airline_samples} 的航司")

        # 只保留有效航司的记录
        df_filtered = df_airline[df_airline['airline'].isin(valid_airlines)].copy()

        # 构建航司映射
        airlines = sorted(df_filtered['airline'].unique())
        self.airline_to_id = {name: idx for idx, name in enumerate(airlines)}
        self.id_to_airline = {idx: name for idx, name in enumerate(airlines)}

        logger.info(f"有效航司数量: {len(self.airline_to_id)}")
        logger.info(f"航司数据集样本数: {len(df_filtered)}")

        return df_filtered

    def split_by_class(
        self,
        df: pd.DataFrame,
        class_column: str = 'typename'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        按类别划分数据集，确保每个类别在各个子集中都有样本

        Args:
            df: 标注数据
            class_column: 类别列名

        Returns:
            (train_df, val_df, test_df)
        """
        logger.info(
            f"按类别划分数据集: 训练集{self.train_ratio*100:.0f}%, "
            f"验证集{self.val_ratio*100:.0f}%, "
            f"测试集{self.test_ratio*100:.0f}%"
        )

        class_groups = df.groupby(class_column)

        train_samples = []
        val_samples = []
        test_samples = []

        for class_name, class_df in class_groups:
            shuffled_df = class_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

            total = len(shuffled_df)
            train_size = int(total * self.train_ratio)
            val_size = int(total * self.val_ratio)

            # 确保每个类别至少有 1 个样本在 train 中
            if train_size == 0:
                train_size = 1
                val_size = max(0, min(1, total - 1))

            train_df_class = shuffled_df.iloc[:train_size]
            val_df_class = shuffled_df.iloc[train_size:train_size + val_size]
            test_df_class = shuffled_df.iloc[train_size + val_size:]

            train_samples.append(train_df_class)
            val_samples.append(val_df_class)
            test_samples.append(test_df_class)

            logger.debug(
                f"类别 {class_name}: 训练{len(train_df_class)}, "
                f"验证{len(val_df_class)}, 测试{len(test_df_class)}"
            )

        train_df = pd.concat(train_samples, ignore_index=True)
        val_df = pd.concat(val_samples, ignore_index=True)
        test_df = pd.concat(test_samples, ignore_index=True)

        # 打乱合并后的数据
        train_df = train_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        val_df = val_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        logger.info(f"划分完成: 训练{len(train_df)}, 验证{len(val_df)}, 测试{len(test_df)}")

        return train_df, val_df, test_df

    def save_split_csv(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """保存划分后的 CSV 文件"""
        train_df.to_csv(self.output_dir / 'train.csv', index=False, encoding='utf-8-sig')
        val_df.to_csv(self.output_dir / 'val.csv', index=False, encoding='utf-8-sig')
        test_df.to_csv(self.output_dir / 'test.csv', index=False, encoding='utf-8-sig')

        logger.info(f"划分 CSV 已保存到: {self.output_dir}")

    def prepare_aerovision(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """
        准备 Aerovision 分类数据集

        Args:
            train_df: 训练集
            val_df: 验证集
            test_df: 测试集
        """
        logger.info("=" * 60)
        logger.info("准备 Aerovision 分类数据集")
        logger.info("=" * 60)

        # 创建目录结构
        self._create_aerovision_structure()

        # 复制图片到对应目录
        self._copy_aerovision_images(train_df, val_df, test_df)

        # 保存类别映射
        self._save_class_mapping()

        # 保存数据集统计
        self._save_dataset_statistics(train_df, val_df, test_df)

        # 创建配置文件
        self._create_aerovision_config()

        logger.info(f"Aerovision 数据集准备完成: {self.aerovision_root}")

    def _create_aerovision_structure(self) -> None:
        """创建 Aerovision 目录结构"""
        self.aircraft_dir.mkdir(parents=True, exist_ok=True)
        self.labels_output_dir.mkdir(parents=True, exist_ok=True)

        for split in ['train', 'val', 'test']:
            split_dir = self.aircraft_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            for class_name in self.id_to_class.values():
                safe_class_name = class_name.replace(' ', '_').replace('/', '_')
                class_dir = split_dir / safe_class_name
                class_dir.mkdir(parents=True, exist_ok=True)

    def _copy_aerovision_images(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """复制图片到 Aerovision 目录结构"""
        split_data = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

        for split_name, df in split_data.items():
            logger.info(f"处理 {split_name} 数据集...")

            copied = 0
            skipped = 0

            for _, row in df.iterrows():
                filename = row['filename']
                typename = row['typename']
                safe_class_name = typename.replace(' ', '_').replace('/', '_')

                src_path = self.images_dir / filename
                dst_path = self.aircraft_dir / split_name / safe_class_name / filename

                if not src_path.exists():
                    skipped += 1
                    continue

                if not dst_path.exists():
                    shutil.copy2(src_path, dst_path)
                    copied += 1

            logger.info(f"  {split_name}: 复制 {copied}, 跳过 {skipped}")

    def _save_class_mapping(self) -> None:
        """保存类别映射"""
        output_file = self.labels_output_dir / "type_classes.json"

        class_mapping = {
            "num_classes": len(self.id_to_class),
            "classes": [
                {"id": class_id, "name": class_name}
                for class_id, class_name in self.id_to_class.items()
            ]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(class_mapping, f, ensure_ascii=False, indent=2)

        logger.info(f"类别映射已保存: {output_file}")

    def _save_dataset_statistics(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """保存数据集统计"""
        output_file = self.labels_output_dir / "dataset_statistics.json"

        splits_info = {}
        for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            class_counts = df['typename'].value_counts().to_dict()
            splits_info[split_name] = {
                "total": len(df),
                "classes": class_counts
            }

        statistics = {
            "num_classes": len(self.id_to_class),
            "splits": splits_info,
            "total_images": len(train_df) + len(val_df) + len(test_df),
            "timestamp": self.timestamp
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)

        logger.info(f"数据集统计已保存: {output_file}")

    def _create_aerovision_config(self) -> None:
        """创建 Aerovision 配置文件"""
        config_file = self.aerovision_root / "dataset_config.yaml"

        config = {
            "path": str(self.aircraft_dir.absolute()),
            "train": "train",
            "val": "val",
            "test": "test",
            "names": self.id_to_class,
            "nc": len(self.id_to_class),
        }

        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        logger.info(f"配置文件已保存: {config_file}")

    def prepare_detection(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ) -> None:
        """
        准备检测数据集

        Args:
            train_df: 训练集
            val_df: 验证集
        """
        logger.info("=" * 60)
        logger.info("准备检测数据集")
        logger.info("=" * 60)

        if not self.registration_dir.exists():
            logger.warning(f"注册号标注目录不存在: {self.registration_dir}")
            logger.warning("跳过检测数据集准备")
            return

        # 创建目录结构
        for split in ['train', 'val']:
            (self.detection_root / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.detection_root / 'labels' / split).mkdir(parents=True, exist_ok=True)

        # 复制文件
        train_count = self._copy_detection_files(train_df, 'train')
        val_count = self._copy_detection_files(val_df, 'val')

        logger.info(f"检测数据集: 训练{train_count}, 验证{val_count}")

        # 创建配置文件
        self._create_detection_config()

        logger.info(f"检测数据集准备完成: {self.detection_root}")

    def _copy_detection_files(self, df: pd.DataFrame, split_name: str) -> int:
        """复制检测数据集文件"""
        count = 0

        for _, row in df.iterrows():
            filename = row['filename']

            # 检查标注文件是否存在
            label_filename = Path(filename).stem + '.txt'
            label_path = self.registration_dir / label_filename

            if not label_path.exists():
                continue

            # 复制图片
            src_img = self.images_dir / filename
            dst_img = self.detection_root / 'images' / split_name / filename

            if src_img.exists() and not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            # 复制标注
            dst_label = self.detection_root / 'labels' / split_name / label_filename
            if not dst_label.exists():
                shutil.copy2(label_path, dst_label)

            count += 1

        return count

    def _create_detection_config(self) -> None:
        """创建检测配置文件"""
        yaml_content = f"""# YOLOv8 Detection Dataset Configuration
path: {self.detection_root.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: registration
"""

        yaml_path = self.detection_root / 'dataset.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)

        logger.info(f"检测配置已保存: {yaml_path}")

    def prepare_airline(
        self,
        df_airline: pd.DataFrame
    ) -> None:
        """
        准备航司分类数据集

        Args:
            df_airline: 航司标注数据（已过滤）
        """
        logger.info("=" * 60)
        logger.info("准备航司分类数据集")
        logger.info("=" * 60)

        if df_airline.empty:
            logger.warning("没有有效的航司数据，跳过航司数据集准备")
            return

        # 按航司类别划分
        train_df, val_df, test_df = self.split_by_class(df_airline, class_column='airline')

        # 创建目录结构
        self._create_airline_structure()

        # 复制图片到对应目录
        self._copy_airline_images(train_df, val_df, test_df)

        # 保存航司类别映射
        self._save_airline_mapping()

        # 保存航司数据集统计
        self._save_airline_statistics(train_df, val_df, test_df)

        # 创建配置文件
        self._create_airline_config()

        logger.info(f"航司数据集准备完成: {self.airline_dir}")

    def _create_airline_structure(self) -> None:
        """创建航司分类目录结构"""
        self.airline_dir.mkdir(parents=True, exist_ok=True)

        for split in ['train', 'val', 'test']:
            split_dir = self.airline_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            for airline_name in self.id_to_airline.values():
                safe_airline_name = airline_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                airline_class_dir = split_dir / safe_airline_name
                airline_class_dir.mkdir(parents=True, exist_ok=True)

    def _copy_airline_images(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """复制图片到航司分类目录结构"""
        split_data = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

        for split_name, df in split_data.items():
            logger.info(f"处理航司 {split_name} 数据集...")

            copied = 0
            skipped = 0

            for _, row in df.iterrows():
                filename = row['filename']
                airline = row['airline']
                safe_airline_name = airline.replace(' ', '_').replace('/', '_').replace('\\', '_')

                src_path = self.images_dir / filename
                dst_path = self.airline_dir / split_name / safe_airline_name / filename

                if not src_path.exists():
                    skipped += 1
                    continue

                if not dst_path.exists():
                    shutil.copy2(src_path, dst_path)
                    copied += 1

            logger.info(f"  {split_name}: 复制 {copied}, 跳过 {skipped}")

    def _save_airline_mapping(self) -> None:
        """保存航司类别映射"""
        output_file = self.labels_output_dir / "airline_classes.json"

        airline_mapping = {
            "num_classes": len(self.id_to_airline),
            "classes": [
                {"id": airline_id, "name": airline_name}
                for airline_id, airline_name in self.id_to_airline.items()
            ]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(airline_mapping, f, ensure_ascii=False, indent=2)

        logger.info(f"航司类别映射已保存: {output_file}")

    def _save_airline_statistics(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """保存航司数据集统计"""
        output_file = self.labels_output_dir / "airline_statistics.json"

        splits_info = {}
        for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            if not df.empty:
                airline_counts = df['airline'].value_counts().to_dict()
                splits_info[split_name] = {
                    "total": len(df),
                    "airlines": airline_counts
                }

        statistics = {
            "num_airlines": len(self.id_to_airline),
            "min_samples_threshold": self.min_airline_samples,
            "splits": splits_info,
            "total_images": len(train_df) + len(val_df) + len(test_df),
            "timestamp": self.timestamp
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)

        logger.info(f"航司数据集统计已保存: {output_file}")

    def _create_airline_config(self) -> None:
        """创建航司分类配置文件"""
        config_file = self.airline_dir / "dataset_config.yaml"

        config = {
            "path": str(self.airline_dir.absolute()),
            "train": "train",
            "val": "val",
            "test": "test",
            "names": self.id_to_airline,
            "nc": len(self.id_to_airline),
        }

        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        logger.info(f"航司配置文件已保存: {config_file}")

    def print_statistics(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """打印数据集统计信息"""
        logger.info("=" * 60)
        logger.info("数据集划分统计信息")
        logger.info("=" * 60)

        total = len(train_df) + len(val_df) + len(test_df)

        logger.info(f"总样本数: {total}")
        logger.info(f"训练集: {len(train_df)} ({len(train_df)/total*100:.1f}%)")
        logger.info(f"验证集: {len(val_df)} ({len(val_df)/total*100:.1f}%)")
        logger.info(f"测试集: {len(test_df)} ({len(test_df)/total*100:.1f}%)")
        logger.info(f"类别数: {len(self.id_to_class)}")

        logger.info("=" * 60)
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("=" * 60)

    def split(self) -> Path:
        """
        执行数据集划分流程

        Returns:
            输出目录路径
        """
        logger.info("=" * 60)
        logger.info("开始数据集划分")
        logger.info("=" * 60)

        # 1. 加载标注文件
        df = self.load_labels()

        # 2. 构建机型类别映射并过滤数据
        df_type = self.build_class_mapping(df)

        # 3. 按机型类别划分数据集
        train_df, val_df, test_df = self.split_by_class(df_type)

        # 4. 保存划分 CSV
        self.save_split_csv(train_df, val_df, test_df)

        # 5. 准备 Aerovision 数据集（机型分类）
        if self.mode in ["all", "aerovision"]:
            self.prepare_aerovision(train_df, val_df, test_df)

        # 6. 准备航司分类数据集
        if self.mode in ["all", "airline"]:
            # 构建航司映射并过滤数据（使用原始 df，不受机型过滤影响）
            df_airline = self.build_airline_mapping(df)
            if not df_airline.empty:
                self.prepare_airline(df_airline)

        # 7. 准备检测数据集（使用机型过滤后的数据）
        if self.mode in ["all", "detection"]:
            self.prepare_detection(train_df, val_df)

        # 8. 打印统计信息
        self.print_statistics(train_df, val_df, test_df)

        # 9. 创建 latest 链接
        self._create_latest_link()

        return self.output_dir

    def _create_latest_link(self) -> None:
        """创建 latest 软链接/目录引用"""
        # 在 Windows 上，如果没有管理员权限，软链接可能失败
        # 所以我们创建一个 latest.txt 文件来存储最新目录路径
        latest_txt = self.output_root / 'latest.txt'
        with open(latest_txt, 'w', encoding='utf-8') as f:
            f.write(str(self.output_dir))

        logger.info(f"最新目录已记录: {latest_txt}")


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="数据集划分脚本 - 划分数据并组织目录结构",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 prepare_dataset.py 的输出
  python split_dataset.py --prepare-dir ../data/prepared/20250104_120000

  # 指定划分比例
  python split_dataset.py --prepare-dir path/to/prepared --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

  # 只生成机型分类数据集
  python split_dataset.py --prepare-dir path/to/prepared --mode aerovision

  # 只生成航司分类数据集
  python split_dataset.py --prepare-dir path/to/prepared --mode airline

  # 只生成检测数据集
  python split_dataset.py --prepare-dir path/to/prepared --mode detection

  # 指定机型和航司最小样本数
  python split_dataset.py --prepare-dir path/to/prepared --min-type-samples 5 --min-airline-samples 5

注意:
  本脚本是数据处理流程的第二步
  请先运行 prepare_dataset.py 准备数据
        """
    )

    parser.add_argument(
        '--prepare-dir',
        type=str,
        default=None,
        help='prepare_dataset.py 输出的目录（默认从配置文件读取 data.prepared.latest）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出根目录（默认为 prepare_dir 同级的 splits/）'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'aerovision', 'airline', 'detection'],
        default='all',
        help='输出模式: all(全部), aerovision(机型分类), airline(航司分类), detection(检测) (默认: all)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='训练集比例 (默认: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='验证集比例 (默认: 0.15)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='测试集比例 (默认: 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )
    parser.add_argument(
        '--min-type-samples',
        type=int,
        default=None,
        help='机型最小样本数，低于此数量的机型将被过滤 (默认: 从配置文件读取或10)'
    )
    parser.add_argument(
        '--min-airline-samples',
        type=int,
        default=None,
        help='航司最小样本数，低于此数量的航司将被过滤 (默认: 从配置文件读取或10)'
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
        config = load_config(modules=['paths', 'airline', 'training'], load_all_modules=False)

    # 获取随机种子
    random_seed = args.seed
    if config.get('seed.random'):
        random_seed = config.get('seed.random')

    # 获取机型最小样本数
    min_type_samples = args.min_type_samples
    if min_type_samples is None:
        min_type_samples = config.get('training.min_samples_per_class') or 10

    # 获取航司最小样本数
    min_airline_samples = args.min_airline_samples
    if min_airline_samples is None:
        min_airline_samples = config.get('airline_data.min_samples_per_class') or 10

    # 获取 prepare_dir（优先使用命令行参数）
    prepare_dir = args.prepare_dir
    if not prepare_dir:
        # 尝试从 latest.txt 读取
        prepared_root = config.get_path('data.prepared.root')
        if prepared_root:
            latest_txt = Path(prepared_root) / 'latest.txt'
            if latest_txt.exists():
                with open(latest_txt, 'r', encoding='utf-8') as f:
                    prepare_dir = f.read().strip()
                logger.info(f"从 latest.txt 读取 prepare 目录: {prepare_dir}")

        if not prepare_dir:
            raise ValueError(
                "请提供 --prepare-dir 参数，或先运行 prepare_dataset.py\n"
                "示例: python split_dataset.py --prepare-dir ../data/prepared/20250104_120000"
            )

    # 转换为绝对路径
    if not Path(prepare_dir).is_absolute():
        prepare_dir = (Path(__file__).parent / prepare_dir).resolve()

    logger.info("=" * 60)
    logger.info("配置信息:")
    logger.info(f"  Prepare 目录: {prepare_dir}")
    logger.info(f"  输出目录: {args.output or '(默认)'}")
    logger.info(f"  模式: {args.mode}")
    logger.info(f"  训练集比例: {args.train_ratio}")
    logger.info(f"  验证集比例: {args.val_ratio}")
    logger.info(f"  测试集比例: {args.test_ratio}")
    logger.info(f"  随机种子: {random_seed}")
    logger.info(f"  机型最小样本数: {min_type_samples}")
    logger.info(f"  航司最小样本数: {min_airline_samples}")
    logger.info("=" * 60)

    # 创建数据集划分器
    splitter = DatasetSplitter(
        prepare_dir=str(prepare_dir),
        output_root=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        mode=args.mode,
        random_seed=random_seed,
        min_type_samples=min_type_samples,
        min_airline_samples=min_airline_samples
    )

    # 执行数据集划分
    output_dir = splitter.split()

    logger.info("")
    logger.info("完成! 数据集已准备好用于训练。")
    logger.info(f"  机型分类 (Aerovision): {output_dir / 'aerovision' / 'aircraft'}")
    logger.info(f"  航司分类 (Airline): {output_dir / 'aerovision' / 'airline'}")
    logger.info(f"  检测 (Detection): {output_dir / 'detection'}")


if __name__ == '__main__':
    main()
