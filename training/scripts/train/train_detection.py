#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 Registration Detection Training Script

训练 YOLOv8 检测模型用于注册号区域检测。
使用模块化配置系统，自动读取最新的数据集路径。

配置说明：
本脚本使用模块化配置系统，自动加载以下配置模块：
- training.yaml: 训练参数配置
- paths.yaml: 路径配置
- base.yaml: 基础配置（随机种子等）

Usage:
    # 使用默认配置（自动读取最新的 split 数据集）
    python train_detection.py

    # 指定参数
    python train_detection.py --epochs 100 --batch-size 16 --imgsz 640 --device 0

    # 指定数据集路径
    python train_detection.py --data path/to/dataset.yaml

    # 使用自定义配置文件
    python train_detection.py --config my_config.yaml
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config

from ultralytics import YOLO


def setup_logging(log_dir: Path) -> logging.Logger:
    """
    Configure structured logging for the training process.

    Args:
        log_dir: Directory to save log files.

    Returns:
        Configured logger instance.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("RegistrationDetector")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 model for registration area detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Pre-trained model path or model name (e.g., yolov8n.pt, yolov8s.pt)'
    )
    parser.add_argument(
        '--model-size',
        type=str,
        default='n',
        choices=['n', 's', 'm', 'l', 'x'],
        help='Model size (n=nano, s=small, m=medium, l=large, x=extra large)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )

    # Data arguments
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to dataset YAML file (auto-detected from splits/latest.txt if not specified)'
    )

    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for training'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size'
    )

    # Optimizer arguments
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.01,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='auto',
        choices=['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'],
        help='Optimizer type'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.937,
        help='SGD momentum or Adam beta1'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='Weight decay (L2 regularization)'
    )

    # Early stopping
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience (epochs without improvement)'
    )

    # Device and performance
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device to use (e.g., 0, 1, cpu, mps)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of dataloader workers'
    )

    # Reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    # Saving
    parser.add_argument(
        '--project',
        type=str,
        default=None,
        help='Project directory for saving results'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment name'
    )
    parser.add_argument(
        '--save-period',
        type=int,
        default=10,
        help='Save checkpoint every N epochs (-1 to disable)'
    )

    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )

    return parser.parse_args()


class RegistrationDetectorTrainer:
    """Trainer class for registration area detection model."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the registration detector trainer.

        Args:
            config: Training configuration dictionary.
            logger: Logger instance.
        """
        self.config = config
        self.logger = logger

        self.training_root = Path(__file__).parent.parent
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Setup checkpoint directory
        checkpoint_base = config.get('checkpoint_dir')
        if checkpoint_base:
            if not Path(checkpoint_base).is_absolute():
                checkpoint_dir = (self.training_root / 'configs' / checkpoint_base).resolve()
            else:
                checkpoint_dir = Path(checkpoint_base)
        else:
            checkpoint_dir = self.training_root / 'ckpt' / 'detection'

        self.checkpoint_dir = checkpoint_dir / self.timestamp
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self._init_model()

    def _init_model(self) -> None:
        """Initialize the YOLO model for training."""
        resume_checkpoint = self.config.get('resume')

        if resume_checkpoint:
            self.logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")
            self.model = YOLO(resume_checkpoint)
        else:
            model_path = self.config['model']
            model_dir = self.training_root / 'model'
            model_dir.mkdir(parents=True, exist_ok=True)

            if Path(model_path).is_absolute() and Path(model_path).exists():
                self.logger.info(f"Loading pre-trained model: {model_path}")
                self.model = YOLO(model_path)
            else:
                model_filename = Path(model_path).name
                local_model_path = model_dir / model_filename

                if local_model_path.exists():
                    self.logger.info(f"Loading local model: {local_model_path}")
                    self.model = YOLO(str(local_model_path))
                else:
                    self.logger.info(f"Model not found locally, will download: {model_filename}")
                    original_dir = os.getcwd()
                    try:
                        os.chdir(str(model_dir))
                        self.model = YOLO(model_filename)
                        os.chdir(original_dir)
                        self.logger.info(f"Model downloaded to: {model_dir / model_filename}")
                    except Exception as e:
                        os.chdir(original_dir)
                        raise e

    def train(self) -> None:
        """Execute the training process."""
        self.logger.info("=" * 60)
        self.logger.info("Starting Registration Detector Training")
        self.logger.info("=" * 60)

        # Log configuration
        self.logger.info("Configuration:")
        for key, value in self.config.items():
            self.logger.info(f"  {key}: {value}")

        # Prepare training arguments
        train_args = {
            'data': self.config['data'],
            'epochs': self.config['epochs'],
            'batch': self.config['batch_size'],
            'imgsz': self.config['imgsz'],
            'lr0': self.config['lr0'],
            'optimizer': self.config['optimizer'],
            'momentum': self.config['momentum'],
            'weight_decay': self.config['weight_decay'],
            'patience': self.config['patience'],
            'device': self.config['device'],
            'workers': self.config['workers'],
            'seed': self.config['seed'],
            'project': self.config['project'],
            'name': self.config['name'],
            'save_period': self.config['save_period'],
            'verbose': True,
            'plots': True,

            # Data augmentation for text detection
            'hsv_h': 0.01,      # 色调变化（文字不需要太多）
            'hsv_s': 0.3,       # 饱和度变化
            'hsv_v': 0.3,       # 明度变化
            'degrees': 5,       # 旋转角度（文字不要旋转太多）
            'translate': 0.1,   # 平移
            'scale': 0.2,       # 缩放
            'fliplr': 0.0,      # 不左右翻转（文字方向重要）
            'flipud': 0.0,      # 不上下翻转
            'mosaic': 0.5,      # 马赛克增强
            'mixup': 0.0,       # 不使用 mixup

            # Warmup
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
        }

        # Start training
        self.logger.info("Starting training...")
        start_time = datetime.now()

        try:
            results = self.model.train(**train_args)

            self.logger.info("Training completed successfully!")
            self.logger.info(f"Training time: {datetime.now() - start_time}")

            if results:
                self.logger.info("Final metrics:")
                if hasattr(results, 'results_dict'):
                    for key, value in results.results_dict.items():
                        if isinstance(value, (int, float)):
                            self.logger.info(f"  {key}: {value:.4f}")

            # Log output paths
            self.logger.info(f"Best model: {self.config['project']}/{self.config['name']}/weights/best.pt")
            self.logger.info(f"Last model: {self.config['project']}/{self.config['name']}/weights/last.pt")

        except Exception as e:
            self.logger.error(f"Training failed with error: {e}", exc_info=True)
            raise


def main() -> None:
    """Main entry point for the training script."""
    training_root = Path(__file__).parent.parent
    model_dir = training_root / 'model'
    model_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variables
    os.environ['YOLO_CONFIG_DIR'] = str(training_root / 'model')
    os.environ.setdefault('TORCH_HOME', str(model_dir))

    # Parse arguments
    args = parse_arguments()

    # Load configuration
    try:
        if args.config and Path(args.config).exists():
            config_obj = load_config(args.config)
        else:
            config_obj = load_config(modules=['training', 'paths'], load_all_modules=False)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Using default configuration...")
        config_obj = load_config(modules=['training', 'paths'], load_all_modules=False)

    # Get data path
    data_path = args.data

    if not data_path:
        # Try to read from splits/latest.txt
        splits_root = config_obj.get_path('data.splits.root')
        if splits_root:
            latest_txt = Path(splits_root) / 'latest.txt'
            if latest_txt.exists():
                with open(latest_txt, 'r', encoding='utf-8') as f:
                    latest_split_dir = f.read().strip()
                # Detection dataset.yaml path
                detection_yaml = Path(latest_split_dir) / 'detection' / 'dataset.yaml'
                if detection_yaml.exists():
                    data_path = str(detection_yaml)

    if not data_path:
        print("Error: No dataset found. Please specify --data or run split_dataset.py first.")
        sys.exit(1)

    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Helper function to resolve paths
    def resolve_config_path(path_str: str) -> str:
        if Path(path_str).is_absolute():
            return path_str
        config_dir = training_root / 'configs'
        return str((config_dir / path_str).resolve())

    # Resolve data path if relative
    if not Path(data_path).is_absolute():
        data_path = resolve_config_path(data_path)

    # Build configuration
    model_size = args.model_size
    model_name = args.model or f'yolov8{model_size}.pt'

    config = {
        'model': model_name,
        'resume': args.resume,
        'data': data_path,
        'epochs': config_obj.get('training.detection.epochs') or args.epochs or 100,
        'batch_size': config_obj.get('training.detection.batch_size') or args.batch_size or 16,
        'imgsz': config_obj.get('training.detection.image_size') or args.imgsz or 640,
        'lr0': config_obj.get('training.detection.lr0') or args.lr0 or 0.01,
        'optimizer': args.optimizer or 'auto',
        'momentum': args.momentum or 0.937,
        'weight_decay': args.weight_decay or 0.0005,
        'patience': config_obj.get('training.detection.patience') or args.patience or 20,
        'device': config_obj.get('device.default') or args.device or '0',
        'workers': args.workers or 8,
        'seed': config_obj.get('seed.random') or args.seed or 42,
        'project': resolve_config_path(config_obj.get('output.detection') or '../output/detection'),
        'name': f"registration_detector_{timestamp}",
        'save_period': args.save_period or 10,
        'checkpoint_dir': config_obj.get('checkpoints.detection') or '../ckpt/detection',
    }

    # Setup logging
    log_dir_path = config_obj.get('logs.detection') or '../logs/detection'
    if not Path(log_dir_path).is_absolute():
        log_dir_path = resolve_config_path(log_dir_path)
    log_dir = Path(log_dir_path) / timestamp
    logger = setup_logging(log_dir)

    # Log startup information
    logger.info("=" * 60)
    logger.info("Registration Detector Training Script")
    logger.info("=" * 60)
    logger.info(f"Dataset: {config['data']}")
    logger.info(f"Model: {config['model']}")
    logger.info(f"Device: {config['device']}")
    logger.info(f"Epochs: {config['epochs']}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Image size: {config['imgsz']}")
    logger.info(f"Learning rate: {config['lr0']}")
    logger.info(f"Seed: {config['seed']}")
    logger.info("=" * 60)

    # Verify dataset
    data_yaml = Path(config['data'])
    if not data_yaml.exists():
        logger.error(f"Dataset YAML not found: {data_yaml}")
        logger.error("Please run split_dataset.py first to prepare the detection dataset.")
        sys.exit(1)

    logger.info(f"Dataset YAML verified: {data_yaml}")

    # Initialize and run trainer
    try:
        trainer = RegistrationDetectorTrainer(config, logger)
        trainer.train()

        logger.info("=" * 60)
        logger.info("Training script completed successfully!")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
