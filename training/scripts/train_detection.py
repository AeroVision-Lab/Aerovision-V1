#!/usr/bin/env python3
"""
YOLOv8 Registration Detection Training Script

This script fine-tunes a pre-trained YOLOv8 detection model for aircraft registration area detection.
It supports custom configurations via modular YAML configuration system and command-line arguments.

配置说明：
本脚本使用新的模块化配置系统，自动加载以下配置模块：
- detection.yaml: 检测训练参数配置
- paths.yaml: 路径配置
- base.yaml: 基础配置（随机种子等）

Usage:
    # Basic usage (uses config from YAML)
    python train_detection.py

    # Custom parameters
    python train_detection.py --epochs 100 --batch-size 16 --imgsz 640 --device 0

    # Custom dataset path
    python train_detection.py --data path/to/dataset.yaml

    # Resume from checkpoint
    python train_detection.py --resume checkpoints/detection/last.pt

    # Use custom config file
    python train_detection.py --config my_config.yaml
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config

from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr


def setup_logging(log_dir: Path) -> logging.Logger:
    """
    Configure structured logging for the training process.

    Args:
        log_dir: Directory to save log files.

    Returns:
        Configured logger instance.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("RegistrationDetector")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers.clear()

    # File handler - detailed logs
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - info level
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
    """
    Parse command-line arguments for training configuration.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description='Fine-tune YOLOv8 model for registration area detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Pre-trained model path or model name (default: from config)'
    )
    parser.add_argument(
        '--model-size',
        type=str,
        default=None,
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
        help='Path to dataset YAML file (default: from config)'
    )

    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (default: from config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for training (default: from config)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=None,
        help='Input image size (default: from config)'
    )

    # Optimizer arguments
    parser.add_argument(
        '--lr0',
        type=float,
        default=None,
        help='Initial learning rate (default: from config)'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'],
        default=None,
        help='Optimizer type (default: from config)'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=None,
        help='SGD momentum or Adam beta1 (default: from config)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=None,
        help='Weight decay (L2 regularization) (default: from config)'
    )

    # Learning rate scheduler
    parser.add_argument(
        '--cos-lr',
        action='store_true',
        default=None,
        help='Use cosine learning rate scheduler'
    )
    parser.add_argument(
        '--lrf',
        type=float,
        default=None,
        help='Final learning rate fraction (default: from config)'
    )

    # Early stopping
    parser.add_argument(
        '--patience',
        type=int,
        default=None,
        help='Early stopping patience (epochs without improvement) (default: from config)'
    )

    # Warmup
    parser.add_argument(
        '--warmup-epochs',
        type=float,
        default=None,
        help='Warmup epochs (default: from config)'
    )

    # Device and performance
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (e.g., 0, 1, cpu, mps) (default: from config)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of dataloader workers (default: from config)'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        default=None,
        help='Use Automatic Mixed Precision training'
    )

    # Reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: from config)'
    )

    # Saving
    parser.add_argument(
        '--project',
        type=str,
        default=None,
        help='Project directory for saving results (default: from config)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment name (default: from config)'
    )
    parser.add_argument(
        '--save-period',
        type=int,
        default=None,
        help='Save checkpoint every N epochs (-1 to disable) (default: from config)'
    )

    # Checkpoint directory (custom)
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory to save custom checkpoints (default: from config)'
    )

    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file (optional, will use modular config if not specified)'
    )

    # Logging
    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help='Directory to save log files (default: from config)'
    )
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        default=None,
        help='Enable TensorBoard logging'
    )

    # Validation
    parser.add_argument(
        '--val',
        action='store_true',
        default=None,
        help='Run validation during training'
    )

    # Plots
    parser.add_argument(
        '--plots',
        action='store_true',
        default=None,
        help='Save plots and images during training'
    )

    return parser.parse_args()


def save_custom_checkpoint(
    model: YOLO,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    val_map: float,
    checkpoint_path: Path,
    is_best: bool = False
) -> None:
    """
    Save custom checkpoint with specific format.

    Args:
        model: YOLO model instance.
        epoch: Current epoch number.
        optimizer: Optimizer instance.
        val_map: Validation mAP.
        checkpoint_path: Path to save the checkpoint.
        is_best: Whether this is the best model so far.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the underlying PyTorch model
    pt_model = model.model

    # Prepare checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': pt_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_map': val_map,
        'model_args': pt_model.args if hasattr(pt_model, 'args') else {},
        'names': pt_model.names if hasattr(pt_model, 'names') else {},
        'date': datetime.now().isoformat(),
    }

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)

    # Save best checkpoint separately
    if is_best:
        best_path = checkpoint_path.parent / 'best.pt'
        torch.save(checkpoint, best_path)


class RegistrationDetectorTrainer:
    """
    Wrapper class for training registration detector with custom logging and checkpointing.
    """

    def __init__(self, config: Dict[str, Any], args: argparse.Namespace, logger: logging.Logger):
        """
        Initialize the registration detector trainer.

        Args:
            config: Training configuration dictionary.
            args: Parsed command-line arguments.
            logger: Logger instance.
        """
        self.config = config
        self.args = args
        self.logger = logger

        # Get training root directory (parent of scripts/)
        self.training_root = Path(__file__).parent.parent

        # Generate timestamp for this training session
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Setup checkpoint directory with timestamp
        checkpoint_base = config.get('checkpoint_dir')
        if checkpoint_base:
            checkpoint_dir = self._resolve_training_path(checkpoint_base)
        else:
            checkpoint_dir = self.training_root / 'ckpt' / 'detection'

        # Add timestamp subdirectory
        self.checkpoint_dir = checkpoint_dir / self.timestamp

        self.best_val_map = 0.0

        # Setup directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self._init_model()

        # Setup TensorBoard
        self.tb_writer = None
        use_tensorboard = config.get('tensorboard', True)
        if use_tensorboard:
            self._setup_tensorboard()

    def _resolve_training_path(self, path: str) -> Path:
        """
        Resolve path relative to training/ directory.
        Config paths use ../xxx notation (relative to training/configs),
        which resolves to training/xxx.

        Args:
            path: Path string (can be relative or absolute)

        Returns:
            Resolved absolute Path object
        """
        path_obj = Path(path)

        # If already absolute, return as-is
        if path_obj.is_absolute():
            return path_obj

        # Config paths are relative to training/configs
        # So ../xxx means training/xxx
        config_dir = self.training_root / 'configs'
        return (config_dir / path).resolve()

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

            # Check if model path is absolute and exists
            if Path(model_path).is_absolute() and Path(model_path).exists():
                # Use the absolute path directly
                self.logger.info(f"Loading pre-trained model: {model_path}")
                self.model = YOLO(model_path)
            else:
                # Model is either a name (like "yolov8n.pt") or doesn't exist yet
                # Extract just the filename
                model_filename = Path(model_path).name
                local_model_path = model_dir / model_filename

                if local_model_path.exists():
                    # Model exists in model directory
                    self.logger.info(f"Loading local model: {local_model_path}")
                    self.model = YOLO(str(local_model_path))
                else:
                    # Model doesn't exist, download it
                    self.logger.info(f"Model not found locally, will download to: {model_dir}")
                    original_dir = os.getcwd()
                    try:
                        # Change to model directory temporarily
                        os.chdir(str(model_dir))
                        self.model = YOLO(model_filename)
                        os.chdir(original_dir)
                        self.logger.info(f"Model downloaded to: {model_dir / model_filename}")
                    except Exception as e:
                        os.chdir(original_dir)
                        raise e

                # Update config with actual model path
                self.config['model'] = str(local_model_path)

    def _setup_tensorboard(self) -> None:
        """Setup TensorBoard writer for logging."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            # Get log directory from config or use default
            log_base = self.config.get('log_dir') or (self.training_root / 'logs' / 'detection')
            if isinstance(log_base, str):
                log_base = self._resolve_training_path(log_base)

            # Add timestamp subdirectory (use same timestamp as checkpoint)
            log_dir = log_base / self.timestamp
            self.tb_writer = SummaryWriter(log_dir)
            self.logger.info(f"TensorBoard logging enabled: {log_dir}")
        except ImportError:
            self.logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.tb_writer = None

    def log_metrics(self, metrics: Dict[str, float], epoch: int) -> None:
        """
        Log training metrics to both file and TensorBoard.

        Args:
            metrics: Dictionary of metric names and values.
            epoch: Current epoch number.
        """
        # Log to file
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch} - {metric_str}")

        # Log to TensorBoard
        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f'train/{key}', value, epoch)
            self.tb_writer.flush()

    def train(self) -> None:
        """
        Execute the training process.

        This method runs the complete training loop including validation,
        checkpointing, and logging.
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Registration Detector Fine-tuning")
        self.logger.info("=" * 60)

        # Log configuration
        self.logger.info("Configuration:")
        for key, value in self.config.items():
            self.logger.info(f"  {key}: {value}")

        # Ensure all YOLO downloads go to model directory
        original_dir = os.getcwd()
        model_dir = self.training_root / 'model'

        # Prepare training arguments for YOLO
        train_args = {
            'data': self.config['data'],
            'epochs': self.config['epochs'],
            'batch': self.config['batch_size'],
            'imgsz': self.config['imgsz'],
            'lr0': self.config['lr0'],
            'optimizer': self.config['optimizer'],
            'momentum': self.config['momentum'],
            'weight_decay': self.config['weight_decay'],
            'cos_lr': self.config['cos_lr'],
            'lrf': self.config['lrf'],
            'patience': self.config['patience'],
            'warmup_epochs': self.config['warmup_epochs'],
            'device': self.config['device'],
            'workers': self.config['workers'],
            'amp': self.config['amp'],
            'seed': self.config['seed'],
            'project': self.config['project'],
            'name': self.config['name'],
            'save_period': self.config['save_period'],
            'val': self.config['val'],
            'plots': self.config['plots'],
            'verbose': True,

            # Data augmentation for text detection
            'hsv_h': self.config.get('hsv_h', 0.01),
            'hsv_s': self.config.get('hsv_s', 0.3),
            'hsv_v': self.config.get('hsv_v', 0.3),
            'degrees': self.config.get('degrees', 5),
            'translate': self.config.get('translate', 0.1),
            'scale': self.config.get('scale', 0.2),
            'fliplr': self.config.get('fliplr', 0.0),
            'flipud': self.config.get('flipud', 0.0),
            'mosaic': self.config.get('mosaic', 0.5),
            'mixup': self.config.get('mixup', 0.0),
        }

        # Start training
        self.logger.info("Starting training...")
        start_time = datetime.now()

        try:
            # Change to model directory to ensure any downloads go there
            os.chdir(str(model_dir))
            self.logger.info(f"Changed working directory to: {os.getcwd()}")

            # Train the model
            results = self.model.train(**train_args)

            # Change back to original directory
            os.chdir(original_dir)

            # Log final results
            self.logger.info("Training completed successfully!")
            self.logger.info(f"Training time: {datetime.now() - start_time}")

            if results:
                self.logger.info("Final metrics:")
                # Get metrics from DetMetrics object
                if hasattr(results, 'results_dict'):
                    metrics_dict = results.results_dict
                    for key, value in metrics_dict.items():
                        if isinstance(value, (int, float)):
                            self.logger.info(f"  {key}: {value:.4f}")
                elif hasattr(results, 'box'):
                    # Detection metrics
                    if hasattr(results.box, 'map'):
                        self.logger.info(f"  mAP50: {results.box.map50:.4f}")
                        self.logger.info(f"  mAP50-95: {results.box.map:.4f}")
                    if hasattr(results, 'fitness'):
                        self.logger.info(f"  fitness: {results.fitness:.4f}")

            # Save final checkpoint
            if hasattr(self.model, 'trainer') and self.model.trainer:
                self._save_final_checkpoint()

            # Log output paths
            self.logger.info(f"Best model: {self.config['project']}/{self.config['name']}/weights/best.pt")
            self.logger.info(f"Last model: {self.config['project']}/{self.config['name']}/weights/last.pt")

        except Exception as e:
            # Make sure to restore working directory on error
            os.chdir(original_dir)
            self.logger.error(f"Training failed with error: {e}", exc_info=True)
            raise

        finally:
            # Close TensorBoard writer
            if self.tb_writer:
                self.tb_writer.close()
                self.logger.info("TensorBoard writer closed")

    def _save_final_checkpoint(self) -> None:
        """Save the final checkpoint after training completion."""
        trainer = self.model.trainer

        # Get validation mAP from metrics
        val_map = 0.0
        if hasattr(trainer, 'metrics') and trainer.metrics:
            # Try to find mAP in metrics
            for key in ['metrics/mAP50-95(B)', 'mAP50-95', 'map', 'val_map']:
                if key in trainer.metrics:
                    val_map = float(trainer.metrics[key])
                    break

        # Get optimizer
        optimizer = trainer.optimizer if hasattr(trainer, 'optimizer') else None

        # Save last checkpoint
        last_path = self.checkpoint_dir / 'last.pt'
        if optimizer:
            save_custom_checkpoint(
                self.model,
                trainer.epoch,
                optimizer,
                val_map,
                last_path,
                is_best=False
            )
            self.logger.info(f"Saved last checkpoint to: {last_path}")

        # Save best checkpoint if available
        best_path = self.checkpoint_dir / 'best.pt'
        if hasattr(trainer, 'best') and trainer.best.exists():
            import shutil
            shutil.copy(trainer.best, best_path)
            self.logger.info(f"Copied best checkpoint to: {best_path}")


def resolve_config_path(path_str: str, training_root: Path) -> str:
    """
    Resolve path from config (relative to training/configs) to absolute path.

    Args:
        path_str: Path string from config
        training_root: Training root directory

    Returns:
        Absolute path string
    """
    if Path(path_str).is_absolute():
        return path_str

    config_dir = training_root / 'configs'
    resolved = (config_dir / path_str).resolve()
    return str(resolved)


def main() -> None:
    """
    Main entry point for the training script.

    This function:
    1. Parses command-line arguments
    2. Loads configuration from modular YAML system
    3. Merges config with command-line arguments
    4. Sets up logging
    5. Initializes and runs the trainer
    """
    # Set environment variables for YOLO download paths
    training_root = Path(__file__).parent.parent
    model_dir = training_root / 'model'
    model_dir.mkdir(parents=True, exist_ok=True)

    # Set YOLO environment variables
    os.environ['YOLO_CONFIG_DIR'] = str(training_root / 'model')
    os.environ.setdefault('TORCH_HOME', str(model_dir))

    # Parse arguments
    args = parse_arguments()

    # Load configuration from modular system
    try:
        if args.config and Path(args.config).exists():
            config_obj = load_config(args.config)
        else:
            # Load detection-specific config along with paths
            config_obj = load_config(modules=['detection', 'paths'], load_all_modules=False)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Using default modular configuration...")
        config_obj = load_config(modules=['detection', 'paths'], load_all_modules=False)

    # Get data path - YOLOv8 detection requires a YAML file
    data_path = None

    # 1. Try command-line argument first
    if args.data:
        data_path = args.data

    # 2. Try from splits/latest.txt
    if not data_path:
        splits_root = config_obj.get_path('data.splits.root')
        if splits_root:
            latest_txt = Path(splits_root) / 'latest.txt'
            if latest_txt.exists():
                with open(latest_txt, 'r', encoding='utf-8') as f:
                    latest_split_dir = f.read().strip()
                # Detection dataset.yaml path
                dataset_subdir = config_obj.get('detection_data.dataset_subdir', 'detection')
                detection_yaml = Path(latest_split_dir) / dataset_subdir / 'dataset.yaml'
                if detection_yaml.exists():
                    data_path = str(detection_yaml)

    # 3. Fallback: use default path
    if not data_path:
        data_path = '../data/splits/latest/detection/dataset.yaml'

    # Extract training configuration with defaults
    # 优先级：命令行参数 > config yaml > 默认值
    # 注意：使用显式 None 检查而非 or 运算符，避免 0 值被错误处理

    def get_config_value(config_key: str, arg_value, default):
        """
        获取配置值，正确处理 0 和 False 等 falsy 值。
        优先级：命令行参数 > config yaml > 默认值
        """
        # 命令行参数优先（如果用户显式指定）
        if arg_value is not None:
            return arg_value
        # 然后是配置文件
        config_val = config_obj.get(config_key)
        if config_val is not None:
            return config_val
        # 最后是默认值
        return default

    # Determine model name
    model_size = args.model_size or config_obj.get('detection_training.model.size', 'n')
    model_name = args.model or config_obj.get('detection_training.model.name') or f'yolov8{model_size}.pt'

    config = {
        # Model configuration
        'model': model_name,
        'resume': get_config_value('detection_training.resume', args.resume, None),

        # Data configuration
        'data': data_path,

        # Training parameters
        'epochs': get_config_value('detection_training.epochs', args.epochs, 100),
        'batch_size': get_config_value('detection_training.batch_size', args.batch_size, 16),
        'imgsz': get_config_value('detection_training.image_size', args.imgsz, 640),

        # Optimizer
        'lr0': get_config_value('detection_training.optimizer.lr0', args.lr0, 0.01),
        'optimizer': get_config_value('detection_training.optimizer.type', args.optimizer, 'auto'),
        'momentum': get_config_value('detection_training.optimizer.momentum', args.momentum, 0.937),
        'weight_decay': get_config_value('detection_training.optimizer.weight_decay', args.weight_decay, 0.0005),

        # Learning rate scheduler
        'cos_lr': get_config_value('detection_training.scheduler.cosine', args.cos_lr, True),
        'lrf': get_config_value('detection_training.scheduler.lrf', args.lrf, 0.01),

        # Early stopping - 注意 patience 为 0 表示禁用早停
        'patience': get_config_value('detection_training.early_stopping.patience', args.patience, 20),

        # Warmup - 注意 warmup_epochs 为 0 表示不预热
        'warmup_epochs': get_config_value('detection_training.warmup.epochs', args.warmup_epochs, 3.0),

        # Device
        'device': get_config_value('device.default', args.device, '0'),
        'workers': get_config_value('detection_training.workers', args.workers, 8),
        'amp': get_config_value('detection_training.amp', args.amp, True),

        # Reproducibility
        'seed': get_config_value('seed.random', args.seed, 42),

        # Saving
        'project': get_config_value('detection_training.output.project', args.project, '../output/detection'),
        'name': get_config_value('detection_training.output.name', args.name, 'registration_detector'),
        'save_period': get_config_value('detection_training.save_period', args.save_period, 10),

        # Checkpoint directory
        'checkpoint_dir': get_config_value('checkpoints.detection', args.checkpoint_dir, '../ckpt/detection'),

        # Log directory
        'log_dir': get_config_value('logs.detection', args.log_dir, '../logs/detection'),

        # Validation and plots
        'val': get_config_value('detection_training.validation.enabled', args.val, True),
        'plots': get_config_value('detection_training.plots', args.plots, True),

        # TensorBoard logging
        'tensorboard': get_config_value('detection_training.tensorboard', args.tensorboard, True),

        # Data augmentation for text detection (registration numbers)
        'hsv_h': get_config_value('detection_training.augmentation.hsv_h', None, 0.01),
        'hsv_s': get_config_value('detection_training.augmentation.hsv_s', None, 0.3),
        'hsv_v': get_config_value('detection_training.augmentation.hsv_v', None, 0.3),
        'degrees': get_config_value('detection_training.augmentation.degrees', None, 5),
        'translate': get_config_value('detection_training.augmentation.translate', None, 0.1),
        'scale': get_config_value('detection_training.augmentation.scale', None, 0.2),
        'fliplr': get_config_value('detection_training.augmentation.fliplr', None, 0.0),
        'flipud': get_config_value('detection_training.augmentation.flipud', None, 0.0),
        'mosaic': get_config_value('detection_training.augmentation.mosaic', None, 0.5),
        'mixup': get_config_value('detection_training.augmentation.mixup', None, 0.0),
    }

    # Resolve data path
    if not Path(config['data']).is_absolute():
        config['data'] = resolve_config_path(config['data'], training_root)

    # Generate timestamp for this training session
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Resolve project path
    if not Path(config['project']).is_absolute():
        config['project'] = resolve_config_path(config['project'], training_root)

    # Add timestamp to experiment name
    config['name'] = f"{config['name']}_{timestamp}"
    config['timestamp'] = timestamp

    # Resolve log directory path
    log_dir_path = config['log_dir']
    if not Path(log_dir_path).is_absolute():
        log_dir_path = resolve_config_path(log_dir_path, training_root)

    log_dir = Path(log_dir_path)

    # Add timestamp subdirectory for this training session
    log_dir = log_dir / timestamp
    logger = setup_logging(log_dir)

    # Log startup information
    logger.info("=" * 60)
    logger.info("Registration Detector Training Script")
    logger.info("=" * 60)
    logger.info(f"Dataset path: {config['data']}")
    logger.info(f"Model: {config['model']}")
    logger.info(f"Device: {config['device']}")
    logger.info(f"Epochs: {config['epochs']}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Image size: {config['imgsz']}")
    logger.info(f"Learning rate: {config['lr0']}")
    logger.info(f"Optimizer: {config['optimizer']}")
    logger.info(f"Patience: {config['patience']}")
    logger.info(f"Seed: {config['seed']}")
    logger.info("=" * 60)
    logger.info(f"Training timestamp: {timestamp}")
    logger.info(f"Output paths:")
    logger.info(f"  Project (YOLO output): {config['project']}")
    logger.info(f"  Experiment name: {config['name']}")
    logger.info(f"  Checkpoints: Will be created as ckpt/detection/{timestamp}/")
    logger.info(f"  Logs: {log_dir}")
    logger.info("=" * 60)

    # Verify dataset paths
    data_yaml = Path(config['data'])

    if not data_yaml.exists():
        logger.error(f"Dataset YAML not found: {data_yaml}")
        logger.error("Please prepare the detection dataset first using prepare_dataset.py and split_dataset.py")
        sys.exit(1)

    if not data_yaml.is_file() or not data_yaml.suffix in ['.yaml', '.yml']:
        logger.error(f"Data path must be a YAML file: {data_yaml}")
        logger.error("YOLOv8 detection requires a dataset.yaml file")
        sys.exit(1)

    logger.info(f"Dataset YAML verified: {data_yaml}")

    # Parse dataset.yaml to show dataset info
    try:
        import yaml
        with open(data_yaml, 'r', encoding='utf-8') as f:
            dataset_info = yaml.safe_load(f)

        if 'names' in dataset_info:
            logger.info(f"  Classes: {dataset_info['names']}")
        if 'nc' in dataset_info:
            logger.info(f"  Number of classes: {dataset_info['nc']}")
        if 'train' in dataset_info:
            train_path = Path(dataset_info['train'])
            if not train_path.is_absolute():
                train_path = data_yaml.parent / train_path
            if train_path.exists():
                train_images = list(train_path.glob('*'))
                logger.info(f"  Training images: {len(train_images)}")
        if 'val' in dataset_info:
            val_path = Path(dataset_info['val'])
            if not val_path.is_absolute():
                val_path = data_yaml.parent / val_path
            if val_path.exists():
                val_images = list(val_path.glob('*'))
                logger.info(f"  Validation images: {len(val_images)}")
    except Exception as e:
        logger.warning(f"Could not parse dataset.yaml for info: {e}")

    # Initialize and run trainer
    try:
        trainer = RegistrationDetectorTrainer(config, args, logger)

        # Override timestamp with the one from main (for consistency)
        trainer.timestamp = timestamp

        # Recreate checkpoint directory with correct timestamp
        checkpoint_base = config.get('checkpoint_dir')
        if checkpoint_base:
            checkpoint_dir = trainer._resolve_training_path(checkpoint_base)
        else:
            checkpoint_dir = trainer.training_root / 'ckpt' / 'detection'
        trainer.checkpoint_dir = checkpoint_dir / timestamp
        trainer.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Log final checkpoint path
        logger.info(f"Checkpoint directory: {trainer.checkpoint_dir}")

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
