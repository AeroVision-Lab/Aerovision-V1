#!/usr/bin/env python3
"""
YOLOv8 Airline Classifier Fine-tuning Training Script

This script fine-tunes a pre-trained YOLOv8 classification model for airline identification.
It supports custom configurations via modular YAML configuration system and command-line arguments.

配置说明：
本脚本使用新的模块化配置系统，自动加载以下配置模块：
- airline.yaml: 航司识别训练参数配置
- paths.yaml: 路径配置
- base.yaml: 基础配置（随机种子等）

Usage:
    # Basic usage (uses config from YAML)
    python train_airline.py

    # Custom parameters
    python train_airline.py --epochs 100 --batch-size 32 --imgsz 224 --device 0

    # Custom dataset path
    python train_airline.py --data path/to/dataset

    # Resume from checkpoint
    python train_airline.py --resume checkpoints/airline/last.pt

    # Use custom config file
    python train_airline.py --config my_config.yaml
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
    logger = logging.getLogger("AirlineClassifier")
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
        description='Fine-tune YOLOv8 model for airline identification',
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
        help='Path to dataset root directory (default: from config)'
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

    # Regularization
    parser.add_argument(
        '--dropout',
        type=float,
        default=None,
        help='Dropout for classification head (default: from config)'
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
    val_acc: float,
    checkpoint_path: Path,
    is_best: bool = False
) -> None:
    """
    Save custom checkpoint with specific format.

    Args:
        model: YOLO model instance.
        epoch: Current epoch number.
        optimizer: Optimizer instance.
        val_acc: Validation accuracy.
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
        'val_acc': val_acc,
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


class AirlineClassifierTrainer:
    """
    Wrapper class for training airline classifier with custom logging and checkpointing.
    """

    def __init__(self, config: Dict[str, Any], args: argparse.Namespace, logger: logging.Logger):
        """
        Initialize the airline classifier trainer.

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
            checkpoint_dir = self.training_root / 'ckpt' / 'airline'

        # Add timestamp subdirectory
        self.checkpoint_dir = checkpoint_dir / self.timestamp

        self.best_val_acc = 0.0

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
                # Model is either a name (like "yolov8n-cls.pt") or doesn't exist yet
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
            log_base = self.config.get('log_dir') or (self.training_root / 'logs' / 'airline')
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
        self.logger.info("Starting Airline Classifier Fine-tuning")
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
            'dropout': self.config['dropout'],
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
                # Get metrics from ClassifyMetrics object
                if hasattr(results, 'results_dict'):
                    metrics_dict = results.results_dict
                    for key, value in metrics_dict.items():
                        if isinstance(value, (int, float)):
                            self.logger.info(f"  {key}: {value:.4f}")
                elif hasattr(results, 'top1'):
                    self.logger.info(f"  top1_acc: {results.top1:.4f}")
                    if hasattr(results, 'top5'):
                        self.logger.info(f"  top5_acc: {results.top5:.4f}")
                    if hasattr(results, 'fitness'):
                        self.logger.info(f"  fitness: {results.fitness:.4f}")

            # Save final checkpoint
            if hasattr(self.model, 'trainer') and self.model.trainer:
                self._save_final_checkpoint()

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

        # Get validation accuracy from metrics
        val_acc = 0.0
        if hasattr(trainer, 'metrics') and trainer.metrics:
            # Try to find accuracy in metrics
            for key in ['metrics/accuracy_top1', 'accuracy_top1', 'acc', 'val_acc']:
                if key in trainer.metrics:
                    val_acc = float(trainer.metrics[key])
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
                val_acc,
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
            # Load airline-specific config along with paths
            config_obj = load_config(modules=['airline', 'paths'], load_all_modules=False)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Using default modular configuration...")
        config_obj = load_config(modules=['airline', 'paths'], load_all_modules=False)

    # Get data path - YOLOv8 classification requires a directory, not a YAML file
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
                # Airline classification dataset subdir from config
                dataset_subdir = config_obj.get('airline_data.dataset_subdir', 'aerovision/airline')
                aerovision_path = Path(latest_split_dir) / dataset_subdir
                if aerovision_path.exists():
                    data_path = str(aerovision_path)

    # 3. Fallback: use default path
    if not data_path:
        # Build default path
        data_path = '../data/splits/latest/aerovision/airline'

    # Extract training configuration with defaults
    # Priority: command-line args > config yaml > defaults
    config = {
        # Model configuration
        'model': args.model or config_obj.get('airline_training.model.name') or 'yolov8m-cls.pt',
        'resume': args.resume or config_obj.get('airline_training.resume') or None,

        # Data configuration
        'data': data_path,

        # Training parameters
        'epochs': args.epochs or config_obj.get('airline_training.epochs') or 100,
        'batch_size': args.batch_size or config_obj.get('airline_training.batch_size') or 32,
        'imgsz': args.imgsz or config_obj.get('airline_training.image_size') or 224,

        # Optimizer
        'lr0': args.lr0 or config_obj.get('airline_training.optimizer.lr0') or 0.001,
        'optimizer': args.optimizer or config_obj.get('airline_training.optimizer.type') or 'AdamW',
        'momentum': args.momentum or config_obj.get('airline_training.optimizer.momentum') or 0.937,
        'weight_decay': args.weight_decay or config_obj.get('airline_training.optimizer.weight_decay') or 0.0005,

        # Learning rate scheduler
        'cos_lr': args.cos_lr if args.cos_lr is not None else (
            config_obj.get('airline_training.scheduler.cosine') if config_obj.get('airline_training.scheduler.cosine') is not None else True
        ),
        'lrf': args.lrf or config_obj.get('airline_training.scheduler.lrf') or 0.01,

        # Regularization
        'dropout': args.dropout if args.dropout is not None else (
            config_obj.get('airline_training.regularization.dropout') if config_obj.get('airline_training.regularization.dropout') is not None else 0.1
        ),

        # Early stopping
        'patience': args.patience or config_obj.get('airline_training.early_stopping.patience') or 30,

        # Warmup
        'warmup_epochs': args.warmup_epochs or config_obj.get('airline_training.warmup.epochs') or 3.0,

        # Device
        'device': args.device or config_obj.get('device.default') or '0',
        'workers': args.workers or config_obj.get('airline_training.workers') or 8,
        'amp': args.amp if args.amp is not None else (
            config_obj.get('airline_training.amp') if config_obj.get('airline_training.amp') is not None else True
        ),

        # Reproducibility
        'seed': args.seed or config_obj.get('seed.random') or 42,

        # Saving
        'project': args.project or config_obj.get('airline_training.output.project') or '../output/airline',
        'name': args.name or config_obj.get('airline_training.output.name') or 'airline_classifier',
        'save_period': args.save_period if args.save_period is not None else (
            config_obj.get('airline_training.save_period') if config_obj.get('airline_training.save_period') is not None else -1
        ),

        # Checkpoint directory
        'checkpoint_dir': args.checkpoint_dir or config_obj.get('airline_checkpoints.dir') or config_obj.get('checkpoints.airline') or '../ckpt/airline',

        # Log directory
        'log_dir': args.log_dir or config_obj.get('airline_checkpoints.log_dir') or config_obj.get('logs.airline') or '../logs/airline',

        # Validation and plots
        'val': args.val if args.val is not None else (
            config_obj.get('airline_training.validation.enabled') if config_obj.get('airline_training.validation.enabled') is not None else True
        ),
        'plots': args.plots if args.plots is not None else (
            config_obj.get('airline_training.plots') if config_obj.get('airline_training.plots') is not None else True
        ),

        # TensorBoard logging
        'tensorboard': args.tensorboard if args.tensorboard is not None else (
            config_obj.get('airline_training.tensorboard') if config_obj.get('airline_training.tensorboard') is not None else True
        ),
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
    logger.info("Airline Classifier Training Script")
    logger.info("=" * 60)
    logger.info(f"Dataset path: {config['data']}")
    logger.info(f"Model: {config['model']}")
    logger.info(f"Device: {config['device']}")
    logger.info(f"Epochs: {config['epochs']}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Learning rate: {config['lr0']}")
    logger.info(f"Optimizer: {config['optimizer']}")
    logger.info(f"Dropout: {config['dropout']}")
    logger.info(f"Patience: {config['patience']}")
    logger.info(f"Seed: {config['seed']}")
    logger.info("=" * 60)
    logger.info(f"Training timestamp: {timestamp}")
    logger.info(f"Output paths:")
    logger.info(f"  Project (YOLO output): {config['project']}")
    logger.info(f"  Experiment name: {config['name']}")
    logger.info(f"  Checkpoints: Will be created as ckpt/airline/{timestamp}/")
    logger.info(f"  Logs: {log_dir}")
    logger.info("=" * 60)

    # Verify dataset paths
    data_dir = Path(config['data'])

    if not data_dir.exists():
        logger.error(f"Dataset directory not found: {data_dir}")
        logger.error("Please prepare the airline dataset first using prepare_dataset.py and split_dataset.py")
        sys.exit(1)

    if not data_dir.is_dir():
        logger.error(f"Data path must be a directory, not a file: {data_dir}")
        logger.error("YOLOv8 classification requires directory structure: data_dir/train/, data_dir/val/, etc.")
        sys.exit(1)

    # Check for train/val directories
    train_path = data_dir / 'train'
    val_path = data_dir / 'val'

    if not train_path.exists():
        logger.warning(f"Training path not found: {train_path}")
        logger.warning("Dataset may need to be prepared first using prepare_dataset.py and split_dataset.py")
    else:
        # Count classes and samples
        train_classes = [d for d in train_path.iterdir() if d.is_dir()]
        total_train_samples = sum(len(list(c.glob('*'))) for c in train_classes)
        logger.info(f"Training path verified: {train_path}")
        logger.info(f"  Found {len(train_classes)} airline classes with {total_train_samples} total samples")

    if not val_path.exists():
        logger.warning(f"Validation path not found: {val_path}")
    else:
        val_classes = [d for d in val_path.iterdir() if d.is_dir()]
        total_val_samples = sum(len(list(c.glob('*'))) for c in val_classes)
        logger.info(f"Validation path verified: {val_path}")
        logger.info(f"  Found {len(val_classes)} airline classes with {total_val_samples} total samples")

    # Initialize and run trainer
    try:
        trainer = AirlineClassifierTrainer(config, args, logger)

        # Override timestamp with the one from main (for consistency)
        trainer.timestamp = timestamp

        # Recreate checkpoint directory with correct timestamp
        checkpoint_base = config.get('checkpoint_dir')
        if checkpoint_base:
            checkpoint_dir = trainer._resolve_training_path(checkpoint_base)
        else:
            checkpoint_dir = trainer.training_root / 'ckpt' / 'airline'
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
