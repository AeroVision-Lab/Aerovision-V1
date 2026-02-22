#!/usr/bin/env python3
"""
YOLOv8 Aircraft Classifier Fine-tuning Training Script

This script fine-tunes a pre-trained YOLOv8 classification model for aircraft type classification.
It supports custom configurations via modular YAML configuration system and command-line arguments.

配置说明：
本脚本使用新的模块化配置系统，自动加载以下配置模块：
- training.yaml: 训练参数配置
- paths.yaml: 路径配置
- base.yaml: 基础配置（随机种子等）

Usage:
    # Basic usage (uses config from YAML)
    python train_classify.py

    # Custom parameters
    python train_classify.py --epochs 100 --batch-size 32 --imgsz 224 --device 0

    # Custom dataset path
    python train_classify.py --data path/to/dataset

    # Resume from checkpoint
    python train_classify.py --resume checkpoints/stage2/last.pt

    # Use custom config file
    python train_classify.py --config my_config.yaml
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
from training_utils import (
    FocalLoss,
    ConfidencePenalty,
    Mixup,
    CombinedLoss,
    apply_gradient_accumulation,
    ElasticFaceArcFace,
    ElasticFaceLossWrapper,
    ElasticFaceWithYOLO,
    stack_embeddings,
    l2_norm,
)

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
    logger = logging.getLogger("AircraftClassifier")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers.clear()

    # File handler - detailed logs
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - info level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
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
        description="Fine-tune YOLOv8 model for aircraft type classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-cls.pt",
        help="Pre-trained model path or model name",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        default="../data/prepared/20260102_221524/aerovision/aircraft",
        help="Path to dataset root directory",
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--imgsz", type=int, default=224, help="Input image size")

    # Optimizer arguments
    parser.add_argument(
        "--lr0", type=float, default=0.001, help="Initial learning rate"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        choices=["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"],
        help="Optimizer type",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.937, help="SGD momentum or Adam beta1"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
        help="Weight decay (L2 regularization)",
    )

    # Learning rate scheduler
    parser.add_argument(
        "--cos-lr",
        action="store_true",
        default=True,
        help="Use cosine learning rate scheduler",
    )
    parser.add_argument(
        "--lrf", type=float, default=0.01, help="Final learning rate fraction"
    )

    # Regularization
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout for classification head"
    )

    # Early stopping
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (epochs without improvement)",
    )

    # Warmup
    parser.add_argument(
        "--warmup-epochs", type=float, default=3.0, help="Warmup epochs"
    )

    # Device and performance
    parser.add_argument(
        "--device", type=str, default="0", help="Device to use (e.g., 0, 1, cpu, mps)"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use Automatic Mixed Precision training",
    )

    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Saving
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Project directory for saving results (relative to training/)",
    )
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Save checkpoint every N epochs (-1 to disable)",
    )

    # Checkpoint directory (custom)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save custom checkpoints (relative to training/)",
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file (optional, will use modular config if not specified)",
    )

    # Logging
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to save log files (relative to training/)",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=True,
        help="Enable TensorBoard logging",
    )

    # Validation
    parser.add_argument(
        "--val",
        action="store_true",
        default=True,
        help="Run validation during training",
    )

    # Plots
    parser.add_argument(
        "--plots",
        action="store_true",
        default=True,
        help="Save plots and images during training",
    )

    # Advanced features
    parser.add_argument(
        "--accumulate",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1, no accumulation)",
    )
    parser.add_argument(
        "--confidence-penalty",
        type=float,
        default=0.0,
        help="Confidence penalty weight (default: 0.0, disabled)",
    )
    parser.add_argument(
        "--focal-loss",
        action="store_true",
        default=False,
        help="Use Focal Loss instead of Cross Entropy",
    )
    parser.add_argument(
        "--focal-alpha", type=float, default=0.25, help="Focal Loss alpha parameter"
    )
    parser.add_argument(
        "--focal-gamma", type=float, default=2.0, help="Focal Loss gamma parameter"
    )
    parser.add_argument(
        "--mixup",
        action="store_true",
        default=False,
        help="Use Mixup data augmentation",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.4,
        help="Mixup alpha parameter for Beta distribution",
    )

    # ElasticFace loss arguments
    parser.add_argument(
        "--elastic-face",
        action="store_true",
        default=False,
        help="Enable ElasticFace loss (combined with CE loss)",
    )
    parser.add_argument(
        "--elastic-lambda",
        type=float,
        default=0.5,
        help="Lambda weight for ElasticFace loss (total = CE + lambda * ElasticFace)",
    )
    parser.add_argument(
        "--elastic-s",
        type=float,
        default=30.0,
        help="Scale factor for ElasticFace logits",
    )
    parser.add_argument(
        "--elastic-m",
        type=float,
        default=0.30,
        help="Base margin for ElasticFace angular penalty",
    )
    parser.add_argument(
        "--elastic-std",
        type=float,
        default=0.01,
        help="Standard deviation for ElasticFace margin sampling",
    )
    parser.add_argument(
        "--elastic-plus",
        action="store_true",
        default=False,
        help="Use ElasticArcFace+ (adaptive margin based on difficulty)",
    )
    parser.add_argument(
        "--elastic-type",
        type=str,
        default="arc",
        choices=["cos", "arc", "cos+", "arc+"],
        help="ElasticFace loss type: cos (CosFace), arc (ArcFace), cos+ (CosFace+), arc+ (ArcFace+)",
    )

    return parser.parse_args()


def save_custom_checkpoint(
    model: YOLO,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    val_acc: float,
    checkpoint_path: Path,
    is_best: bool = False,
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
        "epoch": epoch,
        "model_state_dict": pt_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "model_args": pt_model.args if hasattr(pt_model, "args") else {},
        "names": pt_model.names if hasattr(pt_model, "names") else {},
        "date": datetime.now().isoformat(),
    }

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)

    # Save best checkpoint separately
    if is_best:
        best_path = checkpoint_path.parent / "best.pt"
        torch.save(checkpoint, best_path)


class AircraftClassifierTrainer:
    """
    Wrapper class for training aircraft classifier with custom logging and checkpointing.
    """

    def __init__(
        self, config: Dict[str, Any], args: argparse.Namespace, logger: logging.Logger
    ):
        """
        Initialize the aircraft classifier trainer.

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
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup checkpoint directory with timestamp
        # 优先从 config 读取（config 已经合并了 yaml 和 args）
        checkpoint_base = config.get("checkpoint_dir") or args.checkpoint_dir
        if checkpoint_base:
            checkpoint_dir = self._resolve_training_path(checkpoint_base)
        else:
            checkpoint_dir = self.training_root / "ckpt" / "classify"

        # Add timestamp subdirectory
        self.checkpoint_dir = checkpoint_dir / self.timestamp

        self.best_val_acc = 0.0

        # Setup directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize advanced features
        self._init_advanced_features()

        # Initialize model
        self._init_model()

        # Setup TensorBoard
        self.tb_writer = None
        # 优先使用 config 中的配置
        use_tensorboard = config.get(
            "tensorboard", args.tensorboard if hasattr(args, "tensorboard") else True
        )
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
        config_dir = self.training_root / "configs"
        return (config_dir / path).resolve()

    def _init_advanced_features(self) -> None:
        """Initialize advanced training features."""
        # Gradient accumulation
        self.accumulate = self.config.get("accumulate", 1)
        if self.accumulate > 1:
            self.logger.info(f"Gradient accumulation enabled: {self.accumulate} steps")

        # Confidence Penalty
        self.confidence_penalty_weight = self.config.get("confidence_penalty", 0.0)
        if self.confidence_penalty_weight > 0:
            self.logger.info(
                f"Confidence Penalty enabled: weight={self.confidence_penalty_weight}"
            )

        # Focal Loss
        self.use_focal_loss = self.config.get("focal_loss", False)
        self.focal_alpha = self.config.get("focal_alpha", 0.25)
        self.focal_gamma = self.config.get("focal_gamma", 2.0)
        if self.use_focal_loss:
            self.logger.info(
                f"Focal Loss enabled: alpha={self.focal_alpha}, gamma={self.focal_gamma}"
            )

        # Mixup
        self.use_mixup = self.config.get("mixup", False)
        self.mixup_alpha = self.config.get("mixup_alpha", 0.4)
        if self.use_mixup:
            self.mixup = Mixup(alpha=self.mixup_alpha)
            self.logger.info(f"Mixup enabled: alpha={self.mixup_alpha}")
        else:
            self.mixup = None

        # ElasticFace Loss
        self.use_elastic_face = self.config.get("elastic_face", False)
        self.elastic_lambda = self.config.get("elastic_lambda", 0.5)
        self.elastic_s = self.config.get("elastic_s", 30.0)
        self.elastic_m = self.config.get("elastic_m", 0.30)
        self.elastic_std = self.config.get("elastic_std", 0.01)
        self.elastic_plus = self.config.get("elastic_plus", False)
        self.elastic_type = self.config.get("elastic_type", "arc")
        self.elastic_face_wrapper = None  # Will be initialized after model is loaded

        if self.use_elastic_face:
            self.logger.info(
                f"ElasticFace enabled: type={self.elastic_type}, lambda={self.elastic_lambda}, "
                f"s={self.elastic_s}, m={self.elastic_m}, std={self.elastic_std}"
            )

        # Initialize loss function (will be configured after model is loaded)
        self.loss_fn = None

    def _init_model(self) -> None:
        """Initialize the YOLO model for training."""
        # 优先从 config 读取 resume 参数
        resume_checkpoint = self.config.get("resume") or (
            self.args.resume if self.args else None
        )

        if resume_checkpoint:
            self.logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")
            self.model = YOLO(resume_checkpoint)
        else:
            model_path = self.config["model"]
            model_dir = self.training_root / "model"
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
                    # Use absolute path to ensure it downloads to model directory
                    self.logger.info(
                        f"Model not found locally, will download to: {model_dir}"
                    )
                    # Download to model directory by using the absolute path
                    import os

                    original_dir = os.getcwd()
                    try:
                        # Change to model directory temporarily
                        os.chdir(str(model_dir))
                        self.model = YOLO(model_filename)
                        # Change back
                        os.chdir(original_dir)
                        self.logger.info(
                            f"Model downloaded to: {model_dir / model_filename}"
                        )
                    except Exception as e:
                        os.chdir(original_dir)
                        raise e

                # Update config with actual model path
                self.config["model"] = str(local_model_path)

        # Initialize ElasticFace wrapper after model is loaded
        self._init_elastic_face()

    def _init_elastic_face(self) -> None:
        """Initialize ElasticFace loss wrapper after model is loaded."""
        if not self.use_elastic_face:
            return

        try:
            # Get number of classes from dataset
            data_path = Path(self.config["data"])
            train_path = data_path / "train"
            if train_path.exists():
                # Count number of class directories
                num_classes = len([d for d in train_path.iterdir() if d.is_dir()])
            else:
                # Fallback to a default value
                num_classes = 10
                self.logger.warning(
                    f"Could not determine num_classes from dataset, using default: {num_classes}"
                )

            self.elastic_face_wrapper = ElasticFaceLossWrapper.from_yolo_model(
                yolo_model=self.model,
                num_classes=num_classes,
                lambda_weight=self.elastic_lambda,
                s=self.elastic_s,
                m=self.elastic_m,
                std=self.elastic_std,
                type=self.elastic_type,
                plus=self.elastic_plus,
            )
            self.logger.info(
                f"ElasticFace wrapper initialized: num_classes={num_classes}, "
                f"embedding_dim={self.elastic_face_wrapper.embedding_dim}"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize ElasticFace wrapper: {e}")
            self.elastic_face_wrapper = None
            self.use_elastic_face = False

    def _setup_tensorboard(self) -> None:
        """Setup TensorBoard writer for logging."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            # Get log directory from config or use default
            log_base = self.config.get("log_dir") or (
                self.training_root / "logs" / "classify"
            )
            if isinstance(log_base, str):
                log_base = self._resolve_training_path(log_base)

            # Add timestamp subdirectory (use same timestamp as checkpoint)
            log_dir = log_base / self.timestamp
            self.tb_writer = SummaryWriter(log_dir)
            self.logger.info(f"TensorBoard logging enabled: {log_dir}")
        except ImportError:
            self.logger.warning(
                "TensorBoard not available. Install with: pip install tensorboard"
            )
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
                self.tb_writer.add_scalar(f"train/{key}", value, epoch)
            self.tb_writer.flush()

    def train(self) -> None:
        """
        Execute the training process.

        This method runs the complete training loop including validation,
        checkpointing, and logging.
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Aircraft Classifier Fine-tuning")
        self.logger.info("=" * 60)

        # Log configuration
        self.logger.info("Configuration:")
        for key, value in self.config.items():
            self.logger.info(f"  {key}: {value}")

        # Ensure all YOLO downloads go to model directory
        import os

        original_dir = os.getcwd()
        model_dir = self.training_root / "model"

        # Prepare training arguments for YOLO
        train_args = {
            "data": self.config["data"],
            "epochs": self.config["epochs"],
            "batch": self.config["batch_size"],
            "imgsz": self.config["imgsz"],
            "lr0": self.config["lr0"],
            "optimizer": self.config["optimizer"],
            "momentum": self.config["momentum"],
            "weight_decay": self.config["weight_decay"],
            "cos_lr": self.config["cos_lr"],
            "lrf": self.config["lrf"],
            "dropout": self.config["dropout"],
            "patience": self.config["patience"],
            "warmup_epochs": self.config["warmup_epochs"],
            "device": self.config["device"],
            "workers": self.config["workers"],
            "amp": self.config["amp"],
            "seed": self.config["seed"],
            "project": self.config["project"],
            "name": self.config["name"],
            "save_period": self.config["save_period"],
            "val": self.config["val"],
            "plots": self.config["plots"],
            "verbose": True,
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
                if hasattr(results, "results_dict"):
                    metrics_dict = results.results_dict
                    for key, value in metrics_dict.items():
                        if isinstance(value, (int, float)):
                            self.logger.info(f"  {key}: {value:.4f}")
                elif hasattr(results, "top1"):
                    self.logger.info(f"  top1_acc: {results.top1:.4f}")
                    if hasattr(results, "top5"):
                        self.logger.info(f"  top5_acc: {results.top5:.4f}")
                    if hasattr(results, "fitness"):
                        self.logger.info(f"  fitness: {results.fitness:.4f}")

            # Save final checkpoint
            if hasattr(self.model, "trainer") and self.model.trainer:
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
        if hasattr(trainer, "metrics") and trainer.metrics:
            # Try to find accuracy in metrics
            for key in ["metrics/accuracy_top1", "accuracy_top1", "acc", "val_acc"]:
                if key in trainer.metrics:
                    val_acc = float(trainer.metrics[key])
                    break

        # Get optimizer
        optimizer = trainer.optimizer if hasattr(trainer, "optimizer") else None

        # Save last checkpoint
        last_path = self.checkpoint_dir / "last.pt"
        if optimizer:
            save_custom_checkpoint(
                self.model, trainer.epoch, optimizer, val_acc, last_path, is_best=False
            )
            self.logger.info(f"Saved last checkpoint to: {last_path}")

        # Save best checkpoint if available
        best_path = self.checkpoint_dir / "best.pt"
        if hasattr(trainer, "best") and trainer.best.exists():
            import shutil

            shutil.copy(trainer.best, best_path)
            self.logger.info(f"Copied best checkpoint to: {best_path}")


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
    # This ensures models are downloaded to training/model directory
    import os

    training_root = Path(__file__).parent.parent
    model_dir = training_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Set YOLO environment variables
    os.environ["YOLO_CONFIG_DIR"] = str(training_root / "model")
    # Ultralytics uses this for model downloads
    os.environ.setdefault("TORCH_HOME", str(model_dir))

    # Parse arguments
    args = parse_arguments()

    # Load configuration from modular system
    try:
        if args.config and Path(args.config).exists():
            config_obj = load_config(args.config)
        else:
            config_obj = load_config(
                modules=["training", "paths"], load_all_modules=False
            )
    except FileNotFoundError as e:
        # If config file not found, use default modular config
        print(f"Warning: {e}")
        print("Using default modular configuration...")
        config_obj = load_config(modules=["training", "paths"], load_all_modules=False)

    # Get data path - YOLOv8 classification requires a directory, not a YAML file
    # 优先从 config yaml 读取（尝试从 latest.txt 读取最新的 split 目录）
    data_path = None

    # 1. 尝试从 splits/latest.txt 读取
    splits_root = config_obj.get_path("data.splits.root")
    if splits_root:
        latest_txt = Path(splits_root) / "latest.txt"
        if latest_txt.exists():
            with open(latest_txt, "r", encoding="utf-8") as f:
                latest_split_dir = f.read().strip()
            # Aerovision 分类数据集路径
            aerovision_path = Path(latest_split_dir) / "aerovision" / "aircraft"
            if aerovision_path.exists():
                data_path = str(aerovision_path)

    # 2. 如果没找到，使用命令行参数或默认值
    if not data_path:
        data_path = args.data

    # Extract training configuration with defaults
    # 优先级：config yaml > 命令行参数 > 默认值
    config = {
        # Model configuration - resolve model path from config
        "model": config_obj.get("training.model.name")
        or args.model
        or "yolov8n-cls.pt",
        "resume": config_obj.get("training.resume") or args.resume or None,
        # Data configuration (must be a directory for classification)
        "data": data_path,
        # Training parameters
        "epochs": config_obj.get("training.epochs") or args.epochs or 100,
        "batch_size": config_obj.get("training.batch_size") or args.batch_size or 32,
        "imgsz": config_obj.get("training.image_size") or args.imgsz or 224,
        # Optimizer
        "lr0": config_obj.get("training.optimizer.lr0") or args.lr0 or 0.001,
        "optimizer": config_obj.get("training.optimizer.type")
        or args.optimizer
        or "AdamW",
        "momentum": config_obj.get("training.optimizer.momentum")
        or args.momentum
        or 0.937,
        "weight_decay": config_obj.get("training.optimizer.weight_decay")
        or args.weight_decay
        or 0.0005,
        # Learning rate scheduler
        "cos_lr": config_obj.get("training.scheduler.cosine")
        if config_obj.get("training.scheduler.cosine") is not None
        else (args.cos_lr if hasattr(args, "cos_lr") else True),
        "lrf": config_obj.get("training.scheduler.lrf") or args.lrf or 0.01,
        # Regularization
        "dropout": config_obj.get("training.regularization.dropout")
        if config_obj.get("training.regularization.dropout") is not None
        else (args.dropout or 0.0),
        # Early stopping
        "patience": config_obj.get("training.early_stopping.patience")
        or args.patience
        or 50,
        # Warmup
        "warmup_epochs": config_obj.get("training.warmup.epochs")
        or args.warmup_epochs
        or 3.0,
        # Device
        "device": config_obj.get("device.default") or args.device or "0",
        "workers": config_obj.get("training.workers") or args.workers or 8,
        "amp": config_obj.get("training.amp")
        if config_obj.get("training.amp") is not None
        else (args.amp if hasattr(args, "amp") else True),
        # Reproducibility
        "seed": config_obj.get("seed.random") or args.seed or 42,
        # Saving - resolve all paths relative to training/ directory
        # Note: timestamp will be added to name later
        "project": config_obj.get("training.output.project")
        or args.project
        or "../output/classify",
        "name": config_obj.get("training.output.name")
        or args.name
        or "aircraft_classifier",
        "save_period": config_obj.get("training.save_period")
        if config_obj.get("training.save_period") is not None
        else (args.save_period or -1),
        # Store timestamp for directory naming
        "timestamp": None,  # Will be set later
        # Checkpoint directory
        "checkpoint_dir": config_obj.get("checkpoints.classify")
        or args.checkpoint_dir
        or "../ckpt/classify",
        # Log directory
        "log_dir": config_obj.get("logs.classify")
        or args.log_dir
        or "../logs/classify",
        # Validation and plots
        "val": config_obj.get("training.validation.enabled")
        if config_obj.get("training.validation.enabled") is not None
        else (args.val if hasattr(args, "val") else True),
        "plots": config_obj.get("training.plots")
        if config_obj.get("training.plots") is not None
        else (args.plots if hasattr(args, "plots") else True),
        # TensorBoard logging
        "tensorboard": config_obj.get("training.tensorboard")
        if config_obj.get("training.tensorboard") is not None
        else (args.tensorboard if hasattr(args, "tensorboard") else True),
        # Advanced features
        "accumulate": config_obj.get("training.advanced.accumulate")
        or args.accumulate
        or 1,
        "confidence_penalty": config_obj.get("training.advanced.confidence_penalty")
        or args.confidence_penalty
        or 0.0,
        "focal_loss": config_obj.get("training.advanced.focal_loss.enabled")
        if config_obj.get("training.advanced.focal_loss.enabled") is not None
        else args.focal_loss,
        "focal_alpha": config_obj.get("training.advanced.focal_loss.alpha")
        or args.focal_alpha
        or 0.25,
        "focal_gamma": config_obj.get("training.advanced.focal_loss.gamma")
        or args.focal_gamma
        or 2.0,
        "mixup": config_obj.get("training.advanced.mixup.enabled")
        if config_obj.get("training.advanced.mixup.enabled") is not None
        else args.mixup,
        "mixup_alpha": config_obj.get("training.advanced.mixup.alpha")
        or args.mixup_alpha
        or 0.4,
        # ElasticFace loss configuration
        "elastic_face": config_obj.get("training.advanced.elastic_face.enabled")
        if config_obj.get("training.advanced.elastic_face.enabled") is not None
        else args.elastic_face,
        "elastic_lambda": config_obj.get("training.advanced.elastic_face.lambda_weight")
        or args.elastic_lambda
        or 0.5,
        "elastic_s": config_obj.get("training.advanced.elastic_face.s")
        or args.elastic_s
        or 30.0,
        "elastic_m": config_obj.get("training.advanced.elastic_face.m")
        or args.elastic_m
        or 0.30,
        "elastic_std": config_obj.get("training.advanced.elastic_face.std")
        or args.elastic_std
        or 0.01,
        "elastic_plus": config_obj.get("training.advanced.elastic_face.plus")
        if config_obj.get("training.advanced.elastic_face.plus") is not None
        else args.elastic_plus,
        "elastic_type": config_obj.get("training.advanced.elastic_face.type")
        if config_obj.get("training.advanced.elastic_face.type") is not None
        else args.elastic_type,
    }

    # Helper function to resolve config paths
    # Config paths are relative to /training/configs, so ../xxx means /training/xxx
    def resolve_config_path(path_str: str) -> str:
        """Resolve path from config (relative to training/configs) to absolute path."""
        if Path(path_str).is_absolute():
            return path_str

        training_root = Path(__file__).parent.parent  # /training
        config_dir = training_root / "configs"  # /training/configs

        # Resolve relative to config dir
        resolved = (config_dir / path_str).resolve()
        return str(resolved)

    # Resolve data path
    if not Path(data_path).is_absolute():
        data_path = resolve_config_path(data_path)
    config["data"] = data_path

    # Resolve model path
    model_path = config["model"]
    if not Path(model_path).is_absolute() and not Path(model_path).exists():
        # Try to find model in config paths
        model_name = Path(model_path).stem.replace(
            "-", "_"
        )  # yolov8n-cls -> yolov8n_cls
        config_model_path = config_obj.get(f"models.pretrained.{model_name}")

        if config_model_path:
            # Resolve from config
            resolved_model = resolve_config_path(config_model_path)
            if Path(resolved_model).exists():
                config["model"] = resolved_model
            # else: keep original path, will be downloaded by YOLO
        else:
            # Check in model directory from config
            model_dir = config_obj.get("models.root")
            if model_dir:
                resolved_model = resolve_config_path(
                    f"{model_dir.rstrip('/')}/{model_path}"
                )
                if Path(resolved_model).exists():
                    config["model"] = resolved_model
        # If model doesn't exist locally, YOLO will auto-download it

    # Generate timestamp for this training session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve project path
    project_path = config["project"]
    if not Path(project_path).is_absolute():
        config["project"] = resolve_config_path(project_path)

    # Add timestamp to experiment name
    config["name"] = f"{config['name']}_{timestamp}"
    config["timestamp"] = timestamp

    # Resolve log directory path
    log_dir_path = config["log_dir"]
    if not Path(log_dir_path).is_absolute():
        log_dir_path = resolve_config_path(log_dir_path)

    log_dir = Path(log_dir_path)

    # Add timestamp subdirectory for this training session
    log_dir = log_dir / timestamp
    logger = setup_logging(log_dir)

    # Log startup information
    logger.info("=" * 60)
    logger.info("Aircraft Classifier Training Script")
    logger.info("=" * 60)
    logger.info(f"Dataset path: {config['data']}")
    logger.info(f"Model: {config['model']}")
    logger.info(f"Device: {config['device']}")
    logger.info(f"Epochs: {config['epochs']}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Learning rate: {config['lr0']}")
    logger.info(f"Optimizer: {config['optimizer']}")
    logger.info(f"Seed: {config['seed']}")
    logger.info("=" * 60)
    logger.info(f"Training timestamp: {timestamp}")
    logger.info(f"Output paths:")
    logger.info(f"  Project (YOLO output): {config['project']}")
    logger.info(f"  Experiment name: {config['name']}")
    logger.info(f"  Checkpoints: Will be created as ckpt/classify/{timestamp}/")
    logger.info(f"  Logs: {log_dir}")
    logger.info("=" * 60)

    # Verify dataset paths
    # YOLOv8 classification requires a directory structure with train/val/test subdirectories
    data_dir = Path(config["data"])

    if not data_dir.exists():
        logger.error(f"Dataset directory not found: {data_dir}")
        sys.exit(1)

    if not data_dir.is_dir():
        logger.error(f"Data path must be a directory, not a file: {data_dir}")
        logger.error(
            "YOLOv8 classification requires directory structure: data_dir/train/, data_dir/val/, etc."
        )
        sys.exit(1)

    # Check for train/val directories
    train_path = data_dir / "train"
    val_path = data_dir / "val"

    if not train_path.exists():
        logger.warning(f"Training path not found: {train_path}")
        logger.warning("Dataset may need to be prepared first using prepare_dataset.py")
    else:
        logger.info(f"Training path verified: {train_path}")

    if not val_path.exists():
        logger.warning(f"Validation path not found: {val_path}")
    else:
        logger.info(f"Validation path verified: {val_path}")

    # Initialize and run trainer
    try:
        # Pass timestamp from config to ensure consistency
        trainer = AircraftClassifierTrainer(config, args, logger)

        # Override timestamp with the one from main (for consistency)
        trainer.timestamp = timestamp

        # Recreate checkpoint directory with correct timestamp
        # 优先从 config 读取（config 已经合并了 yaml 和 args）
        checkpoint_base = config.get("checkpoint_dir") or args.checkpoint_dir
        if checkpoint_base:
            checkpoint_dir = trainer._resolve_training_path(checkpoint_base)
        else:
            checkpoint_dir = trainer.training_root / "ckpt" / "classify"
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


if __name__ == "__main__":
    main()
