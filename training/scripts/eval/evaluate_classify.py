#!/usr/bin/env python3
"""
YOLOv8 Aircraft Classifier Evaluation Script

This script evaluates a trained YOLOv8 classification model on the test dataset.
It computes the following metrics:
- Overall Accuracy
- Recall (per class and macro-average)
- Expected Calibration Error (ECE)
- Per-class accuracy for each aircraft type

Usage:
    # Basic usage
    python evaluate_classify.py

    # Specify model path
    python evaluate_classify.py --model ckpt/classify/20260122_090826/best.pt

    # Specify dataset path
    python evaluate_classify.py --data path/to/dataset

    # Specify number of bins for ECE calculation
    python evaluate_classify.py --bins 15
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import recall_score, confusion_matrix

from ultralytics import YOLO


def setup_logging() -> logging.Logger:
    """
    Configure logging for the evaluation process.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("AircraftClassifierEval")
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
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
    Parse command-line arguments for evaluation configuration.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8 aircraft classifier on test dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model checkpoint (.pt file)",
    )

    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset root directory",
    )

    # ECE arguments
    parser.add_argument(
        "--bins",
        type=int,
        default=15,
        help="Number of bins for ECE calculation",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use (e.g., 0, 1, cpu, mps)",
    )

    # Batch size
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )

    # Image size
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Input image size",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results",
    )

    return parser.parse_args()


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, List[float], List[float]]:
    """
    Compute Expected Calibration Error (ECE).

    Args:
        confidences: Model confidence scores (N,)
        predictions: Predicted class indices (N,)
        labels: Ground truth class indices (N,)
        n_bins: Number of bins for ECE calculation

    Returns:
        Tuple of (ece, bin_boundaries, bin_accuracies)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_accuracies = []
    bin_confidences = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            # Compute accuracy and average confidence in this bin
            accuracy_in_bin = predictions[in_bin] == labels[in_bin]
            avg_accuracy = accuracy_in_bin.mean()
            avg_confidence = confidences[in_bin].mean()

            # Weighted contribution to ECE
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

            bin_accuracies.append(avg_accuracy)
            bin_confidences.append(avg_confidence)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)

    return ece, bin_boundaries.tolist(), bin_accuracies


def evaluate_model(
    model: YOLO,
    data_path: Path,
    batch_size: int,
    imgsz: int,
    device: str,
    n_bins: int,
    logger: logging.Logger,
) -> Dict:
    """
    Evaluate the model on the test dataset.

    Args:
        model: YOLO model instance
        data_path: Path to dataset directory
        batch_size: Batch size for evaluation
        imgsz: Input image size
        device: Device to use
        n_bins: Number of bins for ECE calculation
        logger: Logger instance

    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info("=" * 60)
    logger.info("Starting Model Evaluation")
    logger.info("=" * 60)

    # Check for test directory
    test_path = data_path / "test"
    if not test_path.exists():
        logger.warning(f"Test directory not found: {test_path}")
        logger.info("Falling back to validation directory...")
        test_path = data_path / "val"

    if not test_path.exists():
        logger.error(f"Neither test nor val directory found in {data_path}")
        sys.exit(1)

    logger.info(f"Using test data from: {test_path}")

    # Get class names
    class_names = list(sorted(test_path.iterdir()))
    class_names = [d.name for d in class_names if d.is_dir()]
    num_classes = len(class_names)

    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {class_names}")

    # Load test data and perform inference
    logger.info(f"Running inference with batch_size={batch_size}, imgsz={imgsz}")

    all_predictions = []
    all_labels = []
    all_confidences = []

    # Process each class
    for class_idx, class_name in enumerate(class_names):
        class_path = test_path / class_name
        image_files = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))

        if not image_files:
            logger.warning(f"No images found for class: {class_name}")
            continue

        logger.info(f"Processing class '{class_name}': {len(image_files)} images")

        # Batch inference
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]

            # Run inference
            results = model.predict(
                [str(f) for f in batch_files],
                imgsz=imgsz,
                device=device,
                verbose=False,
            )

            for result, img_file in zip(results, batch_files):
                # Get prediction
                probs = result.probs
                pred_class = probs.top1
                confidence = probs.data[pred_class].item()

                all_predictions.append(pred_class)
                all_labels.append(class_idx)
                all_confidences.append(confidence)

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)

    total_samples = len(all_labels)
    logger.info(f"Total samples evaluated: {total_samples}")

    if total_samples == 0:
        logger.error("No samples found for evaluation!")
        sys.exit(1)

    # Compute metrics
    logger.info("\nComputing metrics...")

    # 1. Overall Accuracy
    accuracy = (all_predictions == all_labels).mean()
    logger.info(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # 2. Recall (per class and macro-average)
    recall_per_class = recall_score(
        all_labels, all_predictions, average=None, zero_division=0
    )
    macro_recall = recall_per_class.mean()
    logger.info(f"Macro Recall: {macro_recall:.4f} ({macro_recall*100:.2f}%)")

    # 3. ECE
    ece, bin_boundaries, bin_accuracies = compute_ece(
        all_confidences, all_predictions, all_labels, n_bins
    )
    logger.info(f"Expected Calibration Error (ECE, {n_bins} bins): {ece:.4f}")

    # 4. Per-class accuracy
    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(num_classes)))
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    per_class_accuracy = np.nan_to_num(per_class_accuracy, nan=0.0)

    logger.info("\nPer-class Accuracy:")
    for class_name, acc in zip(class_names, per_class_accuracy):
        logger.info(f"  {class_name}: {acc:.4f} ({acc*100:.2f}%)")

    # 5. Additional statistics
    logger.info("\nConfidence Statistics:")
    logger.info(f"  Mean confidence: {all_confidences.mean():.4f}")
    logger.info(f"  Min confidence: {all_confidences.min():.4f}")
    logger.info(f"  Max confidence: {all_confidences.max():.4f}")
    logger.info(f"  Std confidence: {all_confidences.std():.4f}")

    # Return results
    results = {
        "accuracy": accuracy,
        "macro_recall": macro_recall,
        "ece": ece,
        "n_bins": n_bins,
        "total_samples": total_samples,
        "num_classes": num_classes,
        "class_names": class_names,
        "per_class_accuracy": per_class_accuracy.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "bin_boundaries": bin_boundaries,
        "bin_accuracies": bin_accuracies,
        "confidence_mean": float(all_confidences.mean()),
        "confidence_std": float(all_confidences.std()),
        "confusion_matrix": cm.tolist(),
    }

    return results


def save_results(results: Dict, output_dir: Path, logger: logging.Logger) -> None:
    """
    Save evaluation results to a JSON file.

    Args:
        results: Dictionary containing evaluation metrics
        output_dir: Directory to save results
        logger: Logger instance
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"evaluation_{timestamp}.json"

    import json

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to: {output_file}")


def print_summary(results: Dict, logger: logging.Logger) -> None:
    """
    Print a summary of evaluation results.

    Args:
        results: Dictionary containing evaluation metrics
        logger: Logger instance
    """
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Samples: {results['total_samples']}")
    logger.info(f"Number of Classes: {results['num_classes']}")
    logger.info("")
    logger.info(f"Overall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    logger.info(f"Macro Recall: {results['macro_recall']:.4f} ({results['macro_recall']*100:.2f}%)")
    logger.info(f"ECE ({results['n_bins']} bins): {results['ece']:.4f}")
    logger.info("")
    logger.info("Per-class Accuracy (Top 10 best/worst):")

    # Sort classes by accuracy
    class_acc = list(zip(results['class_names'], results['per_class_accuracy']))
    class_acc_sorted = sorted(class_acc, key=lambda x: x[1], reverse=True)

    # Top 5
    logger.info("  Top 5:")
    for class_name, acc in class_acc_sorted[:5]:
        logger.info(f"    {class_name}: {acc:.4f} ({acc*100:.2f}%)")

    # Bottom 5
    if len(class_acc_sorted) > 5:
        logger.info("  Bottom 5:")
        for class_name, acc in class_acc_sorted[-5:]:
            logger.info(f"    {class_name}: {acc:.4f} ({acc*100:.2f}%)")

    logger.info("=" * 60)


def main() -> None:
    """
    Main entry point for the evaluation script.
    """
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("Aircraft Classifier Evaluation Script")
    logger.info("=" * 60)

    # Get training root directory
    training_root = Path(__file__).parent.parent

    # Determine model path
    if args.model:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = (training_root / args.model).resolve()
    else:
        # Try to find the latest best.pt checkpoint
        checkpoint_dir = training_root / "ckpt" / "classify"
        if checkpoint_dir.exists():
            # Find all timestamp directories
            timestamp_dirs = sorted(
                [d for d in checkpoint_dir.iterdir() if d.is_dir()],
                key=lambda x: x.name,
                reverse=True,
            )

            if timestamp_dirs:
                # Try to find best.pt in the latest directory
                latest_dir = timestamp_dirs[0]
                best_pt = latest_dir / "best.pt"
                if best_pt.exists():
                    model_path = best_pt
                    logger.info(f"Auto-detected model: {model_path}")
                else:
                    # Try last.pt
                    last_pt = latest_dir / "last.pt"
                    if last_pt.exists():
                        model_path = last_pt
                        logger.info(f"Auto-detected model: {model_path}")
                    else:
                        logger.error(f"No checkpoint found in {latest_dir}")
                        sys.exit(1)
            else:
                logger.error(f"No checkpoint directories found in {checkpoint_dir}")
                sys.exit(1)
        else:
            logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
            sys.exit(1)

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)

    # Determine data path
    if args.data:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = (training_root / args.data).resolve()
    else:
        # Try to load from config
        try:
            sys.path.insert(0, str(training_root / "configs"))
            from configs import load_config

            config_obj = load_config(modules=["training", "paths"], load_all_modules=False)

            # Try to read from splits/latest.txt
            splits_root = config_obj.get_path("data.splits.root")
            if splits_root:
                latest_txt = Path(splits_root) / "latest.txt"
                if latest_txt.exists():
                    with open(latest_txt, "r", encoding="utf-8") as f:
                        latest_split_dir = f.read().strip()
                    aerovision_path = Path(latest_split_dir) / "aerovision" / "aircraft"
                    if aerovision_path.exists():
                        data_path = aerovision_path
                        logger.info(f"Auto-detected dataset: {data_path}")
                    else:
                        logger.error(f"Dataset not found: {aerovision_path}")
                        sys.exit(1)
                else:
                    logger.error(f"Latest split file not found: {latest_txt}")
                    sys.exit(1)
            else:
                logger.error("Could not determine data path from config")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            logger.error("Please specify --data argument")
            sys.exit(1)

    if not data_path.exists():
        logger.error(f"Dataset directory not found: {data_path}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = (training_root / args.output_dir).resolve()
    else:
        output_dir = training_root / "logs" / "evaluate"

    # Load model
    logger.info(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))

    # Evaluate
    results = evaluate_model(
        model=model,
        data_path=data_path,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        n_bins=args.bins,
        logger=logger,
    )

    # Save results
    save_results(results, output_dir, logger)

    # Print summary
    print_summary(results, logger)

    logger.info("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
