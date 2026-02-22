"""
训练增强工具模块

实现以下功能：
1. 梯度累计 (Gradient Accumulation)
2. Confidence Penalty 正则化
3. Focal Loss
4. Mixup 数据增强
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance

    Args:
        alpha (float): Weighting factor in range (0, 1) to balance positive/negative examples.
            Default: 0.25
        gamma (float): Focusing parameter for modulating loss. Higher gamma puts more focus
            on hard, misclassified examples. Default: 2.0
        reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', 'sum'.
            Default: 'mean'
    """

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) where C is the number of classes
            targets: (N,) where each value is in range [0, C-1]

        Returns:
            Loss tensor
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        probs = F.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ConfidencePenalty(nn.Module):
    """Confidence Penalty for preventing overconfidence

    Encourages the model to maintain higher entropy, preventing overconfidence
    on incorrect predictions.

    Args:
        num_classes (int): Number of classes in the classification task
        weight (float): Weight for the confidence penalty term. Default: 0.1
    """

    def __init__(self, num_classes: int, weight: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.max_entropy = np.log(num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) logits from the model

        Returns:
            Confidence penalty loss
        """
        probs = F.softmax(inputs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

        penalty = entropy - self.max_entropy
        return self.weight * torch.mean(penalty)


class Mixup:
    """Mixup data augmentation

    Implements mixup augmentation from "mixup: Beyond Empirical Risk Minimization"
    (https://arxiv.org/abs/1710.09412)

    Args:
        alpha (float): Alpha parameter for Beta distribution. Default: 0.4
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def __call__(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup to a batch of images and labels.

        Args:
            images: (N, C, H, W) batch of images
            labels: (N,) batch of labels

        Returns:
            mixed_images: (N, C, H, W) mixed images
            labels_a: (N,) original labels
            labels_b: (N,) shuffled labels for mixing
            lam: mixing coefficient
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        mixed_images = lam * images + (1 - lam) * images[index, :]
        labels_a, labels_b = labels, labels[index]

        return mixed_images, labels_a, labels_b, lam

    def mixup_criterion(
        self,
        criterion: nn.Module,
        pred: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """
        Compute loss for mixed labels.

        Args:
            criterion: Loss function (e.g., CrossEntropyLoss, FocalLoss)
            pred: (N, C) model predictions
            labels_a: (N,) first set of labels
            labels_b: (N,) second set of labels
            lam: mixing coefficient

        Returns:
            Mixed loss
        """
        return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


class CombinedLoss(nn.Module):
    """Combined loss with multiple components

    Combines cross-entropy/focal loss with confidence penalty.

    Args:
        use_focal (bool): Whether to use focal loss. Default: False
        focal_alpha (float): Alpha parameter for focal loss. Default: 0.25
        focal_gamma (float): Gamma parameter for focal loss. Default: 2.0
        confidence_penalty_weight (float): Weight for confidence penalty. Default: 0.0
        num_classes (int): Number of classes. Required if confidence_penalty_weight > 0
    """

    def __init__(
        self,
        use_focal: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        confidence_penalty_weight: float = 0.0,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.use_focal = use_focal

        if use_focal:
            self.base_loss = FocalLoss(
                alpha=focal_alpha, gamma=focal_gamma, reduction="mean"
            )
        else:
            self.base_loss = nn.CrossEntropyLoss(reduction="mean")

        if confidence_penalty_weight > 0:
            if num_classes is None:
                raise ValueError(
                    "num_classes must be provided when using confidence penalty"
                )
            self.confidence_penalty = ConfidencePenalty(
                num_classes=num_classes, weight=confidence_penalty_weight
            )
        else:
            self.confidence_penalty = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) logits from the model
            targets: (N,) ground truth labels

        Returns:
            Combined loss
        """
        loss = self.base_loss(inputs, targets)

        if self.confidence_penalty is not None:
            cp_loss = self.confidence_penalty(inputs)
            loss = loss + cp_loss

        return loss

    def compute_with_mixup(
        self,
        inputs: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """
        Compute loss with mixup augmentation.

        Args:
            inputs: (N, C) logits from the model
            labels_a: (N,) first set of labels
            labels_b: (N,) second set of labels
            lam: mixing coefficient

        Returns:
            Combined loss with mixup
        """
        if self.use_focal:
            loss_a = self.base_loss(inputs, labels_a)
            loss_b = self.base_loss(inputs, labels_b)
            base_loss = lam * loss_a + (1 - lam) * loss_b
        else:
            base_loss = lam * F.cross_entropy(inputs, labels_a) + (
                1 - lam
            ) * F.cross_entropy(inputs, labels_b)

        loss = base_loss

        if self.confidence_penalty is not None:
            cp_loss = self.confidence_penalty(inputs)
            loss = loss + cp_loss

        return loss


def apply_gradient_accumulation(
    loss: torch.Tensor,
    accumulation_steps: int,
    optimizer: torch.optim.Optimizer,
    batch_idx: int,
) -> bool:
    """
    Apply gradient accumulation.

    Args:
        loss: Loss tensor
        accumulation_steps: Number of steps to accumulate gradients
        optimizer: PyTorch optimizer
        batch_idx: Current batch index (0-based)

    Returns:
        True if optimizer step was performed, False otherwise
    """
    normalized_loss = loss / accumulation_steps
    normalized_loss.backward()

    should_step = (batch_idx + 1) % accumulation_steps == 0

    if should_step:
        optimizer.step()
        optimizer.zero_grad()

    return should_step


def l2_norm(input: torch.Tensor, axis: int = 1) -> torch.Tensor:
    """L2 normalize tensor along specified axis.

    Args:
        input: Input tensor
        axis: Axis along which to normalize

    Returns:
        Normalized tensor
    """
    norm = torch.norm(input, 2, axis, keepdim=True)
    output = torch.div(input, norm)
    return output


class ElasticFaceArcFace(nn.Module):
    """ElasticArcFace loss for face recognition / fine-grained classification.

    Implements ElasticFace: Elastic Margin Loss for Face Recognition
    (https://arxiv.org/abs/2109.09138)

    Uses dynamically sampled margins from a Gaussian distribution instead of
    fixed margins, providing better generalization.

    Supports four types:
        - cos: CosFace (margin in cosine space)
        - arc: ArcFace (margin in angular space)
        - cos+: CosFace+ (adaptive margin in cosine space)
        - arc+: ArcFace+ (adaptive margin in angular space)

    Args:
        in_features: Dimension of input embeddings
        out_features: Number of classes
        s: Scale factor for logits. Default: 30.0 (YOLO-adapted)
        m: Base margin for angular penalty. Default: 0.30 (YOLO-adapted)
        std: Standard deviation for margin sampling. Default: 0.01 (YOLO-adapted)
        type: Loss type - "cos", "arc", "cos+", "arc+". Default: "arc"
        plus: (Deprecated) Whether to use adaptive margin. Use type="arc+" instead.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.30,
        std: float = 0.01,
        type: str = "arc",
        plus: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.std = std

        # Parse type parameter and set plus flag
        valid_types = ["cos", "arc", "cos+", "arc+"]
        type_lower = type.lower()

        if type_lower not in valid_types:
            raise ValueError(
                f"Invalid type '{type}'. Must be one of {valid_types}"
            )

        # Handle type variants
        if type_lower == "cos+":
            self.type = "cos+"
            self.plus = True
        elif type_lower == "arc+":
            self.type = "arc+"
            self.plus = True
        else:
            self.type = type_lower
            # For backward compatibility: if plus=True is explicitly set,
            # upgrade the type
            if plus and type_lower == "cos":
                self.type = "cos+"
                self.plus = True
            elif plus and type_lower == "arc":
                self.type = "arc+"
                self.plus = True
            else:
                self.plus = plus

        # Learnable class centers (kernel weights)
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute ElasticFace logits.

        Args:
            embeddings: L2-normalized embeddings, shape (N, in_features)
            labels: Ground truth labels, shape (N,)

        Returns:
            Logits with margin penalty, shape (N, out_features)
        """
        embeddings = l2_norm(embeddings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)

        # Compute cosine similarity
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        # Apply dynamic margin only to positive samples
        index = torch.where(labels != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)

        # Sample margin from Gaussian distribution
        margin = torch.normal(
            mean=self.m, std=self.std, size=labels[index, None].size(), device=cos_theta.device
        )

        if self.plus:
            # ElasticFace+: adaptive margin based on sample difficulty
            with torch.no_grad():
                distmat = cos_theta[index, labels.view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)
            m_hot.scatter_(1, labels[index, None], margin[idicate_cosie])
        else:
            m_hot.scatter_(1, labels[index, None], margin)

        # Apply margin based on type
        if self.type in ["arc", "arc+"]:
            # ArcFace: apply margin in angular space
            cos_theta.acos_()
            cos_theta[index] -= m_hot  # Subtract margin in angular space
            cos_theta.cos_().mul_(self.s)
        else:  # cos, cos+
            # CosFace: apply margin directly in cosine space
            # cos_theta - m (subtract margin from cosine)
            cos_theta[index] -= m_hot
            cos_theta.mul_(self.s)

        return cos_theta


def stack_embeddings(embedding_list: list) -> torch.Tensor:
    """Stack YOLO embed() output into a single tensor.

    YOLO's embed() returns a list of tensors, one per sample.
    This function stacks them into a (batch_size, embedding_dim) tensor.

    Args:
        embedding_list: List of tensors from YOLO embed()

    Returns:
        Stacked embeddings tensor, shape (batch_size, embedding_dim)
    """
    if not embedding_list:
        raise ValueError("embedding_list is empty")

    if len(embedding_list) == 1:
        return embedding_list[0].unsqueeze(0)

    return torch.stack(embedding_list, dim=0)


class ElasticFaceLossWrapper(nn.Module):
    """Wrapper for ElasticFace loss with YOLO embeddings.

    Combines standard Cross-Entropy loss with ElasticFace loss:
        total_loss = CE_loss + lambda * ElasticFace_loss

    Args:
        num_classes: Number of classes
        embedding_dim: Dimension of embeddings (auto-detected from YOLO model)
        lambda_weight: Weight for ElasticFace loss. Default: 0.5
        s: Scale factor for logits. Default: 30.0
        m: Base margin for angular penalty. Default: 0.30
        std: Standard deviation for margin sampling. Default: 0.01
        type: Loss type - "cos", "arc", "cos+", "arc+". Default: "arc"
        plus: (Deprecated) Whether to use ElasticArcFace+. Use type="arc+" instead.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        lambda_weight: float = 0.5,
        s: float = 30.0,
        m: float = 0.30,
        std: float = 0.01,
        type: str = "arc",
        plus: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.lambda_weight = lambda_weight

        # ElasticFace loss
        self.elastic_face = ElasticFaceArcFace(
            in_features=embedding_dim,
            out_features=num_classes,
            s=s,
            m=m,
            std=std,
            type=type,
            plus=plus,
        )

        # Standard cross-entropy loss
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")

        # Simple classifier for standard CE (L2-normalized weights)
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        # Normalize weights for cosine similarity
        with torch.no_grad():
            nn.init.normal_(self.classifier.weight, std=0.01)
            self.classifier.weight.data = l2_norm(self.classifier.weight.data, axis=1)

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Compute combined loss.

        Args:
            embeddings: L2-normalized embeddings, shape (N, embedding_dim)
            labels: Ground truth labels, shape (N,)

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains:
                - total_loss: Total combined loss
                - ce_loss: Standard cross-entropy loss
                - elastic_loss: ElasticFace loss
        """
        # Standard CE loss (cosine classifier)
        ce_logits = self.classifier(embeddings) * self.elastic_face.s
        ce_loss_value = self.ce_loss(ce_logits, labels)

        # ElasticFace loss
        elastic_logits = self.elastic_face(embeddings, labels)
        elastic_loss_value = self.ce_loss(elastic_logits, labels)

        # Combined loss
        total_loss = ce_loss_value + self.lambda_weight * elastic_loss_value

        loss_dict = {
            "total_loss": total_loss.item(),
            "ce_loss": ce_loss_value.item(),
            "elastic_loss": elastic_loss_value.item(),
        }

        return total_loss, loss_dict

    @classmethod
    def from_yolo_model(
        cls,
        yolo_model,
        num_classes: int,
        lambda_weight: float = 0.5,
        s: float = 30.0,
        m: float = 0.30,
        std: float = 0.01,
        type: str = "arc",
        plus: bool = False,
    ) -> "ElasticFaceLossWrapper":
        """Create wrapper by auto-detecting embedding dimension from YOLO model.

        Args:
            yolo_model: YOLO model instance
            num_classes: Number of classes
            lambda_weight: Weight for ElasticFace loss
            s: Scale factor for logits
            m: Base margin for angular penalty
            std: Standard deviation for margin sampling
            type: Loss type - "cos", "arc", "cos+", "arc+"
            plus: (Deprecated) Whether to use ElasticArcFace+

        Returns:
            ElasticFaceLossWrapper instance
        """
        # Create dummy input to get embedding dimension
        dummy_input = torch.randn(1, 3, 224, 224) / 255.0
        dummy_embeddings = yolo_model.embed(dummy_input)
        embedding_dim = dummy_embeddings[0].shape[0]

        return cls(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            lambda_weight=lambda_weight,
            s=s,
            m=m,
            std=std,
            type=type,
            plus=plus,
        )


class ElasticFaceWithYOLO(nn.Module):
    """Complete ElasticFace loss module with YOLO embedding extraction.

    This module handles the full pipeline:
    1. Extract embeddings from YOLO model via embed()
    2. Stack embeddings into tensor
    3. Compute combined CE + ElasticFace loss

    Args:
        yolo_model: YOLO classification model
        num_classes: Number of classes
        lambda_weight: Weight for ElasticFace loss. Default: 0.5
        s: Scale factor for logits. Default: 30.0
        m: Base margin for angular penalty. Default: 0.30
        std: Standard deviation for margin sampling. Default: 0.01
        type: Loss type - "cos", "arc", "cos+", "arc+". Default: "arc"
        plus: (Deprecated) Whether to use ElasticArcFace+. Default: False
    """

    def __init__(
        self,
        yolo_model,
        num_classes: int,
        lambda_weight: float = 0.5,
        s: float = 30.0,
        m: float = 0.30,
        std: float = 0.01,
        type: str = "arc",
        plus: bool = False,
    ):
        super().__init__()
        self.yolo_model = yolo_model

        # Auto-detect embedding dimension
        dummy_input = torch.randn(1, 3, 224, 224) / 255.0
        dummy_embeddings = yolo_model.embed(dummy_input)
        embedding_dim = dummy_embeddings[0].shape[0]

        self.elastic_wrapper = ElasticFaceLossWrapper(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            lambda_weight=lambda_weight,
            s=s,
            m=m,
            std=std,
            type=type,
            plus=plus,
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute loss from images and labels.

        Args:
            images: Input images, shape (N, C, H, W)
            labels: Ground truth labels, shape (N,)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Get embeddings from YOLO model
        embedding_list = self.yolo_model.embed(images)

        # Stack into tensor
        embeddings = stack_embeddings(embedding_list)

        # L2 normalize (required for ElasticFace)
        embeddings = l2_norm(embeddings, axis=1)

        # Compute loss
        return self.elastic_wrapper(embeddings, labels)
