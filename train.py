"""
train.py
========
Training loop, data loading, and loss functions.

Responsibilities:
  1. Build transform pipelines for training and evaluation.
  2. Create DataLoaders for the selected dataset.
  3. Instantiate the correct loss function based on config.
  4. Run the per-epoch training and validation loops.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from parameters import TrainingConfig


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Transforms
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(config: TrainingConfig, train: bool = True) -> transforms.Compose:
    """
    Build the torchvision transform pipeline for the given dataset and split.

    Training augmentations applied to CIFAR-10:
      - RandomCrop(32, padding=4): pad by 4 pixels, then crop back to 32x32.
        Forces the model to be invariant to small translations.
      - RandomHorizontalFlip: mirrors the image with 50% probability,
        effectively doubling the number of training examples.

    No augmentation is applied during evaluation — the test set must represent
    the real-world distribution without any artificial modification.

    Args:
        config: TrainingConfig instance carrying dataset name, mean, and std.
        train:  If True, return the training pipeline; otherwise the eval pipeline.

    Returns:
        transforms.Compose: A sequential transform pipeline.
    """
    mean, std = config.mean, config.std

    if config.dataset == "mnist":
        # MNIST is single-channel and small — augmentation is unnecessary.
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    # Transfer learning option 1 requires 224x224 to match ImageNet-pretrained conv layers.
    # Model names can be either the short form ("resnet", "vgg") used in standard training
    # or the pretrained form ("resnet18", "vgg16") used in transfer learning mode.
    resize_needed: bool = (
        config.model in ("resnet", "vgg", "mobilenet", "resnet18", "vgg16")
        and config.transfer_option == 1
    )

    if train:
        aug = []
        if resize_needed:
            aug.append(transforms.Resize(224))
        crop_size = 224 if resize_needed else 32
        aug += [
            transforms.RandomCrop(crop_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        return transforms.Compose(aug)

    base = []
    if resize_needed:
        base.append(transforms.Resize(224))
    base += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(base)


def get_loaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create and return train, validation, and test DataLoaders.

    Split strategy:
      The original training set is split into train and val using
      torch.utils.data.random_split with a fixed seed so the split is
      reproducible across runs.

      - train : 90% of the original training set  (45,000 for CIFAR-10)
      - val   : 10% of the original training set  ( 5,000 for CIFAR-10)
      - test  : the official held-out test set     (10,000 for CIFAR-10)

    Why separate val from test?
      Using the test set for early stopping (saving the best model) leaks
      information: the model is implicitly selected based on test performance,
      making the final reported accuracy over-optimistic.  A separate val set
      is used for all epoch-level decisions; the test set is touched only once
      at the very end to report the final number.

    Important: val split receives val_tf (no augmentation) even though it
      originates from the training dataset.  Augmentation must not be applied
      at evaluation time because it would add randomness to the accuracy
      estimate.  We achieve this by applying transforms at __getitem__ time
      via a thin wrapper that overrides the transform of the subset.

    Args:
        config: TrainingConfig instance.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader).
    """
    from torch.utils.data import random_split, Subset
    import numpy as np

    train_tf = get_transforms(config, train=True)
    val_tf   = get_transforms(config, train=False)
    pin      = torch.cuda.is_available()

    # ── Load full training set (with train transforms for now) ────────────────
    if config.dataset == "mnist":
        full_train_ds = datasets.MNIST(config.data_dir, train=True,  download=True, transform=train_tf)
        test_ds       = datasets.MNIST(config.data_dir, train=False, download=True, transform=val_tf)
    else:
        full_train_ds = datasets.CIFAR10(config.data_dir, train=True,  download=True, transform=train_tf)
        test_ds       = datasets.CIFAR10(config.data_dir, train=False, download=True, transform=val_tf)

    # ── Split train -> train + val ────────────────────────────────────────────
    n_total = len(full_train_ds)
    n_val   = int(n_total * 0.1)       # 10% validation
    n_train = n_total - n_val          # 90% training

    generator = torch.Generator().manual_seed(config.seed)
    train_subset, val_subset = random_split(full_train_ds, [n_train, n_val], generator=generator)

    # ── Apply val transforms to the val subset ────────────────────────────────
    # random_split returns a Subset whose underlying dataset still has train_tf.
    # We wrap it so __getitem__ applies val_tf instead (no augmentation).
    class _TransformSubset(torch.utils.data.Dataset):
        """Wraps a Subset and overrides its transform at getitem time."""
        def __init__(self, subset: Subset, transform) -> None:
            self.subset    = subset
            self.transform = transform
        def __len__(self) -> int:
            return len(self.subset)
        def __getitem__(self, idx):
            img, label = self.subset.dataset.data[self.subset.indices[idx]],                          self.subset.dataset.targets[self.subset.indices[idx]]
            # data is a numpy array (H,W,C) for CIFAR-10 or (H,W) for MNIST.
            from PIL import Image
            if hasattr(img, 'numpy'):
                img = img.numpy()
            img = Image.fromarray(img)
            return self.transform(img), int(label)

    val_ds_wrapped = _TransformSubset(val_subset, val_tf)

    print(f"  Dataset split  —  train: {n_train:,}  |  val: {n_val:,}  |  test: {len(test_ds):,}")

    train_loader = DataLoader(train_subset,   batch_size=config.batch_size,
                              shuffle=True,  num_workers=config.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds_wrapped, batch_size=config.batch_size,
                              shuffle=False, num_workers=config.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,        batch_size=config.batch_size,
                              shuffle=False, num_workers=config.num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# 2. Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing (Szegedy et al., 2016).

    Standard CE assigns a hard target of 1.0 to the correct class and 0.0 to
    all others, which can make the model overconfident.  Label smoothing
    distributes a small probability mass epsilon uniformly across all classes:

        target[correct]  = 1 - epsilon
        target[others]   = epsilon / (K - 1)

    Example with K=10, epsilon=0.1:
        Hard:     [0, 0, 0, 1.0,  0,    0,    0,    0,    0,    0   ]
        Smoothed: [0.011, 0.011, 0.011, 0.9, 0.011, ...]

    Benefits:
      - Prevents the model from becoming overconfident on training examples.
      - Acts as a regulariser, improving generalisation on unseen data.
      - More robust to label noise in the training set.

    Args:
        num_classes: Number of output classes.
        smoothing:   Epsilon value. 0.0 reduces to standard CE; 0.1 is typical.
    """

    def __init__(self, num_classes: int = 10, smoothing: float = 0.1) -> None:
        super().__init__()
        self.smoothing   = smoothing
        self.num_classes = num_classes
        self.confidence  = 1.0 - smoothing   # probability assigned to the correct class

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  Raw model outputs before softmax, shape (B, C).
            targets: Ground-truth class indices, shape (B,).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # log_softmax is numerically more stable than log(softmax(x)).
        log_probs = F.log_softmax(logits, dim=1)            # (B, C)

        # Uniform smoothing term: -mean(log_probs) over all classes.
        smooth_loss = -log_probs.mean(dim=1)                # (B,)

        # NLL loss: picks the log-prob of the correct class for each sample.
        nll_loss = F.nll_loss(log_probs, targets, reduction="none")  # (B,)

        # Weighted combination of the hard and smooth targets.
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class KnowledgeDistillationLoss(nn.Module):
    """
    Classic knowledge distillation loss (Hinton et al., 2015).

    The student is trained on two objectives simultaneously:
      1. Hard loss: standard CE against the ground-truth labels.
         Ensures the student learns the correct classification task.
      2. Soft loss: KL divergence between student and teacher soft predictions.
         Transfers the teacher's "dark knowledge" — inter-class similarity
         information hidden in the non-maximum probabilities.

    Temperature T controls the softness of the distributions:
      T = 1  ->  standard softmax; one class dominates.
      T = 4  ->  smoother distribution; minority class probabilities become
                 visible and carry relational information (e.g. a cat image
                 might score 0.15 on "dog" at T=4, revealing similarity).

    Total loss:
        L = (1 - alpha) * CE(student, labels)
          + alpha       * KL(student_soft || teacher_soft) * T^2

    The T^2 factor compensates for the gradient scaling introduced by dividing
    logits by T before softmax.

    Args:
        temperature: Softening temperature T. Typical range: 2 – 8.
        alpha:       Weight of the soft distillation loss. Typical: 0.3 – 0.7.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha:       float = 0.3,
        num_classes: int   = 10,
    ) -> None:
        super().__init__()
        self.T           = temperature
        self.alpha       = alpha
        self.num_classes = num_classes
        self.ce          = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets:        torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            student_logits: Student model output, shape (B, C).
            teacher_logits: Teacher model output, shape (B, C).
                            Gradients are not computed for the teacher.
            targets:        Ground-truth class indices, shape (B,).

        Returns:
            Tuple of (total_loss, hard_loss, soft_loss) for logging.
        """
        hard_loss = self.ce(student_logits, targets)

        # Both distributions are softened by dividing by T before softmax.
        student_soft = F.log_softmax(student_logits / self.T, dim=1)
        teacher_soft = F.softmax(teacher_logits    / self.T, dim=1)

        # KL divergence: measures how much student_soft diverges from teacher_soft.
        # kl_div(log_p, q) = sum(q * (log_q - log_p)); student is optimised to match teacher.
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
        soft_loss = soft_loss * (self.T ** 2)   # gradient magnitude correction

        total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        return total_loss, hard_loss, soft_loss


class TeacherProbDistillationLoss(nn.Module):
    """
    Hybrid label-smoothing + knowledge distillation loss (HW Part B.4).

    Instead of using the full teacher soft-label distribution, only the
    teacher's predicted probability for the TRUE class is retained.
    The remaining probability mass is distributed equally among all other classes.

    This encodes per-example difficulty:
      - Teacher predicts p_true = 0.95  ->  easy example  ->  near-hard label.
      - Teacher predicts p_true = 0.55  ->  hard example  ->  more mass on others.

    Constructed soft target for class k:
        soft_target[true_class] = p_true
        soft_target[k != true]  = (1 - p_true) / (K - 1)

    Args:
        num_classes: Number of output classes.
        alpha:       Weight of the soft distillation loss.
    """

    def __init__(self, num_classes: int = 10, alpha: float = 0.3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha       = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets:        torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            student_logits: (B, C)
            teacher_logits: (B, C)
            targets:        (B,)

        Returns:
            Tuple of (total_loss, hard_loss, soft_loss).
        """
        B, C   = student_logits.shape
        device = student_logits.device

        teacher_probs = F.softmax(teacher_logits, dim=1)   # (B, C)

        # gather: selects teacher_probs[:, targets[i]] for each row i.
        p_true = teacher_probs.gather(1, targets.unsqueeze(1))  # (B, 1)

        # Fill all classes with the equal share of remaining probability.
        soft_targets = (1 - p_true).expand(B, C) / (C - 1)     # (B, C)

        # Overwrite the true-class column with p_true.
        # scatter_ writes p_true into soft_targets at the column given by targets.
        soft_targets = soft_targets.clone()
        soft_targets.scatter_(1, targets.unsqueeze(1), p_true)  # (B, C)

        log_probs = F.log_softmax(student_logits, dim=1)        # (B, C)

        soft_loss = F.kl_div(log_probs, soft_targets, reduction="batchmean")
        hard_loss = F.nll_loss(log_probs, targets)

        total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        return total_loss, hard_loss, soft_loss


def build_criterion(config: TrainingConfig) -> nn.Module:
    """
    Select and return the appropriate loss function based on config.

    Decision tree:
      distillation=True,  mode=teacher_prob  ->  TeacherProbDistillationLoss
      distillation=True,  mode=standard      ->  KnowledgeDistillationLoss
      distillation=False, smoothing > 0      ->  LabelSmoothingLoss
      distillation=False, smoothing = 0      ->  CrossEntropyLoss

    Args:
        config: TrainingConfig instance.

    Returns:
        nn.Module: The selected loss function.
    """
    if config.distillation:
        if config.distill_mode == "teacher_prob":
            print(f"  Loss: TeacherProbDistillation  (alpha={config.distill_alpha})")
            return TeacherProbDistillationLoss(
                num_classes=config.num_classes,
                alpha=config.distill_alpha,
            )
        print(f"  Loss: KnowledgeDistillation  (T={config.temperature}, alpha={config.distill_alpha})")
        return KnowledgeDistillationLoss(
            temperature=config.temperature,
            alpha=config.distill_alpha,
            num_classes=config.num_classes,
        )

    if config.label_smoothing > 0.0:
        print(f"  Loss: LabelSmoothingLoss  (epsilon={config.label_smoothing})")
        return LabelSmoothingLoss(
            num_classes=config.num_classes,
            smoothing=config.label_smoothing,
        )

    print("  Loss: CrossEntropyLoss  (standard)")
    return nn.CrossEntropyLoss()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training and Validation Loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    optimizer:    torch.optim.Optimizer,
    criterion:    nn.Module,
    device:       torch.device,
    log_interval: int,
    teacher:      Optional[nn.Module] = None,
) -> Tuple[float, float]:
    """
    Run one full training epoch over the provided DataLoader.

    When knowledge distillation is active, the teacher performs a forward pass
    inside torch.no_grad() — its weights are not updated and no gradient graph
    is built, saving both memory and compute.

    Training step sequence:
      1. Move batch to device.
      2. optimizer.zero_grad()  — clear gradients from the previous step.
      3. Forward pass through student model.
      4. (KD) Forward pass through teacher model (no_grad).
      5. Compute loss.
      6. loss.backward()        — compute gradients via autograd.
      7. optimizer.step()       — update weights.

    Args:
        model:        Student model being trained.
        loader:       Training DataLoader.
        optimizer:    Optimiser (e.g. Adam).
        criterion:    Loss function returned by build_criterion().
        device:       Compute device (cpu / cuda / mps).
        log_interval: Print a progress line every N batches.
        teacher:      Pre-trained teacher model for KD; None if unused.

    Returns:
        Tuple[float, float]: (mean_loss, accuracy) over the epoch.
    """
    model.train()
    if teacher is not None:
        teacher.eval()   # teacher must always stay in inference mode

    total_loss, correct, n = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        student_logits = model(imgs)

        if teacher is not None:
            with torch.no_grad():
                teacher_logits = teacher(imgs)
            loss, _, _ = criterion(student_logits, teacher_logits, labels)
        else:
            loss = criterion(student_logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += student_logits.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"    [{batch_idx + 1:>4}/{len(loader)}]  "
                  f"loss: {total_loss / n:.4f}  acc: {correct / n:.4f}")

    return total_loss / n, correct / n


@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    device:    torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on the validation split.

    The @torch.no_grad() decorator disables gradient computation for the entire
    function, reducing memory usage and speeding up inference by ~30%.
    Validation never updates weights, so gradients are never needed.

    Standard CrossEntropyLoss is always used here — not the KD loss — so that
    validation accuracy is directly comparable across all training modes.

    Args:
        model:  Model to evaluate.
        loader: Validation DataLoader.
        device: Compute device.

    Returns:
        Tuple[float, float]: (mean_loss, accuracy).
    """
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = ce(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        correct    += logits.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

    return total_loss / n, correct / n


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main Training Function
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    model:   nn.Module,
    config:  TrainingConfig,
    device:  torch.device,
    teacher: Optional[nn.Module] = None,
) -> None:
    """
    Run the full training loop and save the best model weights to disk.

    Data splits:
      - train_loader : used for gradient updates every epoch.
      - val_loader   : used after each epoch to pick the best checkpoint.
                       No gradients, no augmentation.
      - test_loader  : used ONCE at the very end to report final accuracy.
                       Never used for any decision during training.

    This three-way split prevents test-set leakage: the model is selected
    based on val performance, so the test score is a true held-out estimate.

    Learning rate schedule (StepLR):
      Multiplies lr by gamma every step_size epochs.
      Example: lr=0.001, step_size=5, gamma=0.5
        Epochs  1-5 : lr = 0.001
        Epochs  6-10: lr = 0.0005

    Args:
        model:   Student model to train.
        config:  TrainingConfig instance.
        device:  Compute device.
        teacher: Pre-trained teacher model for KD; None if unused.
    """
    train_loader, val_loader, test_loader = get_loaders(config)
    criterion = build_criterion(config)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc:     float          = 0.0
    best_weights: Optional[dict] = None

    for epoch in range(1, config.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{config.epochs}  (lr={current_lr:.2e})")

        tr_loss, tr_acc   = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, config.log_interval, teacher,
        )
        val_loss, val_acc = validate(model, val_loader, device)
        scheduler.step()

        print(f"  train  loss={tr_loss:.4f}  acc={tr_acc:.4f}")
        print(f"  val    loss={val_loss:.4f}  acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, config.save_path)
            print(f"  [saved]  val_acc={best_acc:.4f}  ->  {config.save_path}")

    # ── Restore best weights, then evaluate on held-out test set ─────────────
    if best_weights is not None:
        model.load_state_dict(best_weights)

    test_loss, test_acc = validate(model, test_loader, device)
    print(f"\nTraining complete.")
    print(f"  Best val  accuracy : {best_acc:.4f}")
    print(f"  Final test accuracy: {test_acc:.4f}  (held-out, reported once)")


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers  (Part B comparisons)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunHistory:
    """
    Stores per-epoch training metrics for a single experiment run.

    Used by plot_comparison_curves() and print_comparison_table() to draw
    side-by-side comparisons between experiments (e.g. ResNet with and
    without label smoothing, or SimpleCNN vs SimpleCNN+KD).

    Fields:
        label:      Short human-readable name shown in legends and tables.
        train_loss: Training loss recorded at the end of each epoch.
        train_acc:  Training accuracy recorded at the end of each epoch.
        val_acc:    Validation accuracy recorded at the end of each epoch.
        test_acc:   Final held-out test accuracy (single scalar, set after training).
    """
    label:      str
    train_loss: List[float] = field(default_factory=list)
    train_acc:  List[float] = field(default_factory=list)
    val_acc:    List[float] = field(default_factory=list)
    test_acc:   float       = 0.0


def run_training_tracked(
    model:   nn.Module,
    config:  TrainingConfig,
    device:  torch.device,
    label:   str,
    teacher: Optional[nn.Module] = None,
) -> "RunHistory":
    """
    Like run_training() but returns a RunHistory for post-hoc plotting.

    This wrapper replaces run_training() when you need to compare multiple
    experiments on the same axes.  The training logic is identical; we only
    add metric collection.

    Args:
        model:   Student model to train.
        config:  TrainingConfig instance.
        device:  Compute device.
        label:   Short name for this run (e.g. "ResNet + LS").
        teacher: Optional teacher model for KD.

    Returns:
        RunHistory: Per-epoch metrics + final test accuracy.
    """
    train_loader, val_loader, test_loader = get_loaders(config)
    criterion = build_criterion(config)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history      = RunHistory(label=label)
    best_acc:    float          = 0.0
    best_weights: Optional[dict] = None

    for epoch in range(1, config.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{config.epochs}  (lr={current_lr:.2e})")

        tr_loss, tr_acc   = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, config.log_interval, teacher,
        )
        val_loss, val_acc = validate(model, val_loader, device)
        scheduler.step()

        history.train_loss.append(tr_loss)
        history.train_acc.append(tr_acc)
        history.val_acc.append(val_acc)

        print(f"  train  loss={tr_loss:.4f}  acc={tr_acc:.4f}")
        print(f"  val    loss={val_loss:.4f}  acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, config.save_path)
            print(f"  [saved]  val_acc={best_acc:.4f}  ->  {config.save_path}")

    if best_weights is not None:
        model.load_state_dict(best_weights)

    _, test_acc = validate(model, test_loader, device)
    history.test_acc = test_acc

    print(f"\nTraining complete  [{label}]")
    print(f"  Best val  accuracy : {best_acc:.4f}")
    print(f"  Final test accuracy: {test_acc:.4f}  (held-out, reported once)")

    # Save history to JSON so plots can be generated after independent runs.
    import json, re
    fname = "history_" + re.sub(r"[^a-zA-Z0-9]", "_", label) + ".json"
    with open(fname, "w") as f:
        json.dump({
            "label":      history.label,
            "train_loss": history.train_loss,
            "train_acc":  history.train_acc,
            "val_acc":    history.val_acc,
            "test_acc":   history.test_acc,
        }, f, indent=2)
    print(f"  History saved to: {fname}")

    return history


def plot_comparison_curves(
    histories:   List["RunHistory"],
    title:       str       = "Training Comparison",
    save_path:   str       = "comparison_curves.png",
) -> None:
    """
    Plot validation accuracy curves for multiple runs on the same axes.

    Each RunHistory is drawn as a separate line so you can visually compare
    convergence speed and final performance between experiments.

    Args:
        histories:  List of RunHistory objects (one per experiment).
        title:      Chart title shown at the top.
        save_path:  File name for the saved PNG.
    """
    try:
        import matplotlib.pyplot as plt

        epochs = list(range(1, len(histories[0].val_acc) + 1))
        colors = ["#4f86c6", "#e07b39", "#5aab61", "#c45c8a", "#8855bb"]

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        for i, h in enumerate(histories):
            c = colors[i % len(colors)]
            axes[0].plot(epochs, h.train_loss, marker="o", color=c, label=h.label)
            axes[1].plot(epochs, h.val_acc,    marker="o", color=c, label=h.label)

        for ax, ylabel, ytitle in zip(
            axes,
            ["Loss", "Accuracy"],
            ["Training Loss per Epoch", "Validation Accuracy per Epoch"],
        ):
            ax.set_title(ytitle)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_xticks(epochs)
            ax.legend()
            ax.grid(linestyle="--", alpha=0.5)

        plt.suptitle(title, fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Comparison curves saved to: {save_path}")

    except ImportError:
        print("matplotlib not found — skipping curve plot.")


def print_comparison_table(
    histories:   List["RunHistory"],
    flops_dict:  Optional[Dict[str, str]] = None,
) -> None:
    """
    Print a formatted results table suitable for the assignment report.

    Optionally includes a FLOPs column when flops_dict is provided.

    Args:
        histories:  List of RunHistory objects from run_training_tracked().
        flops_dict: Optional {label: flops_string} mapping (e.g. {"SimpleCNN": "55.17 MMac"}).
    """
    has_flops = flops_dict is not None
    width     = 72 if has_flops else 55

    print(f"\n{'=' * width}")
    header = f"  {'Experiment':<28} {'Test Acc':>9}"
    if has_flops:
        header += f"  {'FLOPs':>12}"
    print(header)
    print(f"{'─' * width}")

    for h in histories:
        row = f"  {h.label:<28} {h.test_acc:>8.4f}"
        if has_flops:
            flops = flops_dict.get(h.label, "N/A")
            row  += f"  {flops:>12}"
        print(row)

    print(f"{'=' * width}")