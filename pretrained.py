"""
pretrained.py
=============
Transfer learning from ImageNet-pretrained models to CIFAR-10.

Two strategies:

  Option 1 — Resize + Freeze:
    Resize CIFAR-10 images to 224x224 to match the pretrained backbone's
    expected input resolution.  Freeze all conv layers; train only the FC head.

  Option 2 — Modify Early Conv:
    Keep images at native 32x32.  Replace the first conv + remove maxpool so
    the spatial resolution survives the early layers.  Fine-tune all layers.

Usage:
    python pretrained.py --option 0 --model resnet18 --epochs 10   # both + compare
    python pretrained.py --option 1 --model resnet18 --epochs 10
    python pretrained.py --option 2 --model vgg16    --epochs 10
"""

import argparse
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CIFAR10_MEAN:    Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD:     Tuple[float, float, float] = (0.2023, 0.1994, 0.2010)
NUM_CLASSES:     int                        = 10
CIFAR10_CLASSES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ─────────────────────────────────────────────────────────────────────────────
# History dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingHistory:
    """
    Stores per-epoch metrics for a single training run.

    Collected during run_transfer() and consumed by the plotting functions.
    Separating metrics from the training loop makes it easy to compare
    multiple runs (e.g. Option 1 vs Option 2) on the same axes.

    Fields:
        name:       Human-readable experiment label (e.g. "resnet18_option1").
        train_loss: Training loss at the end of each epoch.
        train_acc:  Training accuracy at the end of each epoch.
        test_loss:  Test loss at the end of each epoch.
        test_acc:   Test accuracy at the end of each epoch.
        best_acc:   Best test accuracy observed across all epochs.
    """
    name:       str
    train_loss: List[float] = field(default_factory=list)
    train_acc:  List[float] = field(default_factory=list)
    test_loss:  List[float] = field(default_factory=list)
    test_acc:   List[float] = field(default_factory=list)
    best_acc:   float       = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def get_cifar10_loaders(
    option:      int,
    batch_size:  int = 64,
    num_workers: int = 2,
    val_ratio:   float = 0.1,
    seed:        int   = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build CIFAR-10 DataLoaders with a proper train / val / test split.

    Split strategy:
      - train : 90% of the official training set (45,000 samples)
      - val   : 10% of the official training set ( 5,000 samples)
      - test  : the official held-out test set   (10,000 samples)

    val and test splits receive no augmentation (only normalisation) so
    accuracy estimates are deterministic and unbiased.  train split receives
    full augmentation (flip, crop).

    The val split is separated from the test set so that:
      - epoch-level decisions (best checkpoint) are made on val.
      - the test set is touched only once at the very end.
      This prevents test-set leakage into the model selection process.

    Args:
        option:      1 = resize to 224x224  |  2 = keep native 32x32.
        batch_size:  Samples per mini-batch.
        num_workers: Parallel CPU workers for data prefetching.
        val_ratio:   Fraction of training data to use for validation.
        seed:        Random seed for the train/val split (reproducibility).

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader).
    """
    from torch.utils.data import random_split
    from PIL import Image as _Image

    norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    pin  = torch.cuda.is_available()

    if option == 1:
        train_tf = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), norm,
        ])
        eval_tf = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(), norm,
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), norm,
        ])
        eval_tf = transforms.Compose([transforms.ToTensor(), norm])

    # Load full training set with train transforms.
    full_train = datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
    test_ds    = datasets.CIFAR10("./data", train=False, download=True, transform=eval_tf)

    # Split into train and val subsets.
    n_val   = int(len(full_train) * val_ratio)
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_train, [n_train, n_val], generator=generator)

    # Override val subset transforms — apply eval_tf instead of train_tf.
    class _EvalSubset(torch.utils.data.Dataset):
        """Subset wrapper that applies eval transforms (no augmentation)."""
        def __init__(self, subset, transform) -> None:
            self.subset    = subset
            self.transform = transform
        def __len__(self) -> int:
            return len(self.subset)
        def __getitem__(self, idx):
            img   = self.subset.dataset.data[self.subset.indices[idx]]
            label = self.subset.dataset.targets[self.subset.indices[idx]]
            img   = _Image.fromarray(img)
            return self.transform(img), int(label)

    val_ds = _EvalSubset(val_subset, eval_tf)

    print(f"  Dataset split  —  train: {n_train:,}  |  val: {n_val:,}  |  test: {len(test_ds):,}")

    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,       batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,      batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Model Builders
# ─────────────────────────────────────────────────────────────────────────────

def build_resnet18_option1(freeze_backbone: bool = True) -> nn.Module:
    """
    ResNet-18 for Option 1: replace FC head, optionally freeze backbone.

    Only model.fc is replaced (512 -> 10).  When freeze_backbone=True,
    all other parameters have requires_grad=False, so only the 5,130-param
    head is updated — ideal when training data is small relative to ImageNet.

    Args:
        freeze_backbone: Freeze all layers except the new FC head.

    Returns:
        nn.Module: Modified ResNet-18 for 224x224 CIFAR-10 input.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def build_resnet18_option2() -> nn.Module:
    """
    ResNet-18 for Option 2: modify stem for 32x32 input, fine-tune all layers.

    Changes:
      conv1:   7x7 stride=2  ->  3x3 stride=1  (no spatial downsampling)
      maxpool: stride=2 pool ->  nn.Identity()  (skip the pool entirely)
      fc:      1000 classes  ->  10 classes

    The four residual stages still downsample via stride=2 in their first
    block: 32->16->8->4.  Global avg pool then gives a 512-dim vector.

    Returns:
        nn.Module: Modified ResNet-18 for 32x32 CIFAR-10 input.
    """
    model         = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def build_vgg16_option1(freeze_backbone: bool = True) -> nn.Module:
    """
    VGG-16 for Option 1: replace classifier head, optionally freeze features.

    VGG-16 expects 224x224 input; the five max-pools reduce it to 7x7 before
    the FC head.  We replace only classifier[6] (4096->1000) with (4096->10).

    Args:
        freeze_backbone: Freeze model.features (all conv layers).

    Returns:
        nn.Module: Modified VGG-16 for 224x224 CIFAR-10 input.
    """
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    return model


def build_vgg16_option2() -> nn.Module:
    """
    VGG-16 for Option 2: remove two max-pools to handle 32x32 input.

    Five max-pools on 32x32 would give 1x1 — unusable.  We keep only the
    first three pools (32->16->8->4), then rebuild the classifier with
    512*4*4=8192 input features instead of 512*7*7=25088.

    Returns:
        nn.Module: Modified VGG-16 for 32x32 CIFAR-10 input.
    """
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    new_features, pool_count = [], 0
    for layer in model.features:
        if isinstance(layer, nn.MaxPool2d):
            pool_count += 1
            if pool_count <= 3:
                new_features.append(layer)
        else:
            new_features.append(layer)
    model.features = nn.Sequential(*new_features)
    model.avgpool  = nn.AdaptiveAvgPool2d((4, 4))
    model.classifier = nn.Sequential(
        nn.Linear(512 * 4 * 4, 4096), nn.ReLU(inplace=True), nn.Dropout(),
        nn.Linear(4096, 4096),         nn.ReLU(inplace=True), nn.Dropout(),
        nn.Linear(4096, NUM_CLASSES),
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    optimizer:    torch.optim.Optimizer,
    criterion:    nn.Module,
    device:       torch.device,
    log_interval: int = 100,
) -> Tuple[float, float]:
    """
    Run one training epoch.

    Args:
        model:        Model being trained.
        loader:       Training DataLoader.
        optimizer:    Optimiser instance.
        criterion:    Loss function.
        device:       Compute device.
        log_interval: Print a progress line every N batches.

    Returns:
        Tuple[float, float]: (mean_loss, accuracy) over the epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += logits.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (i + 1) % log_interval == 0:
            print(f"    [{i + 1:>4}/{len(loader)}]  "
                  f"loss: {total_loss / n:.4f}  acc: {correct / n:.4f}")

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, float]:
    """
    Evaluate on the given DataLoader without updating weights.

    Args:
        model:     Model to evaluate.
        loader:    DataLoader (test split).
        criterion: Loss function.
        device:    Compute device.

    Returns:
        Tuple[float, float]: (mean_loss, accuracy).
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        correct    += logits.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

    return total_loss / n, correct / n


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Function  (now returns TrainingHistory)
# ─────────────────────────────────────────────────────────────────────────────

def run_transfer(
    model:      nn.Module,
    option:     int,
    model_name: str,
    epochs:     int,
    device:     torch.device,
    lr:         float = 1e-4,
    batch_size: int   = 64,
) -> TrainingHistory:
    """
    Fine-tune a pretrained model on CIFAR-10 and return full training history.

    Returns TrainingHistory instead of just a float so callers can:
      - Plot loss / accuracy curves across epochs.
      - Compare Option 1 vs Option 2 on the same axes.
      - Report best_acc in the assignment table.

    Args:
        model:      Model to fine-tune (already on the target device).
        option:     1 or 2 — determines which DataLoader to use.
        model_name: Label for logging and file names.
        epochs:     Number of training epochs.
        device:     Compute device.
        lr:         Initial learning rate (1e-4 is standard for fine-tuning).
        batch_size: Mini-batch size.

    Returns:
        TrainingHistory: Per-epoch metrics + best accuracy.
    """
    train_loader, val_loader, test_loader = get_cifar10_loaders(option, batch_size)
    criterion = nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    history      = TrainingHistory(name=model_name)
    best_weights = copy.deepcopy(model.state_dict())
    save_path    = f"best_{model_name}.pth"

    n_trainable = sum(p.numel() for p in trainable_params)
    n_total     = sum(p.numel() for p in model.parameters())

    print(f"\n{'=' * 55}")
    print(f"  Transfer Learning — {model_name}")
    print(f"  Trainable params : {n_trainable:,} / {n_total:,}")
    print(f"{'=' * 55}")

    for epoch in range(1, epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{epochs}  (lr={lr_now:.2e})")

        tr_loss, tr_acc   = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # Record per-epoch metrics for plotting.
        history.train_loss.append(tr_loss)
        history.train_acc.append(tr_acc)
        history.test_loss.append(val_loss)
        history.test_acc.append(val_acc)

        print(f"  train  loss={tr_loss:.4f}  acc={tr_acc:.4f}")
        print(f"  val    loss={val_loss:.4f}  acc={val_acc:.4f}  <- checkpoint decision")

        # Checkpoint uses VAL — test set is never touched here.
        if val_acc > history.best_acc:
            history.best_acc = val_acc
            best_weights     = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_path)
            print(f"  [saved]  val_acc={history.best_acc:.4f}  ->  {save_path}")

    # Restore best checkpoint then evaluate once on held-out test set.
    model.load_state_dict(best_weights)
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)

    # best_acc stores the FINAL TEST accuracy for the results table.
    # The best validation accuracy is preserved separately for logging.
    best_val_acc     = history.best_acc
    history.best_acc = final_test_acc

    print(f"\nBest val  accuracy : {best_val_acc:.4f}")
    print(f"Final test accuracy: {final_test_acc:.4f}  (held-out, reported once)")

    # Persist history so plot_results.py can load it without re-running training.
    import json, re
    fname = "history_" + re.sub(r"[^a-zA-Z0-9]", "_", model_name) + ".json"
    with open(fname, "w") as f:
        json.dump({
            "label":      history.name,
            "train_loss": history.train_loss,
            "train_acc":  history.train_acc,
            "val_acc":    history.test_acc,   # test_acc field holds per-epoch val acc
            "test_acc":   history.best_acc,
        }, f, indent=2)
    print(f"  History saved to: {fname}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(histories: List[TrainingHistory], save_prefix: str = "transfer") -> None:
    """
    Plot loss and accuracy curves for one or more training runs.

    Produces two side-by-side subplots:
      Left:  Test loss per epoch for all runs.
      Right: Test accuracy per epoch for all runs.

    Each run is drawn as a separate line so Option 1 vs Option 2 convergence
    behaviour is directly comparable.  The plot is saved as a PNG file.

    Args:
        histories:    List of TrainingHistory objects (one per experiment).
        save_prefix:  File name prefix for the saved PNG.
    """
    try:
        import matplotlib.pyplot as plt

        n_epochs = len(histories[0].test_loss)
        epochs   = list(range(1, n_epochs + 1))
        colors   = ["#4f86c6", "#e07b39", "#5aab61", "#c45c8a"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for i, h in enumerate(histories):
            c = colors[i % len(colors)]
            axes[0].plot(epochs, h.test_loss, marker="o", color=c, label=h.name)
            axes[1].plot(epochs, h.test_acc,  marker="o", color=c, label=h.name)

        axes[0].set_title("Validation Loss per Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_xticks(epochs)          # integer ticks: 1, 2, 3 ...
        axes[0].legend()
        axes[0].grid(linestyle="--", alpha=0.5)

        axes[1].set_title("Validation Accuracy per Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_xticks(epochs)          # integer ticks: 1, 2, 3 ...
        axes[1].legend()
        axes[1].grid(linestyle="--", alpha=0.5)

        plt.suptitle("Transfer Learning — Option 1 vs Option 2", fontsize=13)
        plt.tight_layout()

        fname = f"{save_prefix}_curves.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Training curves saved to: {fname}")

    except ImportError:
        print("matplotlib not found — skipping curve plot.")


def plot_accuracy_bar(histories: List[TrainingHistory], save_prefix: str = "transfer") -> None:
    """
    Bar chart comparing the best test accuracy of each experiment.

    Args:
        histories:   List of TrainingHistory objects.
        save_prefix: File name prefix for the saved PNG.
    """
    try:
        import matplotlib.pyplot as plt

        labels = [h.name for h in histories]
        accs   = [h.best_acc * 100 for h in histories]
        colors = ["#4f86c6", "#e07b39", "#5aab61", "#c45c8a"]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, accs, color=colors[:len(labels)], width=0.4)

        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{acc:.2f}%", ha="center", va="bottom", fontsize=11)

        ax.set_ylabel("Final Test Accuracy (%)")
        ax.set_title("Transfer Learning — Final Test Accuracy Comparison")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        fname = f"{save_prefix}_comparison.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Comparison bar chart saved to: {fname}")

    except ImportError:
        print("matplotlib not found — skipping bar chart.")


def plot_tsne(
    model:       nn.Module,
    option:      int,
    model_name:  str,
    device:      torch.device,
    n_samples:   int = 1000,
    save_prefix: str = "transfer",
) -> None:
    """
    Extract penultimate-layer features and visualise with t-SNE.

    t-SNE (t-distributed Stochastic Neighbour Embedding) projects high-
    dimensional feature vectors into 2D while preserving local structure.
    Clusters in the t-SNE plot correspond to classes that the network has
    learned to separate in its feature space.

    A well-trained transfer learning model should show tight, well-separated
    clusters — demonstrating that ImageNet features transfer effectively to
    CIFAR-10.  Option 1 vs Option 2 t-SNE plots reveal which strategy
    produces a more discriminative feature space.

    Implementation:
      - Register a forward hook on the layer just before the final FC to
        capture the 512-dim feature vector without modifying the model.
      - Run inference on n_samples test images (no gradients needed).
      - Fit t-SNE on the collected features.
      - Colour each point by its true class label.

    Args:
        model:       Trained model (ResNet-18 or VGG-16).
        option:      1 or 2 — determines DataLoader (image size).
        model_name:  Label used in the plot title and filename.
        device:      Compute device.
        n_samples:   Number of test images to embed (default 1000).
        save_prefix: File name prefix for the saved PNG.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.manifold import TSNE
    except ImportError:
        print("matplotlib / sklearn not found — skipping t-SNE plot.")
        return

    model.eval()
    _, _, test_loader = get_cifar10_loaders(option, batch_size=256)

    # ── Register hook to capture features before the FC layer ────────────────
    features_list: List[torch.Tensor] = []
    labels_list:   List[torch.Tensor] = []

    def _hook(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        features_list.append(input[0].detach().cpu())

    # For ResNet: the FC layer is model.fc.
    # For VGG:   the final classifier linear is model.classifier[-1].
    if hasattr(model, "fc"):
        handle = model.fc.register_forward_hook(_hook)
    else:
        handle = model.classifier[-1].register_forward_hook(_hook)

    # ── Collect features ─────────────────────────────────────────────────────
    collected = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            if collected >= n_samples:
                break
            imgs.to(device)
            model(imgs.to(device))
            labels_list.append(labels)
            collected += imgs.size(0)

    handle.remove()

    features = torch.cat(features_list)[:n_samples].numpy()   # (N, D)
    labels   = torch.cat(labels_list)[:n_samples].numpy()     # (N,)

    # ── Fit t-SNE ────────────────────────────────────────────────────────────
    print(f"  Fitting t-SNE on {len(features)} samples ...")
    tsne    = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords  = tsne.fit_transform(features)                     # (N, 2)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap    = plt.get_cmap("tab10")

    for cls_idx in range(NUM_CLASSES):
        mask = labels == cls_idx
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=10, color=cmap(cls_idx), label=CIFAR10_CLASSES[cls_idx], alpha=0.7)

    ax.set_title(f"t-SNE Feature Space — {model_name}")
    ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.axis("off")
    plt.tight_layout()

    fname = f"{save_prefix}_{model_name}_tsne.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"t-SNE plot saved to: {fname}")


def print_results_table(histories: List[TrainingHistory]) -> None:
    """
    Print a formatted results table for the assignment report.

    Args:
        histories: List of TrainingHistory objects from run_transfer().
    """
    print(f"\n{'=' * 60}")
    print(f"  {'Experiment':<30} {'Best Val Acc':>12} {'Test Acc':>9}")
    print(f"{'─' * 60}")
    for h in histories:
        best_val = max(h.test_acc) if h.test_acc else 0.0   # test_acc stores val metrics
        print(f"  {h.name:<30} {best_val:>11.4f}  {h.best_acc:>8.4f}")
    print(f"{'=' * 60}")
    print(f"  Note: Best Val Acc = best checkpoint epoch | Test Acc = held-out final")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Run transfer learning experiments from the command line.

    --option 0  runs both Option 1 and Option 2, then plots comparison charts
                and t-SNE feature space visualisations for each.
    """
    parser = argparse.ArgumentParser(
        description="Transfer Learning on CIFAR-10 (Options 1 & 2)"
    )
    parser.add_argument("--model",      choices=["resnet18", "vgg16"], default="resnet18")
    parser.add_argument("--option",     type=int, choices=[0, 1, 2],  default=0,
                        help="0 = run both options and compare (default)")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--no_freeze",  action="store_true",
                        help="Option 1: fine-tune all layers instead of freezing backbone")
    parser.add_argument("--tsne",       action="store_true",
                        help="Generate t-SNE feature space plots after training")
    args = parser.parse_args()

    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cpu"))
    print(f"Device: {device}")

    options_to_run = [1, 2] if args.option == 0 else [args.option]
    histories: List[TrainingHistory] = []
    models_trained: List[Tuple[nn.Module, int, str]] = []

    for opt in options_to_run:
        if args.model == "resnet18":
            model = (build_resnet18_option1(not args.no_freeze) if opt == 1
                     else build_resnet18_option2())
        else:
            model = (build_vgg16_option1(not args.no_freeze) if opt == 1
                     else build_vgg16_option2())

        model = model.to(device)
        label = f"{args.model}_option{opt}"

        history = run_transfer(
            model=model, option=opt, model_name=label,
            epochs=args.epochs, device=device,
            lr=args.lr, batch_size=args.batch_size,
        )
        histories.append(history)
        models_trained.append((model, opt, label))

    # ── Report ───────────────────────────────────────────────────────────────
    print_results_table(histories)

    # ── Plots ────────────────────────────────────────────────────────────────
    prefix = args.model
    plot_training_curves(histories, save_prefix=prefix)
    plot_accuracy_bar(histories,    save_prefix=prefix)

    if args.tsne:
        for model, opt, label in models_trained:
            plot_tsne(model, opt, label, device, save_prefix=prefix)


if __name__ == "__main__":
    main()