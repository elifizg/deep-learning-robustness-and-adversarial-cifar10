"""
test.py
=======
Loads saved model weights and evaluates the model on the test split.

Outputs:
  - Overall accuracy
  - Per-class accuracy with an ASCII bar chart
  - FLOPs and parameter count via ptflops
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from parameters import TrainingConfig
from train import get_transforms

CIFAR10_CLASSES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ─────────────────────────────────────────────────────────────────────────────
# FLOPs Counting
# ─────────────────────────────────────────────────────────────────────────────

def count_flops(model: nn.Module, config: TrainingConfig) -> None:
    """
    Report the model's FLOPs (MACs) and parameter count using ptflops.

    Why FLOPs matter:
      Accuracy alone does not tell the full story.  A MobileNet achieving 88%
      with 300M MACs is far more practical than a ResNet at 90% with 1.8B MACs
      on resource-constrained hardware (mobile, embedded systems).

    ptflops feeds a synthetic input tensor through the model and counts
    multiply-accumulate operations (MACs) in each layer.
      1 MAC = 1 multiplication + 1 addition  ≈  2 FLOPs.

    Falls back to a manual parameter count if ptflops is not installed.

    Args:
        model:  The model to profile.
        config: Used to determine the correct input shape.
    """
    if config.dataset == "mnist":
        input_shape: Tuple[int, int, int] = (1, 28, 28)
    elif config.transfer_option == 1:
        input_shape = (3, 224, 224)   # resized for ImageNet-pretrained backbones
    else:
        input_shape = (3, 32, 32)

    try:
        from ptflops import get_model_complexity_info

        macs, params = get_model_complexity_info(
            model, input_shape,
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
        print(f"\n{'─' * 40}")
        print(f"  FLOPs (MACs) : {macs}")
        print(f"  Parameters   : {params}")
        print(f"{'─' * 40}")

    except ImportError:
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  [ptflops not found — manual count]")
        print(f"  Total parameters     : {total:,}")
        print(f"  Trainable parameters : {trainable:,}")
        print("  Install with: pip install ptflops")


# ─────────────────────────────────────────────────────────────────────────────
# Per-Class Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_per_class(
    model:       nn.Module,
    loader:      DataLoader,
    device:      torch.device,
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> Dict[int, float]:
    """
    Compute and display per-class accuracy on the given DataLoader.

    Why per-class accuracy?
      Overall accuracy can be misleading: a model could perform perfectly on
      nine out of ten classes while completely failing on the tenth, yet still
      report 90% overall accuracy.  Per-class breakdown exposes these blind spots.

      On CIFAR-10, "cat" and "dog" are typically the weakest classes because
      they share similar visual features (fur, ears, body posture).

    Args:
        model:       Model to evaluate.
        loader:      DataLoader (typically the test split).
        device:      Compute device.
        num_classes: Total number of classes.
        class_names: Optional human-readable class labels.

    Returns:
        Dict[int, float]: Mapping from class index to accuracy.
    """
    model.eval()
    class_correct: List[int] = [0] * num_classes
    class_total:   List[int] = [0] * num_classes

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)

            for pred, label in zip(preds, labels):
                class_correct[label] += int(pred == label)
                class_total[label]   += 1

    print(f"\n{'─' * 52}")
    print(f"  {'Class':<14} {'Correct':>7} {'Total':>6} {'Acc':>6}  Bar")
    print(f"{'─' * 52}")

    results: Dict[int, float] = {}
    for i in range(num_classes):
        acc  = class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        name = class_names[i] if class_names else str(i)
        bar  = "█" * int(acc * 20)
        print(f"  {name:<14} {class_correct[i]:>7} {class_total[i]:>6} {acc:.4f}  {bar}")
        results[i] = acc

    print(f"{'─' * 52}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main Test Function
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_test(
    model:  nn.Module,
    config: TrainingConfig,
    device: torch.device,
) -> Tuple[float, Dict[int, float]]:
    """
    Load saved weights, run evaluation on the test split, and report results.

    Steps:
      1. Load weights from config.save_path.
      2. Build the test DataLoader.
      3. Compute overall accuracy.
      4. Compute per-class accuracy.
      5. Report FLOPs and parameter count.

    Note on map_location:
      If a model was saved on a GPU but is now loaded on a CPU machine,
      PyTorch would raise a RuntimeError without map_location.
      Passing map_location=device remaps tensor storage automatically.

    Args:
        model:  Model architecture (weights will be loaded into it).
        config: TrainingConfig instance.
        device: Compute device.

    Returns:
        Tuple[float, Dict[int, float]]: (overall_accuracy, per_class_accuracy).
    """
    state = torch.load(config.save_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Weights loaded from: {config.save_path}")

    tf = get_transforms(config, train=False)
    if config.dataset == "mnist":
        test_ds     = datasets.MNIST(config.data_dir, train=False, download=True, transform=tf)
        class_names = None
    else:
        test_ds     = datasets.CIFAR10(config.data_dir, train=False, download=True, transform=tf)
        class_names = CIFAR10_CLASSES

    loader = DataLoader(
        test_ds, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers,
    )

    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds    = model(imgs).argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)

    overall_acc = correct / total

    print(f"\n{'=' * 52}")
    print(f"  Test Results")
    print(f"{'=' * 52}")
    print(f"  Overall accuracy : {overall_acc:.4f}  ({correct}/{total})")

    per_class = evaluate_per_class(
        model, loader, device, config.num_classes, class_names,
    )

    count_flops(model, config)

    return overall_acc, per_class


# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-10-C Robustness Evaluation
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import TensorDataset
from torchvision import transforms

CORRUPTIONS: List[str] = [
    "brightness", "contrast", "defocus_blur", "elastic_transform",
    "fog", "frost", "gaussian_blur", "gaussian_noise", "glass_blur",
    "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
    "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur",
]


def load_cifar10c_severity(
    data_dir:   str,
    corruption: str,
    severity:   int,
    mean:       Tuple[float, ...] = (0.4914, 0.4822, 0.4465),
    std:        Tuple[float, ...] = (0.2023, 0.1994, 0.2010),
) -> DataLoader:
    """
    Load one (corruption, severity) slice of CIFAR-10-C as a DataLoader.

    CIFAR-10-C stores all 5 severity levels stacked along axis 0:
        severity 1 -> indices     0 –  9 999
        severity 2 -> indices 10 000 – 19 999
        severity 3 -> indices 20 000 – 29 999
        severity 4 -> indices 30 000 – 39 999
        severity 5 -> indices 40 000 – 49 999

    The same labels.npy is shared across all corruptions and severities;
    the ordering is identical to the standard CIFAR-10 test set repeated
    five times.

    Args:
        data_dir:   Path to the CIFAR-10-C folder containing .npy files.
        corruption: Corruption type name, e.g. "fog" or "gaussian_noise".
        severity:   Severity level, integer in [1, 5].
        mean:       Per-channel mean used during training normalisation.
        std:        Per-channel std  used during training normalisation.

    Returns:
        DataLoader: Yields (image_tensor, label) pairs for the 10 000 images
                    of the requested (corruption, severity) combination.
    """
    assert 1 <= severity <= 5, "severity must be an integer between 1 and 5"

    images = np.load(Path(data_dir) / f"{corruption}.npy")  # (50000, 32, 32, 3) uint8
    labels = np.load(Path(data_dir) / "labels.npy")         # (50000,)

    start  = (severity - 1) * 10_000
    images = images[start : start + 10_000]
    labels = labels[start : start + 10_000]

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    tensors      = torch.stack([tf(Image.fromarray(img)) for img in images])
    label_tensor = torch.tensor(labels, dtype=torch.long)

    return DataLoader(
        TensorDataset(tensors, label_tensor),
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate_cifar10c(
    model:       nn.Module,
    data_dir:    str,
    device:      torch.device,
    corruptions: List[str] = CORRUPTIONS,
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate model accuracy across all corruptions and all 5 severity levels.

    For each (corruption, severity) pair, builds a DataLoader from the
    corresponding .npy slice and computes top-1 accuracy. Prints a
    formatted table to stdout and returns the full results dict.

    Args:
        model:       Fine-tuned model; will be set to eval mode internally.
        data_dir:    Path to the CIFAR-10-C folder.
        device:      Compute device (cuda / mps / cpu).
        corruptions: List of corruption names to evaluate. Defaults to all 19.

    Returns:
        Dict[str, Dict[int, float]]: results[corruption][severity] = accuracy.
    """
    model.eval()
    results: Dict[str, Dict[int, float]] = {}

    print(f"\n{'─' * 66}")
    print(f"  {'Corruption':<22} {'Sev1':>6} {'Sev2':>6} {'Sev3':>6} "
          f"{'Sev4':>6} {'Sev5':>6}  {'Mean':>6}")
    print(f"{'─' * 66}")

    for corruption in corruptions:
        results[corruption] = {}
        for severity in range(1, 6):
            loader = load_cifar10c_severity(data_dir, corruption, severity)
            correct, total = 0, 0
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                correct += model(imgs).argmax(1).eq(lbls).sum().item()
                total   += lbls.size(0)
            results[corruption][severity] = correct / total

        accs     = [results[corruption][s] for s in range(1, 6)]
        mean_acc = sum(accs) / len(accs)
        print(f"  {corruption:<22} " +
              " ".join(f"{a:>6.3f}" for a in accs) +
              f"  {mean_acc:>6.3f}")

    print(f"{'─' * 66}")
    overall = sum(
        sum(sv.values()) / len(sv) for sv in results.values()
    ) / len(results)
    print(f"  {'Mean Corruption Acc':<22} {'':>41}  {overall:>6.3f}")
    print(f"{'─' * 66}\n")

    return results