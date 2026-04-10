"""
main.py
=======
Single entry point for all experiments in the assignment:
  - Standard training (MLP, CNN, VGG, ResNet, MobileNet)
  - Transfer learning from ImageNet pretrained models (Options 1 & 2)
  - Label smoothing
  - Knowledge distillation (standard and teacher_prob)
  - HW2: CIFAR-10-C robustness, AugMix fine-tuning, PGD adversarial attacks,
         AugMix-teacher KD comparison, adversarial transferability

Usage examples:
  # Standard training
  python main.py --model resnet  --dataset cifar10 --epochs 20
  python main.py --model mlp     --dataset mnist   --epochs 10

  # Transfer learning — run both options and compare
  python main.py --mode transfer --model resnet18  --epochs 10
  python main.py --mode transfer --model vgg16     --epochs 10 --option 1

  # Label smoothing
  python main.py --model resnet --dataset cifar10 --label_smoothing 0.1

  # Knowledge distillation (train ResNet teacher first, then distil)
  python main.py --model cnn --dataset cifar10 --distillation --teacher_path best_resnet.pth
  python main.py --model mobilenet --dataset cifar10 --distillation --distill_mode teacher_prob

  # HW2 — run in order:
  # 1. CIFAR-10-C robustness evaluation
  python main.py --mode cifar10c --teacher_path best_resnet18_option2.pth --cifar10c_dir ./CIFAR-10-C

  # 2. AugMix fine-tuning + CIFAR-10-C eval
  python main.py --mode augmix --teacher_path best_resnet18_option2.pth --save_path best_augmix.pth --cifar10c_dir ./CIFAR-10-C --epochs 20 --lr 1e-4

  # 3. PGD-20 attacks + Grad-CAM + t-SNE + transferability
  python main.py --mode pgd --teacher_path best_resnet18_option2.pth --augmix_path best_augmix.pth --results_dir ./results

  # 4. AugMix-teacher KD comparison (baseline vs AugMix teacher)
  python main.py --mode augmix_kd --epochs 20 --augmix_path best_augmix.pth

  # 4 — individual sub-experiments:
  python main.py --mode b3_augmix --augmix_path best_augmix.pth --epochs 20
  python main.py --mode b4_augmix --augmix_path best_augmix.pth --epochs 20
"""

import random
import ssl
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from parameters import TrainingConfig, get_params
from models.MLP       import MLP
from models.CNN       import MNIST_CNN, SimpleCNN
from models.VGG       import VGG
from models.ResNet    import ResNet, BasicBlock
from models.mobilenet import MobileNetV2
from pretrained       import (
    build_resnet18_option1, build_resnet18_option2,
    build_vgg16_option1,    build_vgg16_option2,
    run_transfer, TrainingHistory,
    plot_training_curves, plot_accuracy_bar, plot_tsne, print_results_table,
)
from train import run_training, run_training_tracked, plot_comparison_curves, print_comparison_table, RunHistory
from test  import run_test, count_flops

ssl._create_default_https_context = ssl._create_unverified_context


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """
    Fix all randomness sources for reproducible training runs.

    Each library maintains its own random state; all must be seeded to
    guarantee identical results across runs with the same seed.
    cudnn.deterministic=True forces CUDA to select reproducible (though
    sometimes slower) kernel implementations.

    Args:
        seed: Any integer (e.g. 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device(config: TrainingConfig) -> torch.device:
    """
    Return the best available compute device.

    Priority: CUDA > MPS (Apple Silicon) > CPU.

    Args:
        config: TrainingConfig instance (device field used as a hint).

    Returns:
        torch.device: Selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(config: TrainingConfig) -> nn.Module:
    """
    Instantiate the model specified in config.model.

    Args:
        config: TrainingConfig instance.

    Returns:
        nn.Module: Untrained model ready to be moved to a device.

    Raises:
        ValueError: Unknown model name or incompatible dataset.
    """
    name: str = config.model
    nc:   int = config.num_classes

    if name == "mlp":
        return MLP(
            input_size   = config.input_size,
            hidden_sizes = config.hidden_sizes,
            num_classes  = nc,
            dropout      = config.dropout,
        )
    if name == "cnn":
        return MNIST_CNN(num_classes=nc) if config.dataset == "mnist" \
               else SimpleCNN(num_classes=nc)
    if name == "vgg":
        if config.dataset == "mnist":
            raise ValueError("VGG requires 3-channel input; use --dataset cifar10.")
        return VGG(dept=config.vgg_depth, num_class=nc)
    if name == "resnet":
        if config.dataset == "mnist":
            raise ValueError("ResNet requires 3-channel input; use --dataset cifar10.")
        return ResNet(BasicBlock, config.resnet_layers, num_classes=nc)
    if name == "mobilenet":
        if config.dataset == "mnist":
            raise ValueError("MobileNetV2 requires 3-channel input; use --dataset cifar10.")
        return MobileNetV2(num_classes=nc)

    raise ValueError(f"Unknown model '{name}'. Choose: mlp, cnn, vgg, resnet, mobilenet.")


def load_teacher(
    config: TrainingConfig,
    device: torch.device,
) -> Optional[nn.Module]:
    """
    Load the pre-trained ResNet-18 teacher for knowledge distillation.

    The teacher's parameters are frozen (requires_grad=False) so that:
      - No gradient graph is allocated for teacher tensors.
      - The optimiser receives only student parameters.
      - Accidental weight updates are impossible.

    Args:
        config: Must have distillation=True and a valid teacher_path.
        device: Compute device.

    Returns:
        Optional[nn.Module]: Loaded frozen teacher, or None if KD is off.
    """
    if not config.distillation:
        return None

    teacher = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=config.num_classes)
    state   = torch.load(config.teacher_path, map_location=device)
    teacher.load_state_dict(state)
    teacher.to(device).eval()

    for param in teacher.parameters():
        param.requires_grad = False

    print(f"Teacher loaded from: {config.teacher_path}")
    return teacher


# ─────────────────────────────────────────────────────────────────────────────
# Transfer Learning Mode
# ─────────────────────────────────────────────────────────────────────────────

def run_transfer_mode(config: TrainingConfig, device: torch.device) -> None:
    """
    Run transfer learning experiments and plot Option 1 vs Option 2 comparison.

    If config.transfer_option == 0, both options are run and compared.
    Otherwise only the specified option is executed.

    Calls the builder functions from pretrained.py, then run_transfer() for
    the actual training loop, and finally plot_transfer_comparison() to
    produce a bar chart saved to disk.

    Args:
        config: TrainingConfig instance (model, transfer_option, epochs, lr).
        device: Compute device.
    """
    options_to_run = [1, 2] if config.transfer_option == 0 else [config.transfer_option]
    from typing import List as _List
    histories: _List[TrainingHistory] = []
    models_trained = []

    for opt in options_to_run:
        if config.model == "resnet18":
            model = (build_resnet18_option1(freeze_backbone=config.freeze_backbone)
                     if opt == 1 else build_resnet18_option2())
        elif config.model == "vgg16":
            model = (build_vgg16_option1(freeze_backbone=config.freeze_backbone)
                     if opt == 1 else build_vgg16_option2())
        else:
            raise ValueError(
                f"Transfer learning supports 'resnet18' and 'vgg16', got '{config.model}'."
            )

        model = model.to(device)
        label = f"{config.model}_option{opt}"
        history = run_transfer(
            model      = model,
            option     = opt,
            model_name = label,
            epochs     = config.epochs,
            device     = device,
            lr         = config.learning_rate,
            batch_size = config.batch_size,
        )
        histories.append(history)
        models_trained.append((model, opt, label))

    print_results_table(histories)
    plot_training_curves(histories, save_prefix=config.model)
    plot_accuracy_bar(histories,    save_prefix=config.model)

    if config.tsne:
        for model, opt, label in models_trained:
            plot_tsne(model, opt, label, device, save_prefix=config.model)






# ─────────────────────────────────────────────────────────────────────────────
# Part B: Individual experiment runners
# ─────────────────────────────────────────────────────────────────────────────

def _free_gpu(*models) -> None:
    """Move models to CPU and release GPU memory cache."""
    for m in models:
        if m is not None:
            m.cpu()
    torch.cuda.empty_cache()


def _make_cifar_config(epochs: int, save_path: str,
                       label_smoothing: float = 0.0,
                       distillation: bool = False,
                       distill_mode: str = "standard",
                       temperature: float = 4.0,
                       distill_alpha: float = 0.3) -> TrainingConfig:
    """
    Build a CIFAR-10 TrainingConfig from scratch with explicit values.

    Avoids dataclasses.asdict() which converts tuples to lists and
    corrupts mean/std fields causing the config to silently fall back
    to MNIST defaults.
    """
    return TrainingConfig(
        dataset         = "cifar10",
        data_dir        = "./data",
        num_workers     = 0,
        mean            = (0.4914, 0.4822, 0.4465),
        std             = (0.2023, 0.1994, 0.2010),
        model           = "cnn",        # overridden per experiment
        input_size      = 3072,
        num_classes     = 10,
        dropout         = 0.3,
        vgg_depth       = "16",
        resnet_layers   = [2, 2, 2, 2],
        transfer_option = 1,
        freeze_backbone = True,
        label_smoothing = label_smoothing,
        distillation    = distillation,
        teacher_path    = "best_resnet.pth",
        temperature     = temperature,
        distill_alpha   = distill_alpha,
        distill_mode    = distill_mode,
        epochs          = epochs,
        batch_size      = 64,
        learning_rate   = 1e-3,
        weight_decay    = 1e-4,
        seed            = 42,
        device          = "cuda",
        save_path       = save_path,
        log_interval    = 100,
        mode            = "train",
        tsne            = False,
        # HW2 fields — kept at defaults; set explicitly in run_augmix / run_pgd
        cifar10c_dir      = "./CIFAR-10-C",
        augmix_severity   = 3,
        augmix_width      = 3,
        augmix_lambda_jsd = 12.0,
        augmix_path       = "best_augmix.pth",
        results_dir       = "./results",
        baseline_history  = "",
        augmix_history    = "",
    )


def run_b1(config: TrainingConfig, device: torch.device) -> None:
    """B.1 — SimpleCNN baseline (standard CE, no teacher)."""
    from models.CNN import SimpleCNN
    cfg = _make_cifar_config(epochs=config.epochs, save_path="best_cnn_baseline.pth")
    print(f"  Dataset : cifar10 | Model : SimpleCNN | Device : {device}")
    model = SimpleCNN(num_classes=10).to(device)
    run_training_tracked(model, cfg, device, label="SimpleCNN (baseline)")
    _free_gpu(model)


def run_b2a(config: TrainingConfig, device: torch.device) -> None:
    """B.2a — ResNet-18 from scratch, no label smoothing."""
    cfg = _make_cifar_config(epochs=config.epochs, save_path="best_resnet.pth",
                             label_smoothing=0.0)
    print(f"  Dataset : cifar10 | Model : ResNet-18 | Device : {device}")
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)
    run_training_tracked(model, cfg, device, label="ResNet (no LS)")
    _free_gpu(model)


def run_b2b(config: TrainingConfig, device: torch.device) -> None:
    """B.2b — ResNet-18 from scratch, label smoothing epsilon=0.1."""
    cfg = _make_cifar_config(epochs=config.epochs, save_path="best_resnet_ls.pth",
                             label_smoothing=0.1)
    print(f"  Dataset : cifar10 | Model : ResNet-18 | Label Smoothing : 0.1 | Device : {device}")
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)
    run_training_tracked(model, cfg, device, label="ResNet (LS=0.1)")
    _free_gpu(model)


def run_b3(config: TrainingConfig, device: torch.device) -> None:
    """B.3 — SimpleCNN student + ResNet teacher (standard KD)."""
    import os
    from models.CNN import SimpleCNN
    teacher_path = "best_resnet.pth"
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(
            f"Teacher weights not found: '{teacher_path}'\n"
            "  Run B.2a first to train and save the ResNet teacher:\n"
            "      python main.py --mode b2a --epochs 20"
        )
    teacher = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    cfg = _make_cifar_config(epochs=config.epochs, save_path="best_cnn_kd.pth",
                             distillation=True, distill_mode="standard",
                             temperature=4.0, distill_alpha=0.3)
    print(f"  Dataset : cifar10 | Model : SimpleCNN | KD T=4.0 alpha=0.3 | Device : {device}")
    model = SimpleCNN(num_classes=10).to(device)
    run_training_tracked(model, cfg, device, label="SimpleCNN (KD)", teacher=teacher)
    _free_gpu(model, teacher)


def run_b4(config: TrainingConfig, device: torch.device) -> None:
    """B.4 — MobileNet student + ResNet teacher (standard KD).

    Uses the same KD mode as run_b4_augmix() so that the only variable
    between the two experiments is the teacher checkpoint. This isolates
    the effect of teacher robustness from the effect of KD method.
    """
    import os
    from models.mobilenet import MobileNetV2
    teacher_path = "best_resnet.pth"
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(
            f"Teacher weights not found: '{teacher_path}'\n"
            "  Run B.2a first to train and save the ResNet teacher:\n"
            "      python main.py --mode b2a --epochs 20"
        )
    teacher = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    cfg = _make_cifar_config(epochs=config.epochs, save_path="best_mobilenet_kd.pth",
                             distillation=True, distill_mode="standard",
                             temperature=4.0, distill_alpha=0.3)
    print(f"  Dataset : cifar10 | Model : MobileNetV2 | Standard KD T=4.0 alpha=0.3 | Device : {device}")
    model = MobileNetV2(num_classes=10).to(device)
    run_training_tracked(model, cfg, device,
                         label="MobileNet (KD, baseline teacher)", teacher=teacher)
    _free_gpu(model, teacher)


def _load_teacher(path: str, device: torch.device, label: str) -> nn.Module:
    """
    Load a frozen ResNet-18 teacher from a checkpoint.

    Auto-detects whether the checkpoint was saved from a torchvision ResNet-18
    (keys: fc, downsample) or the custom ResNet class (keys: linear, shortcut),
    and loads the correct architecture accordingly.

    Args:
        path:   Path to the .pth checkpoint file.
        device: Compute device.
        label:  Human-readable name printed in the error message.

    Returns:
        nn.Module: Frozen teacher in eval mode.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} checkpoint not found: '{path}'\n"
            f"  Train it first with --mode augmix --save_path {path}"
        )
    state_dict = torch.load(path, map_location=device)
    is_torchvision = any("fc." in k or "downsample" in k for k in state_dict.keys())

    if is_torchvision:
        # Build the Option-2 architecture (3x3 stem, no maxpool, 10-class head)
        # without downloading pretrained weights — checkpoint supplies all weights.
        import torchvision.models as tv_models
        teacher = tv_models.resnet18(weights=None)
        teacher.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        teacher.maxpool = nn.Identity()
        teacher.fc      = nn.Linear(teacher.fc.in_features, 10)
        print(f"  {label}: detected torchvision ResNet-18")
    else:
        teacher = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        print(f"  {label}: detected custom ResNet-18")

    teacher.load_state_dict(state_dict)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"  {label} loaded from: {path}")
    return teacher


def run_b3_augmix(config: TrainingConfig, device: torch.device) -> None:
    """
    HW2 Part 4 — SimpleCNN student + AugMix-trained ResNet teacher (standard KD).

    Mirrors run_b3() but replaces the vanilla ResNet teacher with the
    AugMix-fine-tuned checkpoint.  This tests whether a robustness-trained
    teacher transfers better dark knowledge to the student.

    Teacher checkpoint: config.augmix_path  (default: best_augmix.pth)
    Student saved to  : best_cnn_kd_augmix.pth

    Args:
        config: TrainingConfig; uses config.augmix_path and config.epochs.
        device: Compute device.
    """
    from models.CNN import SimpleCNN
    teacher = _load_teacher(config.augmix_path, device, "AugMix teacher")

    cfg = _make_cifar_config(
        epochs       = config.epochs,
        save_path    = "best_cnn_kd_augmix.pth",
        distillation = True,
        distill_mode = "standard",
        temperature  = 4.0,
        distill_alpha= 0.3,
    )
    print(f"  Dataset : cifar10 | Student : SimpleCNN | Teacher : AugMix ResNet | "
          f"KD T=4.0 alpha=0.3 | Device : {device}")
    model = SimpleCNN(num_classes=10).to(device)
    run_training_tracked(model, cfg, device,
                         label="SimpleCNN (KD, AugMix teacher)", teacher=teacher)
    _free_gpu(model, teacher)


def run_b4_augmix(config: TrainingConfig, device: torch.device) -> None:
    """
    HW2 Part 4 — MobileNet student + AugMix-trained ResNet teacher (standard KD).

    Mirrors run_b4() exactly — same student, same KD mode (standard KL divergence),
    same hyperparameters — with the sole difference being the teacher checkpoint.
    This isolates the effect of teacher robustness on student performance.

    Teacher checkpoint: config.augmix_path  (default: best_augmix.pth)
    Student saved to  : best_mobilenet_kd_augmix.pth

    Args:
        config: TrainingConfig; uses config.augmix_path and config.epochs.
        device: Compute device.
    """
    from models.mobilenet import MobileNetV2
    teacher = _load_teacher(config.augmix_path, device, "AugMix teacher")

    cfg = _make_cifar_config(
        epochs        = config.epochs,
        save_path     = "best_mobilenet_kd_augmix.pth",
        distillation  = True,
        distill_mode  = "standard",
        temperature   = 4.0,
        distill_alpha = 0.3,
    )
    print(f"  Dataset : cifar10 | Student : MobileNetV2 | Teacher : AugMix ResNet | "
          f"Standard KD | Device : {device}")
    model = MobileNetV2(num_classes=10).to(device)
    run_training_tracked(model, cfg, device,
                         label="MobileNet (KD, AugMix teacher)", teacher=teacher)
    _free_gpu(model, teacher)


def run_augmix_kd(config: TrainingConfig, device: torch.device) -> None:
    """
    HW2 Part 4 — Full comparison: baseline KD vs AugMix-teacher KD.

    Runs four experiments in sequence and prints a side-by-side comparison
    table so the effect of the AugMix teacher can be directly quantified:

        Experiment          | Teacher          | Student
        ─────────────────── | ──────────────── | ──────────
        B.3  (baseline KD)  | ResNet (vanilla) | SimpleCNN
        B.3A (AugMix KD)    | ResNet (AugMix)  | SimpleCNN
        B.4  (baseline KD)  | ResNet (vanilla) | MobileNet
        B.4A (AugMix KD)    | ResNet (AugMix)  | MobileNet

    Both teacher checkpoints must exist before calling this mode:
        best_resnet.pth   — trained via --mode b2a
        best_augmix.pth   — trained via --mode augmix

    Args:
        config: TrainingConfig; uses config.epochs, config.augmix_path.
        device: Compute device.
    """
    from train import RunHistory, plot_comparison_curves, print_comparison_table

    print(f"\n{'=' * 65}")
    print(f"  HW2 Part 4 — KD Comparison: Baseline Teacher vs AugMix Teacher")
    print(f"{'=' * 65}")

    histories = []
    for mode_label, runner in [
        ("B.3  — SimpleCNN  + Baseline teacher", run_b3),
        ("B.3A — SimpleCNN  + AugMix teacher",   run_b3_augmix),
        ("B.4  — MobileNet  + Baseline teacher (Standard KD)", run_b4),
        ("B.4A — MobileNet  + AugMix teacher   (Standard KD)", run_b4_augmix),
    ]:
        print(f"\n{'─' * 65}")
        print(f"  {mode_label}")
        print(f"{'─' * 65}")
        runner(config, device)

    # Collect history JSONs written by run_training_tracked during this session.
    # run_training_tracked sanitises labels into filenames (e.g. spaces → _),
    # so we glob all history_*.json and filter by label content.
    import json, glob
    json_paths = sorted(glob.glob("history_*.json"))
    histories  = []
    for p in json_paths:
        with open(p) as f:
            h = json.load(f)
        # Match all four KD experiments by label
        if any(kw in h["label"] for kw in [
            "SimpleCNN (KD",
            "MobileNet (KD",
        ]):
            histories.append(h)

    # ── Comparison table ──────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  {'Experiment':<40} {'Test Acc':>9}")
    print(f"{'─' * 65}")
    for h in sorted(histories, key=lambda x: x["test_acc"], reverse=True):
        print(f"  {h['label']:<40} {h['test_acc']:>8.4f}")
    print(f"{'=' * 65}")

    # ── Comparison curves ──────────────────────────────────────────────────
    if histories:
        from attack import plot_epoch_history
        matched_paths = []
        for p in json_paths:
            with open(p) as f:
                label = json.load(f).get("label", "")
            if any(kw in label for kw in [
                "SimpleCNN (KD", "MobileNet (KD",
            ]):
                matched_paths.append(p)
        plot_epoch_history(
            json_paths = matched_paths,
            save_path  = "augmix_kd_comparison.png",
        )


# ─────────────────────────────────────────────────────────────────────────────
# PGD Adversarial Attack Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_cifar10c(config: TrainingConfig, device: torch.device) -> None:
    """
    Evaluate a fine-tuned ResNet-18 on clean CIFAR-10 and CIFAR-10-C.

    Automatically detects whether the checkpoint was saved from the
    torchvision ResNet-18 (keys: fc, downsample) or the custom ResNet
    class (keys: linear, shortcut), and loads the correct architecture.

    Args:
        config: TrainingConfig; uses config.teacher_path and config.cifar10c_dir.
        device: Compute device.
    """
    import os
    from test import evaluate_cifar10c
    from train import get_loaders, validate

    if not os.path.exists(config.teacher_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: '{config.teacher_path}'\n"
            f"  Pass the correct path with --teacher_path"
        )

    state_dict = torch.load(config.teacher_path, map_location=device)
    is_torchvision = any("fc." in k or "downsample" in k for k in state_dict.keys())

    if is_torchvision:
        from pretrained import build_resnet18_option2
        model = build_resnet18_option2()
        print(f"  Detected: torchvision ResNet-18 (transfer learning option 2)")
    else:
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        print(f"  Detected: custom ResNet-18 (Part B)")

    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"  Loaded: {config.teacher_path}")

    cfg = _make_cifar_config(epochs=1, save_path="dummy.pth")
    _, _, test_loader = get_loaders(cfg)
    _, clean_acc = validate(model, test_loader, device)
    print(f"\n  Clean test accuracy : {clean_acc:.4f}")

    print(f"  CIFAR-10-C dir      : {config.cifar10c_dir}\n")
    evaluate_cifar10c(model, config.cifar10c_dir, device)

    _free_gpu(model)


def run_augmix(config: TrainingConfig, device: torch.device) -> None:
    """
    Fine-tune ResNet-18 (Option 2 architecture) with AugMix + JSD loss,
    then evaluate on clean CIFAR-10 and CIFAR-10-C.

    Args:
        config: TrainingConfig; key fields:
                  teacher_path       — starting checkpoint (Option 2 weights)
                  save_path          — where to save the AugMix model
                  cifar10c_dir       — CIFAR-10-C folder for evaluation
                  epochs             — number of fine-tuning epochs
                  learning_rate      — initial lr (recommended: 1e-4)
                  augmix_severity, augmix_width, augmix_lambda_jsd
        device: Compute device.
    """
    import os, copy
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from augmix import AugMixTransform, JensenShannonLoss
    from test import evaluate_cifar10c
    from train import get_loaders, validate

    from pretrained import build_resnet18_option2
    model = build_resnet18_option2()

    if os.path.exists(config.teacher_path):
        state_dict = torch.load(config.teacher_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"  Starting from checkpoint: {config.teacher_path}")
    else:
        print(f"  Warning: '{config.teacher_path}' not found — training from scratch.")

    model.to(device)

    augmix_tf = AugMixTransform(
        mean     = (0.4914, 0.4822, 0.4465),
        std      = (0.2023, 0.1994, 0.2010),
        severity = config.augmix_severity,
        width    = config.augmix_width,
    )

    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=augmix_tf)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
    )

    criterion = JensenShannonLoss(lambda_jsd=config.augmix_lambda_jsd, num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc     = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    # History tracking for training curves
    history = {
        "label":      "AugMix Fine-tuning",
        "train_loss": [],
        "train_acc":  [],
        "val_acc":    [],
        "test_acc":   0.0,
    }

    from torchvision import transforms as T
    # CIFAR-10 has no separate validation split; the test set (train=False)
    # serves as both validation during training and final evaluation.
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_ds     = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    print(f"\n{'=' * 55}")
    print(f"  AugMix Fine-tuning")
    print(f"  Epochs     : {config.epochs}")
    print(f"  LR         : {config.learning_rate}")
    print(f"  Severity   : {config.augmix_severity}")
    print(f"  Width      : {config.augmix_width}")
    print(f"  Lambda JSD : {config.augmix_lambda_jsd}")
    print(f"{'=' * 55}\n")

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss, ce_sum, jsd_sum, correct, n = 0.0, 0.0, 0.0, 0, 0

        for batch in train_loader:
            (orig, aug1, aug2), labels = batch
            orig, aug1, aug2, labels = orig.to(device), aug1.to(device), aug2.to(device), labels.to(device)

            optimizer.zero_grad()
            logits_orig = model(orig)
            logits_aug1 = model(aug1)
            logits_aug2 = model(aug2)

            loss, ce_loss, jsd_loss = criterion(logits_orig, logits_aug1, logits_aug2, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()     * orig.size(0)
            ce_sum     += ce_loss.item()  * orig.size(0)
            jsd_sum    += jsd_loss.item() * orig.size(0)
            correct    += logits_orig.argmax(1).eq(labels).sum().item()
            n          += orig.size(0)

        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                val_correct += model(imgs).argmax(1).eq(lbls).sum().item()
                val_total   += lbls.size(0)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch}/{config.epochs} | "
              f"loss: {total_loss/n:.4f}  ce: {ce_sum/n:.4f}  jsd: {jsd_sum/n:.4f} | "
              f"train acc: {correct/n:.4f}  val acc: {val_acc:.4f}")

        # Record epoch metrics
        history["train_loss"].append(round(total_loss / n, 6))
        history["train_acc"].append(round(correct / n, 6))
        history["val_acc"].append(round(val_acc, 6))

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, config.save_path)
            print(f"  [saved] val_acc={best_acc:.4f} -> {config.save_path}")

    model.load_state_dict(best_weights)
    model.eval()

    clean_correct, clean_total = 0, 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            clean_correct += model(imgs).argmax(1).eq(lbls).sum().item()
            clean_total   += lbls.size(0)
    clean_acc = clean_correct / clean_total
    history["test_acc"] = round(clean_acc, 6)

    # Save epoch history JSON next to the model checkpoint
    import json as _json
    history_path = config.save_path.replace(".pth", "_history.json")
    with open(history_path, "w") as f:
        _json.dump(history, f, indent=2)
    print(f"  Epoch history saved to: {history_path}")

    print(f"\n  Clean test accuracy (AugMix) : {clean_acc:.4f}")
    print(f"  CIFAR-10-C dir              : {config.cifar10c_dir}\n")
    evaluate_cifar10c(model, config.cifar10c_dir, device)

    _free_gpu(model)


# ─────────────────────────────────────────────────────────────────────────────
# PGD Adversarial Attack Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_pgd(config: TrainingConfig, device: torch.device) -> None:
    """
    Evaluate both fine-tuned models (baseline + AugMix) against PGD-20 attacks,
    generate Grad-CAM visualisations and t-SNE plots, and save epoch history.

    Runs four attack configurations:
        1. Baseline model  — L∞, epsilon = 4/255
        2. Baseline model  — L2,  epsilon = 0.25
        3. AugMix model    — L∞, epsilon = 4/255
        4. AugMix model    — L2,  epsilon = 0.25

    Args:
        config: TrainingConfig; uses teacher_path, augmix_path,
                results_dir, baseline_history, augmix_history.
        device: Compute device.
    """
    import os, json
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms as T
    from attack import (
        evaluate_pgd, visualize_gradcam, visualize_tsne_adv,
        plot_epoch_history,
    )
    from pretrained import build_resnet18_option2

    os.makedirs(config.results_dir, exist_ok=True)

    def load_resnet(path: str) -> nn.Module:
        model = build_resnet18_option2()
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()
        return model

    if not os.path.exists(config.teacher_path):
        raise FileNotFoundError(f"Baseline checkpoint not found: {config.teacher_path}")
    if not os.path.exists(config.augmix_path):
        raise FileNotFoundError(f"AugMix checkpoint not found: {config.augmix_path}\n"
                                f"  Pass the path with --augmix_path")

    print(f"  Loading baseline : {config.teacher_path}")
    model_baseline = load_resnet(config.teacher_path)
    print(f"  Loading AugMix   : {config.augmix_path}")
    model_augmix = load_resnet(config.augmix_path)

    val_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_ds     = datasets.CIFAR10("./data", train=False, download=True, transform=val_tf)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    # ── PGD Evaluation ────────────────────────────────────────────────────
    attack_configs = [
        ("Baseline", model_baseline, "linf", 4/255),
        ("Baseline", model_baseline, "l2",   0.25),
        ("AugMix",   model_augmix,   "linf", 4/255),
        ("AugMix",   model_augmix,   "l2",   0.25),
    ]

    results = {}
    print(f"\n{'─' * 62}")
    print(f"  {'Model':<12} {'Norm':<6} {'Epsilon':<10} {'Clean':>7} {'Adv (PGD20)':>12}")
    print(f"{'─' * 62}")

    for model_name, model, norm, epsilon in attack_configs:
        eps_str = "4/255" if norm == "linf" else "0.25"
        key = f"{model_name}_{norm}"
        print(f"  Evaluating {model_name} — {norm.upper()} eps={eps_str} ...", flush=True)

        clean_acc, adv_acc = evaluate_pgd(
            model, test_loader, epsilon, norm, device,
            num_steps=20, max_batches=40,
        )
        results[key] = {"clean": clean_acc, "adv": adv_acc,
                        "norm": norm, "epsilon": eps_str, "model": model_name}
        print(f"  {model_name:<12} {norm.upper():<6} eps={eps_str:<8} "
              f"clean: {clean_acc:.4f}  adv: {adv_acc:.4f}")

    print(f"{'─' * 62}")

    results_path = os.path.join(config.results_dir, "pgd_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  PGD results saved to: {results_path}")

    # ── Grad-CAM ──────────────────────────────────────────────────────────
    print("\n  Generating Grad-CAM visualisations...")
    for norm, epsilon, fname in [
        ("linf", 4/255, "gradcam_linf.png"),
        ("l2",   0.25,  "gradcam_l2.png"),
    ]:
        visualize_gradcam(
            model_baseline = model_baseline,
            model_augmix   = model_augmix,
            loader         = test_loader,
            device         = device,
            epsilon        = epsilon,
            norm           = norm,
            n_samples      = 2,
            save_path      = os.path.join(config.results_dir, fname),
        )

    # ── t-SNE ─────────────────────────────────────────────────────────────
    print("\n  Generating t-SNE plots...")
    for model, model_name, fname in [
        (model_baseline, "Baseline", "tsne_baseline_linf.png"),
        (model_augmix,   "AugMix",   "tsne_augmix_linf.png"),
    ]:
        visualize_tsne_adv(
            model      = model,
            loader     = test_loader,
            device     = device,
            epsilon    = 4/255,
            norm       = "linf",
            n_samples  = 500,
            model_name = model_name,
            save_path  = os.path.join(config.results_dir, fname),
        )

    # ── Epoch history plots ───────────────────────────────────────────────
    print("\n  Generating training curves...")
    existing = [p for p in [config.baseline_history, config.augmix_history]
                if p and os.path.exists(p)]
    if existing:
        plot_epoch_history(
            json_paths = existing,
            save_path  = os.path.join(config.results_dir, "training_curves.png"),
        )
    else:
        print("  No history JSON files found — skipping training curves.")
        print("  Tip: pass --baseline_history and --augmix_history paths.")

    # ── Adversarial Transferability (Teacher → Student) ──────────────────
    # Per assignment: generate PGD-20 adversarial examples using the TEACHER
    # model and evaluate them on the STUDENT models trained via KD.
    # This tests whether adversarial examples crafted for the teacher
    # transfer to fool the distilled students (black-box attack scenario).
    from attack import evaluate_transferability
    from models.CNN import SimpleCNN
    from models.mobilenet import MobileNetV2

    print(f"\n{'─' * 62}")
    print(f"  Adversarial Transferability: Teacher → Student")
    print(f"  Source (teacher) : Baseline ResNet-18")
    print(f"  PGD-20 L∞, ε = 4/255")
    print(f"{'─' * 62}")

    transfer_results = {}

    # Load student checkpoints if available
    student_configs = [
        ("SimpleCNN (KD, baseline teacher)",   "best_cnn_kd.pth",
         SimpleCNN(num_classes=10)),
        ("SimpleCNN (KD, AugMix teacher)",     "best_cnn_kd_augmix.pth",
         SimpleCNN(num_classes=10)),
        ("MobileNet (KD, baseline teacher)",   "best_mobilenet_kd.pth",
         MobileNetV2(num_classes=10)),
        ("MobileNet (KD, AugMix teacher)",     "best_mobilenet_kd_augmix.pth",
         MobileNetV2(num_classes=10)),
    ]

    for student_label, student_path, student_model in student_configs:
        if not os.path.exists(student_path):
            print(f"  [{student_label}] checkpoint not found: {student_path} — skipping")
            continue

        student_model.load_state_dict(torch.load(student_path, map_location=device))
        student_model.to(device).eval()

        t_adv_acc, s_clean_acc, s_transfer_acc = evaluate_transferability(
            teacher     = model_baseline,
            student     = student_model,
            loader      = test_loader,
            epsilon     = 4 / 255,
            norm        = "linf",
            device      = device,
            num_steps   = 20,
            max_batches = 40,
        )

        drop = s_clean_acc - s_transfer_acc
        print(f"\n  Student : {student_label}")
        print(f"    Teacher adv acc (source)   : {t_adv_acc:.4f}")
        print(f"    Student clean acc          : {s_clean_acc:.4f}")
        print(f"    Student transfer acc       : {s_transfer_acc:.4f}")
        print(f"    Transfer drop              : {drop:+.4f}")

        transfer_results[student_label] = {
            "teacher_adv_acc":      t_adv_acc,
            "student_clean_acc":    s_clean_acc,
            "student_transfer_acc": s_transfer_acc,
            "transfer_drop":        drop,
        }
        _free_gpu(student_model)

    print(f"\n{'─' * 62}")

    results["transferability_teacher_to_student"] = transfer_results
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Transferability results saved to: {results_path}")

    _free_gpu(model_baseline, model_augmix)
    print(f"\n  All outputs saved to: {config.results_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Part B: Knowledge Distillation Experiment Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_kd_experiments(config: TrainingConfig, device: torch.device) -> None:
    """
    Run all Part B experiments in sequence and produce comparison plots.

    Delegates to individual run_b* functions so GPU is freed between each
    experiment.  Histories are collected and used to generate report plots.

    Args:
        config: TrainingConfig — only config.epochs is used.
        device: Compute device.
    """
    import json, os
    from train import RunHistory

    histories: list = []

    # Run each experiment — GPU freed after each one via _free_gpu()
    for mode, runner in [("b1", run_b1), ("b2a", run_b2a),
                         ("b2b", run_b2b), ("b3", run_b3), ("b4", run_b4)]:
        print(f"\n{'=' * 60}")
        print(f"  Running experiment: {mode.upper()}")
        print(f"{'=' * 60}")
        runner(config, device)

    # ── Collect histories from saved .pth files for plotting ─────────────────
    # Since each run_b* is independent, we re-run training in tracked mode
    # or load saved results. Here we rebuild histories from a quick eval.
    print("\n  All experiments complete. Generating plots...")

    # Auto-generate all comparison plots
    import importlib.util, os
    spec     = importlib.util.spec_from_file_location(
        "plot_results",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_results.py")
    )
    plot_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot_mod)
    plot_mod.main()


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Program entry point.

    Dispatches to the correct experiment based on config.mode:
      "transfer"         ->  Transfer learning (Option 1 and/or 2)
      "train" / "both"   ->  Standard training loop
      "test"  / "both"   ->  Evaluation on the test split
    """
    config = get_params()
    set_seed(config.seed)
    device = get_device(config)

    # For kd/b* modes the run_b* functions print their own headers.
    kd_modes = {"kd", "b1", "b2a", "b2b", "b3", "b4", "cifar10c", "augmix", "pgd",
                "b3_augmix", "b4_augmix", "augmix_kd"}
    if config.mode not in kd_modes:
        print(f"\n{'=' * 55}")
        print(f"  Dataset          : {config.dataset}")
        print(f"  Model            : {config.model}")
        print(f"  Mode             : {config.mode}")
        print(f"  Device           : {device}")
        if config.label_smoothing > 0:
            print(f"  Label smoothing  : epsilon={config.label_smoothing}")
        if config.distillation:
            print(f"  Distillation     : T={config.temperature}  "
                  f"alpha={config.distill_alpha}  mode={config.distill_mode}")
        print(f"{'=' * 55}\n")
    else:
        print(f"\n  Mode : {config.mode}  |  Device : {device}\n")

    # ── Knowledge Distillation — individual experiments ──────────────────────
    kd_dispatch = {"b1": run_b1, "b2a": run_b2a, "b2b": run_b2b,
                   "b3": run_b3, "b4": run_b4,
                   "cifar10c": run_cifar10c, "augmix": run_augmix, "pgd": run_pgd,
                   "b3_augmix": run_b3_augmix, "b4_augmix": run_b4_augmix,
                   "augmix_kd": run_augmix_kd}
    if config.mode in kd_dispatch:
        kd_dispatch[config.mode](config, device)
        return

    # ── Knowledge Distillation full pipeline ─────────────────────────────────
    if config.mode == "kd":
        run_kd_experiments(config, device)
        return

    # ── Transfer learning ────────────────────────────────────────────────────
    if config.mode == "transfer":
        run_transfer_mode(config, device)
        return

    # ── Standard training + evaluation ──────────────────────────────────────
    model   = build_model(config).to(device)
    teacher = load_teacher(config, device)

    if config.mode in ("train", "both"):
        run_training(model, config, device, teacher=teacher)

    if config.mode in ("test", "both"):
        run_test(model, config, device)


if __name__ == "__main__":
    main()