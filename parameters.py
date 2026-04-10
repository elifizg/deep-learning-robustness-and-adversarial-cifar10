"""
parameters.py
=============
Parses command-line arguments and returns a typed TrainingConfig dataclass.

Why dataclass instead of dict?
  - Access via config.learning_rate instead of params["learning_rate"].
  - Type hints are enforced; IDEs provide autocomplete and static analysis.
  - Accidental key typos are caught at definition time, not at runtime.
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# DataClass Definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """
    Holds all hyperparameters and settings for a training run.

    The @dataclass decorator auto-generates __init__, __repr__, and __eq__.
    Fields with mutable defaults (lists, tuples) must use field(default_factory=...)
    to avoid sharing the same object across all instances — a classic Python bug.
    """

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset:     str                = "mnist"   # "mnist" or "cifar10"
    data_dir:    str                = "./data"
    num_workers: int                = 2
    mean:        Tuple[float, ...]  = field(default_factory=tuple)
    std:         Tuple[float, ...]  = field(default_factory=tuple)

    # ── Model ─────────────────────────────────────────────────────────────────
    model:         str       = "mlp"
    input_size:    int       = 784
    hidden_sizes:  List[int] = field(default_factory=lambda: [512, 256, 128])
    num_classes:   int       = 10
    dropout:       float     = 0.3
    vgg_depth:     str       = "16"
    resnet_layers: List[int] = field(default_factory=lambda: [2, 2, 2, 2])

    # ── Transfer Learning ──────────────────────────────────────────────────────
    transfer_option: int  = 1      # 1 = resize + freeze backbone, 2 = modify conv layers
    freeze_backbone: bool = True   # if True, early layers are frozen in option 1

    # ── Label Smoothing ────────────────────────────────────────────────────────
    label_smoothing: float = 0.0   # 0.0 = disabled; typical value: 0.1

    # ── Knowledge Distillation ─────────────────────────────────────────────────
    distillation:  bool  = False
    teacher_path:  str   = "best_resnet.pth"
    temperature:   float = 4.0    # T > 1 softens the teacher's probability distribution
    distill_alpha: float = 0.3    # weight of the soft loss; (1 - alpha) weights the hard loss
    distill_mode:  str   = "standard"  # "standard" or "teacher_prob"

    # ── Training ──────────────────────────────────────────────────────────────
    epochs:        int   = 10
    batch_size:    int   = 64
    learning_rate: float = 1e-3
    weight_decay:  float = 1e-4

    # ── Misc ──────────────────────────────────────────────────────────────────
    tsne:         bool = False
    seed:         int  = 42
    device:       str  = "cpu"
    save_path:    str  = "best_model.pth"
    log_interval: int  = 100
    mode:         str  = "both"  # train/test/both/transfer/kd/b1/b2a/b2b/b3/b4/cifar10c/augmix/pgd

    # ── HW2: CIFAR-10-C ───────────────────────────────────────────────────────
    cifar10c_dir: str = "./CIFAR-10-C"  # path to CIFAR-10-C folder containing .npy files

    # ── HW2: AugMix ───────────────────────────────────────────────────────────
    augmix_severity:   int   = 3     # AugMix operation magnitude (1-10)
    augmix_width:      int   = 3     # number of augmentation chains
    augmix_lambda_jsd: float = 12.0  # weight of the JSD consistency loss

    # ── HW2: PGD / Visualisation ──────────────────────────────────────────────
    augmix_path:      str = "best_augmix.pth"  # AugMix checkpoint for PGD evaluation
    results_dir:      str = "./results"         # output folder for PGD/GradCAM/tSNE
    baseline_history: str = ""                  # path to baseline epoch history JSON
    augmix_history:   str = ""                  # path to AugMix epoch history JSON


# ─────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────────────────────

def get_params() -> TrainingConfig:
    """
    Parses command-line arguments and returns a populated TrainingConfig.

    Returns:
        TrainingConfig: Fully populated configuration object.

    Examples:
        python main.py --model resnet --dataset cifar10 --epochs 20
        python main.py --model cnn --distillation --teacher_path best_resnet.pth
        python main.py --model resnet --dataset cifar10 --label_smoothing 0.1
        python main.py --model mobilenet --distillation --distill_mode teacher_prob
    """
    parser = argparse.ArgumentParser(
        description="CIFAR-10 / MNIST: Transfer Learning & Knowledge Distillation"
    )

    # ── Core arguments ────────────────────────────────────────────────────────
    parser.add_argument("--mode", default="both",
        choices=["train", "test", "both", "transfer", "kd",
                 "b1", "b2a", "b2b", "b3", "b4",
                 "cifar10c", "augmix", "pgd",
                 "b3_augmix", "b4_augmix", "augmix_kd"])
    parser.add_argument("--dataset",    choices=["mnist", "cifar10"],      default="mnist")
    parser.add_argument("--model",
        choices=["mlp", "cnn", "vgg", "resnet", "mobilenet", "resnet18", "vgg16"], default="mlp")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--device",     type=str,   default="cpu")
    parser.add_argument("--batch_size", type=int,   default=64)

    # ── Model-specific ────────────────────────────────────────────────────────
    parser.add_argument("--vgg_depth",
        choices=["11", "13", "16", "19"], default="16")
    parser.add_argument("--resnet_layers", type=int, nargs=4,
        default=[2, 2, 2, 2], metavar=("L1", "L2", "L3", "L4"),
        help="Number of blocks per ResNet stage (default: 2 2 2 2 = ResNet-18)")

    # ── Transfer Learning ──────────────────────────────────────────────────────
    parser.add_argument("--transfer_option", type=int, choices=[0, 1, 2], default=1,
        help="1: resize images + freeze backbone  |  2: modify early conv layers")
    parser.add_argument("--no_freeze", action="store_true",
        help="In option 1, fine-tune all layers instead of freezing the backbone")

    # ── Label Smoothing ────────────────────────────────────────────────────────
    parser.add_argument("--label_smoothing", type=float, default=0.0,
        help="Label smoothing epsilon (0.0 = disabled; recommended: 0.1)")

    # ── Knowledge Distillation ─────────────────────────────────────────────────
    parser.add_argument("--distillation", action="store_true",
        help="Enable knowledge distillation training mode")
    parser.add_argument("--teacher_path", type=str, default="best_resnet.pth",
        help="Baseline model checkpoint (also used as KD teacher path)")
    parser.add_argument("--temperature",   type=float, default=4.0,
        help="Distillation temperature T")
    parser.add_argument("--distill_alpha", type=float, default=0.3,
        help="Weight of the soft KD loss")
    parser.add_argument("--tsne", action="store_true",
        help="Generate t-SNE feature space plots after transfer learning")
    parser.add_argument("--distill_mode", choices=["standard", "teacher_prob"],
        default="standard")
    parser.add_argument("--save_path", type=str, default="best_model.pth",
        help="Path to save the best model weights")

    # ── HW2: CIFAR-10-C ───────────────────────────────────────────────────────
    parser.add_argument("--cifar10c_dir", type=str, default="./CIFAR-10-C",
        help="Path to CIFAR-10-C folder containing .npy files")

    # ── HW2: AugMix ───────────────────────────────────────────────────────────
    parser.add_argument("--augmix_severity",   type=int,   default=3,
        help="AugMix operation magnitude 1-10 (default: 3)")
    parser.add_argument("--augmix_width",      type=int,   default=3,
        help="Number of AugMix augmentation chains (default: 3)")
    parser.add_argument("--augmix_lambda_jsd", type=float, default=12.0,
        help="Weight of the Jensen-Shannon consistency loss (default: 12.0)")

    # ── HW2: PGD / Visualisation ──────────────────────────────────────────────
    parser.add_argument("--augmix_path", type=str, default="best_augmix.pth",
        help="AugMix model checkpoint path for PGD evaluation")
    parser.add_argument("--results_dir", type=str, default="./results",
        help="Output folder for PGD results, Grad-CAM, and t-SNE plots")
    parser.add_argument("--baseline_history", type=str, default="",
        help="Path to baseline epoch history JSON for training curves plot")
    parser.add_argument("--augmix_history", type=str, default="",
        help="Path to AugMix epoch history JSON for training curves plot")

    args = parser.parse_args()

    # Auto-correct dataset for CIFAR-only modes
    if args.mode in ("transfer", "kd", "b1", "b2a", "b2b", "b3", "b4",
                     "cifar10c", "augmix", "pgd",
                     "b3_augmix", "b4_augmix", "augmix_kd"):
        if args.dataset == "mnist":
            args.dataset = "cifar10"

    # ── Dataset-dependent statistics ──────────────────────────────────────────
    if args.dataset == "mnist":
        input_size: int               = 784
        mean: Tuple[float, ...]       = (0.1307,)
        std:  Tuple[float, ...]       = (0.3081,)
    else:
        input_size = 3072
        mean       = (0.4914, 0.4822, 0.4465)
        std        = (0.2023, 0.1994, 0.2010)

    return TrainingConfig(
        dataset            = args.dataset,
        mean               = mean,
        std                = std,
        input_size         = input_size,
        model              = args.model,
        vgg_depth          = args.vgg_depth,
        resnet_layers      = args.resnet_layers,
        transfer_option    = args.transfer_option,
        freeze_backbone    = not args.no_freeze,
        label_smoothing    = args.label_smoothing,
        distillation       = args.distillation,
        teacher_path       = args.teacher_path,
        temperature        = args.temperature,
        distill_alpha      = args.distill_alpha,
        distill_mode       = args.distill_mode,
        epochs             = args.epochs,
        batch_size         = args.batch_size,
        learning_rate      = args.lr,
        device             = args.device,
        mode               = args.mode,
        tsne               = args.tsne,
        save_path          = args.save_path,
        cifar10c_dir       = args.cifar10c_dir,
        augmix_severity    = args.augmix_severity,
        augmix_width       = args.augmix_width,
        augmix_lambda_jsd  = args.augmix_lambda_jsd,
        augmix_path        = args.augmix_path,
        results_dir        = args.results_dir,
        baseline_history   = args.baseline_history,
        augmix_history     = args.augmix_history,
    )