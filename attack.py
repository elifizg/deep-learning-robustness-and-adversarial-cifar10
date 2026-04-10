"""
attack.py
=========
PGD adversarial attacks, Grad-CAM visualisation, and t-SNE embedding
for adversarial robustness evaluation.

Reference:
    Madry et al., "Towards Deep Learning Models Resistant to Adversarial
    Attacks", ICLR 2018.

Contents:
    pgd_attack()              — PGD-20 with L∞ or L2 norm
    evaluate_pgd()            — evaluate model accuracy under PGD attack
    evaluate_transferability()— test adversarial transferability across models
    grad_cam()                — Grad-CAM heatmap extraction
    visualize_gradcam()       — plot clean vs adversarial + Grad-CAM overlays
    visualize_tsne_adv()      — t-SNE of clean vs adversarial feature embeddings
    save_epoch_history()      — save per-epoch training metrics to JSON
    plot_epoch_history()      — plot training curves from saved JSON
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

CIFAR10_CLASSES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ─────────────────────────────────────────────────────────────────────────────
# PGD Attack
# ─────────────────────────────────────────────────────────────────────────────

def pgd_attack(
    model:     nn.Module,
    images:    torch.Tensor,
    labels:    torch.Tensor,
    epsilon:   float,
    norm:      str   = "linf",
    num_steps: int   = 20,
    step_size: float = None,
    random_start: bool = True,
) -> torch.Tensor:
    """
    PGD-20 adversarial attack (Madry et al., 2018).

    Iteratively perturbs the input to maximise the cross-entropy loss,
    while keeping the perturbation within an epsilon-ball around the
    original image. The ball is defined by either the L∞ or L2 norm.

    L∞ attack:
        Each pixel can be shifted by at most epsilon in absolute value.
        Step size = epsilon / 4 (typical rule of thumb).
        Projection: clip perturbation to [-epsilon, epsilon] per pixel.

    L2 attack:
        The total perturbation vector has L2 norm at most epsilon.
        Step size = epsilon / 10.
        Projection: rescale perturbation to have L2 norm ≤ epsilon.

    Args:
        model:        Model to attack (must be in eval mode).
        images:       Clean input images, shape (B, C, H, W). Values in
                      normalised space (not [0,1]).
        labels:       True class labels, shape (B,).
        epsilon:      Perturbation budget (4/255 for L∞, 0.25 for L2).
        norm:         "linf" or "l2".
        num_steps:    Number of PGD iterations (20 for PGD-20).
        step_size:    Step size per iteration. If None, set automatically.
        random_start: If True, initialise perturbation randomly within
                      the epsilon-ball (recommended for stronger attacks).

    Returns:
        torch.Tensor: Adversarial images, same shape as input.
    """
    assert norm in ("linf", "l2"), f"norm must be 'linf' or 'l2', got '{norm}'"

    if step_size is None:
        step_size = epsilon / 4 if norm == "linf" else epsilon / 10

    model.eval()
    images = images.clone().detach()
    device = images.device

    # Random initialisation within the epsilon-ball
    if random_start:
        if norm == "linf":
            delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
        else:
            delta = torch.randn_like(images)
            delta = delta / delta.view(delta.size(0), -1).norm(dim=1).view(-1, 1, 1, 1)
            delta = delta * torch.empty(delta.size(0), 1, 1, 1,
                                        device=device).uniform_(0, epsilon)
    else:
        delta = torch.zeros_like(images)

    delta = delta.to(device)

    for _ in range(num_steps):
        delta.requires_grad_(True)

        logits = model(images + delta)
        loss   = F.cross_entropy(logits, labels)
        grad   = torch.autograd.grad(loss, delta)[0]

        with torch.no_grad():
            if norm == "linf":
                # Gradient sign step
                delta = delta + step_size * grad.sign()
                # Project onto L∞ ball
                delta = delta.clamp(-epsilon, epsilon)
            else:
                # Normalised gradient step
                grad_norm = grad.view(grad.size(0), -1).norm(dim=1).view(-1, 1, 1, 1)
                delta = delta + step_size * grad / (grad_norm + 1e-8)
                # Project onto L2 ball
                delta_norm = delta.view(delta.size(0), -1).norm(dim=1).view(-1, 1, 1, 1)
                delta = delta * torch.clamp(epsilon / (delta_norm + 1e-8), max=1.0)

        delta = delta.detach()

    return (images + delta).detach()


def evaluate_pgd(
    model:     nn.Module,
    loader:    DataLoader,
    epsilon:   float,
    norm:      str,
    device:    torch.device,
    num_steps: int = 20,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Evaluate model accuracy on clean and PGD-adversarial examples.

    Args:
        model:       Model to evaluate (will be set to eval mode).
        loader:      DataLoader with clean test images.
        epsilon:     Perturbation budget.
        norm:        "linf" or "l2".
        device:      Compute device.
        num_steps:   PGD iterations.
        max_batches: If set, only evaluate on this many batches (faster).

    Returns:
        Tuple[float, float]: (clean_accuracy, adversarial_accuracy).
    """
    model.eval()
    clean_correct, adv_correct, total = 0, 0, 0

    for i, (images, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        images, labels = images.to(device), labels.to(device)

        # Clean accuracy — no gradients needed
        with torch.no_grad():
            logits_clean = model(images)
        clean_correct += logits_clean.argmax(1).eq(labels).sum().item()

        # Adversarial accuracy — PGD needs gradients internally
        adv_images = pgd_attack(model, images, labels, epsilon, norm, num_steps)
        with torch.no_grad():
            logits_adv = model(adv_images)
        adv_correct += logits_adv.argmax(1).eq(labels).sum().item()

        total += labels.size(0)

    return clean_correct / total, adv_correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM: gradient-weighted class activation mapping.

    Registers forward and backward hooks on the target convolutional layer
    to capture activations and gradients. The CAM is computed as:

        weights = global_average_pool(gradients)   # (C,)
        cam     = ReLU(sum_c(weights_c * activations_c))  # (H, W)

    The ReLU discards negative contributions (features that suppress the
    predicted class). The resulting heatmap highlights image regions most
    relevant to the model's decision.

    Args:
        model:      Model to explain.
        target_layer: The convolutional layer to hook (last conv recommended).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model        = model
        self.activations  = None
        self.gradients    = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, images: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for the given images.

        Args:
            images:    Input tensor, shape (B, C, H, W).
            class_idx: Target class index. If None, uses the predicted class.

        Returns:
            np.ndarray: Heatmaps, shape (B, H, W), values in [0, 1].
        """
        self.model.eval()
        images = images.clone().requires_grad_(True)

        logits = self.model(images)

        if class_idx is None:
            class_idx = logits.argmax(dim=1)

        # Backpropagate only the target class score
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        for b in range(logits.size(0)):
            idx = class_idx[b] if isinstance(class_idx, torch.Tensor) else class_idx
            one_hot[b, idx] = 1.0
        logits.backward(gradient=one_hot)

        # weights = global average of gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1)        # (B, H, W)
        cam     = F.relu(cam)

        # Normalise each heatmap to [0, 1]
        cam = cam.cpu().numpy()
        for b in range(cam.shape[0]):
            mn, mx = cam[b].min(), cam[b].max()
            if mx > mn:
                cam[b] = (cam[b] - mn) / (mx - mn)

        return cam

    def remove(self) -> None:
        """Remove hooks to free memory."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_target_layer(model: nn.Module) -> nn.Module:
    """
    Return the last convolutional layer of a ResNet-18 model.

    For torchvision ResNet-18, the last conv is in layer4[-1].conv2.
    For the custom ResNet class, it's the same path.

    Args:
        model: ResNet-18 model (torchvision or custom).

    Returns:
        nn.Module: The target convolutional layer.
    """
    return model.layer4[-1].conv2


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, ...] = (0.4914, 0.4822, 0.4465),
    std:  Tuple[float, ...] = (0.2023, 0.1994, 0.2010),
) -> np.ndarray:
    """
    Reverse CIFAR-10 normalisation and return a uint8 numpy image.

    Args:
        tensor: Normalised image tensor, shape (C, H, W).

    Returns:
        np.ndarray: RGB image, shape (H, W, 3), dtype uint8.
    """
    t = tensor.clone().cpu()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    t = t.clamp(0, 1).permute(1, 2, 0).numpy()
    return (t * 255).astype(np.uint8)


def overlay_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on an RGB image.

    Args:
        image: RGB image, shape (H, W, 3), uint8.
        cam:   Heatmap, shape (H, W), values in [0, 1].
        alpha: Heatmap opacity.

    Returns:
        np.ndarray: Blended image, shape (H, W, 3), uint8.
    """
    from PIL import Image as PILImage
    h, w = image.shape[:2]

    # Resize CAM to match image size
    cam_img = PILImage.fromarray((cam * 255).astype(np.uint8)).resize(
        (w, h), PILImage.BILINEAR
    )
    cam_np = np.array(cam_img) / 255.0

    # Apply a blue→red colormap
    cmap    = plt.cm.jet(cam_np)[..., :3]          # (H, W, 3) float
    overlay = (1 - alpha) * (image / 255.0) + alpha * cmap
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


def visualize_gradcam(
    model_baseline: nn.Module,
    model_augmix:   nn.Module,
    loader:         DataLoader,
    device:         torch.device,
    epsilon:        float   = 4 / 255,
    norm:           str     = "linf",
    n_samples:      int     = 2,
    save_path:      str     = "gradcam_comparison.png",
) -> None:
    """
    Find samples misclassified under PGD attack, then plot:
      clean image | clean Grad-CAM | adv image | adv Grad-CAM

    for both the baseline and AugMix models side-by-side.

    Samples are selected such that:
    - The clean image is correctly classified by the model.
    - The adversarial image is misclassified.

    Args:
        model_baseline: Fine-tuned model without AugMix.
        model_augmix:   Fine-tuned model with AugMix.
        loader:         Test DataLoader.
        device:         Compute device.
        epsilon:        PGD perturbation budget.
        norm:           "linf" or "l2".
        n_samples:      Number of qualifying samples to visualise.
        save_path:      Output PNG filename.
    """
    found_clean, found_adv, found_labels = [], [], []
    found_pred_clean, found_pred_adv     = [], []

    for images, labels in loader:
        if len(found_clean) >= n_samples:
            break
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            preds_clean = model_baseline(images).argmax(1)

        adv_images  = pgd_attack(model_baseline, images, labels, epsilon, norm)
        with torch.no_grad():
            preds_adv = model_baseline(adv_images).argmax(1)

        # Select correctly-classified clean, misclassified adversarial
        for i in range(images.size(0)):
            if len(found_clean) >= n_samples:
                break
            if preds_clean[i] == labels[i] and preds_adv[i] != labels[i]:
                found_clean.append(images[i].unsqueeze(0))
                found_adv.append(adv_images[i].unsqueeze(0))
                found_labels.append(labels[i].item())
                found_pred_clean.append(preds_clean[i].item())
                found_pred_adv.append(preds_adv[i].item())

    if not found_clean:
        print("  No qualifying samples found for Grad-CAM visualisation.")
        return

    # ── Plot ─────────────────────────────────────────────────────────────
    # 4 columns: Clean | Clean CAM | Adv | Adv CAM
    n      = len(found_clean)
    n_rows = n * 2
    fig, axes = plt.subplots(n_rows, 4, figsize=(18, n_rows * 4.5))
    if n_rows == 2:
        axes = axes.reshape(2, 4)

    col_titles = ["Clean image", "Clean Grad-CAM", "Adv image", "Adv Grad-CAM"]

    from PIL import Image as PILImage
    TARGET = (256, 256)

    def upsample(arr_uint8: np.ndarray) -> np.ndarray:
        return np.array(PILImage.fromarray(arr_uint8).resize(TARGET, PILImage.LANCZOS))

    def upsample_cam(cam: np.ndarray) -> np.ndarray:
        return np.array(PILImage.fromarray(
            (cam * 255).astype(np.uint8)).resize(TARGET, PILImage.BILINEAR)) / 255.

    for si in range(n):
        img_t  = found_clean[si]
        adv_t  = found_adv[si]
        true_c = CIFAR10_CLASSES[found_labels[si]]
        pred_c = CIFAR10_CLASSES[found_pred_clean[si]]
        pred_a = CIFAR10_CLASSES[found_pred_adv[si]]

        img_np = denormalize(img_t[0])
        adv_np = denormalize(adv_t[0])
        img_up = upsample(img_np)
        adv_up = upsample(adv_np)

        for mi, (model, label) in enumerate([
            (model_baseline, "Baseline"),
            (model_augmix,   "AugMix"),
        ]):
            gcam = GradCAM(model, get_target_layer(model))
            row  = si * 2 + mi

            cam_clean = gcam(img_t.clone())[0]
            cam_adv   = gcam(adv_t.clone())[0]
            gcam.remove()

            cam_c_up = upsample_cam(cam_clean)
            cam_a_up = upsample_cam(cam_adv)

            imgs_row = [
                img_up,
                overlay_heatmap(img_up, cam_c_up),
                adv_up,
                overlay_heatmap(adv_up, cam_a_up),
            ]

            for col, (ax, img_show) in enumerate(zip(axes[row], imgs_row)):
                ax.imshow(img_show, interpolation="lanczos")
                ax.axis("off")
                if row == 0:
                    ax.set_title(col_titles[col], fontsize=12, fontweight="bold", pad=8)
                if col == 0:
                    ax.text(
                        -0.22, 0.5,
                        f"{label}\nTrue: {true_c}\n✓ {pred_c} → ✗ {pred_a}",
                        transform=ax.transAxes,
                        fontsize=10, fontweight="bold",
                        va="center", ha="right",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="#f0f0f0", edgecolor="#aaaaaa", alpha=0.9),
                    )

    norm_label = "L∞ ε=4/255" if norm == "linf" else "L2 ε=0.25"
    plt.suptitle(
        f"Grad-CAM: Clean vs PGD-20 Adversarial  [{norm_label}]",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout(rect=[0.2, 0, 1, 1])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Grad-CAM plot saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# t-SNE Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualize_tsne_adv(
    model:       nn.Module,
    loader:      DataLoader,
    device:      torch.device,
    epsilon:     float = 4 / 255,
    norm:        str   = "linf",
    n_samples:   int   = 500,
    model_name:  str   = "model",
    save_path:   str   = "tsne_adversarial.png",
) -> None:
    """
    t-SNE visualisation comparing clean and adversarial feature embeddings.

    Extracts penultimate-layer features for n_samples clean images and their
    PGD-adversarial counterparts. Projects both sets to 2D with t-SNE and
    plots them with different markers:
        • filled circle  = clean sample
        • cross (×)      = adversarial sample
    Colour encodes the true class label (10 colours for CIFAR-10).

    Adversarial samples that are misclassified will appear far from their
    clean counterparts in the t-SNE space, showing how PGD pushes embeddings
    across decision boundaries.

    Args:
        model:      Model to extract features from.
        loader:     Test DataLoader.
        device:     Compute device.
        epsilon:    PGD perturbation budget.
        norm:       "linf" or "l2".
        n_samples:  Number of clean/adversarial pairs to embed.
        model_name: Label for the plot title.
        save_path:  Output PNG filename.
    """
    from sklearn.manifold import TSNE

    model.eval()
    features_clean, features_adv = [], []
    labels_all = []

    # Hook to capture features before the final FC layer
    feat_buf: List[torch.Tensor] = []

    def _hook(module, inp, out):
        feat_buf.append(inp[0].detach().cpu())

    # Works for both torchvision ResNet (model.fc) and custom ResNet (model.linear)
    fc_layer = model.fc if hasattr(model, "fc") else model.linear
    handle   = fc_layer.register_forward_hook(_hook)

    collected = 0
    for images, labels in loader:
        if collected >= n_samples:
            break
        images, labels = images.to(device), labels.to(device)
        take = min(images.size(0), n_samples - collected)
        images, labels = images[:take], labels[:take]

        # Clean features
        feat_buf.clear()
        with torch.no_grad():
            model(images)
        features_clean.append(feat_buf[0].clone())

        # Adversarial features
        adv_images = pgd_attack(model, images, labels, epsilon, norm)
        feat_buf.clear()
        with torch.no_grad():
            model(adv_images)
        features_adv.append(feat_buf[0].clone())

        labels_all.append(labels.cpu())
        collected += take

    handle.remove()

    feats_c  = torch.cat(features_clean).numpy()
    feats_a  = torch.cat(features_adv).numpy()
    lbls     = torch.cat(labels_all).numpy()

    # Stack clean + adv for joint t-SNE (preserves relative distances)
    all_feats = np.concatenate([feats_c, feats_a], axis=0)
    n         = len(feats_c)

    print(f"  Fitting t-SNE on {2 * n} points ({n} clean + {n} adversarial)...")
    tsne   = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(all_feats)

    coords_c = coords[:n]
    coords_a = coords[n:]

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap    = plt.cm.tab10

    for cls in range(10):
        mask = lbls == cls
        color = cmap(cls)
        ax.scatter(coords_c[mask, 0], coords_c[mask, 1],
                   s=18, color=color, marker="o", alpha=0.7,
                   label=f"{CIFAR10_CLASSES[cls]} (clean)")
        ax.scatter(coords_a[mask, 0], coords_a[mask, 1],
                   s=18, color=color, marker="x", alpha=0.7)

    # Legend: class colours only (clean=circle, adv=cross explained in title)
    handles, lbl_texts = ax.get_legend_handles_labels()
    ax.legend(handles, [CIFAR10_CLASSES[c] for c in range(10)],
              markerscale=1.5, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=8)

    norm_label = "L∞ ε=4/255" if norm == "linf" else "L2 ε=0.25"
    ax.set_title(
        f"t-SNE: clean (●) vs PGD-20 adversarial (×) — {model_name} [{norm_label}]",
        fontsize=12
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  t-SNE plot saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Epoch History — Save & Plot
# ─────────────────────────────────────────────────────────────────────────────

def save_epoch_history(
    history: Dict,
    save_path: str = "epoch_history.json",
) -> None:
    """
    Save per-epoch training metrics to a JSON file.

    Expected history dict format:
        {
            "label":      str,
            "train_loss": [float, ...],
            "train_acc":  [float, ...],
            "val_acc":    [float, ...],
            "test_acc":   float,
        }

    Args:
        history:   Dict with per-epoch metrics.
        save_path: Output JSON filename.
    """
    with open(save_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Epoch history saved to: {save_path}")


def plot_epoch_history(
    json_paths: List[str],
    save_path:  str = "training_curves.png",
) -> None:
    """
    Load one or more epoch history JSON files and plot training curves.

    Produces two side-by-side subplots:
        Left:  Training loss per epoch
        Right: Validation accuracy per epoch

    Each JSON file is drawn as a separate line so multiple runs (e.g.
    baseline vs AugMix) can be compared on the same axes.

    Args:
        json_paths: List of JSON file paths produced by save_epoch_history().
        save_path:  Output PNG filename.
    """
    histories = []
    for p in json_paths:
        if os.path.exists(p):
            with open(p) as f:
                histories.append(json.load(f))
        else:
            print(f"  Warning: history file not found: {p}")

    if not histories:
        print("  No history files found — skipping plot.")
        return

    colors = ["#4f86c6", "#e07b39", "#5aab61", "#c45c8a"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for i, h in enumerate(histories):
        c      = colors[i % len(colors)]
        epochs = list(range(1, len(h["train_loss"]) + 1))
        axes[0].plot(epochs, h["train_loss"], marker="o", color=c, label=h["label"])
        axes[1].plot(epochs, h["val_acc"],    marker="o", color=c, label=h["label"])

    for ax, title, ylabel in zip(
        axes,
        ["Training Loss per Epoch", "Validation Accuracy per Epoch"],
        ["Loss", "Accuracy"],
    ):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(linestyle="--", alpha=0.5)

    # Add overall title listing model names
    model_names = " vs ".join(h["label"] for h in histories)
    plt.suptitle(f"Training Curves — {model_names}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial Transferability
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_transferability(
    teacher:    nn.Module,
    student:    nn.Module,
    loader:     DataLoader,
    epsilon:    float,
    norm:       str,
    device:     torch.device,
    num_steps:  int = 20,
    max_batches: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Test adversarial transferability: generate PGD examples against the teacher
    model and evaluate them on both the teacher and the student.

    This measures how well adversarial perturbations crafted for one model
    transfer to fool a different model — a key property studied in the
    black-box attack literature (Papernot et al., 2017).

    A high transfer rate (low student accuracy on teacher-adversarial examples)
    indicates that the two models share similar decision boundaries, which often
    happens when they are trained on the same dataset.

    Args:
        teacher:     Model used to generate adversarial examples (white-box).
        student:     Model on which the transferability is tested (black-box).
        loader:      Test DataLoader with clean images.
        epsilon:     PGD perturbation budget.
        norm:        "linf" or "l2".
        device:      Compute device.
        num_steps:   PGD iterations (20 for PGD-20).
        max_batches: If set, evaluate only this many batches (faster).

    Returns:
        Tuple[float, float, float]:
            - teacher_adv_acc   : teacher accuracy on its own adversarial examples.
            - student_clean_acc : student accuracy on clean images.
            - student_transfer_acc : student accuracy on teacher-adversarial examples.
    """
    teacher.eval()
    student.eval()

    t_adv_correct, s_clean_correct, s_transfer_correct, total = 0, 0, 0, 0

    for i, (images, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        images, labels = images.to(device), labels.to(device)

        # Generate adversarial examples using the teacher (white-box)
        adv_images = pgd_attack(teacher, images, labels, epsilon, norm, num_steps)

        with torch.no_grad():
            t_adv_correct      += teacher(adv_images).argmax(1).eq(labels).sum().item()
            s_clean_correct    += student(images).argmax(1).eq(labels).sum().item()
            s_transfer_correct += student(adv_images).argmax(1).eq(labels).sum().item()

        total += labels.size(0)

    return (
        t_adv_correct      / total,
        s_clean_correct    / total,
        s_transfer_correct / total,
    )