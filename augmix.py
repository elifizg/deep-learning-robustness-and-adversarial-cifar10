"""
augmix.py
=========
AugMix data augmentation and Jensen-Shannon Consistency loss.

Reference:
    Hendrycks et al., "AugMix: A Simple Method to Improve Robustness and
    Uncertainty under Data Shift", ICLR 2020.
    https://openreview.net/forum?id=S1gmrxHFvB

Core idea:
    Standard augmentation applies one random transform per image.
    AugMix creates k augmented versions of each image by chaining 1–3
    randomly sampled operations with random magnitudes, then mixes them
    with the original using Dirichlet-sampled weights.

    This produces diverse, semantically consistent augmentations that
    improve robustness to distribution shift without degrading clean accuracy.

    The Jensen-Shannon Consistency loss enforces that the model produces
    similar predictions across the original and all augmented views,
    acting as a consistency regulariser on top of standard cross-entropy.

Loss:
    L = CE(p_orig, y) + lambda * JS(p_orig, p_aug1, p_aug2)

    where JS divergence for k distributions is:
        JS(p1,...,pk) = H(mean(pi)) - mean(H(pi))
        H(p) = -sum(p * log(p))   (entropy)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageEnhance
from torchvision import transforms
from typing import Tuple, List


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation Operations
# ─────────────────────────────────────────────────────────────────────────────

def int_parameter(level: float, maxval: float) -> int:
    """Scale a level in [0, 10] to an integer in [0, maxval]."""
    return int(level * maxval / 10)


def float_parameter(level: float, maxval: float) -> float:
    """Scale a level in [0, 10] to a float in [0, maxval]."""
    return float(level) * maxval / 10.


def autocontrast(pil_img: Image.Image, level: float) -> Image.Image:
    """Maximise image contrast by stretching the histogram."""
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img: Image.Image, level: float) -> Image.Image:
    """Equalise image histogram."""
    return ImageOps.equalize(pil_img)


def posterize(pil_img: Image.Image, level: float) -> Image.Image:
    """Reduce bits per channel."""
    level = int_parameter(level, 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img: Image.Image, level: float) -> Image.Image:
    """Rotate image by up to 30 degrees."""
    degrees = int_parameter(level, 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR, fillcolor=(128, 128, 128))


def solarize(pil_img: Image.Image, level: float) -> Image.Image:
    """Invert pixels above a threshold."""
    level = int_parameter(level, 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img: Image.Image, level: float) -> Image.Image:
    """Shear image along the x-axis."""
    level = float_parameter(level, 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        pil_img.size, Image.AFFINE, (1, level, 0, 0, 1, 0),
        resample=Image.BILINEAR, fillcolor=(128, 128, 128)
    )


def shear_y(pil_img: Image.Image, level: float) -> Image.Image:
    """Shear image along the y-axis."""
    level = float_parameter(level, 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        pil_img.size, Image.AFFINE, (1, 0, 0, level, 1, 0),
        resample=Image.BILINEAR, fillcolor=(128, 128, 128)
    )


def translate_x(pil_img: Image.Image, level: float) -> Image.Image:
    """Translate image along the x-axis."""
    level = int_parameter(level, pil_img.size[0] // 3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        pil_img.size, Image.AFFINE, (1, 0, level, 0, 1, 0),
        resample=Image.BILINEAR, fillcolor=(128, 128, 128)
    )


def translate_y(pil_img: Image.Image, level: float) -> Image.Image:
    """Translate image along the y-axis."""
    level = int_parameter(level, pil_img.size[1] // 3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, level),
        resample=Image.BILINEAR, fillcolor=(128, 128, 128)
    )


def color(pil_img: Image.Image, level: float) -> Image.Image:
    """Adjust colour balance."""
    level = float_parameter(level, 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


def contrast(pil_img: Image.Image, level: float) -> Image.Image:
    """Adjust contrast."""
    level = float_parameter(level, 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


def brightness(pil_img: Image.Image, level: float) -> Image.Image:
    """Adjust brightness."""
    level = float_parameter(level, 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


def sharpness(pil_img: Image.Image, level: float) -> Image.Image:
    """Adjust sharpness."""
    level = float_parameter(level, 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


# All available augmentation operations
AUGMENTATIONS = [
    autocontrast, equalize, posterize, rotate, solarize,
    shear_x, shear_y, translate_x, translate_y,
    color, contrast, brightness, sharpness,
]


# ─────────────────────────────────────────────────────────────────────────────
# AugMix Transform
# ─────────────────────────────────────────────────────────────────────────────

def augment_and_mix(
    image: Image.Image,
    severity: int = 3,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
) -> Image.Image:
    """
    Apply AugMix to a single PIL image.

    Creates `width` augmented versions of the image, each produced by
    chaining 1–3 randomly sampled operations. Mixes them with the
    original image using Dirichlet-sampled weights.

    Args:
        image:    Input PIL image.
        severity: Magnitude of each operation (1–10).
        width:    Number of augmentation chains (k in the paper).
        depth:    Chain length; -1 = random in [1, 3].
        alpha:    Dirichlet / Beta concentration parameter.

    Returns:
        PIL.Image: AugMix-augmented image.
    """
    ws = np.float32(np.random.dirichlet([alpha] * width))   # chain weights
    m  = np.float32(np.random.beta(alpha, alpha))           # mix weight

    mix = np.zeros_like(np.array(image, dtype=np.float32))

    for i in range(width):
        image_aug = image.copy()
        d = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(d):
            op = np.random.choice(AUGMENTATIONS)
            image_aug = op(image_aug, severity)
        mix += ws[i] * np.array(image_aug, dtype=np.float32)

    # Blend: m * original + (1 - m) * augmented_mix
    mixed = (1 - m) * np.array(image, dtype=np.float32) + m * mix
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))


class AugMixTransform:
    """
    AugMix wrapper for use in a DataLoader.

    Returns a tuple of (original, augmented_1, augmented_2) tensors
    instead of a single tensor, so the Jensen-Shannon loss can be computed
    over all three views.

    The pre-process pipeline (crop + flip) is applied to the original PIL
    image before AugMix so all three views share the same spatial crop.
    The normalisation is applied after conversion to tensor.

    Args:
        mean:     Channel mean for normalisation.
        std:      Channel std for normalisation.
        severity: AugMix operation magnitude (1–10).
        width:    Number of augmentation chains.
        depth:    Chain length (-1 = random).
        alpha:    Dirichlet concentration parameter.
    """

    def __init__(
        self,
        mean: Tuple[float, ...] = (0.4914, 0.4822, 0.4465),
        std:  Tuple[float, ...] = (0.2023, 0.1994, 0.2010),
        severity: int   = 3,
        width:    int   = 3,
        depth:    int   = -1,
        alpha:    float = 1.0,
    ) -> None:
        self.severity = severity
        self.width    = width
        self.depth    = depth
        self.alpha    = alpha

        self.preprocess = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __call__(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image: PIL image from the dataset.

        Returns:
            Tuple of (orig, aug1, aug2) normalised tensors, each (C, H, W).
        """
        # Apply spatial augmentation once — all views share the same crop/flip
        image = self.preprocess(image)

        aug1 = augment_and_mix(image, self.severity, self.width, self.depth, self.alpha)
        aug2 = augment_and_mix(image, self.severity, self.width, self.depth, self.alpha)

        return self.to_tensor(image), self.to_tensor(aug1), self.to_tensor(aug2)


# ─────────────────────────────────────────────────────────────────────────────
# Jensen-Shannon Consistency Loss
# ─────────────────────────────────────────────────────────────────────────────

class JensenShannonLoss(nn.Module):
    """
    Jensen-Shannon Consistency loss for AugMix training.

    Enforces that the model produces similar predictions across the original
    image and its two AugMix augmented views:

        JS(p_orig, p_aug1, p_aug2) = H(M) - (H(p_orig) + H(p_aug1) + H(p_aug2)) / 3

    where M = (p_orig + p_aug1 + p_aug2) / 3  and  H(p) = -sum(p * log(p)).

    Total AugMix loss:
        L = CE(p_orig, y) + lambda_jsd * JS(p_orig, p_aug1, p_aug2)

    Args:
        lambda_jsd: Weight of the JS consistency term (default 12, as in paper).
        num_classes: Number of output classes.
    """

    def __init__(self, lambda_jsd: float = 12.0, num_classes: int = 10) -> None:
        super().__init__()
        self.lambda_jsd  = lambda_jsd
        self.num_classes = num_classes
        self.ce          = nn.CrossEntropyLoss()

    def forward(
        self,
        logits_orig: torch.Tensor,
        logits_aug1: torch.Tensor,
        logits_aug2: torch.Tensor,
        targets:     torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            logits_orig: Model output for original images, shape (B, C).
            logits_aug1: Model output for first augmented view, shape (B, C).
            logits_aug2: Model output for second augmented view, shape (B, C).
            targets:     Ground-truth labels, shape (B,).

        Returns:
            Tuple of (total_loss, ce_loss, jsd_loss) for logging.
        """
        ce_loss = self.ce(logits_orig, targets)

        # Convert logits to probabilities
        p_orig = F.softmax(logits_orig, dim=1)
        p_aug1 = F.softmax(logits_aug1, dim=1)
        p_aug2 = F.softmax(logits_aug2, dim=1)

        # Mixture distribution M
        M = (p_orig + p_aug1 + p_aug2) / 3.0

        # JS divergence = H(M) - mean(H(pi))
        # Clamp to avoid log(0)
        eps = 1e-8
        H_M    = -(M * torch.log(M + eps)).sum(dim=1).mean()
        H_orig = -(p_orig * torch.log(p_orig + eps)).sum(dim=1).mean()
        H_aug1 = -(p_aug1 * torch.log(p_aug1 + eps)).sum(dim=1).mean()
        H_aug2 = -(p_aug2 * torch.log(p_aug2 + eps)).sum(dim=1).mean()

        jsd_loss = H_M - (H_orig + H_aug1 + H_aug2) / 3.0

        total_loss = ce_loss + self.lambda_jsd * jsd_loss
        return total_loss, ce_loss, jsd_loss