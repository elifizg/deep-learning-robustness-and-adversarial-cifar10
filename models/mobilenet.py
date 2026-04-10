"""
models/mobilenet.py
===================
MobileNetV2 adapted for CIFAR-10 (3-channel, 32x32 images).

Reference:
    Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks",
    CVPR 2018.  https://arxiv.org/abs/1801.04381

Key design principle — Inverted Residual Block:
    Standard residual blocks expand channels in the middle (narrow->wide->narrow).
    MobileNetV2 inverts this: it starts narrow, expands wide for depthwise conv,
    then projects back to narrow.  The wide intermediate representation is never
    stored in memory across the residual connection, reducing peak memory usage.

    Each block:
        1. Pointwise conv (1x1): expand channels by factor t  (cheap, mixes channels)
        2. Depthwise conv (3x3, groups=channels): spatial filtering (very cheap)
        3. Pointwise conv (1x1): project back to out_planes    (cheap, mixes channels)
        No ReLU after step 3 — the "linear bottleneck" preserves information
        in low-dimensional spaces that ReLU would otherwise destroy.

CIFAR-10 adaptations vs the original ImageNet MobileNetV2:
    - conv1 stride 2 -> 1  (keeps 32x32 instead of reducing to 16x16 immediately)
    - Stage 2 stride 2 -> 1
    - Final avg pool kernel 7 -> 4  (matches the 4x4 feature map size)
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """
    Inverted Residual Block (the core building unit of MobileNetV2).

    Structure:
        expand  : Conv1x1(in_planes -> t*in_planes) + BN + ReLU6
        depthwise: Conv3x3(t*in_planes, groups=t*in_planes) + BN + ReLU6
        project : Conv1x1(t*in_planes -> out_planes) + BN   (NO ReLU)
        shortcut: identity if stride==1 and in_planes==out_planes,
                  else 1x1 conv if stride==1 and channels differ.

    ReLU6 clips activations at 6, which is important for fixed-point
    quantisation (e.g. running on mobile CPUs with 8-bit arithmetic).

    The skip connection is only applied when stride==1 because a stride>1
    depthwise conv changes the spatial resolution, making a residual addition
    between input and output impossible without a projection.

    Args:
        in_planes:  Number of input channels.
        out_planes: Number of output channels.
        expansion:  Channel expansion factor t for the intermediate representation.
        stride:     Depthwise conv stride (1 = same size, 2 = halve spatial dims).
    """

    def __init__(
        self,
        in_planes:  int,
        out_planes: int,
        expansion:  int,
        stride:     int,
    ) -> None:
        super().__init__()
        self.stride = stride
        planes = expansion * in_planes

        # Step 1: expand channels with pointwise conv.
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        # Step 2: depthwise conv — each channel filtered independently.
        # groups=planes means each channel has its own 3x3 filter.
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=planes, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        # Step 3: project back to out_planes — linear bottleneck (no ReLU).
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn3   = nn.BatchNorm2d(out_planes)

        # Shortcut: only when stride==1; may need 1x1 conv for channel mismatch.
        self.shortcut: nn.Module = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, in_planes, H, W).

        Returns:
            torch.Tensor: Output tensor, shape (B, out_planes, H_out, W_out).
        """
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))          # linear bottleneck — no activation

        # Add skip connection only when spatial size is unchanged (stride==1).
        if self.stride == 1:
            out = out + self.shortcut(x)
        return out


class MobileNetV2(nn.Module):
    """
    MobileNetV2 for CIFAR-10 classification.

    Stage configuration: (expansion t, out_channels, num_blocks, stride)
    Strides in stages 1 and 2 are reduced to 1 (vs 2 in the ImageNet version)
    to prevent the 32x32 feature maps from collapsing too early.

    Args:
        num_classes: Number of output classes (default 10 for CIFAR-10).
    """

    # (expansion, out_planes, num_blocks, stride)
    _CFG: List[Tuple[int, int, int, int]] = [
        (1,  16, 1, 1),
        (6,  24, 2, 1),   # stride 2->1 for CIFAR-10
        (6,  32, 3, 2),
        (6,  64, 4, 2),
        (6,  96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Stem: stride=1 for CIFAR-10 (ImageNet uses stride=2).
        self.conv1  = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)

        # Pointwise conv to project to 1280 channels before classification.
        self.conv2  = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2    = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes: int) -> nn.Sequential:
        """
        Build all inverted residual stages from the config table.

        For each stage with num_blocks > 1, only the first block uses the
        specified stride; all subsequent blocks use stride=1.

        Args:
            in_planes: Input channels for the first block (32 after the stem).

        Returns:
            nn.Sequential: All inverted residual blocks in order.
        """
        layers: List[nn.Module] = []
        for expansion, out_planes, num_blocks, stride in self._CFG:
            strides = [stride] + [1] * (num_blocks - 1)
            for s in strides:
                layers.append(Block(in_planes, out_planes, expansion, s))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, 3, 32, 32).

        Returns:
            torch.Tensor: Class logits, shape (B, num_classes).
        """
        out = F.relu6(self.bn1(self.conv1(x)))    # (B,   32, 32, 32)
        out = self.layers(out)                     # (B,  320,  4,  4)
        out = F.relu6(self.bn2(self.conv2(out)))   # (B, 1280,  4,  4)
        out = F.avg_pool2d(out, 4)                 # (B, 1280,  1,  1)  kernel=4 for CIFAR
        out = out.view(out.size(0), -1)            # (B, 1280)
        return self.linear(out)                    # (B, num_classes)
