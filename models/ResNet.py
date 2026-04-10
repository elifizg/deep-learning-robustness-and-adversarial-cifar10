"""
models/ResNet.py
================
ResNet implementation adapted for CIFAR-10 (3-channel, 32x32 images).

Reference:
    He et al., "Deep Residual Learning for Image Recognition",
    CVPR 2016.  https://arxiv.org/abs/1512.03385
"""

from typing import Callable, List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    """
    Wraps an arbitrary callable as an nn.Module.

    Used to implement the Option A shortcut in BasicBlock: zero-padding
    is applied inline without defining a separate named layer class.

    Args:
        lambd: Any callable that accepts and returns a torch.Tensor.
    """

    def __init__(self, lambd: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambd(x)


class BasicBlock(nn.Module):
    """
    Basic residual block for shallow ResNets (ResNet-18, ResNet-34).

    Contains two 3x3 conv layers with BatchNorm and ReLU.  A skip connection
    adds the block input directly to the conv output before the final ReLU:

        out = ReLU(F(x) + shortcut(x))

    This additive skip connection is the core contribution of ResNet.  In very
    deep networks, gradients tend to vanish as they are multiplied through
    dozens of layers during backprop.  The skip connection creates a direct
    gradient highway from the loss back to early layers:

        d_loss/d_x  =  d_loss/d_out  *  (d_F/d_x  +  1)

    The constant 1 term ensures that gradients always have a path to flow
    regardless of how small d_F/d_x becomes.

    Shortcut options when spatial size or channel count changes:
      Option A: zero-pad channels and subsample spatially — no extra params.
      Option B: 1x1 conv with stride — learns the projection, more expressive.

    Args:
        in_channels: Number of input feature map channels.
        channels:    Number of output channels for both conv layers.
        stride:      Stride for the first conv (2 = downsample spatial size).
        norm:        Normalisation layer constructor.
        option:      "A" or "B" shortcut type when dimensions change.

    Attributes:
        expansion: Output channel multiplier (1 for BasicBlock).
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        channels:    int,
        stride:      int             = 1,
        norm:        Type[nn.Module] = nn.BatchNorm2d,
        option:      str             = "B",
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = norm(channels)

        self.shortcut: nn.Module = nn.Sequential()

        if stride != 1 or in_channels != channels:
            if option == "A":
                # Subsample spatially with slice [::2, ::2], then zero-pad channels.
                # x[:, :, ::2, ::2] : take every 2nd pixel -> halves H and W.
                # F.pad(..., (0,0,0,0, ch//4, ch//4)) : pad channel dim on both sides.
                c = channels
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, c // 4, c // 4),
                        "constant", 0,
                    )
                )
            elif option == "B":
                # Learnable 1x1 conv + BN to match output dimensions.
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, self.expansion * channels,
                              kernel_size=1, stride=stride, bias=False),
                    norm(self.expansion * channels),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor, shape (B, channels, H_out, W_out).
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)    # residual addition
        return F.relu(out)


class ResNet(nn.Module):
    """
    Generic ResNet builder for CIFAR-10 (32x32 input).

    Stem uses a 3x3 conv with stride=1 (no downsampling) so that the small
    spatial resolution is preserved through the early layers.  Compare with
    the ImageNet ResNet stem (7x7 conv, stride=2, then maxpool) which is
    designed for 224x224 inputs.

    After the stem, four residual stages progressively double the channel
    count and halve the spatial size via stride=2 in the first block of
    stages 2-4:
        32x32 -> 32x32 -> 16x16 -> 8x8 -> 4x4

    Global average pooling then collapses the spatial dimensions to 1x1
    before the final linear classifier.

    Args:
        block:       Residual block class (BasicBlock or a Bottleneck variant).
        num_blocks:  List of 4 ints — blocks per stage (e.g. [2,2,2,2] = ResNet-18).
        norm:        Normalisation layer constructor.
        num_classes: Number of output classes.

    Example:
        >>> model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        >>> logits = model(torch.randn(8, 3, 32, 32))
        >>> logits.shape
        torch.Size([8, 10])
    """

    def __init__(
        self,
        block:       Type[BasicBlock],
        num_blocks:  List[int],
        norm:        Type[nn.Module] = nn.BatchNorm2d,
        num_classes: int             = 10,
    ) -> None:
        super().__init__()
        self.in_channels: int = 64

        # Stem: single 3x3 conv, no downsampling (stride=1 for CIFAR-10).
        self.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = norm(64)

        # Four residual stages.
        self.layer1  = self._make_layer(block, 64,  num_blocks[0], norm, stride=1)
        self.layer2  = self._make_layer(block, 128, num_blocks[1], norm, stride=2)
        self.layer3  = self._make_layer(block, 256, num_blocks[2], norm, stride=2)
        self.layer4  = self._make_layer(block, 512, num_blocks[3], norm, stride=2)

        # Global average pooling: (B, 512, H, W) -> (B, 512, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear  = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block:      Type[BasicBlock],
        channels:   int,
        num_blocks: int,
        norm:       Type[nn.Module],
        stride:     int,
    ) -> nn.Sequential:
        """
        Build one residual stage consisting of `num_blocks` BasicBlocks.

        The first block uses the given stride (may downsample).
        All subsequent blocks use stride=1 (no spatial change).

        Args:
            block:      Block class to instantiate.
            channels:   Output channels for this stage.
            num_blocks: How many blocks to stack.
            norm:       Normalisation layer constructor.
            stride:     Stride for the first block only.

        Returns:
            nn.Sequential: The complete residual stage.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers: List[nn.Module] = []

        for s in strides:
            layers.append(block(self.in_channels, channels, s, norm))
            self.in_channels = channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, 3, H, W).

        Returns:
            torch.Tensor: Class logits, shape (B, num_classes).
        """
        out = F.relu(self.bn1(self.conv1(x)))   # (B, 64, 32, 32)
        out = self.layer1(out)                   # (B,  64, 32, 32)
        out = self.layer2(out)                   # (B, 128, 16, 16)
        out = self.layer3(out)                   # (B, 256,  8,  8)
        out = self.layer4(out)                   # (B, 512,  4,  4)
        out = self.avgpool(out)                  # (B, 512,  1,  1)
        out = out.view(out.size(0), -1)          # (B, 512)
        return self.linear(out)                  # (B, num_classes)
