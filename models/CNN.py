"""
models/CNN.py
=============
Convolutional Neural Networks for image classification.

  MNIST_CNN  — two conv layers for single-channel 28x28 input (MNIST).
  SimpleCNN  — two conv layers for three-channel 32x32 input (CIFAR-10),
               with Kaiming (He) weight initialisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_CNN(nn.Module):
    """
    Lightweight CNN for MNIST (1-channel, 28x28 images).

    Spatial flow:
        Input  : (B,  1, 28, 28)
        conv1  : (B, 20, 24, 24)   formula: (28 - 5) / 1 + 1 = 24
        pool1  : (B, 20, 12, 12)   kernel=2, stride=2
        conv2  : (B, 50,  8,  8)   formula: (12 - 5) / 1 + 1 = 8
        pool2  : (B, 50,  4,  4)   kernel=2, stride=2
        flatten: (B, 800)
        fc1    : (B, 500)
        fc2    : (B, num_classes)

    Args:
        num_classes: Number of output classes (default 10 for MNIST).
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # (in_channels, out_channels, kernel_size, stride)
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.fc1   = nn.Linear(4 * 4 * 50, 500)
        self.fc2   = nn.Linear(500, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, 1, 28, 28).

        Returns:
            torch.Tensor: Class logits, shape (B, num_classes).
        """
        x = F.relu(self.conv1(x))      # -> (B, 20, 24, 24)
        x = F.max_pool2d(x, 2, 2)      # -> (B, 20, 12, 12)
        x = F.relu(self.conv2(x))      # -> (B, 50,  8,  8)
        x = F.max_pool2d(x, 2, 2)      # -> (B, 50,  4,  4)
        x = x.view(x.size(0), -1)      # -> (B, 800)
        x = F.relu(self.fc1(x))        # -> (B, 500)
        return self.fc2(x)             # -> (B, num_classes)


class SimpleCNN(nn.Module):
    """
    Two-layer CNN for CIFAR-10 (3-channel, 32x32 images).

    Uses Kaiming (He) initialisation for all Conv2d and Linear weights.

    Kaiming initialisation (for ReLU):
        w ~ N(0, sqrt(2 / fan_in))

    The factor 2 compensates for ReLU zeroing roughly half of all activations,
    which would otherwise halve the signal variance at each layer.  Without
    this correction, activations in deep networks either vanish or explode.
    fan_in mode bases the scale on the number of input connections, which
    is the standard choice when the forward signal stability matters more
    than the backward signal.

    Spatial flow:
        Input  : (B,  3, 32, 32)
        conv1  : (B, 32, 32, 32)   padding=1 preserves spatial size
        pool1  : (B, 32, 16, 16)
        conv2  : (B, 64, 16, 16)   padding=1 preserves spatial size
        pool2  : (B, 64,  8,  8)
        flatten: (B, 4096)
        fc1    : (B, 128)
        fc2    : (B, num_classes)

    Args:
        num_classes: Number of output classes (default 10 for CIFAR-10).
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,  out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(64 * 8 * 8, 128)
        self.fc2   = nn.Linear(128, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Apply Kaiming normal initialisation to all Conv2d and Linear layers.

        Biases are zeroed so that all neurons start with the same offset,
        preventing any artificial symmetry breaking in the bias direction.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, 3, 32, 32).

        Returns:
            torch.Tensor: Class logits, shape (B, num_classes).
        """
        x = F.relu(self.conv1(x))   # -> (B, 32, 32, 32)
        x = F.max_pool2d(x, 2)      # -> (B, 32, 16, 16)
        x = F.relu(self.conv2(x))   # -> (B, 64, 16, 16)
        x = F.max_pool2d(x, 2)      # -> (B, 64,  8,  8)
        x = x.view(x.size(0), -1)   # -> (B, 4096)
        x = F.relu(self.fc1(x))     # -> (B, 128)
        return self.fc2(x)          # -> (B, num_classes)
