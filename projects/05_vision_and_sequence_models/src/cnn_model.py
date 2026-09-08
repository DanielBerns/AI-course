"""
Convolutional Neural Network (CNN) with Residual Connections in PyTorch.

Implements 2D convolutions, spatial pooling, batch normalization, and residual skip connections.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual Block with skip connection (y = x + F(x))."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class PyTorchResNetCNN(nn.Module):
    """CNN Classifier with Residual Blocks and Adaptive Pooling."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10, base_filters: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(),
        )

        self.layer1 = ResidualBlock(base_filters)
        self.downsample = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(),
        )
        self.layer2 = ResidualBlock(base_filters * 2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_filters * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.layer1(out)
        out = self.downsample(out)
        out = self.layer2(out)
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)
