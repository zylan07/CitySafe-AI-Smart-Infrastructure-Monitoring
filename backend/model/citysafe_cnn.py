"""
CitySafe / ZylanZest V2 — Production CNN Architecture
======================================================
Inference-safe PyTorch neural network specification for CitySafeCNN.
Contains only model architecture components required for runtime prediction.
Decoupled from training, evaluation, plotting, and dataset dependencies.
"""

import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.dw = ConvBNReLU(in_planes, in_planes, kernel_size=3, stride=stride, groups=in_planes)
        self.pw = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.pw(self.dw(x))


class CitySafeCNN(nn.Module):
    """
    Lightweight, MobileNet-inspired CNN architecture for edge infrastructure inspection.
    Takes 224x224 RGB image, downsamples through inverted depthwise stages,
    and classifies concrete integrity into Safe (0) vs Damaged (1).
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3, 16, stride=2),            # 224 -> 112
            DepthwiseSeparableConv(16, 24, stride=2), # 112 -> 56
            DepthwiseSeparableConv(24, 32, stride=2), # 56 -> 28
            DepthwiseSeparableConv(32, 64, stride=2), # 28 -> 14
            DepthwiseSeparableConv(64, 96, stride=2), # 14 -> 7
            ConvBNReLU(96, 128, kernel_size=1)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)
        logits = self.classifier(pooled)
        return logits

