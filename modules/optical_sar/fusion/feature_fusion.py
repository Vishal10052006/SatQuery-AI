"""PyTorch feature-level two-stream fusion neural network architecture."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Standard 2D Convolutional block with Batch Normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class OpticalEncoder(nn.Module):
    """Lightweight CNN encoder for multi-spectral optical imagery."""

    def __init__(self, in_channels: int = 4, feature_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        self.backbone = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=3, pool=True),
            ConvBlock(32, 64, kernel_size=3, pool=True),
            ConvBlock(64, feature_dim, kernel_size=3, pool=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(feature_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode optical raster tensor (B, C, H, W) to feature vector (B, feature_dim)."""
        b = x.size(0)
        feats = self.backbone(x).view(b, -1)
        return F.relu(self.fc(feats))


class SAREncoder(nn.Module):
    """Lightweight CNN encoder for polarimetric SAR imagery."""

    def __init__(self, in_channels: int = 2, feature_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        self.backbone = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=3, pool=True),
            ConvBlock(32, 64, kernel_size=3, pool=True),
            ConvBlock(64, feature_dim, kernel_size=3, pool=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(feature_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode SAR raster tensor (B, C, H, W) to feature vector (B, feature_dim)."""
        b = x.size(0)
        feats = self.backbone(x).view(b, -1)
        return F.relu(self.fc(feats))


class MultimodalFusionModule(nn.Module):
    """Feature fusion layer merging optical and SAR embedding representations."""

    def __init__(self, optical_dim: int = 128, sar_dim: int = 128, fusion_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        combined_dim = optical_dim + sar_dim
        self.fusion_fc = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, opt_feat: torch.Tensor, sar_feat: torch.Tensor) -> torch.Tensor:
        """Concatenate optical and SAR feature vectors and project through fusion MLP."""
        concat = torch.cat([opt_feat, sar_feat], dim=1)
        return self.fusion_fc(concat)


class ClassificationHead(nn.Module):
    """Prediction classification head mapping fused features to class logits."""

    def __init__(self, feature_dim: int = 128, num_classes: int = 5):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class FeatureFusionNetwork(nn.Module):
    """Complete two-stream multimodal Optical + SAR deep neural network."""

    def __init__(
        self,
        optical_channels: int = 4,
        sar_channels: int = 2,
        feature_dim: int = 128,
        num_classes: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.optical_encoder = OpticalEncoder(in_channels=optical_channels, feature_dim=feature_dim)
        self.sar_encoder = SAREncoder(in_channels=sar_channels, feature_dim=feature_dim)
        self.fusion_module = MultimodalFusionModule(
            optical_dim=feature_dim, sar_dim=feature_dim, fusion_dim=feature_dim, dropout=dropout
        )
        self.classifier = ClassificationHead(feature_dim=feature_dim, num_classes=num_classes)

    def forward(self, optical_x: torch.Tensor, sar_x: torch.Tensor) -> torch.Tensor:
        """Forward pass taking separate optical and SAR inputs."""
        opt_feat = self.optical_encoder(optical_x)
        sar_feat = self.sar_encoder(sar_x)
        fused = self.fusion_module(opt_feat, sar_feat)
        return self.classifier(fused)
