"""Small Siamese U-Net for bi-temporal binary change segmentation."""
from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two-convolution block using GroupNorm for stable batch-size-1 training."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        groups = 8 if cout >= 8 else 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(groups, cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(groups, cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SiameseUNet(nn.Module):
    """Shared-encoder U-Net with absolute bi-temporal feature differences."""

    def __init__(self, in_channels: int = 3, base: int = 32):
        super().__init__()
        self.e1 = ConvBlock(in_channels, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.b = ConvBlock(base * 8, base * 16)
        self.d4 = ConvBlock(base * 16 + base * 8, base * 8)
        self.d3 = ConvBlock(base * 8 + base * 4, base * 4)
        self.d2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.d1 = ConvBlock(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def _enc(self, x):
        a = self.e1(x)
        b = self.e2(self.pool(a))
        c = self.e3(self.pool(b))
        d = self.e4(self.pool(c))
        z = self.b(self.pool(d))
        return a, b, c, d, z

    @staticmethod
    def _up(x, skip):
        return torch.nn.functional.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(self, before, after):
        a1, b1, c1, d1, z1 = self._enc(before)
        a2, b2, c2, d2, z2 = self._enc(after)
        z = torch.abs(z1 - z2)
        x = self.d4(torch.cat([self._up(z, d1), torch.abs(d1 - d2)], 1))
        x = self.d3(torch.cat([self._up(x, c1), torch.abs(c1 - c2)], 1))
        x = self.d2(torch.cat([self._up(x, b1), torch.abs(b1 - b2)], 1))
        x = self.d1(torch.cat([self._up(x, a1), torch.abs(a1 - a2)], 1))
        return self.out(x)
