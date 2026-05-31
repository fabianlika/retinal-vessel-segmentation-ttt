"""
U-Net model with an auxiliary rotation prediction head for Test-Time Adaptation.

Architecture:
  - Shared encoder (ResNet-like double-conv blocks)
  - U-Net decoder for vessel segmentation
  - Rotation head attached to the bottleneck for self-supervised TTT
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downsampling: MaxPool then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsampling then DoubleConv (with skip connection)."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 if spatial dims don't match x2
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                                      diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class UNetWithRotationHead(nn.Module):
    """
    U-Net for retinal vessel segmentation with a rotation-prediction head
    used during test-time adaptation (TTT).

    Args:
        n_channels: Number of input image channels (3 for RGB).
        n_classes:  Number of segmentation output channels (1 for binary mask).
        bilinear:   Use bilinear upsampling instead of transposed convolutions.
    """

    def __init__(self, n_channels: int = 3, n_classes: int = 1, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- Encoder ---
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)   # bottleneck

        # --- Decoder ---
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512,  128, bilinear)
        self.up3 = Up(256,  64,  bilinear)
        self.up4 = Up(128,  64,  bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

        # --- Auxiliary rotation head (4 classes: 0°, 90°, 180°, 270°) ---
        self.rotation_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 4),
        )

    def forward(self, x: torch.Tensor, return_rotation_logits: bool = False):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)   # bottleneck features

        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        seg_logits = self.outc(x)

        if return_rotation_logits:
            rot_logits = self.rotation_head(x5)
            return seg_logits, rot_logits

        return seg_logits

    def encoder_parameters(self):
        """Parameters of the shared encoder (used for TTT updates)."""
        params = []
        for module in [self.inc, self.down1, self.down2, self.down3, self.down4]:
            params += list(module.parameters())
        return params

    def rotation_head_parameters(self):
        return list(self.rotation_head.parameters())

    def segmentation_parameters(self):
        params = []
        for module in [self.up1, self.up2, self.up3, self.up4, self.outc]:
            params += list(module.parameters())
        return params
