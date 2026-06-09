"""
Test-Time Adaptation (TTT) for retinal vessel segmentation.

Strategy: At inference time, for each test image:
  1. Rotate the image 4 ways (0°, 90°, 180°, 270°).
  2. Feed through model → get rotation logits.
  3. Compute rotation prediction loss (cross-entropy).
  4. Backprop through the ENCODER only (keep decoder + seg head frozen).
  5. Run segmentation on the adapted model.
  6. Restore original weights for the next image.

This allows per-image adaptation without any labels.
"""

import copy
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import UNetWithRotationHead
from training.dataset import apply_rotation


# ---------------------------------------------------------------------------
# Core TTT function
# ---------------------------------------------------------------------------

def test_time_adapt(
    model: UNetWithRotationHead,
    image: torch.Tensor,          # [1, C, H, W] on device
    n_steps: int = 10,
    lr: float = 1e-5,
    adapt_mode: str = "bn",       # "bn" | "encoder" | "full"
) -> torch.Tensor:
    """
    Adapt the model using rotation prediction on a single image.

    Args:
        model:       The trained U-Net.
        image:       A single image tensor [1, C, H, W].
        n_steps:     Number of gradient steps for adaptation.
        lr:          Learning rate for TTT optimiser.
        adapt_mode:  Which parameters to adapt:
                       "bn"      — BatchNorm affine params only (default)
                       "encoder" — all encoder layers (inc, down1-4)
                       "full"    — all model parameters

    Returns:
        seg_logits: [1, 1, H, W] segmentation logits after adaptation.
    """
    original_state = copy.deepcopy(model.state_dict())
    rot_criterion  = nn.CrossEntropyLoss()

    if adapt_mode == "bn":
        params = [p for m in model.modules()
                  if isinstance(m, nn.BatchNorm2d)
                  for p in [m.weight, m.bias] if p is not None]
    elif adapt_mode == "encoder":
        encoder_modules = [model.inc, model.down1, model.down2, model.down3, model.down4]
        params = [p for m in encoder_modules for p in m.parameters()]
    else:  # "full"
        params = list(model.parameters())

    optimizer = torch.optim.Adam(params, lr=lr)

    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.momentum = None

    img = image.squeeze(0)
    rotations = [apply_rotation(img, k).unsqueeze(0) for k in range(4)]

    for _ in range(n_steps):
        optimizer.zero_grad()
        # Process each rotation separately and accumulate gradients.
        # Mathematically identical to a batch of 4 (mean cross-entropy) but
        # uses ~4x less peak memory, avoiding OOM on low-RAM machines.
        for k, rot_img in enumerate(rotations):
            label = torch.tensor([k], dtype=torch.long, device=image.device)
            _, rot_logits = model(rot_img, return_rotation_logits=True)
            loss = rot_criterion(rot_logits, label) / len(rotations)
            loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        seg_logits = model(image)

    model.load_state_dict(original_state)
    return seg_logits


# ---------------------------------------------------------------------------
# Batch inference with TTT
# ---------------------------------------------------------------------------

def run_ttt_inference(
    model: UNetWithRotationHead,
    loader,
    device: torch.device,
    n_steps: int = 10,
    lr: float = 1e-5,
):
    """
    Run TTT on every image in loader.

    Yields:
        (seg_prob, mask) tuples where seg_prob is a probability map [1, 1, H, W]
        in [0, 1] (apply a 0.5 threshold for a binary mask).
    """
    model.to(device)

    for batch_idx, (imgs, masks) in enumerate(loader):
        imgs  = imgs.to(device)
        masks = masks.to(device)

        batch_probs = []
        for i in range(imgs.size(0)):
            single_img = imgs[i].unsqueeze(0)           # [1, C, H, W]
            logits = test_time_adapt(
                model, single_img, n_steps=n_steps, lr=lr
            )
            batch_probs.append(torch.sigmoid(logits))

        probs = torch.cat(batch_probs, dim=0)
        yield probs, masks

        if (batch_idx + 1) % 5 == 0:
            print(f"  TTT progress: {batch_idx + 1}/{len(loader)} batches done", flush=True)


# ---------------------------------------------------------------------------
# Baseline inference (no adaptation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_baseline_inference(model, loader, device):
    """Standard inference without TTT. Yields (prob, mask) pairs where prob is
    a probability map in [0, 1]."""
    model.eval()
    model.to(device)

    for imgs, masks in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)
        logits = model(imgs)
        probs  = torch.sigmoid(logits)
        yield probs, masks


# ---------------------------------------------------------------------------
# Test-Time Augmentation (TTA) — simple baseline for comparison
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_tta_inference(model, loader, device):
    """
    Test-Time Augmentation: average predictions over 8 augmentations
    (4 rotations × 2 horizontal flips). No gradient updates — purely
    ensembling. Used as a simple baseline to compare against TTT.

    Yields:
        (seg_prob, mask) where seg_prob is the averaged probability map
        [B, 1, H, W] in [0, 1] (apply a 0.5 threshold for a binary mask).
    """
    model.eval()
    model.to(device)

    # Define augment/de-augment pairs: (aug_fn, deaug_fn)
    augmentations = []
    for k in range(4):
        augmentations.append((
            lambda img, k=k: torch.rot90(img, k,  dims=[2, 3]),   # aug
            lambda msk, k=k: torch.rot90(msk, -k, dims=[2, 3]),   # de-aug
        ))
        augmentations.append((
            lambda img, k=k: torch.flip(torch.rot90(img, k, dims=[2, 3]), dims=[3]),
            lambda msk, k=k: torch.rot90(torch.flip(msk, dims=[3]), -k, dims=[2, 3]),
        ))

    for imgs, masks in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)

        prob_sum = torch.zeros_like(imgs[:, :1])  # [B, 1, H, W]
        for aug_fn, deaug_fn in augmentations:
            aug_imgs    = aug_fn(imgs)
            aug_logits  = model(aug_imgs)
            aug_probs   = torch.sigmoid(aug_logits)
            prob_sum   += deaug_fn(aug_probs)

        avg_probs = prob_sum / len(augmentations)
        yield avg_probs, masks


# ---------------------------------------------------------------------------
# Intensity normalization (CLAHE) — reduce domain gap before adaptation
# ---------------------------------------------------------------------------

def apply_clahe(pil_image, clip_limit: float = 2.0, tile_grid: int = 8):
    """
    Contrast-Limited Adaptive Histogram Equalization for fundus images.

    Normalizes brightness/contrast to reduce the intensity domain gap
    (e.g. CHASE images are ~40% darker than DRIVE). Applied on the L
    channel of LAB space when OpenCV is available; otherwise falls back
    to per-image luminance equalization via PIL so the function works
    on any platform.

    Args:
        pil_image:  Input PIL image (any mode; converted to RGB).
        clip_limit: CLAHE clip limit (higher = more contrast).
        tile_grid:  CLAHE tile grid size (NxN).

    Returns:
        A PIL RGB Image with normalized intensity.
    """
    from PIL import Image, ImageOps
    import numpy as np

    rgb = pil_image.convert("RGB")
    try:
        import cv2
        arr = np.array(rgb)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
        l = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
        return Image.fromarray(out)
    except Exception:
        # OpenCV unavailable — fall back to global luminance equalization
        return ImageOps.equalize(rgb)
