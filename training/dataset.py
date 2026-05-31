"""
Dataset loaders for DRIVE, STARE, and CHASE_DB1 retinal vessel datasets.

Expected folder layout (after downloading):
  data/
    DRIVE/
      images/   *.tif  (or *.png)
      masks/    *.gif  (or *.png)
    STARE/
      images/   *.ppm
      masks/    *.ppm
    CHASE/
      images/   *.jpg
      masks/    *.png

All images are resized to `img_size` and normalised to [0, 1].
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUPPORTED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".ppm", ".gif", ".bmp"}


def _collect_paths(folder: str) -> List[Path]:
    folder = Path(folder)
    paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
    return paths


def _to_tensor_img(pil_img: Image.Image, img_size: int) -> torch.Tensor:
    """Resize RGB image → float tensor [3, H, W] in [0, 1]."""
    pil_img = pil_img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    return transforms.ToTensor()(pil_img)  # [3, H, W]


def _to_tensor_mask(pil_img: Image.Image, img_size: int) -> torch.Tensor:
    """Resize binary mask → float tensor [1, H, W] with values 0 or 1."""
    pil_img = pil_img.convert("L").resize((img_size, img_size), Image.NEAREST)
    mask = torch.from_numpy(np.array(pil_img)).float().unsqueeze(0)
    mask = (mask > 127).float()
    return mask


# ---------------------------------------------------------------------------
# Rotation augmentation for TTT auxiliary task
# ---------------------------------------------------------------------------

ROTATIONS = [0, 90, 180, 270]


def apply_rotation(img: torch.Tensor, angle_idx: int) -> torch.Tensor:
    """Rotate image tensor by 0/90/180/270 degrees (label = angle_idx)."""
    k = angle_idx  # torch.rot90 uses k quarters
    return torch.rot90(img, k=k, dims=[1, 2])


# ---------------------------------------------------------------------------
# Base dataset
# ---------------------------------------------------------------------------

class RetinalDataset(Dataset):
    """
    Generic retinal vessel dataset.

    Args:
        image_dir:   Path to folder containing retinal images.
        mask_dir:    Path to folder containing vessel masks.
        img_size:    Spatial size to resize images/masks to.
        augment:     Apply random horizontal/vertical flips and rotation.
        ttt_mode:    If True, also return 4 rotated copies + rotation labels
                     (used during test-time adaptation).
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        img_size: int = 512,
        augment: bool = False,
        ttt_mode: bool = False,
    ):
        self.img_paths  = _collect_paths(image_dir)
        self.mask_paths = _collect_paths(mask_dir)
        assert len(self.img_paths) == len(self.mask_paths), (
            f"Mismatch: {len(self.img_paths)} images vs {len(self.mask_paths)} masks in "
            f"{image_dir} / {mask_dir}"
        )
        self.img_size = img_size
        self.augment  = augment
        self.ttt_mode = ttt_mode

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img  = Image.open(self.img_paths[idx])
        mask = Image.open(self.mask_paths[idx])

        img_t  = _to_tensor_img(img, self.img_size)
        mask_t = _to_tensor_mask(mask, self.img_size)

        if self.augment:
            img_t, mask_t = self._random_augment(img_t, mask_t)

        if self.ttt_mode:
            # Return 4 rotations with labels for the auxiliary task
            rotated_imgs = [apply_rotation(img_t, k) for k in range(4)]
            rot_labels   = torch.arange(4)
            return torch.stack(rotated_imgs), rot_labels  # [4, C, H, W], [4]

        return img_t, mask_t

    def _random_augment(
        self, img: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > 0.5:
            img  = torch.flip(img,  [2])
            mask = torch.flip(mask, [2])
        if random.random() > 0.5:
            img  = torch.flip(img,  [1])
            mask = torch.flip(mask, [1])
        k = random.randint(0, 3)
        img  = torch.rot90(img,  k=k, dims=[1, 2])
        mask = torch.rot90(mask, k=k, dims=[1, 2])
        return img, mask


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def get_drive_loaders(
    data_root: str = "data/DRIVE",
    img_size: int = 512,
    batch_size: int = 4,
    val_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for the DRIVE dataset."""
    ds = RetinalDataset(
        image_dir=os.path.join(data_root, "images"),
        mask_dir=os.path.join(data_root, "masks"),
        img_size=img_size,
        augment=True,
    )
    n_val   = max(1, int(len(ds) * val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    # Disable augmentation for val
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader


def get_test_loader(
    image_dir: str,
    mask_dir: str,
    img_size: int = 512,
    batch_size: int = 1,
    ttt_mode: bool = False,
) -> DataLoader:
    """Return a test DataLoader for STARE or CHASE."""
    ds = RetinalDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        img_size=img_size,
        augment=False,
        ttt_mode=ttt_mode,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
