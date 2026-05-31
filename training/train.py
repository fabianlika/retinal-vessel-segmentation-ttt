"""
Training script for U-Net with auxiliary rotation head.

Loss:
  total_loss = seg_loss (BCE + Dice) + λ * rotation_loss (CrossEntropy)

Usage:
  python training/train.py --data_root data/DRIVE --epochs 50 --batch_size 4
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import UNetWithRotationHead
from training.dataset import get_drive_loaders, apply_rotation
from evaluation.metrics import dice_coefficient


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat   = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets) + self.dice(logits, targets)


# ---------------------------------------------------------------------------
# One epoch helpers
# ---------------------------------------------------------------------------

def _generate_rotation_batch(imgs: torch.Tensor):
    """Create a batch of rotated images with rotation labels."""
    device = imgs.device
    rotated, labels = [], []
    for i in range(imgs.size(0)):
        k = torch.randint(0, 4, (1,)).item()
        rotated.append(apply_rotation(imgs[i], k))
        labels.append(k)
    return torch.stack(rotated).to(device), torch.tensor(labels, dtype=torch.long).to(device)


def train_one_epoch(model, loader, seg_criterion, rot_criterion, optimizer, device, lambda_rot=0.3):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        # Rotation auxiliary task
        rot_imgs, rot_labels = _generate_rotation_batch(imgs)

        # Forward
        seg_logits, rot_logits = model(imgs, return_rotation_logits=True)
        # Also forward rotated images through rotation head only
        _, rot_logits_aux = model(rot_imgs, return_rotation_logits=True)

        seg_loss = seg_criterion(seg_logits, masks)
        rot_loss = rot_criterion(rot_logits_aux, rot_labels)
        loss = seg_loss + lambda_rot * rot_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, seg_criterion, device):
    model.eval()
    total_loss, total_dice = 0.0, 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss   = seg_criterion(logits, masks)
        dice   = dice_coefficient(logits, masks)
        total_loss += loss.item()
        total_dice += dice.item()
    n = len(loader)
    return total_loss / n, total_dice / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net with rotation head")
    parser.add_argument("--data_root",   default="data/DRIVE")
    parser.add_argument("--img_size",    type=int,   default=512)
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--lambda_rot",  type=float, default=0.3,
                        help="Weight for rotation auxiliary loss")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--resume",      default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Data
    train_loader, val_loader = get_drive_loaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Model
    model = UNetWithRotationHead(n_channels=3, n_classes=1).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Resumed from {args.resume}")

    # Optimiser & schedulers
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    seg_criterion = CombinedLoss().to(device)
    rot_criterion = nn.CrossEntropyLoss().to(device)

    writer    = SummaryWriter(log_dir="runs/train")
    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, seg_criterion, rot_criterion,
            optimizer, device, lambda_rot=args.lambda_rot
        )
        val_loss, val_dice = validate(model, val_loader, seg_criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:3d}/{args.epochs}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_dice={val_dice:.4f}  ({elapsed:.1f}s)"
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("Dice/val",   val_dice,   epoch)

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_dice": val_dice},
                os.path.join(args.checkpoint_dir, "best_model.pth"),
            )
            print(f"  -> Saved best model (Dice={best_dice:.4f})")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
