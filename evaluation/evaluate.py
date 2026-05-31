"""
Evaluate a trained model on a target domain dataset (STARE or CHASE),
comparing baseline (no adaptation) vs. Test-Time Adaptation (TTT).

Usage:
  python evaluation/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --dataset stare \
    --data_root data/STARE \
    --ttt_steps 10 \
    --ttt_lr 1e-5
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import UNetWithRotationHead
from training.dataset import get_test_loader
from ttt.adapt import run_baseline_inference, run_ttt_inference
from evaluation.metrics import evaluate, print_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model with/without TTT")
    parser.add_argument("--checkpoint",  required=True,  help="Path to trained model checkpoint")
    parser.add_argument("--dataset",     default="stare", choices=["stare", "chase", "hrf"],
                        help="Target domain dataset")
    parser.add_argument("--data_root",   default="data/STARE")
    parser.add_argument("--img_size",    type=int, default=512)
    parser.add_argument("--ttt_steps",   type=int, default=10, help="TTT gradient steps per image")
    parser.add_argument("--ttt_lr",      type=float, default=1e-5, help="TTT learning rate")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Evaluating on: {args.dataset.upper()} | TTT steps: {args.ttt_steps} | LR: {args.ttt_lr}")

    # Load model
    model = UNetWithRotationHead(n_channels=3, n_classes=1)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')}, "
          f"val_dice={ckpt.get('val_dice', 'N/A')})")

    # Data loader
    loader = get_test_loader(
        image_dir=os.path.join(args.data_root, "images"),
        mask_dir=os.path.join(args.data_root,  "masks"),
        img_size=args.img_size,
        batch_size=1,
        ttt_mode=False,
    )

    # --- Baseline ---
    print("\nRunning baseline inference (no TTT)...")
    base_preds, base_masks = [], []
    for preds, masks in run_baseline_inference(model, loader, device):
        base_preds.append(preds.cpu())
        base_masks.append(masks.cpu())

    baseline_metrics = evaluate(base_preds, base_masks)
    print_metrics(baseline_metrics, title=f"Baseline — {args.dataset.upper()}")

    # --- TTT ---
    print(f"Running TTT inference ({args.ttt_steps} steps per image)...")
    ttt_preds, ttt_masks = [], []
    for preds, masks in run_ttt_inference(model, loader, device,
                                          n_steps=args.ttt_steps, lr=args.ttt_lr):
        ttt_preds.append(preds.cpu())
        ttt_masks.append(masks.cpu())

    ttt_metrics = evaluate(ttt_preds, ttt_masks)
    print_metrics(ttt_metrics, title=f"After TTT — {args.dataset.upper()}")

    # --- Delta ---
    print("Improvement (TTT - Baseline):")
    for k in baseline_metrics:
        delta = ttt_metrics[k] - baseline_metrics[k]
        sign  = "+" if delta >= 0 else ""
        print(f"  {k:<15}: {sign}{delta:.4f}")


if __name__ == "__main__":
    main()
