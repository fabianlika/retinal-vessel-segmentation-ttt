"""
Ablation study: compare four adaptation strategies across all test datasets.

Conditions:
  1. Baseline  — no adaptation
  2. TTA       — test-time augmentation (8 augmented views, no gradient)
  3. TTT-BN    — adapt BatchNorm affine params only  (proposed method)
  4. TTT-Enc   — adapt all encoder layers
  5. TTT-Full  — adapt all model parameters

Usage (run from repo root):
  python ablation_study.py --checkpoint checkpoints/best_model.pth
"""

import argparse
import os
import sys
import copy

import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.unet import UNetWithRotationHead
from training.dataset import get_test_loader, apply_rotation
from evaluation.metrics import evaluate
from ttt.adapt import test_time_adapt, run_tta_inference

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASETS = {
    "STARE": ("data/STARE/images", "data/STARE/masks"),
    "CHASE": ("data/CHASE/images", "data/CHASE/masks"),
    "HRF":   ("data/HRF/images",   "data/HRF/masks"),
}

CONDITIONS = [
    ("Baseline",  None),
    ("TTA",       None),          # handled separately
    ("TTT-BN",    "bn"),
    ("TTT-Enc",   "encoder"),
    ("TTT-Full",  "full"),
]

TTT_LR    = 1e-6
TTT_STEPS = 5


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_baseline(model, loader, device):
    model.eval()
    probs, masks = [], []
    for imgs, msks in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs.append(torch.sigmoid(logits).cpu())
        masks.append(msks)
    return probs, masks


def run_ttt(model, loader, device, mode):
    probs, masks = [], []
    for imgs, msks in loader:
        imgs = imgs.to(device)
        batch_probs = []
        for i in range(imgs.size(0)):
            single = imgs[i].unsqueeze(0)
            logits = test_time_adapt(model, single, n_steps=TTT_STEPS,
                                     lr=TTT_LR, adapt_mode=mode)
            batch_probs.append(torch.sigmoid(logits).cpu())
        probs.append(torch.cat(batch_probs, dim=0))
        masks.append(msks)
    return probs, masks


def run_tta(model, loader, device):
    preds, masks = [], []
    for p, m in run_tta_inference(model, loader, device):
        preds.append(p.cpu())
        masks.append(m)
    return preds, masks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--img_size",   type=int, default=512)
    parser.add_argument("--output",     default="ablation_results.txt")
    return parser.parse_args()


def load_model(checkpoint, device):
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model = UNetWithRotationHead(n_channels=3, n_classes=1)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)

    results = {}   # dataset -> condition -> metrics

    for ds_name, (img_dir, mask_dir) in DATASETS.items():
        if not Path(img_dir).exists():
            print(f"Skipping {ds_name} — data not found.")
            continue

        print(f"\n{'='*50}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*50}")

        loader = get_test_loader(img_dir, mask_dir, img_size=args.img_size, batch_size=1)
        results[ds_name] = {}

        for cond_name, mode in CONDITIONS:
            print(f"  Running {cond_name}...", end=" ", flush=True)

            if cond_name == "Baseline":
                preds, masks = run_baseline(model, loader, device)
            elif cond_name == "TTA":
                preds, masks = run_tta(model, loader, device)
            else:
                preds, masks = run_ttt(model, loader, device, mode)

            metrics = evaluate(preds, masks)
            results[ds_name][cond_name] = metrics
            print(f"Dice={metrics['dice']:.4f}")

    # ---------------------------------------------------------------------------
    # Print and save results table
    # ---------------------------------------------------------------------------
    cols = list(CONDITIONS[0][0].__class__.__mro__)  # just a trick — use list
    cond_names = [c for c, _ in CONDITIONS]

    lines = []
    lines.append("\n" + "="*90)
    lines.append("ABLATION STUDY RESULTS")
    lines.append("="*90)

    for ds_name, ds_results in results.items():
        lines.append(f"\n{ds_name}")
        header = f"  {'Condition':<12}  {'Dice':>7}  {'IoU':>7}  {'Sens':>7}  {'Spec':>7}  {'AUC':>7}"
        lines.append(header)
        lines.append("  " + "-"*60)

        baseline_dice = ds_results.get("Baseline", {}).get("dice", 0)
        for cond_name in cond_names:
            if cond_name not in ds_results:
                continue
            m = ds_results[cond_name]
            delta = m["dice"] - baseline_dice
            sign  = f"({'+'if delta>=0 else ''}{delta:.4f})"
            line = (f"  {cond_name:<12}  {m['dice']:>7.4f}  {m['iou']:>7.4f}  "
                    f"{m['sensitivity']:>7.4f}  {m['specificity']:>7.4f}  "
                    f"{m.get('auc_roc', float('nan')):>7.4f}  {sign}")
            lines.append(line)

    output = "\n".join(lines)
    print(output)

    with open(args.output, "w") as f:
        f.write(output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
