"""
Evaluation metrics for binary vessel segmentation.

Metrics computed:
  - Dice Coefficient (F1)
  - IoU (Jaccard Index)
  - Accuracy
  - Sensitivity (Recall)
  - Specificity
  - AUC-ROC
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score


EPS = 1e-7


# ---------------------------------------------------------------------------
# Tensor-based metrics (used during training)
# ---------------------------------------------------------------------------

def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Dice coefficient from raw logits and binary target masks."""
    preds = (torch.sigmoid(logits) > threshold).float()
    preds_f   = preds.view(-1)
    targets_f = targets.view(-1)
    intersection = (preds_f * targets_f).sum()
    return (2.0 * intersection + EPS) / (preds_f.sum() + targets_f.sum() + EPS)


def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Intersection-over-Union (Jaccard) from logits."""
    preds = (torch.sigmoid(logits) > threshold).float()
    preds_f   = preds.view(-1)
    targets_f = targets.view(-1)
    intersection = (preds_f * targets_f).sum()
    union        = preds_f.sum() + targets_f.sum() - intersection
    return (intersection + EPS) / (union + EPS)


# ---------------------------------------------------------------------------
# Full evaluation over a dataset
# ---------------------------------------------------------------------------

def evaluate(preds_list, masks_list):
    """
    Compute all metrics given lists of prediction and mask tensors.

    Args:
        preds_list: list of [B, 1, H, W] binary prediction tensors (0/1).
        masks_list: list of [B, 1, H, W] binary ground truth tensors.

    Returns:
        dict with keys: dice, iou, accuracy, sensitivity, specificity, auc_roc
    """
    all_preds  = torch.cat(preds_list, dim=0).cpu().numpy().ravel()
    all_masks  = torch.cat(masks_list, dim=0).cpu().numpy().ravel()

    tp = np.sum((all_preds == 1) & (all_masks == 1))
    tn = np.sum((all_preds == 0) & (all_masks == 0))
    fp = np.sum((all_preds == 1) & (all_masks == 0))
    fn = np.sum((all_preds == 0) & (all_masks == 1))

    dice        = (2 * tp + EPS) / (2 * tp + fp + fn + EPS)
    iou         = (tp + EPS) / (tp + fp + fn + EPS)
    accuracy    = (tp + tn + EPS) / (tp + tn + fp + fn + EPS)
    sensitivity = (tp + EPS) / (tp + fn + EPS)
    specificity = (tn + EPS) / (tn + fp + EPS)

    try:
        auc = roc_auc_score(all_masks.astype(int), all_preds.astype(float))
    except ValueError:
        auc = float("nan")

    return {
        "dice":        round(float(dice), 4),
        "iou":         round(float(iou), 4),
        "accuracy":    round(float(accuracy), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "auc_roc":     round(float(auc), 4),
    }


def print_metrics(metrics: dict, title: str = "Results"):
    print(f"\n{'='*40}")
    print(f"  {title}")
    print(f"{'='*40}")
    for k, v in metrics.items():
        print(f"  {k:<15}: {v:.4f}")
    print(f"{'='*40}\n")
