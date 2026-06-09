"""
Regenerate every report figure from the corrected results and the trained
checkpoint. Bar charts are drawn from the final metric numbers; the domain-gap,
prediction grids and ROC curves are produced by running the model locally.

Run (from repo root):  py report/make_figures.py
Outputs -> report/figures/*.png
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIG = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)

import torch
from models.unet import UNetWithRotationHead
from training.dataset import RetinalDataset, get_test_loader
from ttt.adapt import test_time_adapt, run_baseline_inference
from evaluation.metrics import evaluate  # noqa: F401  (kept for parity)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512
DATASETS = {
    "STARE": ("data/STARE/images", "data/STARE/masks"),
    "CHASE": ("data/CHASE/images", "data/CHASE/masks"),
    "HRF":   ("data/HRF/images",   "data/HRF/masks"),
}

# ---------------------------------------------------------------------------
# Final metric numbers (mirror results/*.txt)
# ---------------------------------------------------------------------------
EVAL = {  # baseline vs TTT
    "STARE": dict(base=[0.5349, 0.3651, 0.5164, 0.9659, 0.8850],
                  ttt =[0.5544, 0.3835, 0.5440, 0.9656, 0.8985]),
    "CHASE": dict(base=[0.3923, 0.2440, 0.2921, 0.9853, 0.8031],
                  ttt =[0.4195, 0.2654, 0.3257, 0.9831, 0.8151]),
    "HRF":   dict(base=[0.5501, 0.3794, 0.6357, 0.9436, 0.8977],
                  ttt =[0.5526, 0.3818, 0.6359, 0.9444, 0.9004]),
}
METRIC_NAMES = ["Dice", "IoU", "Sensitivity", "Specificity", "AUC-ROC"]

ABLATION = {  # Baseline, TTA, TTT-BN, TTT-Enc, TTT-Full  (Dice)
    "STARE": [0.5349, 0.5396, 0.5544, 0.5527, 0.5527],
    "CHASE": [0.3923, 0.4114, 0.4195, 0.4190, 0.4190],
    "HRF":   [0.5501, 0.5540, 0.5526, 0.5526, 0.5526],
}
ABLATION_COND = ["Baseline", "TTA", "TTT-BN", "TTT-Enc", "TTT-Full"]

CLAHE = {  # Baseline(orig), TTA(orig), Baseline(CLAHE), TTA(CLAHE)  (Dice)
    "STARE": [0.5349, 0.5396, 0.4582, 0.4561],
    "CHASE": [0.3923, 0.4114, 0.3908, 0.3963],
    "HRF":   [0.5501, 0.5540, 0.3779, 0.3668],
}
CLAHE_COND = ["Baseline (orig)", "TTA (orig)", "Baseline (CLAHE)", "TTA (CLAHE)"]

DARK_BG, DARK_AX, DARK_GRID = "#0d1117", "#161b22", "#30363d"
PALETTE = ["#8b95a7", "#4aa3df", "#4cc38a", "#f5a623", "#f56b6b"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dark_ax(ax, title, ymax=None):
    ax.set_facecolor(DARK_AX)
    ax.set_title(title, color="white", fontsize=13, pad=8)
    ax.tick_params(colors="white", labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(DARK_GRID)
    ax.yaxis.label.set_color("white")
    if ymax:
        ax.set_ylim(0, ymax)
    ax.grid(axis="y", color=DARK_GRID, alpha=0.4, linewidth=0.7)
    ax.set_axisbelow(True)


def _bars_with_labels(ax, xs, vals, colors):
    bars = ax.bar(xs, vals, color=colors, width=0.7, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.4f}",
                ha="center", va="bottom", color="white", fontsize=8.5)
    return bars


# ---------------------------------------------------------------------------
# 1. Baseline vs TTT — Dice per dataset (light)
# ---------------------------------------------------------------------------

def fig_dice_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.suptitle("Baseline vs TTT — Dice Score per Dataset", fontsize=15, fontweight="bold")
    for ax, ds in zip(axes, DATASETS):
        b, t = EVAL[ds]["base"][0], EVAL[ds]["ttt"][0]
        bars = ax.bar(["Baseline", "TTT"], [b, t], color=["#4c72b0", "#dd8452"], width=0.6)
        for bar, v in zip(bars, [b, t]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=10)
        ax.set_title(ds, fontsize=12)
        ax.set_ylabel("Dice Score")
        lo = min(b, t); hi = max(b, t)
        ax.set_ylim(max(0, lo - 0.06), hi + 0.05)
        ax.text(0.5, -0.18, f"TTT delta: +{t - b:.4f}", transform=ax.transAxes,
                ha="center", color="green", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(FIG, "dice_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  dice_comparison.png")


# ---------------------------------------------------------------------------
# 2. All metrics — Baseline vs TTT (light)
# ---------------------------------------------------------------------------

def fig_all_metrics():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle("All Metrics: Baseline vs TTT", fontsize=15, fontweight="bold")
    x = np.arange(len(METRIC_NAMES)); w = 0.38
    for ax, ds in zip(axes, DATASETS):
        ax.bar(x - w / 2, EVAL[ds]["base"], w, label="Baseline", color="#4c72b0")
        ax.bar(x + w / 2, EVAL[ds]["ttt"],  w, label="TTT",      color="#dd8452")
        ax.set_title(ds, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(METRIC_NAMES, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(FIG, "all_metrics_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  all_metrics_comparison.png")


# ---------------------------------------------------------------------------
# 3. Ablation (dark)
# ---------------------------------------------------------------------------

def fig_ablation():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor=DARK_BG)
    fig.suptitle("Ablation Study — Dice Coefficient by Adaptation Strategy",
                 color="white", fontsize=15, fontweight="bold")
    for ax, ds in zip(axes, DATASETS):
        vals = ABLATION[ds]
        ymax = max(vals) * 1.18
        _dark_ax(ax, ds, ymax=ymax)
        _bars_with_labels(ax, ABLATION_COND, vals, PALETTE)
        ax.set_xticklabels(ABLATION_COND, rotation=20, ha="right", color="white", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Dice")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(FIG, "ablation_study.png"), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  ablation_study.png")


# ---------------------------------------------------------------------------
# 4. Intensity correction (dark)
# ---------------------------------------------------------------------------

def fig_intensity():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), facecolor=DARK_BG)
    fig.suptitle("Intensity Correction (CLAHE) — Dice by Condition",
                 color="white", fontsize=15, fontweight="bold")
    cols = ["#8b95a7", "#f5a623", "#4aa3df", "#4cc38a"]
    for ax, ds in zip(axes, DATASETS):
        vals = CLAHE[ds]
        _dark_ax(ax, ds, ymax=max(vals) * 1.22)
        _bars_with_labels(ax, CLAHE_COND, vals, cols)
        ax.set_xticklabels(CLAHE_COND, rotation=20, ha="right", color="white", fontsize=8.5)
        if ax is axes[0]:
            ax.set_ylabel("Dice")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(FIG, "intensity_correction.png"), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  intensity_correction.png")


# ---------------------------------------------------------------------------
# 5. Domain-gap analysis (dark, 0-255 scale)
# ---------------------------------------------------------------------------

EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".ppm", ".gif", ".bmp"}


def _paths(folder):
    from pathlib import Path
    return sorted([p for p in Path(os.path.join(ROOT, folder)).iterdir()
                   if p.suffix.lower() in EXTS])


def fig_domain():
    order = ["DRIVE", "STARE", "CHASE", "HRF"]
    dirs = {"DRIVE": "data/DRIVE/images", "STARE": "data/STARE/images",
            "CHASE": "data/CHASE/images", "HRF": "data/HRF/images"}
    stats = {d: {"bright": [], "contrast": [], "gmean": [], "gstd": []} for d in order}
    for d in order:
        for p in _paths(dirs[d]):
            arr = np.array(Image.open(p).convert("RGB"), dtype=np.float32)
            gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
            g = arr[..., 1]
            stats[d]["bright"].append(gray.mean())
            stats[d]["contrast"].append(gray.std())
            stats[d]["gmean"].append(g.mean())
            stats[d]["gstd"].append(g.std())

    colors = ["#4aa3df", "#4cc38a", "#f5a623", "#f56b6b"]
    panels = [("bright", "Brightness (mean)"), ("contrast", "Contrast (std)"),
              ("gmean", "Green Channel Mean"), ("gstd", "Green Channel Std")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=DARK_BG)
    fig.suptitle("Domain Gap Analysis", color="white", fontsize=16, fontweight="bold")
    for ax, (key, title) in zip(axes.ravel(), panels):
        _dark_ax(ax, title)
        ax.grid(False)
        for i, d in enumerate(order):
            ys = stats[d][key]
            xs = np.random.normal(i, 0.06, size=len(ys))
            ax.scatter(xs, ys, s=45, color=colors[i], alpha=0.55, edgecolors="none", zorder=2)
            ax.scatter([i], [np.mean(ys)], marker="D", s=220, color=colors[i],
                       edgecolors="white", linewidths=1.2, zorder=4)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, color="white", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(FIG, "domain_analysis.png"), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    # print brightness means for the report table
    for d in order:
        print(f"    {d}: brightness={np.mean(stats[d]['bright']):.1f} "
              f"green_mean={np.mean(stats[d]['gmean']):.1f}")
    print("  domain_analysis.png")


# ---------------------------------------------------------------------------
# 6. Prediction grids (light) — run baseline + TTT on first 3 images
# ---------------------------------------------------------------------------

def fig_predictions(model):
    for ds, (img_dir, mask_dir) in DATASETS.items():
        ds_obj = RetinalDataset(os.path.join(ROOT, img_dir), os.path.join(ROOT, mask_dir),
                                img_size=IMG_SIZE, augment=False)
        n = min(3, len(ds_obj))
        fig, axes = plt.subplots(n, 4, figsize=(13, 3.3 * n))
        fig.suptitle(f"{ds} — Predictions (first {n} images)", fontsize=15, fontweight="bold")
        col_titles = ["Input Image", "Ground Truth", "Baseline Pred", "TTT Pred"]
        for r in range(n):
            img_t, mask_t = ds_obj[r]
            x = img_t.unsqueeze(0).to(DEVICE)
            model.eval()
            with torch.no_grad():
                base = (torch.sigmoid(model(x)) > 0.5).float().cpu().squeeze().numpy()
            ttt_logits = test_time_adapt(model, x, n_steps=5, lr=1e-6, adapt_mode="bn")
            ttt = (torch.sigmoid(ttt_logits) > 0.5).float().cpu().squeeze().numpy()
            img_np = img_t.permute(1, 2, 0).numpy()
            gt = mask_t.squeeze().numpy()
            for c, im in enumerate([img_np, gt, base, ttt]):
                ax = axes[r, c] if n > 1 else axes[c]
                ax.imshow(im, cmap=None if c == 0 else "gray")
                ax.axis("off")
                if r == 0:
                    ax.set_title(col_titles[c], fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(FIG, f"{ds.lower()}_predictions.png")
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  {os.path.basename(out)}")


# ---------------------------------------------------------------------------
# 7. ROC curves (light) — baseline probabilities per dataset
# ---------------------------------------------------------------------------

def fig_roc(model):
    from sklearn.metrics import roc_curve, auc
    plt.figure(figsize=(7, 6.2))
    colors = {"STARE": "#4c72b0", "CHASE": "#dd8452", "HRF": "#55a868"}
    for ds, (img_dir, mask_dir) in DATASETS.items():
        loader = get_test_loader(os.path.join(ROOT, img_dir), os.path.join(ROOT, mask_dir),
                                 img_size=IMG_SIZE, batch_size=1)
        probs, masks = [], []
        for p, m in run_baseline_inference(model, loader, DEVICE):
            probs.append(p.cpu().numpy().ravel())
            masks.append(m.cpu().numpy().ravel())
        y_score = np.concatenate(probs)
        y_true = np.concatenate(masks).astype(int)
        # subsample for a manageable ROC computation
        if y_true.size > 3_000_000:
            idx = np.random.RandomState(0).choice(y_true.size, 3_000_000, replace=False)
            y_true, y_score = y_true[idx], y_score[idx]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, color=colors[ds], linewidth=2,
                 label=f"{ds} (AUC = {auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Baseline (per dataset)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "roc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  roc_curves.png")


def main():
    print("Charts:")
    fig_dice_comparison()
    fig_all_metrics()
    fig_ablation()
    fig_intensity()
    print("Domain analysis (reading images):")
    fig_domain()

    print(f"Model figures (device={DEVICE}):")
    model = UNetWithRotationHead(n_channels=3, n_classes=1)
    ckpt = torch.load(os.path.join(ROOT, "checkpoints", "best_model.pth"), map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    fig_predictions(model)
    fig_roc(model)
    print("All figures written to report/figures/")


if __name__ == "__main__":
    main()
