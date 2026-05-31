"""
Domain gap analysis: quantify the distributional difference between
DRIVE (training domain) and STARE / CHASE / HRF (test domains).

Statistics computed per dataset:
  - Mean & std of green channel intensity (most informative for vessels)
  - Overall image brightness (mean luminance)
  - Image contrast (std of luminance)
  - Vessel density (% of pixels that are vessel, from GT masks)
  - Green channel histogram (plotted)

Usage (run from repo root):
  python domain_analysis.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASETS = {
    "DRIVE (train)": ("data/DRIVE/images",  "data/DRIVE/masks"),
    "STARE":         ("data/STARE/images",   "data/STARE/masks"),
    "CHASE":         ("data/CHASE/images",   "data/CHASE/masks"),
    "HRF":           ("data/HRF/images",     "data/HRF/masks"),
}

SUPPORTED = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".ppm", ".gif", ".bmp"}


def collect_paths(folder):
    return sorted([p for p in Path(folder).iterdir()
                   if p.suffix.lower() in SUPPORTED])


# ---------------------------------------------------------------------------
# Per-image statistics
# ---------------------------------------------------------------------------

def image_stats(img_path, mask_path):
    img  = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    mask = np.array(Image.open(mask_path).convert("L"))

    gray   = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    green  = img[:,:,1]
    binary = (mask > 127).astype(np.float32)

    return {
        "brightness_mean": float(np.mean(gray)),
        "brightness_std":  float(np.std(gray)),
        "contrast":        float(np.std(gray)),
        "green_mean":      float(np.mean(green)),
        "green_std":       float(np.std(green)),
        "vessel_density":  float(np.mean(binary)),
        "green_hist":      np.histogram(green.ravel(), bins=64, range=(0, 1))[0].astype(np.float32),
    }


def dataset_stats(img_dir, mask_dir):
    imgs  = collect_paths(img_dir)
    masks = collect_paths(mask_dir)
    assert len(imgs) == len(masks), f"Mismatch in {img_dir}"

    all_stats = [image_stats(i, m) for i, m in zip(imgs, masks)]

    agg = {}
    for key in all_stats[0]:
        vals = [s[key] for s in all_stats]
        if key == "green_hist":
            agg[key] = np.mean(vals, axis=0)
        else:
            agg[key]          = float(np.mean(vals))
            agg[key + "_sem"] = float(np.std(vals) / np.sqrt(len(vals)))
    agg["n"] = len(imgs)
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    stats = {}
    for ds_name, (img_dir, mask_dir) in DATASETS.items():
        if not Path(img_dir).exists():
            print(f"Skipping {ds_name} (not found)")
            continue
        print(f"Computing stats for {ds_name}...", end=" ", flush=True)
        stats[ds_name] = dataset_stats(img_dir, mask_dir)
        print(f"done ({stats[ds_name]['n']} images)")

    # ---------------------------------------------------------------------------
    # Print summary table
    # ---------------------------------------------------------------------------
    print("\n" + "="*80)
    print("DOMAIN GAP ANALYSIS")
    print("="*80)
    header = f"{'Dataset':<20} {'N':>4}  {'Brightness':>12}  {'Contrast':>10}  {'Green mean':>12}  {'Vessel %':>10}"
    print(header)
    print("-"*80)
    for ds_name, s in stats.items():
        print(f"{ds_name:<20} {s['n']:>4}  "
              f"{s['brightness_mean']:>6.4f}±{s['brightness_mean_sem']:.4f}  "
              f"{s['contrast']:>8.4f}  "
              f"{s['green_mean']:>10.4f}±{s['green_mean_sem']:.4f}  "
              f"{s['vessel_density']*100:>8.2f}%")

    # ---------------------------------------------------------------------------
    # Visualise: histogram comparison + bar charts
    # ---------------------------------------------------------------------------
    n_ds = len(stats)
    colors = ["#4299e1", "#48bb78", "#f6ad55", "#fc8181"]
    ds_names = list(stats.keys())

    fig = plt.figure(figsize=(16, 10), facecolor="#0f1117")
    fig.suptitle("Domain Gap Analysis: DRIVE vs Test Datasets",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97, top=0.92, bottom=0.08)

    def styled_ax(ax, title):
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors="white", labelsize=8)
        ax.set_title(title, color="white", fontsize=9, pad=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3748")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        return ax

    # 1. Green channel histograms
    ax1 = styled_ax(fig.add_subplot(gs[0, :2]), "Green Channel Intensity Distribution")
    bins = np.linspace(0, 1, 65)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    for i, (ds_name, s) in enumerate(stats.items()):
        h = s["green_hist"] / s["green_hist"].sum()
        ax1.plot(bin_centers, h, label=ds_name, color=colors[i % len(colors)], linewidth=1.8)
    ax1.legend(facecolor="#1a1d2e", edgecolor="#2d3748", labelcolor="white", fontsize=8)
    ax1.set_xlabel("Pixel intensity")
    ax1.set_ylabel("Frequency")

    # 2. Brightness bar chart
    ax2 = styled_ax(fig.add_subplot(gs[0, 2]), "Mean Brightness")
    vals = [stats[d]["brightness_mean"] for d in ds_names]
    errs = [stats[d]["brightness_mean_sem"] for d in ds_names]
    bars = ax2.bar(range(n_ds), vals, yerr=errs, color=colors[:n_ds], capsize=4, width=0.6)
    ax2.set_xticks(range(n_ds))
    ax2.set_xticklabels([d.split(" ")[0] for d in ds_names], rotation=20, ha="right")
    ax2.set_ylabel("Mean luminance")

    # 3. Contrast bar chart
    ax3 = styled_ax(fig.add_subplot(gs[1, 0]), "Image Contrast (std of luminance)")
    vals = [stats[d]["contrast"] for d in ds_names]
    ax3.bar(range(n_ds), vals, color=colors[:n_ds], width=0.6)
    ax3.set_xticks(range(n_ds))
    ax3.set_xticklabels([d.split(" ")[0] for d in ds_names], rotation=20, ha="right")
    ax3.set_ylabel("Std of luminance")

    # 4. Green channel mean
    ax4 = styled_ax(fig.add_subplot(gs[1, 1]), "Green Channel Mean")
    vals = [stats[d]["green_mean"] for d in ds_names]
    errs = [stats[d]["green_mean_sem"] for d in ds_names]
    ax4.bar(range(n_ds), vals, yerr=errs, color=colors[:n_ds], capsize=4, width=0.6)
    ax4.set_xticks(range(n_ds))
    ax4.set_xticklabels([d.split(" ")[0] for d in ds_names], rotation=20, ha="right")
    ax4.set_ylabel("Mean green intensity")

    # 5. Vessel density
    ax5 = styled_ax(fig.add_subplot(gs[1, 2]), "Vessel Density (% vessel pixels)")
    vals = [stats[d]["vessel_density"] * 100 for d in ds_names]
    ax5.bar(range(n_ds), vals, color=colors[:n_ds], width=0.6)
    ax5.set_xticks(range(n_ds))
    ax5.set_xticklabels([d.split(" ")[0] for d in ds_names], rotation=20, ha="right")
    ax5.set_ylabel("% vessel pixels")

    plt.savefig("domain_analysis.png", dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    print("\nSaved domain_analysis.png")
    plt.show()


if __name__ == "__main__":
    main()
