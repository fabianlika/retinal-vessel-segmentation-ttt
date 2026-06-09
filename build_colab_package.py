"""
Build colab_package.zip for training / evaluation on Google Colab.

Bundles the CURRENT local source tree (so any local, uncommitted fixes are
included), the datasets under data/, and the trained checkpoint into a single
zip whose members live under a top-level ``colab_package/`` folder — matching
what notebooks/colab_train.ipynb and notebooks/colab_eval.ipynb expect.

Usage (from repo root):
    python build_colab_package.py                  # full package (all datasets)
    python build_colab_package.py --train-only     # code + DRIVE only (small, fast upload)
    python build_colab_package.py --no-checkpoint  # omit the 60 MB checkpoint

Then upload the resulting colab_package.zip in the notebook's first cell.
"""

import argparse
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARC_PREFIX = "colab_package"   # top-level folder inside the zip

# Directory names that are never useful inside the package.
EXCLUDE_DIRS = {
    ".git", "__pycache__", ".ipynb_checkpoints", ".pytest_cache",
    "venv", ".venv", "env", "ENV", ".vs", ".vscode", ".idea",
    "runs", "report", "colab_package",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log", ".zip"}


def should_skip(path: Path) -> bool:
    parts = set(path.relative_to(ROOT).parts)
    if parts & EXCLUDE_DIRS:
        return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    # CLAHE datasets are regenerated in the notebook at runtime.
    rel = path.relative_to(ROOT).as_posix()
    if "_clahe/" in rel:
        return True
    return False


def iter_files(train_only: bool, include_checkpoint: bool):
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for name in filenames:
            p = Path(dirpath) / name
            if should_skip(p):
                continue
            rel = p.relative_to(ROOT).as_posix()

            if not include_checkpoint and rel.startswith("checkpoints/"):
                continue
            # Training needs only DRIVE; --train-only drops the other datasets.
            if train_only and rel.startswith("data/") and not rel.startswith("data/DRIVE/"):
                continue
            yield p, rel


def main():
    ap = argparse.ArgumentParser(description="Build colab_package.zip")
    ap.add_argument("--train-only", action="store_true",
                    help="Include only the DRIVE dataset (smaller, faster upload).")
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="Exclude checkpoints/ (train from scratch in Colab).")
    ap.add_argument("--output", default="colab_package.zip")
    args = ap.parse_args()

    out = ROOT / args.output
    if out.exists():
        out.unlink()

    n = 0
    total = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for path, rel in iter_files(args.train_only, not args.no_checkpoint):
            z.write(path, arcname=f"{ARC_PREFIX}/{rel}")
            n += 1
            total += path.stat().st_size

    size_mb = out.stat().st_size / 1e6
    print(f"Wrote {out.name}: {n} files, {total/1e6:.1f} MB raw -> {size_mb:.1f} MB zipped")
    if args.train_only:
        print("Mode: TRAIN-ONLY (DRIVE only). Use the full build for evaluation.")
    print("Upload this file in the notebook's first cell.")


if __name__ == "__main__":
    main()
