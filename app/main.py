"""
Flask web app for retinal vessel segmentation with TTT.

Routes:
  GET  /         — serves the frontend HTML page
  POST /predict  — upload a fundus image, returns baseline + TTT masks and metrics
"""

import io
import os
import sys
import base64
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from flask import Flask, request, jsonify, send_from_directory

# ---------------------------------------------------------------------------
# Path setup — allow importing from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.unet import UNetWithRotationHead
from ttt.adapt import test_time_adapt, apply_clahe

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pth"
IMG_SIZE   = 512
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TTT_LR     = 1e-6
TTT_STEPS  = 5

# ---------------------------------------------------------------------------
# Load model once at module import
# ---------------------------------------------------------------------------
print(f"Loading model from {CHECKPOINT} on {DEVICE}...")
if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

_state = torch.load(CHECKPOINT, map_location=DEVICE)
if isinstance(_state, dict) and "model_state" in _state:
    _state = _state["model_state"]
elif isinstance(_state, dict) and "model_state_dict" in _state:
    _state = _state["model_state_dict"]

model = UNetWithRotationHead(n_channels=3, n_classes=1)
model.load_state_dict(_state)
model.to(DEVICE)
model.eval()
print("Model loaded.")


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def preprocess(pil_image: Image.Image) -> torch.Tensor:
    """Return [1, 3, H, W] tensor on DEVICE."""
    tensor = TRANSFORM(pil_image.convert("RGB"))
    return tensor.unsqueeze(0).to(DEVICE)


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------

def run_baseline(image_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(image_tensor)
    return (torch.sigmoid(logits) > 0.5).float()


def run_ttt(image_tensor: torch.Tensor) -> torch.Tensor:
    logits = test_time_adapt(model, image_tensor, n_steps=TTT_STEPS, lr=TTT_LR)
    return (torch.sigmoid(logits) > 0.5).float()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
EPS = 1e-7

def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    tp = np.sum((pred == 1) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    dice        = round(float((2 * tp + EPS) / (2 * tp + fp + fn + EPS)), 4)
    iou         = round(float((tp + EPS) / (tp + fp + fn + EPS)), 4)
    accuracy    = round(float((tp + tn + EPS) / (tp + tn + fp + fn + EPS)), 4)
    sensitivity = round(float((tp + EPS) / (tp + fn + EPS)), 4)
    specificity = round(float((tn + EPS) / (tn + fp + EPS)), 4)

    try:
        auc = round(roc_auc_score(gt.astype(int), pred.astype(float)), 4)
    except ValueError:
        auc = None

    return dict(dice=dice, iou=iou, accuracy=accuracy,
                sensitivity=sensitivity, specificity=specificity, auc_roc=auc)


# ---------------------------------------------------------------------------
# Image → base64 PNG helper
# ---------------------------------------------------------------------------

def tensor_to_b64(mask_tensor: torch.Tensor, original_size: tuple) -> str:
    mask_np = mask_tensor.squeeze().cpu().numpy()
    pil_mask = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
    pil_mask = pil_mask.resize(original_size, Image.NEAREST)
    rgba = np.zeros((*pil_mask.size[::-1], 4), dtype=np.uint8)
    m = np.array(pil_mask) > 127
    rgba[m] = [0, 200, 80, 200]
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_to_b64(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")


@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # --- Load image ---
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    try:
        pil_img = Image.open(request.files["image"].stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    # --- Optional CLAHE intensity normalization (reduces domain gap) ---
    normalize = request.form.get("normalize", "").lower() in ("1", "true", "on", "yes")
    if normalize:
        pil_img = apply_clahe(pil_img).convert("RGB")

    original_size = pil_img.size
    img_tensor = preprocess(pil_img)

    # --- Optional GT mask ---
    gt_np = None
    mask_file = request.files.get("mask")
    if mask_file and mask_file.filename:
        try:
            pil_mask = Image.open(mask_file.stream).convert("L")
            pil_mask = pil_mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
            gt_np = (np.array(pil_mask) > 127).astype(np.float32).ravel()
        except Exception:
            return jsonify({"error": "Invalid mask file"}), 400

    # --- Run inference ---
    baseline_mask = run_baseline(img_tensor)
    ttt_mask      = run_ttt(img_tensor)

    # --- Metrics ---
    baseline_metrics = None
    ttt_metrics      = None
    if gt_np is not None:
        def mask_to_np(t):
            arr = t.squeeze().cpu().numpy()
            pil = Image.fromarray((arr * 255).astype(np.uint8))
            pil = pil.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
            return (np.array(pil) > 127).astype(np.float32).ravel()
        baseline_metrics = compute_metrics(mask_to_np(baseline_mask), gt_np)
        ttt_metrics      = compute_metrics(mask_to_np(ttt_mask),      gt_np)

    return jsonify({
        "original_image":   image_to_b64(pil_img),
        "baseline_overlay": tensor_to_b64(baseline_mask, original_size),
        "ttt_overlay":      tensor_to_b64(ttt_mask,      original_size),
        "baseline_metrics": baseline_metrics,
        "ttt_metrics":      ttt_metrics,
        "device":           str(DEVICE),
        "normalized":       normalize,
    })


if __name__ == "__main__":
    from waitress import serve
    print("Starting server on http://127.0.0.1:8000")
    serve(app, host="127.0.0.1", port=8000, threads=4)
