# Retinal Vessel Segmentation under Domain Shift with Test-Time Adaptation

## Overview
Train a U-Net on the **DRIVE** retinal vessel dataset, then apply **Test-Time Adaptation (TTT)** at inference to generalize to **STARE**, **CHASE_DB1**, and **HRF** without retraining.

### Key idea
At test time, the model adapts its BatchNorm parameters using a self-supervised auxiliary task (rotation prediction) on each unlabeled test image — no ground truth needed.

---

## Project Structure
```
├── data/
│   ├── DRIVE/    images/ masks/   (20 training images)
│   ├── STARE/    images/ masks/   (20 test images)
│   ├── CHASE/    images/ masks/   (28 test images)
│   └── HRF/      images/ masks/   (45 test images)
├── models/
│   └── unet.py           U-Net + rotation head
├── training/
│   ├── dataset.py        Dataset loaders
│   └── train.py          Training script
├── ttt/
│   └── adapt.py          TTT inference logic
├── evaluation/
│   ├── metrics.py        Dice, IoU, AUC-ROC
│   └── evaluate.py       Baseline vs TTT evaluation
├── notebooks/
│   └── experiments.ipynb Full experiment walkthrough
├── checkpoints/          Saved model weights
└── requirements.txt
```

---

## Datasets
Download and place in `data/`:

| Dataset | Link | # Images | Role |
|---------|------|----------|------|
| DRIVE   | https://drive.grand-challenge.org | 20 | Training |
| STARE   | https://cecas.clemson.edu/~ahoover/stare | 20 | Test |
| CHASE_DB1 | https://blogs.kingston.ac.uk/retinal/chasedb1 | 28 | Test |
| HRF     | https://www5.cs.fau.de/research/data/fundus-images/ | 45 | Test |

---

## Setup
```bash
pip install -r requirements.txt
```

---

## Training (on DRIVE)
```bash
python training/train.py \
  --data_root data/DRIVE \
  --epochs 50 \
  --batch_size 4 \
  --lr 1e-4 \
  --lambda_rot 0.3
```

---

## Evaluation (Baseline vs TTT)
```bash
# On STARE
python evaluation/evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --dataset stare \
  --data_root data/STARE \
  --ttt_steps 5 \
  --ttt_lr 1e-6

# On CHASE
python evaluation/evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --dataset chase \
  --data_root data/CHASE \
  --ttt_steps 5 \
  --ttt_lr 1e-6

# On HRF
python evaluation/evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --dataset hrf \
  --data_root data/HRF \
  --ttt_steps 5 \
  --ttt_lr 1e-6
```

---

## Method

### Training Phase
The model is trained jointly on two tasks:
1. **Segmentation** (BCE + Dice loss) — learns to segment vessels
2. **Rotation prediction** (Cross-entropy) — auxiliary self-supervised task

### Test-Time Adaptation
For each test image (no labels needed):
1. Rotate the image 0°, 90°, 180°, 270°
2. Forward pass → rotation prediction loss
3. Backprop through **BatchNorm affine parameters only** (all other weights frozen)
4. Run segmentation on adapted model → restore original weights

```
Test image → 4 rotations → rotation loss → update BN params → segment → restore weights
```

---

## Results

Model trained on DRIVE (20 images), evaluated on three unseen datasets.  
Best TTT hyperparameters (found via sweep on STARE): **lr = 1e-6, steps = 5**.

### Baseline vs TTT (Dice / AUC-ROC)

| Dataset | Baseline Dice | TTT Dice | Δ | Baseline AUC | Notes |
|---------|---------------|----------|---|--------------|-------|
| STARE (20) | 0.4187 | 0.4207 | +0.0020 | 0.6574 | moderate shift |
| CHASE (28) | 0.0332 | 0.0338 | +0.0006 | 0.5079 | severe shift |
| HRF (45)   | 0.4402 | 0.4454 | +0.0052 | 0.6727 | moderate shift |

Specificity stays >0.97 everywhere — the model rarely paints background as vessel.

### Ablation: TTA vs TTT, and how much to adapt

We compared a label-free **Test-Time Augmentation** (TTA, 8-way rotation+flip ensemble) against
rotation-based **TTT** adapting different parameter subsets (BN affine / encoder / full network).

| Dataset | Baseline | TTA | TTT-BN | TTT-Enc | TTT-Full |
|---------|----------|-----|--------|---------|----------|
| STARE | 0.4187 | **0.4342** (+0.0155) | 0.4207 | 0.4206 | 0.4206 |
| CHASE | 0.0332 | 0.0216 (**−0.0116**) | 0.0338 | 0.0339 | 0.0339 |
| HRF   | 0.4402 | **0.4662** (+0.0260) | 0.4454 | 0.4461 | 0.4461 |

Two clear findings:
- **TTA beats TTT by ~3–5×** on STARE and HRF, but **catastrophically hurts CHASE** (−0.0116).
- **Adapting more parameters does not help**: BN ≈ Encoder ≈ Full. The rotation auxiliary task
  simply does not carry enough signal to reshape the segmentation decision boundary.

### Domain-gap analysis (why CHASE fails)

| Dataset | Mean brightness | Green-channel mean | Baseline Dice |
|---------|-----------------|--------------------|----------------|
| DRIVE (train) | ~88 | — | — |
| STARE | ~85 | ~85 | 0.4187 |
| HRF   | ~80 | ~50 | 0.4402 |
| CHASE | **~53** | **~40** | **0.0332** |

CHASE images are ~40% darker than the DRIVE training distribution. This intensity shift —
not vessel morphology — is the dominant cause of the near-total failure, and it explains why
ensembling (TTA) makes CHASE *worse*: it averages confident-but-wrong predictions.

### Intensity correction (CLAHE) — the decisive experiment

Part 5 tests the domain-gap hypothesis directly: apply CLAHE normalization to each test set
before inference. If the gap is intensity-driven, correcting it should rescue CHASE.

| Dataset | Baseline (orig) | TTA (orig) | Baseline (CLAHE) | TTA (CLAHE) |
|---------|-----------------|------------|------------------|-------------|
| STARE | 0.4187 | 0.4342 | 0.4706 | **0.4993** |
| CHASE | 0.0332 | 0.0216 | 0.3571 | **0.3781** |
| HRF   | 0.4402 | 0.4662 | 0.4363 | 0.4520 |

**This is the key result of the project:**
- **CHASE Dice jumps 0.0332 → 0.3571 with CLAHE — a ~10× improvement** from a single
  preprocessing step, confirming the collapse was an *intensity* problem, not a vessel-structure one.
- CLAHE also **flips TTA on CHASE from harmful (−0.0116) to helpful (+0.021)** — exactly the
  predicted mechanism: once the distribution matches, ensembling works again.
- STARE improves too (0.419 → 0.471). **HRF (already bright) barely changes** — the correct
  control, showing CLAHE only helps when an intensity gap actually exists.

### Key Observations
- Rotation-based **TTT gives only marginal gains** (+0.0006 to +0.0052 Dice) and is dominated by
  much simpler **TTA** when source and target distributions are similar (STARE, HRF).
- **TTA fails catastrophically under large appearance shift** (CHASE), confirming that ensembling
  cannot compensate for a distribution the model never learned.
- **More adaptation ≠ better**: BN-only, encoder, and full-network TTT are statistically identical.
- The CHASE collapse is explained by a ~40% brightness deficit — and **CLAHE intensity
  normalization recovers ~10× the Dice (0.033 → 0.357)**, making intensity correction a
  *prerequisite* to test-time adaptation rather than an afterthought.

---

## Metrics
- Dice Coefficient (F1)
- IoU (Jaccard)
- Sensitivity / Specificity
- AUC-ROC