import sys, torch, torch.nn as nn
sys.path.insert(0, '.')
from models.unet import UNetWithRotationHead
from training.dataset import get_test_loader
from ttt.adapt import run_baseline_inference, run_ttt_inference
from evaluation.metrics import evaluate

LOG = 'sweep_results.log'

def log(msg):
    print(msg, flush=True)
    with open(LOG, 'a') as f:
        f.write(msg + '\n')

device = torch.device('cpu')
model = UNetWithRotationHead(n_channels=3, n_classes=1)
ckpt = torch.load('checkpoints/best_model.pth', map_location=device)
model.load_state_dict(ckpt['model_state'])
model.to(device)

loader = get_test_loader('data/STARE/images', 'data/STARE/masks', img_size=512, batch_size=1)

# Baseline
base_preds, base_masks = [], []
for p, m in run_baseline_inference(model, loader, device):
    base_preds.append(p.cpu()); base_masks.append(m.cpu())
base = evaluate(base_preds, base_masks)
log(f'Baseline dice={base["dice"]:.4f}')

# Sweep LR and steps
for lr in [1e-6, 5e-7, 1e-7]:
    for steps in [3, 5]:
        ttt_preds, ttt_masks = [], []
        for p, m in run_ttt_inference(model, loader, device, n_steps=steps, lr=lr):
            ttt_preds.append(p.cpu()); ttt_masks.append(m.cpu())
        res = evaluate(ttt_preds, ttt_masks)
        diff = res['dice'] - base['dice']
        log(f'lr={lr:.0e} steps={steps}  dice={res["dice"]:.4f}  delta={diff:+.4f}')

log('SWEEP COMPLETE')
