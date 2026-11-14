"""
Train BodyMeasureNet on synthetic anthropometric data.

Run:  py -3.13 model/train_body_net.py
      (requires PyTorch 2.6+cu124 in Python 3.13, RTX 4070, ~15 min)

Steps:
  1. Load train_data.npz (generate it first with generate_data.py)
  2. Train BodyMeasureNet with AdamW + cosine-anneal LR + early stopping
  3. Save weights → model/body_weights.pt
  4. Print per-measurement validation MAE in cm
"""

import os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# Allow `import body_net` when run from project root or model/ directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from body_net import BodyMeasureNet, N_FEAT, N_MEASURES, MEASURE_KEYS

DATA_PATH   = os.path.join(_HERE, 'train_data.npz')
WEIGHT_PATH = os.path.join(_HERE, 'body_weights.pt')

# ─── Hyperparameters ──────────────────────────────────────────────────────────
EPOCHS       = 120
BATCH_SIZE   = 2048
LR           = 3e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.10
PATIENCE     = 12        # early stopping patience


def load_data():
    if not os.path.exists(DATA_PATH):
        print(f'ERROR: {DATA_PATH} not found.')
        print('Run first:  py -3.13 model/generate_data.py')
        sys.exit(1)

    d = np.load(DATA_PATH)
    X = torch.from_numpy(d['features'])    # (N, 298)  float32
    Y = torch.from_numpy(d['targets'])     # (N, 8)    float32 — fractions
    print(f'Loaded {X.shape[0]:,} samples  X={tuple(X.shape)}  Y={tuple(Y.shape)}')
    return X, Y


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('WARNING: no GPU found — training on CPU (will be slow)')

    X, Y = load_data()
    N = len(X)
    n_val = int(N * VAL_SPLIT)
    n_trn = N - n_val

    ds = TensorDataset(X, Y)
    trn_ds, val_ds = random_split(ds, [n_trn, n_val],
                                  generator=torch.Generator().manual_seed(0))

    trn_ld = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=(device.type == 'cuda'))
    val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=(device.type == 'cuda'))

    model = BodyMeasureNet().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR,
                               weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS,
                                                         eta_min=LR * 0.01)
    criterion = nn.SmoothL1Loss(beta=0.02)   # Huber; robust to outliers

    best_val = float('inf')
    best_epoch = 0
    patience_left = PATIENCE

    print(f'\nTraining for up to {EPOCHS} epochs '
          f'(batch={BATCH_SIZE}, lr={LR}, patience={PATIENCE}) …\n')

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        trn_loss = 0.0
        for xb, yb in trn_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            trn_loss += loss.item() * len(xb)
        trn_loss /= n_trn

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_ld:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= n_val
        sched.step()

        dt = time.time() - t0
        print(f'Epoch {ep:3d}/{EPOCHS}  trn={trn_loss:.5f}  '
              f'val={val_loss:.5f}  lr={sched.get_last_lr()[0]:.2e}  '
              f't={dt:.1f}s', flush=True)

        if val_loss < best_val - 1e-6:
            best_val, best_epoch = val_loss, ep
            patience_left = PATIENCE
            torch.save({'model': model.state_dict(), 'epoch': ep,
                        'val_loss': val_loss}, WEIGHT_PATH)
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f'\nEarly stop at epoch {ep} (best was {best_epoch}).')
                break

    print(f'\nBest val loss: {best_val:.5f} at epoch {best_epoch}')

    # ── Per-measurement MAE in cm (approximate height 175 cm for scale) ──────
    print('\nLoading best weights for validation report …')
    ckpt = torch.load(WEIGHT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_ld:
            xb = xb.to(device)
            all_pred.append(model(xb).cpu())
            all_true.append(yb)

    pred = torch.cat(all_pred).numpy()    # (n_val, 8) — fractions
    true = torch.cat(all_true).numpy()

    # We need true heights to convert fractions → cm.
    # The last feature is normalised height (always 1.0), so we can't
    # recover actual height from features.  Use population mean 175 cm
    # to get an approximate scale for the MAE printout.
    h_approx = 175.0
    mae_cm = np.abs(pred - true) * h_approx   # per-sample, per-output (cm)

    print('\n  Measurement        MAE (cm, ~175 cm reference)')
    print('  ' + '-' * 45)
    for j, k in enumerate(MEASURE_KEYS):
        print(f'  {k:<20s}  {mae_cm[:, j].mean():.2f} cm')
    print(f'\n  Overall MAE: {mae_cm.mean():.2f} cm')

    print(f'\nWeights saved -> {WEIGHT_PATH}')


if __name__ == '__main__':
    train()
