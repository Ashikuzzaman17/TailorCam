"""
Train and evaluate the BodyM silhouette → measurement network.

  python -m model.bodym.train_bodym              # train, then evaluate
  python -m model.bodym.train_bodym --eval-only  # evaluate saved weights

Data: dataset/bodym/{train,testA,testB}/ downloaded from the public bucket
  aws s3 sync --no-sign-request s3://amazon-bodym/ dataset/bodym/

Augmentation mimics what the app's MediaPipe masks look like: ragged or
slightly fat/thin edges (random dilate/erode), small shifts, scale and
rotation changes, and mirrored views. Weight is hidden for 20% of samples so
the model also works when the customer skips the weight field.
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys as _sys
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.bodym.prep import prep_mask, H, W
from model.bodym.bodym_net import (BodyMNet, MEASUREMENTS, WEIGHTS_PATH,
                                   scalar_features)

DATA = data_path('bodym')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Data ──────────────────────────────────────────────────────────────────────
def load_split(split):
    """→ dict of arrays: front, side (N,H,W uint8), y (N,14), h, w, g, subj."""
    cache = os.path.join(DATA, f'cache_{split}.npz')
    if os.path.exists(cache):
        return dict(np.load(cache, allow_pickle=True))
    d = os.path.join(DATA, split)
    meas = {r['subject_id']: r for r in csv.DictReader(open(os.path.join(d, 'measurements.csv')))}
    meta = {r['subject_id']: r for r in csv.DictReader(open(os.path.join(d, 'hwg_metadata.csv')))}
    fr, sd, y, h, w, g, subj = [], [], [], [], [], [], []
    for r in csv.DictReader(open(os.path.join(d, 'subject_to_photo_map.csv'))):
        sid, pid = r['subject_id'], r['photo_id'].strip()
        a = cv2.imread(os.path.join(d, 'mask', pid + '.png'), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(os.path.join(d, 'mask_left', pid + '.png'), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None or sid not in meas:
            continue
        pa, pb = prep_mask(a), prep_mask(b)
        if pa is None or pb is None:
            continue
        fr.append(pa.astype(np.uint8)); sd.append(pb.astype(np.uint8))
        y.append([float(meas[sid][k]) for k in MEASUREMENTS])
        h.append(float(meta[sid]['height_cm'])); w.append(float(meta[sid]['weight_kg']))
        g.append(meta[sid]['gender']); subj.append(sid)
    out = dict(front=np.array(fr), side=np.array(sd), y=np.array(y, np.float32),
               h=np.array(h, np.float32), w=np.array(w, np.float32),
               g=np.array(g), subj=np.array(subj))
    np.savez_compressed(cache, **out)
    return out


def scalars(D, idx, drop_weight=None):
    rows = []
    for j, i in enumerate(idx):
        wt = None if (drop_weight is not None and drop_weight[j]) else float(D['w'][i])
        rows.append(scalar_features(float(D['h'][i]), wt, str(D['g'][i])))
    return torch.from_numpy(np.array(rows)).to(DEV)


# ── GPU augmentation ──────────────────────────────────────────────────────────
def augment(x):
    """x: N x 1 x H x W in {0,1}."""
    n = x.shape[0]
    ang = (torch.rand(n, device=DEV) - 0.5) * np.radians(6)
    sc = 1 + (torch.rand(n, device=DEV) - 0.5) * 0.08
    tx = (torch.rand(n, device=DEV) - 0.5) * 0.08
    ty = (torch.rand(n, device=DEV) - 0.5) * 0.06
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1),
                         torch.stack([sin, cos, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False),
                      align_corners=False)
    flip = torch.rand(n, device=DEV) < 0.5
    x[flip] = x[flip].flip(-1)
    # segmentation-edge noise: dilate or erode by 0-2 px
    r = int(np.random.randint(0, 3))
    if r:
        k = 2 * r + 1
        if np.random.rand() < 0.5:
            x = F.max_pool2d(x, k, 1, r)
        else:
            x = -F.max_pool2d(-x, k, 1, r)
    return (x > 0.5).float()


def batch(D, idx, aug):
    f = torch.from_numpy(D['front'][idx]).float()[:, None].to(DEV)
    s = torch.from_numpy(D['side'][idx]).float()[:, None].to(DEV)
    if aug:
        f, s = augment(f), augment(s)
    return f, s


# ── Train / evaluate ──────────────────────────────────────────────────────────
@torch.no_grad()
def predict(net, D, use_weight=True, bs=128):
    net.eval()
    out = []
    for i in range(0, len(D['y']), bs):
        idx = np.arange(i, min(i + bs, len(D['y'])))
        f, s = batch(D, idx, False)
        sc = scalars(D, idx, None if use_weight else np.ones(len(idx), bool))
        p = (net(f, s, sc) + net(f, s.flip(-1), sc)) / 2
        out.append(net.to_cm(p).cpu().numpy())
    return np.concatenate(out)


def hw_baseline(train, test, use_weight=True):
    """Ridge regression on height/weight/BMI/gender only (no images)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    def X(D):
        rows = [scalar_features(float(h), float(w) if use_weight else None, str(g))
                for h, w, g in zip(D['h'], D['w'], D['g'])]
        return PolynomialFeatures(2).fit_transform(np.array(rows))
    return Ridge(alpha=1.0).fit(X(train), train['y']).predict(X(test))


def report(name, D, preds):
    print(f'\n== {name}: {len(set(D["subj"]))} people, {len(D["y"])} photo pairs — '
          'mean absolute error (cm), per photo pair')
    print(f'{"measurement":20s}' + ''.join(f'{k:>16s}' for k in preds))
    for j, m in enumerate(MEASUREMENTS):
        print(f'{m:20s}' + ''.join(f'{np.abs(p[:, j] - D["y"][:, j]).mean():16.2f}'
                                    for p in preds.values()))


def train(epochs, bs=64, lr=3e-4):
    tr = load_split('train')
    subj = np.unique(tr['subj'])
    rng = np.random.default_rng(0)
    val_subj = set(rng.choice(subj, size=len(subj) // 10, replace=False))
    val_mask = np.array([s in val_subj for s in tr['subj']])
    tr_idx, va_idx = np.where(~val_mask)[0], np.where(val_mask)[0]
    va = {k: v[va_idx] for k, v in tr.items()}
    print(f'train {len(tr_idx)} pairs / val {len(va_idx)} pairs ({len(val_subj)} people held out)')

    net = BodyMNet(pretrained=True).to(DEV)
    y = torch.from_numpy(tr['y'][tr_idx])
    net.y_mean.copy_(y.mean(0)); net.y_std.copy_(y.std(0))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(tr_idx) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    yt = torch.from_numpy(tr['y']).to(DEV)
    best, t0 = 1e9, time.time()
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(tr_idx)
        for i in range(len(perm) // bs):
            idx = perm[i * bs:(i + 1) * bs]
            f, s = batch(tr, idx, True)
            sc = scalars(tr, idx, rng.random(len(idx)) < 0.2)
            target = (yt[idx] - net.y_mean) / net.y_std
            loss = F.smooth_l1_loss(net(f, s, sc), target, beta=0.5)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        mae = np.abs(predict(net, va) - va['y']).mean(0)
        key = mae[[MEASUREMENTS.index(k) for k in ('chest', 'waist', 'hip')]].mean()
        if key < best:
            best = key
            torch.save(net.state_dict(), WEIGHTS_PATH)
        print(f'epoch {ep + 1:3d}/{epochs}  loss {loss.item():.3f}  '
              f'val chest/waist/hip MAE {key:.2f} cm  (best {best:.2f})  {time.time() - t0:.0f}s',
              flush=True)


def evaluate():
    net = BodyMNet().to(DEV)
    net.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEV, weights_only=True))
    tr = load_split('train')
    for split in ('testA', 'testB'):
        D = load_split(split)
        report(split, D, {
            'height+weight': hw_baseline(tr, D),
            'net+weight': predict(net, D, True),
            'height only': hw_baseline(tr, D, False),
            'net, no weight': predict(net, D, False),
        })


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--eval-only', action='store_true')
    a = ap.parse_args()
    if not a.eval_only:
        train(a.epochs)
    evaluate()
