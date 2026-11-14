"""
Silhouette network trained on BODY-fit (Tampere University, 4,149 real people,
3D scan fits + body measurements; scripts/build_bodyfit_dataset.py).

Same design as model/bodym/hybrid.py (ResNet-18 on front + side silhouettes,
height / weight / gender as numbers, a ridge prior that the image branch
corrects), but it predicts the TailorCam measurements directly, with the
BODY-fit definitions, so no label mixing with BodyM.

  python -m model.bodyfit.train_bodyfit --seed 0     # train one ensemble member
  python -m model.bodyfit.train_bodyfit --eval-only  # held-out BODY-fit test
"""

import os
import sys
import glob
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys as _sys
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.bodym.bodym_net import scalar_features, N_SCALARS
from model.bodym.train_bodym import augment
from model.bodym.hybrid import poly, N_POLY, _real_world_aug

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = data_path('bodyfit', 'cache.npz')
KEYS = ['chest', 'waist', 'hip', 'neck', 'thigh', 'shoulder_width', 'sleeve_length',
        'bicep', 'wrist', 'calf', 'ankle']
NM = len(KEYS)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def member_paths():
    return sorted(glob.glob(os.path.join(_HERE, 'bodyfit_s*.pt')))


class BodyFitNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        conv = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        conv.weight.data = r.conv1.weight.data.sum(1, keepdim=True)
        r.conv1, r.fc = conv, nn.Identity()
        self.encoder = r
        self.view_embed = nn.Parameter(torch.zeros(2, 512))
        self.head = nn.Sequential(nn.Linear(1024 + N_SCALARS + 1, 512), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 2 * NM))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('prior_W', torch.zeros(N_POLY, NM))
        self.register_buffer('y_std', torch.ones(NM))

    def prior(self, s):
        return poly(s) @ self.prior_W

    def forward(self, front, side, s, image_ok):
        """→ (mean cm, Laplace scale cm). image_ok: N x 1, 0 = ignore the photos."""
        p = self.prior(s)
        e = torch.cat([self.encoder(front) + self.view_embed[0], self.encoder(side) + self.view_embed[1]], 1)
        out = self.head(torch.cat([e * image_ok, s, image_ok], 1))
        d, ls = out[:, :NM], out[:, NM:]
        return p + d * self.y_std * image_ok, F.softplus(ls) * self.y_std + 0.3


def load():
    return dict(np.load(DATA))                          # our own cache (no pickles)


def scalars(D, idx, drop_weight=None):
    rows = [scalar_features(float(D['h'][i]),
                            None if (drop_weight is not None and drop_weight[j]) else float(D['w'][i]),
                            'male' if D['g'][i] == 1 else 'female') for j, i in enumerate(idx)]
    return torch.from_numpy(np.array(rows, np.float32)).to(DEV)


def fit_prior(D, idx, mask):
    """Ridge prior per measurement on the people that have it (with and without weight)."""
    Wp = np.zeros((N_POLY, NM), np.float32)
    Pa = np.concatenate([poly(scalars(D, idx, np.full(len(idx), drop)).cpu().numpy()) for drop in (False, True)])
    for k in range(NM):
        ok = np.concatenate([mask[idx, k]] * 2)
        y = np.concatenate([D['y'][idx, k]] * 2)[ok]
        A = Pa[ok]
        Wp[:, k] = np.linalg.solve(A.T @ A + 1.0 * np.eye(N_POLY), A.T @ y)
    return Wp


def train(epochs=40, seed=0, bs=64, lr=3e-4):
    D = load()
    tr = np.where(D['split'] == 0)[0]
    va = np.where(D['split'] == 1)[0]
    mask = ~np.isnan(D['y'])
    torch.manual_seed(seed); np.random.seed(seed)
    net = BodyFitNet(pretrained=True).to(DEV)
    net.prior_W.copy_(torch.from_numpy(fit_prior(D, tr, mask)))
    net.y_std.copy_(torch.from_numpy(np.nanstd(D['y'][tr], 0).astype(np.float32)))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(tr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    Y = torch.from_numpy(np.nan_to_num(D['y'])).to(DEV)
    M = torch.from_numpy(mask.astype(np.float32)).to(DEV)
    best = 1e9
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(tr)
        for b in range(len(tr) // bs):
            idx = perm[b * bs:(b + 1) * bs]
            f = torch.from_numpy(D['front'][idx]).float()[:, None].to(DEV)
            s = torch.from_numpy(D['side'][idx]).float()[:, None].to(DEV)
            f, s = _real_world_aug(augment(f)), _real_world_aug(augment(s))
            drop_w = np.random.rand(len(idx)) < 0.3
            sc = scalars(D, idx, drop_w)
            ok = (torch.rand(len(idx), 1, device=DEV) > 0.15).float()
            mu, b_ = net(f, s, sc, ok)
            nll = (torch.abs(mu - Y[idx]) / b_ + torch.log(b_)) * M[idx]
            loss = nll.sum() / M[idx].sum()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        m = evaluate(net, D, va)
        key = np.nanmean(m[:3])
        if key < best:
            best = key
            torch.save(net.state_dict(), os.path.join(_HERE, f'bodyfit_s{seed}.pt'))
        print(f'epoch {ep + 1:3d}/{epochs}  loss {loss.item():.3f}  val chest/waist/hip {m[0]:.2f}/{m[1]:.2f}/{m[2]:.2f}'
              f'  best {best:.2f}', flush=True)


@torch.no_grad()
def predict(net, D, idx, use_weight=True, bs=128):
    net.eval()
    out = []
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        f = torch.from_numpy(D['front'][j]).float()[:, None].to(DEV)
        s = torch.from_numpy(D['side'][j]).float()[:, None].to(DEV)
        sc = scalars(D, j, np.full(len(j), not use_weight))
        out.append(net(f, s, sc, torch.ones(len(j), 1, device=DEV))[0].cpu().numpy())
    return np.concatenate(out)


def evaluate(net, D, idx, use_weight=True):
    p = predict(net, D, idx, use_weight)
    return np.nanmean(np.abs(p - D['y'][idx]), 0)


class BodyFitModel:
    """Ensemble used by the app / evaluation: predict(height, weight, gender, views)."""

    def __init__(self, paths=None):
        self.nets = []
        for p in paths or member_paths():
            n = BodyFitNet().to(DEV)
            n.load_state_dict(torch.load(p, map_location=DEV, weights_only=True))
            n.eval(); self.nets.append(n)

    @torch.no_grad()
    def predict(self, height_cm, weight_kg, gender, views):
        from model.bodym.prep import prep_mask
        s = torch.from_numpy(np.array([scalar_features(height_cm, weight_kg, gender)], np.float32)).to(DEV)
        pairs = [(prep_mask(a), prep_mask(b)) for a, b in views]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        outs = []
        for n in self.nets:
            if pairs:
                f = torch.from_numpy(np.array([a for a, _ in pairs])).float()[:, None].to(DEV)
                sd = torch.from_numpy(np.array([b for _, b in pairs])).float()[:, None].to(DEV)
                mu, _ = n(f, sd, s.repeat(len(pairs), 1), torch.ones(len(pairs), 1, device=DEV))
                outs.append(mu.mean(0).cpu().numpy())
            else:
                outs.append(n.prior(s)[0].cpu().numpy())
        return dict(zip(KEYS, np.mean(outs, 0).round(2)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-only', action='store_true')
    a = ap.parse_args()
    if not a.eval_only:
        train(a.epochs, a.seed)
    D = load()
    te = np.where(D['split'] == 2)[0]
    for use_w in (True, False):
        preds = []
        for p in member_paths():
            n = BodyFitNet().to(DEV)
            n.load_state_dict(torch.load(p, map_location=DEV, weights_only=True))
            preds.append(predict(n, D, te, use_w))
        e = np.nanmean(np.abs(np.mean(preds, 0) - D['y'][te]), 0)
        print(f'BODY-fit test ({len(te)} people, {"with" if use_w else "no"} weight), MAE cm:',
              dict(zip(KEYS, e.round(2))))
