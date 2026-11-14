"""
Hybrid body-measurement model.

  prior  : ridge regression on height / weight / BMI / gender (no images).
           Learned from BodyM; reliable everywhere (≈2–4 cm on girths).
  image  : ResNet-18 on front + side silhouettes predicts a *correction* to
           the prior plus its own uncertainty (Laplace scale, cm).
  gate   : the image branch can be switched off (image_ok = 0) when the
           silhouettes are unusable — e.g. arms not away from the body — and
           the model then falls back to the prior. It is trained with the
           image branch randomly dropped so both modes are calibrated.

Real camera masks differ from BodyM's clean silhouettes (arms touching the
torso, loose clothes, blobby MediaPipe edges); training augmentation
imitates those so the image branch stays useful on real photos.

  python -m model.bodym.hybrid              # train + evaluate
  python -m model.bodym.hybrid --eval-only
"""

import os
import sys
import time
import argparse
from itertools import combinations_with_replacement

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.bodym.prep import prep_mask
from model.bodym.bodym_net import MEASUREMENTS, scalar_features, N_SCALARS

_HERE = os.path.dirname(os.path.abspath(__file__))


def member_path(seed):
    return os.path.join(_HERE, f'bodym_hybrid_s{seed}.pt')


def member_paths():
    """All trained ensemble members (bodym_hybrid_s*.pt)."""
    import glob
    return sorted(glob.glob(os.path.join(_HERE, 'bodym_hybrid_s*.pt')))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NM = len(MEASUREMENTS)
_PAIRS = list(combinations_with_replacement(range(N_SCALARS), 2))
N_POLY = 1 + N_SCALARS + len(_PAIRS)


def poly(s):
    """Degree-2 polynomial features. s: (N, 5) tensor or array."""
    lib = torch if torch.is_tensor(s) else np
    cols = [lib.ones_like(s[:, :1]), s] + [s[:, i:i + 1] * s[:, j:j + 1] for i, j in _PAIRS]
    return lib.concatenate(cols, 1) if lib is np else torch.cat(cols, 1)


class HybridNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        conv = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        conv.weight.data = r.conv1.weight.data.sum(1, keepdim=True)
        r.conv1, r.fc = conv, nn.Identity()
        self.encoder = r
        self.view_embed = nn.Parameter(torch.zeros(2, 512))
        self.head = nn.Sequential(
            nn.Linear(1024 + N_SCALARS + 1, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 2 * NM))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('prior_W', torch.zeros(N_POLY, NM))   # ridge, in cm
        self.register_buffer('y_std', torch.ones(NM))
        self.register_buffer('prior_mae', torch.ones(NM))          # init scale

    def prior(self, scalars):
        return poly(scalars) @ self.prior_W

    def forward(self, front, side, scalars, image_ok):
        """
        front/side: N x 1 x H x W {0,1}; scalars: N x 5; image_ok: N (0/1).
        Returns (mean_cm, scale_cm), each N x 14.
        """
        f = (self.encoder(front * 2 - 1) + self.view_embed[0]) * image_ok[:, None]
        s = (self.encoder(side * 2 - 1) + self.view_embed[1]) * image_ok[:, None]
        out = self.head(torch.cat([f, s, scalars, image_ok[:, None]], 1))
        delta, log_b = out[:, :NM], out[:, NM:]
        mean = self.prior(scalars) + delta * self.y_std
        scale = self.prior_mae * torch.exp(log_b.clamp(-4, 3))
        return mean, scale


class HybridPredictor:
    """App-side: measurements + per-measurement uncertainty (cm)."""

    def __init__(self, paths=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = []
        for p in paths or member_paths():
            net = HybridNet()
            net.load_state_dict(torch.load(p, map_location='cpu', weights_only=True))
            self.nets.append(net.eval().to(self.device))
        if not self.nets:
            raise FileNotFoundError('no bodym_hybrid_s*.pt model files')

    @torch.no_grad()
    def predict(self, height_cm, weight_kg=None, gender='unknown', views=()):
        """
        views: list of (front_mask, side_mask) pairs from the same scan (any
        number, may be empty). Pairs whose masks can't be read are skipped;
        with no usable pair the prior alone is returned.
        Returns {name: (value_cm, sigma_cm)}.
        Unknown gender: the model only saw male / female, so both are predicted
        and averaged (the spread between them is added to the uncertainty).
        """
        if gender not in ('male', 'female'):
            pm = self.predict(height_cm, weight_kg, 'male', views)
            pf = self.predict(height_cm, weight_kg, 'female', views)
            return {k: (round((pm[k][0] + pf[k][0]) / 2, 1),
                        round(float(np.hypot((pm[k][1] + pf[k][1]) / 2,
                                             (pm[k][0] - pf[k][0]) / 2)), 1))
                    for k in pm}
        sc = torch.from_numpy(scalar_features(height_cm, weight_kg, gender))[None].to(self.device)
        pairs = [(prep_mask(a), prep_mask(b)) for a, b in views]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        if pairs:
            f = torch.from_numpy(np.stack([a for a, _ in pairs]))[:, None].to(self.device)
            s = torch.from_numpy(np.stack([b for _, b in pairs]))[:, None].to(self.device)
            f, s = torch.cat([f, f]), torch.cat([s, s.flip(-1)])
            ok = torch.ones(len(f), device=self.device)
        else:
            f = s = torch.zeros(1, 1, 256, 192, device=self.device)
            ok = torch.zeros(1, device=self.device)
        means, scales = [], []
        for net in self.nets:
            mean, scale = net(f, s, sc.expand(len(f), -1), ok)
            # combine frames of this scan: precision-weighted mean
            w = 1.0 / scale ** 2
            means.append((mean * w).sum(0) / w.sum(0))
            scales.append(scale.mean(0))      # frames aren't independent: no /sqrt(n)
        means, scales = torch.stack(means), torch.stack(scales)
        m = means.mean(0)
        # expected error ≈ model's own scale plus ensemble disagreement
        sig = torch.sqrt(scales.mean(0) ** 2 + means.var(0, unbiased=False))
        return {k: (round(float(a), 1), round(float(b), 1))
                for k, a, b in zip(MEASUREMENTS, m.cpu().numpy(), sig.cpu().numpy())}


# ── Training ──────────────────────────────────────────────────────────────────
def _real_world_aug(x):
    """Make BodyM silhouettes look like MediaPipe masks of clothed people."""
    n = x.shape[0]
    out = x.clone()
    # arms touching / merged with the torso: horizontal closing
    sel = torch.rand(n, device=DEV) < 0.35
    if sel.any():
        k = int(np.random.choice([9, 15, 21, 27]))
        y = F.max_pool2d(out[sel], (3, k), 1, (1, k // 2))
        out[sel] = -F.max_pool2d(-y, (3, k), 1, (1, k // 2))
    # loose clothing / blurry segmentation: blur then re-threshold low
    sel = torch.rand(n, device=DEV) < 0.35
    if sel.any():
        k = int(np.random.choice([5, 9, 13]))
        y = F.avg_pool2d(out[sel], k, 1, k // 2)
        out[sel] = (y > float(np.random.uniform(0.25, 0.5))).float()
    # random holes / missing chunks (segmentation drop-outs)
    sel = torch.rand(n, device=DEV) < 0.15
    for i in torch.where(sel)[0].tolist():
        h, w = np.random.randint(10, 40), np.random.randint(10, 40)
        y0, x0 = np.random.randint(0, 256 - h), np.random.randint(0, 192 - w)
        out[i, :, y0:y0 + h, x0:x0 + w] = 0
    return out


def fit_prior(tr, idx):
    """Ridge on poly(scalars), with and without weight, → (N_POLY, 14) in cm."""
    from sklearn.linear_model import Ridge
    rows, ys = [], []
    for hide in (False, True):
        for i in idx:
            rows.append(scalar_features(float(tr['h'][i]), None if hide else float(tr['w'][i]),
                                        str(tr['g'][i])))
            ys.append(tr['y'][i])
    X, Y = poly(np.array(rows)), np.array(ys)
    r = Ridge(alpha=1.0, fit_intercept=False).fit(X, Y)
    W = r.coef_.T.astype(np.float32)
    return W, np.abs(X @ W - Y).mean(0).astype(np.float32)


def train(epochs, seed=0, bs=64, lr=3e-4):
    from model.bodym.train_bodym import load_split, augment, batch, scalars
    tr = load_split('train')
    subj = np.unique(tr['subj'])
    val_subj = set(np.random.default_rng(0).choice(subj, size=len(subj) // 10, replace=False))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed); np.random.seed(seed)
    vm = np.array([s in val_subj for s in tr['subj']])
    tr_idx, va_idx = np.where(~vm)[0], np.where(vm)[0]
    va = {k: v[va_idx] for k, v in tr.items()}

    net = HybridNet(pretrained=True).to(DEV)
    W, mae = fit_prior(tr, tr_idx)
    net.prior_W.copy_(torch.from_numpy(W)); net.prior_mae.copy_(torch.from_numpy(mae))
    net.y_std.copy_(torch.from_numpy(tr['y'][tr_idx].std(0)))
    print('prior (height/weight) train MAE:', dict(zip(MEASUREMENTS, mae.round(2))))

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
            f, s = _real_world_aug(f), _real_world_aug(s)
            sc = scalars(tr, idx, rng.random(len(idx)) < 0.2)
            ok = (torch.rand(len(idx), device=DEV) > 0.2).float()
            mean, scale = net(f, s, sc, ok)
            # Laplace negative log-likelihood (≈ MAE with learned confidence)
            loss = ((yt[idx] - mean).abs() / scale + torch.log(scale)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        m = evaluate_split(net, va)['image']
        key = m[[MEASUREMENTS.index(k) for k in ('chest', 'waist', 'hip')]].mean()
        if key < best:
            best = key
            torch.save(net.state_dict(), member_path(seed))
        print(f'epoch {ep + 1:3d}/{epochs}  loss {loss.item():.3f}  val chest/waist/hip '
              f'MAE {key:.2f} (best {best:.2f})  {time.time() - t0:.0f}s', flush=True)


@torch.no_grad()
def predict_split(net, D, bs=128, use_weight=True, real_aug=False):
    """→ {'prior': N x 14, 'image': N x 14} predictions in cm."""
    from model.bodym.train_bodym import batch, scalars
    net.eval()
    res = {'prior': [], 'image': []}
    for i in range(0, len(D['y']), bs):
        idx = np.arange(i, min(i + bs, len(D['y'])))
        f, s = batch(D, idx, False)
        if real_aug:
            torch.manual_seed(i); np.random.seed(i)
            f, s = _real_world_aug(f), _real_world_aug(s)
        sc = scalars(D, idx, None if use_weight else np.ones(len(idx), bool))
        res['prior'].append(net.prior(sc).cpu().numpy())
        one = torch.ones(len(idx), device=DEV)
        m1, _ = net(f, s, sc, one)
        m2, _ = net(f, s.flip(-1), sc, one)
        res['image'].append(((m1 + m2) / 2).cpu().numpy())
    return {k: np.concatenate(v) for k, v in res.items()}


def evaluate_split(net, D, **kw):
    return {k: np.abs(v - D['y']).mean(0) for k, v in predict_split(net, D, **kw).items()}


def evaluate():
    from model.bodym.train_bodym import load_split
    nets = []
    for p in member_paths():
        n = HybridNet().to(DEV)
        n.load_state_dict(torch.load(p, map_location=DEV, weights_only=True))
        nets.append(n)
    print(f'ensemble of {len(nets)} model(s)')

    def ens(D, **kw):
        preds = [predict_split(n, D, **kw) for n in nets]
        return {k: np.mean([p[k] for p in preds], 0) for k in preds[0]}, preds

    for split in ('testA', 'testB'):
        D = load_split(split)
        e, members = ens(D)
        e_noisy, _ = ens(D, real_aug=True)
        e_now, _ = ens(D, use_weight=False)
        mae = lambda p: np.abs(p - D['y']).mean(0)
        cols = {'prior h+w': mae(e['prior'])}
        for i, m in enumerate(members):
            cols[f'model {i}'] = mae(m['image'])
        cols['ENSEMBLE'] = mae(e['image'])
        cols['ens,noisy'] = mae(e_noisy['image'])
        cols['ens,no w'] = mae(e_now['image'])
        print(f'\n== {split}: {len(set(D["subj"]))} people - MAE (cm)')
        print(f'{"":20s}' + ''.join(f'{k:>12s}' for k in cols))
        for j, m in enumerate(MEASUREMENTS):
            print(f'{m:20s}' + ''.join(f'{v[j]:12.2f}' for v in cols.values()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--seed', type=int, default=0, help='ensemble member to train')
    ap.add_argument('--no-eval', action='store_true')
    a = ap.parse_args()
    if not a.eval_only:
        train(a.epochs, a.seed)
    if not a.no_eval:
        evaluate()
