"""
Fill the remaining model x dataset gaps that the data allows:
  A. BodyM silhouette CNN + BodyM prior + ANSUR prior on BODY-fit (test split) and NOMO-3D
  B. BODY-fit network on Amazon BodyM testA / testB

Datasets measure in different ways (BodyM scan definitions, TC2 scanner, tape), so
each result is given raw and after one offset per measurement learned on the
TRAINING split of the target dataset (BODY-fit train for BODY-fit / NOMO, BodyM
train for BodyM), so no test person is used to fit anything.

  python experiments/cross_dataset/run.py      -> results.csv
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
sys.path.insert(0, ROOT)
from model.bodym.hybrid import HybridNet, member_paths, DEV
from model.bodym.bodym_net import MEASUREMENTS, scalar_features
from model.bodym.train_bodym import load_split
from model.ansur_prior import AnsurPrior
import model.bodyfit.train_bodyfit as BF

# common measurements: name -> (BodyM key, BODY-fit key)
COMMON = {'chest': ('chest', 'chest'), 'waist': ('waist', 'waist'), 'hip': ('hip', 'hip'), 'thigh': ('thigh', 'thigh'),
          'arm / sleeve': ('arm-length', 'sleeve_length'), 'shoulder': ('shoulder-breadth', 'shoulder_width')}
ANSUR_KEY = {'chest': 'chest', 'waist': 'waist', 'hip': 'hip', 'thigh': 'thigh', 'arm / sleeve': 'sleeve_length',
             'shoulder': 'shoulder_width'}


@torch.no_grad()
def bodym_cnn(D, idx, use_weight):
    """BodyM ensemble on 256x192 silhouettes already prepared (BODY-fit / NOMO caches)."""
    nets = []
    for p in member_paths():
        n = HybridNet().to(DEV); n.load_state_dict(torch.load(p, map_location=DEV, weights_only=True)); n.eval(); nets.append(n)
    out, prior = [], []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        f = torch.from_numpy(D['front'][j]).float()[:, None].to(DEV)
        s = torch.from_numpy(D['side'][j]).float()[:, None].to(DEV)
        sc = torch.from_numpy(np.array([scalar_features(float(D['h'][k]), float(D['w'][k]) if use_weight else None,
                                                        'male' if D['g'][k] == 1 else 'female') for k in j], np.float32)).to(DEV)
        ok = torch.ones(len(j), device=DEV)
        out.append(np.mean([n(f, s, sc, ok)[0].cpu().numpy() for n in nets], 0))
        prior.append(np.mean([n.prior(sc).cpu().numpy() for n in nets], 0))
    return np.concatenate(out), np.concatenate(prior)


def score(rows, ds, model, mode, pred_tr, y_tr, pred_te, y_te, keys):
    for name, k in keys.items():
        a, b = pred_te[name], y_te[name]
        ok = ~np.isnan(a) & ~np.isnan(b)
        off = np.nanmean(y_tr[name] - pred_tr[name]) if pred_tr is not None else 0.0
        rows.append({'dataset': ds, 'model': model, 'mode': mode, 'measurement': name, 'people': int(ok.sum()),
                     'raw MAE': np.abs(a[ok] - b[ok]).mean(), 'MAE after offset': np.abs(a[ok] + off - b[ok]).mean(),
                     'accuracy after offset': 100 - (np.abs(a[ok] + off - b[ok]) / b[ok] * 100).mean()})


def main():
    rows = []
    ap = AnsurPrior()
    bf = dict(np.load(BF.DATA))
    nomo = dict(np.load(data_path('nomo3d', 'cache.npz')))
    tr, te = np.where(bf['split'] == 0)[0], np.where(bf['split'] == 2)[0]
    truth = lambda D, idx: {n: D['y'][idx, BF.KEYS.index(k[1])] for n, k in COMMON.items()}
    # A. BodyM CNN, BodyM prior, ANSUR prior on BODY-fit and NOMO-3D
    for use_w in (True, False):
        mode = 'with weight' if use_w else 'no weight'
        c_tr, p_tr = bodym_cnn(bf, tr, use_w)
        a_tr = {n: np.array([ap.predict(float(bf['h'][i]), float(bf['w'][i]) if use_w else None,
                                        'male' if bf['g'][i] == 1 else 'female', None)[ANSUR_KEY[n]] for i in tr]) for n in COMMON}
        targets = [('BODY-fit test (558)', bf, te)] + ([] if use_w else [('NOMO-3D (356)', nomo, np.arange(len(nomo['h'])))])
        for ds, D, idx in targets:
            c_te, p_te = bodym_cnn(D, idx, use_w)
            for model, Ptr, Pte in (('Silhouette CNN (BodyM)', c_tr, c_te), ('BodyM prior', p_tr, p_te)):
                pick = lambda P: {n: P[:, MEASUREMENTS.index(k[0])] for n, k in COMMON.items()}
                score(rows, ds, model, mode, pick(Ptr), truth(bf, tr), pick(Pte), truth(D, idx), COMMON)
            a_te = {n: np.array([ap.predict(float(D['h'][i]), float(D['w'][i]) if use_w else None,
                                            'male' if D['g'][i] == 1 else 'female', None)[ANSUR_KEY[n]] for i in idx]) for n in COMMON}
            score(rows, ds, 'ANSUR prior', mode, a_tr, truth(bf, tr), a_te, truth(D, idx), COMMON)
        print('A done', mode, flush=True)
    # B. BODY-fit network on BodyM testA / testB (offset from BodyM train)
    nets = []
    for p in BF.member_paths():
        n = BF.BodyFitNet().to(BF.DEV); n.load_state_dict(torch.load(p, map_location=BF.DEV, weights_only=True)); nets.append(n)

    def bf_pred(D, use_w):
        DD = dict(front=D['front'], side=D['side'], h=D['h'], w=D['w'], g=np.array([1 if x == 'male' else 0 for x in D['g']]))
        idx = np.arange(len(D['h']))
        P = np.mean([BF.predict(n, DD, idx, use_w) for n in nets], 0)
        return {nm: P[:, BF.KEYS.index(k[1])] for nm, k in COMMON.items()}

    Dtr = load_split('train')
    ytr = {nm: Dtr['y'][:, MEASUREMENTS.index(k[0])] for nm, k in COMMON.items()}
    for use_w in (True, False):
        mode = 'with weight' if use_w else 'no weight'
        ptr = bf_pred(Dtr, use_w)
        for split in ('testA', 'testB'):
            D = load_split(split)
            score(rows, f'BodyM {split} ({len(D["h"])} pairs)', 'BODY-fit network', mode, ptr, ytr, bf_pred(D, use_w),
                  {nm: D['y'][:, MEASUREMENTS.index(k[0])] for nm, k in COMMON.items()}, COMMON)
        print('B done', mode, flush=True)
    r = pd.DataFrame(rows)
    r.round(3).to_csv(os.path.join(HERE, 'results.csv'), index=False)
    pd.set_option('display.width', 250)
    print(r.pivot_table(index=['dataset', 'mode', 'model'], columns='measurement', values='MAE after offset').round(2).to_string())


if __name__ == '__main__':
    main()
