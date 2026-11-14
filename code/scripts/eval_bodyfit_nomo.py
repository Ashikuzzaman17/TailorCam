"""
Outside test of the BODY-fit network on NOMO-3D (356 real scans it never saw;
BODY-fit people who are also in NOMO were removed from its training).
Weight is not used: NOMO's raw scans are not watertight, so their volume (and
the weight derived from it) is unreliable.

Reference: the height-and-gender prior alone (no photos), same people.

  python scripts/eval_bodyfit_nomo.py
"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
sys.path.insert(0, ROOT)
from model.bodyfit.train_bodyfit import BodyFitNet, KEYS, DEV, member_paths, predict, scalars

D = dict(np.load(data_path('nomo3d', 'cache.npz')))
idx = np.arange(len(D['h']))
preds, priors = [], []
for p in member_paths():
    n = BodyFitNet().to(DEV)
    n.load_state_dict(torch.load(p, map_location=DEV, weights_only=True))
    preds.append(predict(n, D, idx, use_weight=False))
    with torch.no_grad():
        priors.append(n.prior(scalars(D, idx, np.ones(len(idx), bool))).cpu().numpy())
y = D['y']
for name, p in (('BODY-fit network (photos + height)', np.mean(preds, 0)), ('prior only (height + gender)', np.mean(priors, 0))):
    e = np.abs(p - y)
    print(f'{name}: MAE cm', {k: round(float(np.nanmean(e[:, j])), 2) for j, k in enumerate(KEYS)})
    print('   accuracy %', {k: round(float(100 - np.nanmean(e[:, j] / y[:, j] * 100)), 1) for j, k in enumerate(KEYS[:7])})
