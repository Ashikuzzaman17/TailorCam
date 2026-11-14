"""
Honest comparison of combiner settings on the real tape-measured people.

Each setting is scored with nested cross-validation (6 outer test folds,
leave-one-out validation inside) repeated over 5 different random splits, so
a change only counts if it helps on average — not because of one lucky split.

  python scripts/repeated_cv.py
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
sys.path.insert(0, ROOT)
import model.calibration as C

SEEDS = range(5)
KEYS = ('chest', 'waist', 'hip', 'thigh', 'neck', 'sleeve_length')


def first(prefix, targets):
    """Put '<prefix>:<target>' first (it becomes the default model) where available."""
    return {k: ([f'{prefix}:{k}'] + [c for c in v if c != f'{prefix}:{k}'] if f'{prefix}:{k}' in v else v)
            for k, v in targets.items()}


def score(df, mode, targets, scale=False, seeds=SEEDS):
    """→ {target: (mean MAE over splits, mean % accuracy, mean within-2cm %)}."""
    C.ENABLE_SCALE = scale
    x = df[df['mode'] == mode]
    out = {}
    for t in KEYS:
        if t not in targets:
            continue
        comps = [c for c in targets[t] if c in x and x[c].notna().any()]
        u = x[x[comps + [f'true:{t}']].notna().all(axis=1)] if f'true:{t}' in x else x.iloc[:0]
        if len(u) < 8:
            continue
        X, y = u[comps].values.astype(float), u[f'true:{t}'].values.astype(float)
        maes, accs, w2 = [], [], []
        for s in seeds:
            p = C.nested_cv_app(X, y, 6, seed=s)      # fit_all's rule, decided on training people only
            e = np.abs(p - y)
            maes.append(e.mean()); accs.append(100 - (e / y * 100).mean()); w2.append((e <= 2).mean() * 100)
        out[t] = (round(float(np.mean(maes)), 2), round(float(np.mean(accs)), 1), round(float(np.mean(w2))))
    C.ENABLE_SCALE = False
    return out


def table(results):
    rows = []
    for name, res in results.items():
        r = {'setting': name}
        for t, (mae, acc, w2) in res.items():
            r[t] = f'{mae:.2f} ({acc:.1f}%)'
        rows.append(r)
    return pd.DataFrame(rows).set_index('setting')


if __name__ == '__main__':
    d = pd.read_csv(data_path('predictions', 'real24.csv'))
    T = C.TARGETS
    for mode, configs in (
            ('with_weight', {'current (hybrid first)': (T, False),
                             '+ scale': (T, True),
                             'CameraHMR first': (first('camerahmr', T), False),
                             'CameraHMR first + scale': (first('camerahmr', T), True)}),
            ('no_weight', {'current (CameraHMR first)': (first('camerahmr', T), False),
                           '+ scale': (first('camerahmr', T), True)})):
        res = {n: score(d, mode, t, s) for n, (t, s) in configs.items()}
        print(f'\n== {mode}: MAE cm (% accuracy), mean of {len(SEEDS)} splits')
        print(table(res).to_string())
