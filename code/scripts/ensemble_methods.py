"""
Which way of combining the models' answers is best?

Compared on the real tape-measured people with the same honest protocol as the
rest of the project: 6 outer folds (test people never used for fitting),
repeated over 5 random splits. Everything (offsets, weights, choices) is learned
from the training folds only.

  current     combiner (model/calibration.py): picks a rule per measurement by LOO
  mean        average of all models + offset
  median      median of the offset-corrected models
  inv-error   offset-corrected models weighted by 1 / their training error
  nn-stack    learned blend with weights >= 0 that sum to 1 (+ offset)
  best-one    the single offset-corrected model with the lowest training error

  python scripts/ensemble_methods.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import model.calibration as C
from repeated_cv import first

KEYS = ('chest', 'waist', 'hip', 'thigh', 'neck', 'sleeve_length')


def folds_for(n, seed, k=6):
    f = np.empty(n, int)
    f[np.random.default_rng(seed).permutation(n)] = np.arange(n) % k   # same split as calibration.nested_cv
    return f


def fit_predict(method, Xtr, ytr, Xte):
    off = (ytr[:, None] - Xtr).mean(0)              # each model's average offset on training people
    Ctr, Cte = Xtr + off, Xte + off                 # offset-corrected answers
    err = np.abs(Ctr - ytr[:, None]).mean(0)
    if method == 'mean':
        return Xte.mean(1) + (ytr - Xtr.mean(1)).mean()
    if method == 'median':
        return np.median(Cte, 1)
    if method == 'inv-error':
        w = 1 / np.maximum(err, 1e-6); w /= w.sum()
        return Cte @ w
    if method == 'best-one':
        return Cte[:, int(np.argmin(err))]
    if method == 'nn-stack':                        # min ||Ctr w - y||, w >= 0, sum w = 1 (penalty row)
        lam = 100.0
        A = np.vstack([Ctr, lam * np.ones((1, Ctr.shape[1]))])
        b = np.r_[ytr, lam]
        w, _ = nnls(A, b)
        w /= max(w.sum(), 1e-9)
        return Cte @ w
    raise ValueError(method)


def evaluate(df, mode, targets, seeds=range(5)):
    x = df[df['mode'] == mode]
    rows = []
    for t in KEYS:
        comps = [c for c in targets[t] if c in x and x[c].notna().any()]
        u = x[x[comps + [f'true:{t}']].notna().all(axis=1)]
        X, y = u[comps].values.astype(float), u[f'true:{t}'].values.astype(float)
        res = {m: [] for m in ('current', 'mean', 'median', 'inv-error', 'nn-stack', 'best-one')}
        for s in seeds:
            res['current'].append(np.abs(C.nested_cv_app(X, y, 6, seed=s) - y).mean())   # leak-free app rule
            f = folds_for(len(y), s)
            for m in res:
                if m == 'current':
                    continue
                p = np.empty(len(y))
                for k in range(6):
                    te, tr = f == k, f != k
                    p[te] = fit_predict(m, X[tr], y[tr], X[te])
                res[m].append(np.abs(p - y).mean())
        rows.append({'measurement': t, 'models': len(comps), **{m: round(float(np.mean(v)), 2) for m, v in res.items()}})
    return pd.DataFrame(rows).set_index('measurement')


if __name__ == '__main__':
    d = pd.read_csv(data_path('predictions', 'real24.csv'))
    out = []
    for mode, T in (('with_weight', C.TARGETS), ('no_weight', first('camerahmr', C.TARGETS))):
        r = evaluate(d, mode, T)
        r.loc['AVERAGE'] = r.drop(columns='models').mean().round(2)
        print(f'\n== {mode}: average error in cm on unseen people (lower is better)')
        print(r.to_string())
        out.append(r.assign(mode=mode))
    os.makedirs(os.path.join(ROOT, 'reports', 'paper', 'tables'), exist_ok=True)
    pd.concat(out).to_csv(os.path.join(ROOT, 'reports', 'paper', 'tables', 'ensemble_methods.csv'))
