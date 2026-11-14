"""
Combiner trained on ALL datasets at once (real24 + HBW + BEDLAM + agency).

The standard combiner (model/calibration.py) learns from the 24 real-photo
people only. This one also learns from the other datasets: the blend weights
are shared, but every dataset gets its own offset (each one measures chest /
waist / hip a little differently), and the app uses the real24 offset.

Model, per measurement:  pred = mean(models) + Σ b_j (model_j − mean) + offset[dataset]
with ridge strength `alpha` on b and weight `lam` on the external datasets;
(lam, alpha) are chosen by leave-one-out on real24 people only.

Honest test: nested 6-fold cross-validation on real24 (the external data is
always in training; settings chosen without the test people). It replaces the
standard combiner for a measurement only where that test is better.

Tested 2026-10-20 with 5 repeated nested-CV splits: NOT better than the
standard combiner (with weight: chest 2.57 vs 2.53, waist 3.48 vs 3.41, hip 3.58
vs 3.15; no weight: chest 4.19 vs 3.71, waist 4.04 vs 3.72, hip 5.04 vs 5.21).
The other datasets measure differently from tape-measured real people, so the
app keeps the standard combiner. A single split can make it look better —
always compare over several splits.

  python -m model.pooled_combiner          # report only
  python -m model.pooled_combiner --apply  # also write it into the calibration file
"""

import os
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from paths import data_path, DATASETS  # noqa: E402  (dataset/, assets/, app_data/)
PRED = data_path('predictions')
MODELS = ['camerahmr', 'shapy', 'hybrid', 'prior', 'hmr', 'ansur']
DATASETS = ('real24', 'hbw_val', 'bedlam', 'agency')
CFG = [(lam, a) for lam in (0.0, 0.1, 0.3, 1.0) for a in (10.0, 100.0, 1000.0)]


def _table(mode, target):
    frames = []
    for ds in DATASETS:
        p = os.path.join(PRED, f'{ds}.csv')
        if os.path.exists(p):
            d = pd.read_csv(p)
            frames.append(d[d['mode'] == mode].assign(ds=ds))
    d = pd.concat(frames, ignore_index=True)
    cols = [f'{m}:{target}' for m in MODELS]
    d = d[d[cols + [f'true:{target}']].notna().all(axis=1)].reset_index(drop=True)
    return d, cols


def _fit(X, y, ds, tr, lam, alpha):
    """→ (linear weights v over models, real24 offset)."""
    w = np.where(ds[tr] == 'real24', 1.0, lam)
    tr, w = tr[w > 0], w[w > 0]
    names = sorted(set(ds[tr]))
    D = np.stack([(ds[tr] == s).astype(float) for s in names], 1)
    Xm = X[tr] - X[tr].mean(1, keepdims=True)
    A = np.hstack([Xm, D])
    reg = np.diag([alpha] * X.shape[1] + [1e-6] * len(names))
    c = np.linalg.solve(A.T @ (A * w[:, None]) + reg, A.T @ (w * (y[tr] - X[tr].mean(1))))
    b = c[:X.shape[1]]
    v = 1.0 / X.shape[1] + b - b.sum() / X.shape[1]      # pred = X @ v + offset  (Σv = 1)
    return v, float(c[X.shape[1] + names.index('real24')])


def _choose(X, y, ds, inner, ext):
    best = None
    for lam, a in CFG:
        errs = []
        for i in inner:
            v, off = _fit(X, y, ds, np.r_[np.setdiff1d(inner, [i]), ext], lam, a)
            errs.append(abs(X[i] @ v + off - y[i]))
        m = float(np.mean(errs))
        if best is None or m < best[0]:
            best = (m, lam, a)
    return best[1], best[2]


def evaluate(mode, target, folds=6, seed=0):
    """Nested CV on real24 → (test errors, final entry for the calibration file)."""
    d, cols = _table(mode, target)
    X, y, ds = d[cols].values.astype(float), d[f'true:{target}'].values.astype(float), d.ds.values
    r24, ext = np.flatnonzero(ds == 'real24'), np.flatnonzero(ds != 'real24')
    fold = np.empty(len(r24), int)
    fold[np.random.default_rng(seed).permutation(len(r24))] = np.arange(len(r24)) % folds
    pred = np.empty(len(r24))
    for f in range(folds):
        inner = r24[fold != f]
        lam, a = _choose(X, y, ds, inner, ext)
        v, off = _fit(X, y, ds, np.r_[inner, ext], lam, a)
        pred[fold == f] = X[r24[fold == f]] @ v + off
    err = np.abs(pred - y[r24])
    lam, a = _choose(X, y, ds, r24, ext)
    v, off = _fit(X, y, ds, np.r_[r24, ext], lam, a)
    entry = {'components': cols, 'kind': 'pooled', 'alpha': a, 'external_weight': lam,
             'bias': off, 'w': [float(x) for x in v[1:]],          # same format as 'blend'
             'n': int(len(r24)), 'n_external': int(len(ext)),
             'test_mae': round(float(err.mean()), 2), 'test_p80': round(float(np.percentile(err, 80)), 2)}
    return err, entry


def improve(calib, mode, targets=('chest', 'waist', 'hip')):
    """Replace a target's entry by the pooled combiner where the nested test is better."""
    out, report = dict(calib), []
    for t in targets:
        if t not in calib:
            continue
        cur = calib[t].get('test_mae', calib[t].get('test_mae_after'))
        _, e = evaluate(mode, t)
        use = e['test_mae'] < cur - 0.05
        if use:
            out[t] = e
        report.append({'mode': mode, 'measurement': t, 'standard combiner (cm)': cur,
                       'all-datasets combiner (cm)': e['test_mae'],
                       'external rows': e['n_external'], 'used': 'all-datasets' if use else 'standard'})
    return out, report


if __name__ == '__main__':
    import sys
    from model.calibration import _load, save
    allc, rows = _load(), []
    for mode in ('with_weight', 'no_weight'):
        new, rep = improve(allc.get(mode, {}), mode)
        if '--apply' in sys.argv:
            save(new, mode)
        rows += rep
    print(pd.DataFrame(rows).to_string(index=False))
