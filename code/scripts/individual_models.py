"""
Every part of TailorCam tested on its own, on the same people.

  raw      the model's answer as it comes out
  +offset  the model plus one learned offset per measurement (each model measures
           at a slightly different place on the body). The offset is learned on the
           training folds only: 6 folds x 5 repeated splits, like the combiner.

Datasets: real24 (tape, weight known / unknown), HBW (3D scans), model agency test,
plus the three silhouette CNNs one by one on BodyM testB.

Output: reports/paper/tables/individual_models*.csv

  python scripts/individual_models.py
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
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import model.calibration as C
from model.calibration import apply_calibration
from repeated_cv import first

OUT = os.path.join(ROOT, 'reports', 'paper', 'tables')
PRED = data_path('predictions')
KEYS = ('chest', 'waist', 'hip', 'neck', 'thigh', 'sleeve_length')
MODELS = {'HMR (2018)': 'hmr', 'SHAPY (2022)': 'shapy', 'CameraHMR (2025)': 'camerahmr',
          'Silhouette CNN x3 (ours)': 'hybrid', 'BodyM prior (ours)': 'prior', 'ANSUR prior (ours)': 'ansur'}
SEEDS = range(5)


def col(model, t, d):
    c = f'{model}:{t}'
    if model == 'hybrid' and t == 'sleeve_length':
        c = 'hybrid:arm-length'
    if model == 'hybrid' and t == 'shoulder_width':
        c = 'hybrid:shoulder-breadth'
    return c if c in d and d[c].notna().any() else None


def offset_cv(x, y, seeds=SEEDS):
    """Nested CV of 'keep' vs 'model + offset' for one model → mean abs error per person."""
    return np.mean([np.abs(C.nested_cv_app(x[:, None], y, 6, seed=s) - y) for s in seeds], 0)


def combiner_cv(d, mode, t, targets):
    comps = [c for c in targets[t] if c in d and d[c].notna().any()]
    u = d[d[comps + [f'true:{t}']].notna().all(axis=1)]
    X, y = u[comps].values.astype(float), u[f'true:{t}'].values.astype(float)
    return np.mean([np.abs(C.nested_cv_app(X, y, 6, seed=s) - y) for s in SEEDS], 0), y


def real24_table(mode, targets):
    d = pd.read_csv(os.path.join(PRED, 'real24.csv'))
    d = d[d['mode'] == mode]
    rows = []
    for name, m in list(MODELS.items()) + [('Combiner = TailorCam (ours)', None)]:
        r = {'model': name}
        accs = []
        for t in KEYS:
            if f'true:{t}' not in d:
                continue
            if m is None:
                e, y = combiner_cv(d, mode, t, targets)
                r[f'{t} raw'] = np.nan
                r[f'{t} +offset'] = e.mean()
                accs.append(100 - (e / y * 100).mean())
                continue
            c = col(m, t, d)
            if c is None:
                continue
            u = d[d[[c, f'true:{t}']].notna().all(axis=1)]
            x, y = u[c].values.astype(float), u[f'true:{t}'].values.astype(float)
            r[f'{t} raw'] = np.abs(x - y).mean()
            e = offset_cv(x, y)
            r[f'{t} +offset'] = e.mean()
            accs.append(100 - (e / y * 100).mean())
        r['avg accuracy +offset (%)'] = np.mean(accs) if accs else np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index('model')


def external_table():
    rows = []
    for ds, mode, split in (('hbw_val', 'with_weight', None), ('hbw_val', 'no_weight', None), ('agency', 'no_weight', 'test')):
        d = pd.read_csv(os.path.join(PRED, f'{ds}.csv'))
        d = d[d['mode'] == mode]
        if split:
            d = d[d.split == split]
        for name, m in list(MODELS.items()) + [('Combiner = TailorCam (ours)', None)]:
            r = {'dataset': f'{ds} ({mode.replace("_", " ")})', 'model': name}
            for t in ('chest', 'waist', 'hip'):
                if m is None:
                    p = [apply_calibration({k: v for k, v in rr.items() if ':' in k and not k.startswith('true:') and pd.notna(v)},
                                           weight_known=(mode == 'with_weight')).get(t) for rr in d.to_dict('records')]
                    p = np.array([np.nan if v is None else v for v in p], float)
                else:
                    c = col(m, t, d)
                    if c is None:
                        continue
                    p = d[c].values.astype(float)
                y = d[f'true:{t}'].values.astype(float)
                ok = ~np.isnan(p) & ~np.isnan(y)
                r[t] = np.abs(p[ok] - y[ok]).mean()
            rows.append(r)
    return pd.DataFrame(rows).set_index(['dataset', 'model'])


def cnn_members():
    import torch
    from model.bodym.bodym_net import MEASUREMENTS
    from model.bodym.train_bodym import load_split
    from model.bodym.hybrid import HybridNet, member_paths, predict_split, DEV
    D = load_split('testB')
    rows, preds = [], []
    for i, p in enumerate(member_paths()):
        net = HybridNet().to(DEV)
        net.load_state_dict(torch.load(p, map_location=DEV, weights_only=True))
        pr = predict_split(net, D)['image']
        preds.append(pr)
        rows.append({'model': f'CNN copy {i + 1} (seed {i})', **{m: np.abs(pr[:, MEASUREMENTS.index(m)] - D['y'][:, MEASUREMENTS.index(m)]).mean()
                                                          for m in ('chest', 'waist', 'hip')}})
    ens = np.mean(preds, 0)
    rows.append({'model': 'Ensemble (average of 3)', **{m: np.abs(ens[:, MEASUREMENTS.index(m)] - D['y'][:, MEASUREMENTS.index(m)]).mean()
                                                       for m in ('chest', 'waist', 'hip')}})
    return pd.DataFrame(rows).set_index('model')


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    os.makedirs(OUT, exist_ok=True)
    for mode, T in (('with_weight', C.TARGETS), ('no_weight', first('camerahmr', C.TARGETS))):
        t = real24_table(mode, T).round(2)
        t.to_csv(os.path.join(OUT, f'individual_models_real24_{mode}.csv'))
        print(f'\n== real24, {mode}: MAE cm (raw = as is; +offset = with learned offset, nested CV)')
        print(t.to_string())
    e = external_table().round(2)
    e.to_csv(os.path.join(OUT, 'individual_models_external.csv'))
    print('\n== external test sets, raw MAE cm (combiner = final app value)'); print(e.to_string())
    c = cnn_members().round(2)
    c.to_csv(os.path.join(OUT, 'individual_models_cnn_members.csv'))
    print('\n== silhouette CNN copies on BodyM testB (400 people), MAE cm'); print(c.to_string())
