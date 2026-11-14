"""
Accuracy of each photo → Amazon-silhouette variant (experiments/photo_to_amazon/run.py).

  A. The CNN alone: per measurement, nested-CV offset (same rule as the app), real24.
  B. The whole TailorCam hybrid with the CNN column replaced by each variant (real24).
  C. HBW: raw error of the CNN alone (10 people, too few for an offset).

When a variant gives no silhouette for a person (V2 with the arms down), the
app falls back to the prior, so the prior's value is used for that person.

  python experiments/photo_to_amazon/evaluate.py
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
sys.path[:0] = [ROOT, os.path.join(ROOT, 'scripts')]
import model.calibration as C
from model_accuracy import cv_predict
from repeated_cv import first

KEYS = ('chest', 'waist', 'hip', 'thigh', 'arm-length', 'shoulder-breadth')
TK = {'chest': 'chest', 'waist': 'waist', 'hip': 'hip', 'thigh': 'thigh', 'arm-length': 'sleeve_length',
      'shoulder-breadth': 'shoulder_width'}


def filled(d):
    """One row per person / mode / variant; missing variants fall back to the prior (like the app)."""
    base = d[d.variant == 'V0 prior'].set_index(['dataset', 'person_id', 'mode'])
    out = []
    for v in sorted(d.variant.unique()):
        x = d[d.variant == v].set_index(['dataset', 'person_id', 'mode'])
        miss = base.index.difference(x.index)
        x = pd.concat([x, base.loc[miss].assign(variant=v)])
        out.append(x.reset_index())
    return pd.concat(out)


def main():
    d = filled(pd.read_csv(os.path.join(HERE, 'predictions.csv')))
    variants = sorted(d.variant.unique())
    pd.set_option('display.width', 250)

    print('\nA. CNN alone on 24 real people (nested-CV offset), MAE cm / accuracy %')
    for mode in ('with_weight', 'no_weight'):
        rows = []
        for v in variants:
            x = d[(d.dataset == 'real24') & (d['mode'] == mode) & (d.variant == v)]
            r, accs = {'variant': v}, []
            for k in KEYS:
                ok = x[f'pred:{k}'].notna() & x[f'true:{k}'].notna()
                X, y = x.loc[ok, [f'pred:{k}']].values, x.loc[ok, f'true:{k}'].values.astype(float)
                e = np.abs(cv_predict(X, y) - y).mean(0)
                r[k] = f'{e.mean():.2f}'
                accs.append(100 - (e / y * 100).mean())
            r['avg acc %'] = round(float(np.mean(accs)), 2)
            rows.append(r)
        print(f'-- {mode}'); print(pd.DataFrame(rows).set_index('variant').to_string())

    print('\nB. Whole hybrid with the CNN column replaced, 24 real people, overall accuracy %')
    real = pd.read_csv(data_path('predictions', 'real24_before_bodyfit.csv'))
    for mode in ('with_weight', 'no_weight'):
        targets = C.TARGETS if mode == 'with_weight' else first('camerahmr', C.TARGETS)
        res = {}
        for v in ['app (current)'] + variants:
            r = real[real['mode'] == mode].copy()
            if v != 'app (current)':
                x = d[(d.dataset == 'real24') & (d['mode'] == mode) & (d.variant == v)].set_index('person_id')
                for k in KEYS:
                    col = f'hybrid:{k}'
                    if col in r:
                        r[col] = r.person_id.map(x[f'pred:{k}'])
            accs, maes = [], {}
            for t in ('chest', 'waist', 'hip', 'thigh', 'neck', 'sleeve_length'):
                c = [c for c in targets[t] if c in r and r[c].notna().any()]
                ok = r[c + [f'true:{t}']].notna().all(axis=1)
                X, y = r.loc[ok, c].values.astype(float), r.loc[ok, f'true:{t}'].values.astype(float)
                e = np.abs(cv_predict(X, y) - y).mean(0)
                maes[t] = round(float(e.mean()), 2)
                accs.append(100 - (e / y * 100).mean())
            res[v] = {**maes, 'overall %': round(float(np.mean(accs)), 2)}
        print(f'-- {mode}'); print(pd.DataFrame(res).T.to_string())

    print('\nC. HBW (10 people, 3D-scan truth), CNN alone, raw MAE cm')
    for mode in ('with_weight', 'no_weight'):
        rows = []
        for v in variants:
            x = d[(d.dataset == 'hbw') & (d['mode'] == mode) & (d.variant == v)]
            if len(x):
                rows.append({'variant': v, 'people': len(x), **{k: round(float(np.abs(x[f'pred:{k}'] - x[f'true:{k}']).mean()), 2)
                                                                for k in ('chest', 'waist', 'hip')}})
        print(f'-- {mode}'); print(pd.DataFrame(rows).set_index('variant').to_string())

    x = d[(d.dataset == 'real24') & (d.variant == 'V0 prior') & (d['mode'] == 'with_weight')]
    print(f'\nreal24: {int((x.arms_away >= 0.45).sum())} of {len(x)} people have the arms away from the body')


if __name__ == '__main__':
    main()
