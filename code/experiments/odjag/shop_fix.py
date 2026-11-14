"""
How to adapt TailorCam to a new tailor's tape style: three fixes compared on the odjag
set, leave-one-person-out (learn from 5 shop people, test on the 6th, repeat for all 6).

  A  combiner unchanged + one offset per measurement learned on the shop people
  B  combiner refitted on the 24 public people + the shop people (retrain_from_shop.py)
  C  as B, but every shop person counted 5 times

  python experiments/odjag/shop_fix.py      -> shop_fix.csv
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
sys.path.insert(0, ROOT)
from model.calibration import TARGETS, fit_all, apply_calibration, save
import model.calibration as C

KEYS = ['chest', 'waist', 'hip', 'neck', 'thigh', 'sleeve_length', 'shoulder_width']
COLS = {'HMR': 'hmr', 'SHAPY': 'shapy', 'CameraHMR': 'camerahmr', 'Silhouette CNN': 'hybrid', 'BodyM prior': 'prior',
        'ANSUR prior': 'ansur'}
BODYM = {'sleeve_length': 'arm-length', 'shoulder_width': 'shoulder-breadth'}
NW = {k: (['camerahmr:' + k] + [c for c in v if c != 'camerahmr:' + k] if 'camerahmr:' + k in v else v) for k, v in TARGETS.items()}


def shop_rows():
    d = pd.read_csv(os.path.join(HERE, 'results.csv'))
    rows = []
    for pid, g in d.groupby('person_id', sort=False):
        r = {'person_id': pid}
        for k in KEYS:
            x = g[g.measurement == k].iloc[0]
            r[f'true:{k}'] = float(x.tape)
            r[f'app:{k}'] = float(x.TailorCam)
            for name, m in COLS.items():
                if pd.notna(x[name]):
                    r[f'{m}:{BODYM.get(k, k) if m in ("hybrid", "prior") else k}'] = float(x[name])
        rows.append(r)
    return rows


def clean(df):
    return [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()} for r in df.to_dict('records')]


def main():
    shop = shop_rows()
    pub = clean(pd.read_csv(data_path('predictions', 'real24.csv')).query("mode == 'no_weight'"))
    tmp = os.path.join(HERE, '_tmp_calib.json')
    err = {m: {k: [] for k in KEYS} for m in ('app', 'A shop offset', 'B mixed refit', 'C mixed, shop x5')}
    for i, test in enumerate(shop):
        train = [r for j, r in enumerate(shop) if j != i]
        for k in KEYS:
            y = test[f'true:{k}']
            err['app'][k].append(abs(test[f'app:{k}'] - y))
            off = np.mean([r[f'true:{k}'] - r[f'app:{k}'] for r in train])
            err['A shop offset'][k].append(abs(test[f'app:{k}'] + off - y))
        for name, rep in (('B mixed refit', 1), ('C mixed, shop x5', 5)):
            calib, _ = fit_all(pub + train * rep, folds=6, targets=NW)
            save(calib, 'no_weight', path=tmp)
            C._cache['mtime'] = None
            comps = {k: v for k, v in test.items() if ':' in k and not k.startswith(('true:', 'app:'))}
            out = apply_calibration(comps, weight_known=False, path=tmp)
            for k in KEYS:
                p = out.get(k, test.get(f'hmr:{k}'))
                err[name][k].append(abs(p - test[f'true:{k}']))
    os.remove(tmp)
    ys = {k: np.array([r[f'true:{k}'] for r in shop]) for k in KEYS}
    rows = []
    for m, e in err.items():
        r = {'method': m}
        accs = []
        for k in KEYS:
            a = np.array(e[k])
            r[k] = round(float(a.mean()), 2)
            accs.append(100 - (a / ys[k] * 100).mean())
        r['overall accuracy %'] = round(float(np.mean(accs)), 2)
        rows.append(r)
    t = pd.DataFrame(rows).set_index('method')
    t.to_csv(os.path.join(HERE, 'shop_fix.csv'))
    pd.set_option('display.width', 200)
    print(t.to_string())


if __name__ == '__main__':
    main()
