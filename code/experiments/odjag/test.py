"""
Test TailorCam (the app's own rules, nothing retrained) on the odjag set:
6 people, one front photo each, height from the tape sheet, no weight.

  python experiments/odjag/test.py    -> results.csv, summary printed
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
from model.hmr.hmr_body import HMRBody
from model.hmr.body_points import load_rings
from model.hmr.hmr_worker import run_request, _load_hybrid, _load_shapy, _load_camerahmr

DATA = data_path('odjag')
KEYS = ['chest', 'waist', 'hip', 'neck', 'thigh', 'sleeve_length', 'shoulder_width']
MODELS = {'hmr': 'HMR', 'shapy': 'SHAPY', 'camerahmr': 'CameraHMR', 'hybrid': 'Silhouette CNN', 'prior': 'BodyM prior',
          'ansur': 'ANSUR prior'}
BODYM = {'sleeve_length': 'arm-length', 'shoulder_width': 'shoulder-breadth'}


def comp(c, m, k):
    key = f'{m}:{BODYM.get(k, k)}' if m in ('hybrid', 'prior') else f'{m}:{k}'
    return c.get(key, np.nan)


def main():
    people = pd.read_csv(os.path.join(DATA, 'people.csv'))
    people = people[people.has_photo]
    hb, rings = HMRBody(), load_rings()
    hy, sh, cam = _load_hybrid(), _load_shapy(), _load_camerahmr()
    rows = []
    for r in people.itertuples():
        res = run_request(hb, rings, {'height_cm': float(r.height_cm), 'weight_kg': None, 'gender': 'unknown',
                                      'frames': [{'path': os.path.join(DATA, r.person_id, 'front.jpg'), 'view': 'FRONT'}],
                                      'calibrate': True, 'sigma': False, 'shop_offset_path': None}, hy, sh, None, cam)   # no shop offset: plain app
        if not res['ok']:
            print(r.person_id, res.get('error')); continue
        m, c = res['measurements'], res['components']
        for k in KEYS:
            row = {'person_id': r.person_id, 'measurement': k, 'tape': getattr(r, k), 'TailorCam': m.get(k, np.nan)}
            row.update({name: comp(c, mk, k) for mk, name in MODELS.items()})
            rows.append(row)
        rows.append({'person_id': r.person_id, 'measurement': 'belly (vs our waist)', 'tape': r.belly,
                     'TailorCam': m.get('waist', np.nan), **{name: comp(c, mk, 'waist') for mk, name in MODELS.items()}})
        print(r.person_id, 'done', flush=True)
    hb.close()
    d = pd.DataFrame(rows)
    d.round(2).to_csv(os.path.join(HERE, 'results.csv'), index=False)
    cols = ['TailorCam'] + list(MODELS.values())
    out = []
    for k, g in d.groupby('measurement', sort=False):
        e = {c_: (g[c_] - g['tape']).abs().mean() for c_ in cols}
        a = {c_: 100 - ((g[c_] - g['tape']).abs() / g['tape'] * 100).mean() for c_ in cols}
        out.append({'measurement': k, 'people': int(g['tape'].notna().sum()),
                    **{f'{c_} MAE': e[c_] for c_ in cols}, 'TailorCam acc %': a['TailorCam'],
                    'TailorCam bias': (g['TailorCam'] - g['tape']).mean()})
    s = pd.DataFrame(out).set_index('measurement').round(2)
    s.to_csv(os.path.join(HERE, 'summary.csv'))
    pd.set_option('display.width', 250)
    print(s.to_string())
    print('\nper person (TailorCam vs tape, cm):')
    print(d.pivot_table(index='person_id', columns='measurement', values=['tape', 'TailorCam']).round(1).to_string())


if __name__ == '__main__':
    main()
