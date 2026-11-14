"""
End-to-end check of the shop offset inside the real pipeline (run_request), on the
odjag people, leave-one-person-out: offsets learned from 5 people with the app's
own fit_shop_offsets, then the 6th person is measured by the app with them.

Also writes app_data/shop_extra/odjag.csv (the 6 people as shop data for
scripts/retrain_from_shop.py).

  python experiments/odjag/shop_offset_e2e.py
"""

import json
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
from model.calibration import fit_shop_offsets, app_value
from model.hmr.hmr_body import HMRBody
from model.hmr.body_points import load_rings
from model.hmr.hmr_worker import run_request, _load_hybrid, _load_shapy, _load_camerahmr

DATA = data_path('odjag')
KEYS = ['chest', 'waist', 'hip', 'neck', 'thigh', 'sleeve_length', 'shoulder_width']


def main():
    people = pd.read_csv(os.path.join(DATA, 'people.csv'))
    people = people[people.has_photo].reset_index(drop=True)
    hb, rings = HMRBody(), load_rings()
    hy, sh, cam = _load_hybrid(), _load_shapy(), _load_camerahmr()
    req = lambda r, path: {'height_cm': float(r.height_cm), 'weight_kg': None, 'gender': 'unknown',
                           'frames': [{'path': os.path.join(DATA, r.person_id, 'front.jpg'), 'view': 'FRONT'}],
                           'calibrate': True, 'sigma': False, 'shop_offset_path': path}
    # 1. one plain app run per person -> full components (shop rows)
    rows, plain = [], {}
    for r in people.itertuples():
        res = run_request(hb, rings, req(r, None), hy, sh, None, cam)
        plain[r.person_id] = res['measurements']
        rows.append({'person_id': r.person_id, 'mode': 'no_weight', 'height_cm': r.height_cm, 'gender': 'unknown',
                     **res['components'], **{f'true:{k}': float(getattr(r, k)) for k in KEYS}})
        # app_value must reproduce what the app showed
        for k in KEYS:
            assert abs(app_value(res['components'], k, False) - res['measurements'][k]) < 0.11, (r.person_id, k)
    os.makedirs(data_path('shop_extra'), exist_ok=True)
    pd.DataFrame(rows).to_csv(data_path('shop_extra', 'odjag.csv'), index=False)
    # 2. leave one out through the real pipeline
    tmp = os.path.join(HERE, '_shop_tmp.json')
    out = []
    for i, r in enumerate(people.itertuples()):
        json.dump(fit_shop_offsets([x for j, x in enumerate(rows) if j != i]), open(tmp, 'w'))
        m = run_request(hb, rings, req(r, tmp), hy, sh, None, cam)['measurements']
        for k in KEYS:
            y = float(getattr(r, k))
            out.append({'person_id': r.person_id, 'measurement': k, 'tape': y, 'app': plain[r.person_id][k],
                        'app + shop offset': m[k]})
    os.remove(tmp)
    hb.close()
    d = pd.DataFrame(out)
    d.to_csv(os.path.join(HERE, 'shop_offset_e2e.csv'), index=False)
    s = []
    for k, g in d.groupby('measurement', sort=False):
        e1, e2 = (g.app - g.tape).abs(), (g['app + shop offset'] - g.tape).abs()
        s.append({'measurement': k, 'app MAE': e1.mean(), 'with shop offset MAE': e2.mean(),
                  'app acc %': 100 - (e1 / g.tape * 100).mean(), 'with shop offset acc %': 100 - (e2 / g.tape * 100).mean()})
    s = pd.DataFrame(s).set_index('measurement').round(2)
    print(s.to_string())
    print('overall accuracy: app %.2f%%  ->  with shop offset %.2f%%' % (s['app acc %'].mean(), s['with shop offset acc %'].mean()))


if __name__ == '__main__':
    main()
