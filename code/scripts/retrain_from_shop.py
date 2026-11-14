"""
Adapt TailorCam to your shop with your own tape-measured customers.

Workflow in the app:
  1. Scan the customer (enter weight, arms away from the body, fitted clothes).
  2. Tape-measure them and enter the real values with the Feedback button.
  3. After 5+ customers, run:   python scripts/retrain_from_shop.py

Default (recommended): learn one offset per measurement for YOUR way of taping and
save it to model/calibration/shop_offset.json. The app adds it after the combiner.
A measurement is only corrected when the offset lowered the error in
leave-one-customer-out validation. On 6 test people from a local tailor this took
overall accuracy from 92.2 % to 96.7 % (experiments/odjag/shop_fix.py).
Delete shop_offset.json to switch it off.

Extra shop data in the same row format (components + true:<key> + mode) can be put
in app_data/shop_extra/*.csv, e.g. the odjag test people.

  python scripts/retrain_from_shop.py            # shop offsets (default)
  python scripts/retrain_from_shop.py --mixed    # old way: refit the combiner on
                                                 # the 24 public people + your customers
"""

import os
import sys
import glob
import json

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
sys.path.insert(0, ROOT)

from model.calibration import fit_all, save, TARGETS, CALIB_PATH, _load, fit_shop_offsets, SHOP_PATH, SHOP_MIN
from model.shop_data import load_rows


def to_rows(df):
    return [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
            for r in df.to_dict('records')]


def load_extra():
    rows = []
    for p in sorted(glob.glob(data_path('shop_extra', '*.csv'))):
        rows += to_rows(pd.read_csv(p))
        print(f'extra shop data: {os.path.relpath(p, ROOT)}')
    return rows


def shop_offsets(shop):
    res = fit_shop_offsets(shop)
    for mode, r in res.items():
        print(f'\n== {mode}: {sum(1 for x in shop if x.get("mode") == mode)} customers')
        if not r:
            print(f'   no measurement has {SHOP_MIN}+ customers yet')
        for t, v in r.items():
            print(f'   {t:15s} n={v["n"]:3d}  offset {v["offset"]:+6.2f} cm  error {v["mae_before"]:5.2f} -> '
                  f'{v["mae_after_loo"]:5.2f} cm  {"USED" if v["used"] else "not used (did not help)"}')
    with open(SHOP_PATH, 'w') as f:
        json.dump(res, f, indent=1)
    print('\nSaved ->', SHOP_PATH)


def mixed_refit(shop):
    pub = to_rows(pd.read_csv(data_path('predictions', 'real24.csv')))
    nw_targets = {k: (['camerahmr:' + k] + [c for c in v if c != 'camerahmr:' + k]
                      if 'camerahmr:' + k in v else v) for k, v in TARGETS.items()}
    old = _load()
    for mode, targets in (('with_weight', TARGETS), ('no_weight', nw_targets)):
        # (a with-weight scan cannot teach the no-weight case: its 3D answers were
        #  already adjusted to the weight)
        rows = [r for r in pub + shop if r.get('mode') == mode]
        calib, rep = fit_all(rows, folds=min(6, max(3, len(rows) // 5)), targets=targets)
        print(f'\n== {mode}: {len(rows)} people')
        keep = dict(old.get(mode, {}))
        for t, r in rep.items():
            f = r['final']
            o = old.get(mode, {}).get(t, {})
            prev = o.get('test_mae', o.get('test_mae_after'))
            better = prev is None or f['test_mae'] <= prev + 0.05
            print(f'   {t:15s} n={r["n"]:3d}  CV MAE {f["test_mae"]:5.2f} (80% within {f["test_p80"]:.1f})'
                  f'  (previous {prev})  {"UPDATED" if better else "kept previous"}')
            if better:
                keep[t] = f
        save(keep, mode)
    print('\nSaved ->', CALIB_PATH)


def main():
    shop = load_rows() + load_extra()
    print(f'Customers with tape values: {len(shop)}')
    if len(shop) < SHOP_MIN:
        print(f'Need at least {SHOP_MIN} measured customers - keep collecting.')
        return
    (mixed_refit if '--mixed' in sys.argv else shop_offsets)(shop)


if __name__ == '__main__':
    main()
