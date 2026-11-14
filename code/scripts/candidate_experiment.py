"""
Which set of combiner rules generalises best with ~20 tape-measured people?
More rules = more chances to pick one that only looked good by luck.
Scored with 5-split repeated nested CV (scripts/repeated_cv.py).

  python scripts/candidate_experiment.py
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import model.calibration as C
from repeated_cv import score, table, first

SETS = {'full (before)': dict(ENABLE_MEAN=False, ENABLE_BLEND=True),
        'full + mean': dict(ENABLE_MEAN=True, ENABLE_BLEND=True),
        'no blends + mean': dict(ENABLE_MEAN=True, ENABLE_BLEND=False),
        'no blends, no mean': dict(ENABLE_MEAN=False, ENABLE_BLEND=False)}

if __name__ == '__main__':
    d = pd.read_csv(data_path('predictions', 'real24.csv'))
    for mode, T in (('with_weight', C.TARGETS), ('no_weight', first('camerahmr', C.TARGETS))):
        res = {}
        for name, flags in SETS.items():
            for k, v in flags.items():
                setattr(C, k, v)
            res[name] = score(d, mode, T)
        C.ENABLE_MEAN, C.ENABLE_BLEND = True, True
        print(f'\n== {mode}: MAE cm (% accuracy), mean of 5 splits')
        print(table(res).to_string(), flush=True)
