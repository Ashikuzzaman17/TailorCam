"""
Does data augmentation help the combiner (which learns from only ~20 people)?

Two standard augmentations for small tabular training sets, applied ONLY to
the training rows inside every cross-validation fold (test people untouched):
  noise   K jittered copies of each person: each model's answer + Gaussian noise
          (σ = k × that model's spread of errors on the training rows)
  mixup   K blended pairs: λ·person_a + (1−λ)·person_b for answers AND truth

Scored with the same 5-split repeated nested CV as scripts/repeated_cv.py.

  python scripts/augment_experiment.py
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
from repeated_cv import score, table, first

_orig_fit = C.fit_candidate


def make_augmented_fit(kind_aug, k, copies=10, seed=0):
    def fit(kind, X, y, alpha=None):
        rng = np.random.default_rng(seed)
        if kind_aug == 'noise':
            sd = np.std(X - y[:, None], axis=0) * k
            Xa = np.vstack([X] + [X + rng.normal(0, 1, X.shape) * sd for _ in range(copies)])
            ya = np.concatenate([y] * (copies + 1))
        else:                                                   # mixup
            n = len(y)
            a, b = rng.integers(0, n, n * copies), rng.integers(0, n, n * copies)
            lam = rng.beta(k, k, n * copies)[:, None]
            Xa = np.vstack([X, lam * X[a] + (1 - lam) * X[b]])
            ya = np.concatenate([y, lam[:, 0] * y[a] + (1 - lam[:, 0]) * y[b]])
        return _orig_fit(kind, Xa, ya, alpha)
    return fit


if __name__ == '__main__':
    d = pd.read_csv(data_path('predictions', 'real24.csv'))
    for mode, T in (('with_weight', C.TARGETS), ('no_weight', first('camerahmr', C.TARGETS))):
        res = {'no augmentation': score(d, mode, T)}
        for name, aug, k in (('noise k=0.25', 'noise', 0.25), ('noise k=0.5', 'noise', 0.5),
                             ('mixup a=0.4', 'mixup', 0.4), ('mixup a=2', 'mixup', 2.0)):
            C.fit_candidate = make_augmented_fit(aug, k)
            res[name] = score(d, mode, T)
            C.fit_candidate = _orig_fit
        print(f'\n== {mode}: MAE cm (% accuracy), mean of 5 splits')
        print(table(res).to_string(), flush=True)
