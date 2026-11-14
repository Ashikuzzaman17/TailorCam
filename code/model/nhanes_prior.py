"""
Body-measurement prior from NHANES (US CDC National Health and Nutrition
Examination Survey): tens of thousands of ordinary adults measured by trained
staff with a tape. Public domain. A general-population counterpart to ANSUR II
(soldiers).

Data (dataset/nhanes, from wwwn.cdc.gov/Nchs/Data/Nhanes/Public/...):
  P_BMX / BMX_L        2017-2023: waist, hip, arm circumference
  BMX, BMX_B/C/D       1999-2006: waist, thigh, calf, arm circumference
  matching *DEMO*      gender and age
Note NHANES measures the waist just above the hip bone (iliac crest) — lower
than a tailor's waist; the combiner learns that offset.

Same features and model as model/ansur_prior.py (ridge on degree-2 polynomial
features of height, weight, BMI, age, gender; with and without weight).

Tested 2026-10-20: added to the combiner it made real24 waist worse (3.64 →
4.21 cm with weight) and nothing better, so the app does not use it.

  python -m model.nhanes_prior          # train, test, save model/nhanes_prior.joblib
"""

import os
import glob
import numpy as np

from model.ansur_prior import features

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
DATA = data_path('nhanes')
MODEL_PATH = os.path.join(_ROOT, 'model', 'nhanes_prior.joblib')

# TailorCam key → NHANES column (cm)
TARGETS = {'waist': 'BMXWAIST', 'hip': 'BMXHIP', 'thigh': 'BMXTHICR',
           'calf': 'BMXCALF', 'bicep': 'BMXARMC'}


def load_data():
    import pandas as pd
    frames = []
    for bmx in sorted(glob.glob(os.path.join(DATA, '*BMX*.xpt'))):
        demo = bmx.replace('BMX', 'DEMO')
        b, d = pd.read_sas(bmx), pd.read_sas(demo)[['SEQN', 'RIAGENDR', 'RIDAGEYR']]
        frames.append(b.merge(d, on='SEQN'))
    d = pd.concat(frames, ignore_index=True)
    d = d[(d.RIDAGEYR >= 18) & d.BMXHT.notna() & d.BMXWT.notna()].copy()
    d['height_cm'], d['weight_kg'], d['age'] = d.BMXHT, d.BMXWT, d.RIDAGEYR
    d['gender'] = np.where(d.RIAGENDR == 1, 'male', 'female')
    return d


def train(test_frac=0.2, seed=0):
    from sklearn.linear_model import Ridge
    import joblib
    d = load_data()
    is_test = np.random.default_rng(seed).random(len(d)) < test_frac
    models, report = {}, []
    for with_w in (True, False):
        w = lambda r: r.weight_kg if with_w else None
        Xk = np.array([features(r.height_cm, w(r), r.gender, r.age) for r in d.itertuples()])
        Xu = np.array([features(r.height_cm, w(r), 'unknown', r.age) for r in d.itertuples()])
        for key, col in TARGETS.items():
            ok = d[col].notna().values
            tr, te = ok & ~is_test, ok & is_test
            y = d[col].values
            # each person also seen with gender "unknown" (half weight), as in ANSUR
            m = Ridge(alpha=1.0).fit(np.vstack([Xk[tr], Xu[tr]]), np.concatenate([y[tr], y[tr]]),
                                     sample_weight=np.r_[np.ones(tr.sum()), 0.5 * np.ones(tr.sum())])
            models[(key, with_w)] = m
            err = np.abs(m.predict(Xk[te]) - y[te])
            report.append({'measurement': key, 'weight': 'known' if with_w else 'unknown',
                           'train people': int(tr.sum()), 'test people': int(te.sum()),
                           'MAE (cm)': round(err.mean(), 2), '≤3cm (%)': round((err <= 3).mean() * 100, 1)})
    joblib.dump(models, MODEL_PATH)
    return report


class NhanesPrior:
    def __init__(self, path=MODEL_PATH):
        import joblib
        self.models = joblib.load(path)

    def predict(self, height_cm, weight_kg=None, gender='unknown', age=None):
        """→ {measurement: cm}"""
        with_w = bool(weight_kg)
        x = features(height_cm, weight_kg, gender, age)[None]
        return {key: round(float(self.models[(key, with_w)].predict(x)[0]), 1) for key in TARGETS}


if __name__ == '__main__':
    import pandas as pd
    print(pd.DataFrame(train()).to_string(index=False))
    print('saved', MODEL_PATH)
