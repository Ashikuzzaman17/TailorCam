"""
Body-measurement prior from ANSUR II (US Army anthropometric survey, 2012):
6,068 adults measured by trained staff with tape and calipers. Public domain.
Data: dataset/ansur2/ANSUR_II_{MALE,FEMALE}_Public.csv
      from https://www.openlab.psu.edu/ansur2/

Predicts tailoring measurements from height (+ weight, gender, age when
known). Two models per measurement: with weight, and without weight.
Model: ridge regression on degree-2 polynomial features, trained on 80% of
the people and tested on the other 20% (split by person).

  python -m model.ansur_prior          # train, test, save model/ansur_prior.joblib
"""

import os
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
DATA = data_path('ansur2')
MODEL_PATH = os.path.join(_ROOT, 'model', 'ansur_prior.joblib')

# TailorCam key → ANSUR II column (millimetres)
TARGETS = {
    'chest': 'chestcircumference',
    'waist': 'waistcircumference',
    'hip': 'buttockcircumference',
    'neck': 'neckcircumferencebase',
    'shoulder_width': 'biacromialbreadth',
    'sleeve_length': 'sleeveoutseam',
    'back_length': 'waistbacklength',
    'thigh': 'thighcircumference',
    'calf': 'calfcircumference',
    'wrist': 'wristcircumference',
    'bicep': 'bicepscircumferenceflexed',
}
_AGE_MEAN = 30.0


def features(height_cm, weight_kg=None, gender='unknown', age=None):
    """→ 1D feature vector. weight_kg / age may be None (unknown)."""
    g = 1.0 if gender == 'male' else (-1.0 if gender == 'female' else 0.0)
    h = (height_cm - 170.0) / 10.0
    a = ((age or _AGE_MEAN) - _AGE_MEAN) / 10.0
    if weight_kg:
        bmi = weight_kg / (height_cm / 100.0) ** 2
        base = [h, (weight_kg - 75.0) / 15.0, (bmi - 25.0) / 5.0, a, g]
    else:
        base = [h, a, g]
    x = np.array(base, float)
    quad = [x[i] * x[j] for i in range(len(x)) for j in range(i, len(x))]
    return np.concatenate([x, quad])


def load_data():
    import pandas as pd
    frames = []
    for g in ('MALE', 'FEMALE'):
        d = pd.read_csv(os.path.join(DATA, f'ANSUR_II_{g}_Public.csv'), encoding='latin-1')
        d['gender'] = g.lower()
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d['height_cm'] = d['stature'] / 10.0
    d['weight_kg'] = d['weightkg'] / 10.0           # stored in hectograms
    return d


def train(test_frac=0.2, seed=0):
    from sklearn.linear_model import Ridge
    import joblib
    d = load_data()
    rng = np.random.default_rng(seed)
    is_test = rng.random(len(d)) < test_frac
    models, report = {}, []
    for with_w in (True, False):
        def X(rows):
            # gender is always known in ANSUR; train each person both as known gender
            # and (with half weight) as "unknown" so the model handles a missing gender
            out, wts = [], []
            for _, r in rows.iterrows():
                kw = dict(weight_kg=r.weight_kg if with_w else None, age=r.Age)
                out.append(features(r.height_cm, gender=r.gender, **kw)); wts.append(1.0)
            return np.array(out), np.array(wts)
        Xtr, _ = X(d[~is_test])
        Xu = np.array([features(r.height_cm, r.weight_kg if with_w else None, 'unknown', r.Age)
                       for _, r in d[~is_test].iterrows()])
        Xte, _ = X(d[is_test])
        for key, col in TARGETS.items():
            ytr = d.loc[~is_test, col].values / 10.0
            yte = d.loc[is_test, col].values / 10.0
            m = Ridge(alpha=1.0).fit(np.vstack([Xtr, Xu]), np.concatenate([ytr, ytr]),
                                     sample_weight=np.concatenate([np.ones(len(ytr)),
                                                                   0.5 * np.ones(len(ytr))]))
            models[(key, with_w)] = m
            err = np.abs(m.predict(Xte) - yte)
            report.append({'measurement': key, 'weight': 'known' if with_w else 'unknown',
                           'test people': len(yte), 'MAE (cm)': round(err.mean(), 2),
                           '≤2cm (%)': round((err <= 2).mean() * 100, 1),
                           '≤3cm (%)': round((err <= 3).mean() * 100, 1)})
    joblib.dump(models, MODEL_PATH)
    return report


class AnsurPrior:
    def __init__(self, path=MODEL_PATH):
        import joblib
        self.models = joblib.load(path)

    def predict(self, height_cm, weight_kg=None, gender='unknown', age=None):
        """→ {measurement: cm}"""
        with_w = bool(weight_kg)
        x = features(height_cm, weight_kg, gender, age)[None]
        return {key: round(float(self.models[(key, with_w)].predict(x)[0]), 1)
                for key in TARGETS}


if __name__ == '__main__':
    import pandas as pd
    print(pd.DataFrame(train()).to_string(index=False))
    print('saved', MODEL_PATH)
