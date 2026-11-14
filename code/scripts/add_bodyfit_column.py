"""
Run the BODY-fit silhouette network on the real24 photos and add its answers
as 'bodyfit:<measurement>' columns to dataset/predictions/real24.csv, so the
combiner can be tested with it (same masks and rule as the BodyM hybrid:
photos are used only when the arms are away from the body).

  python scripts/add_bodyfit_column.py
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
sys.path.insert(0, ROOT)
from model.hmr.hmr_body import HMRBody
from model.hmr.hmr_worker import arms_away, ARMS_AWAY_MIN
from model.bodyfit.train_bodyfit import BodyFitModel

PRED = data_path('predictions', 'real24.csv')


def main():
    hb, bf = HMRBody(), BodyFitModel()
    people = pd.read_csv(data_path('real_photos', 'people.csv')).set_index('person_id')
    views = {}
    for pid in people.index:
        res = []
        for f in ('front.jpg', 'side.jpg'):
            img = cv2.imread(data_path('real_photos', pid, f))
            res.append(hb.analyze(img))
        (kf, mf), (_, ms) = res
        a = arms_away(kf) if kf is not None else None
        views[pid] = [(mf, ms)] if mf is not None and ms is not None and a is not None and a >= ARMS_AWAY_MIN else []
    hb.close()
    d = pd.read_csv(PRED)
    cols = []
    for _, r in d.iterrows():
        w = r.weight_kg if pd.notna(r.weight_kg) else None
        g = r.gender if r.gender in ('male', 'female') else 'unknown'
        cols.append({f'bodyfit:{k}': v for k, v in bf.predict(float(r.height_cm), w, g, views[r.person_id]).items()})
    a = pd.DataFrame(cols, index=d.index)
    d = pd.concat([d.drop(columns=[c for c in a.columns if c in d]), a], axis=1)
    d.to_csv(PRED, index=False)
    print(f'bodyfit columns added; photos used for {sum(bool(v) for v in views.values())}/{len(views)} people')


if __name__ == '__main__':
    main()
