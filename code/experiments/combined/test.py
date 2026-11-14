"""
Test on the combined dataset as ONE new dataset: every person with a photo (no weight,
since part of it has none). Same protocol as the main test: leak-free nested
cross-validation (6 folds x 5 repeated splits); every single model gets one learned
offset, the combined model learns its rules; nobody is predicted by a model that
learned from them.

Inputs: every model's answer per person, from dataset/predictions/real24.csv and
app_data/shop_extra/odjag.csv. People are renamed P01, P02, ...

  python experiments/combined/test.py     -> people.csv, summary.csv
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'scripts')]
from paths import dataset, app_data
from model.calibration import TARGETS
from model_accuracy import cv_predict

KEYS = ['chest', 'waist', 'hip', 'neck', 'thigh', 'sleeve_length', 'shoulder_width']
MODELS = {'HMR': 'hmr', 'SHAPY': 'shapy', 'CameraHMR': 'camerahmr', 'Silhouette CNN': 'hybrid', 'BodyM prior': 'prior',
          'ANSUR prior': 'ansur', 'TailorCam (combined)': None}
BODYM = {'sleeve_length': 'arm-length', 'shoulder_width': 'shoulder-breadth'}
NW = {k: (['camerahmr:' + k] + [c for c in v if c != 'camerahmr:' + k] if 'camerahmr:' + k in v else v)
      for k, v in TARGETS.items()}


def load():
    r24 = pd.read_csv(dataset('predictions', 'real24.csv'))
    d = pd.concat([r24[r24['mode'] == 'no_weight'], pd.read_csv(app_data('shop_extra', 'odjag.csv'))], ignore_index=True)
    d['person'] = [f'P{i + 1:02d}' for i in range(len(d))]
    return d


def cols_for(m, k, d):
    if m is None:
        if k == 'shoulder_width':
            return ['hmr:shoulder_width']                 # the app has no combiner rule for shoulder
        return [c for c in NW[k] if c in d and d[c].notna().any()]
    c = f'{m}:{BODYM.get(k, k)}' if m in ('hybrid', 'prior') else f'{m}:{k}'
    return [c] if c in d and d[c].notna().any() else []


def main():
    d = load()
    per, summ = [], []
    for name, m in MODELS.items():
        for k in KEYS:
            cols = cols_for(m, k, d)
            if not cols:
                continue
            ok = d[cols + [f'true:{k}']].notna().all(axis=1)
            X, y = d.loc[ok, cols].values.astype(float), d.loc[ok, f'true:{k}'].values.astype(float)
            P = cv_predict(X, y)                          # repeated splits x people
            p, e = P.mean(0), np.abs(P - y).mean(0)
            for person, yy, pp, ee in zip(d.loc[ok, 'person'], y, p, e):
                per.append({'person': person, 'model': name, 'measurement': k, 'tape': yy, 'predicted': pp, 'abs error': ee})
            summ.append({'model': name, 'measurement': k, 'people': int(ok.sum()), 'MAE': e.mean(),
                         'accuracy': 100 - (e / y * 100).mean(), 'bias': (p - y).mean(),
                         'within 2 cm %': (e <= 2).mean() * 100})
        print(name, 'done', flush=True)
    pd.DataFrame(per).round(3).to_csv(os.path.join(HERE, 'people.csv'), index=False)
    s = pd.DataFrame(summ)
    s.round(3).to_csv(os.path.join(HERE, 'summary.csv'), index=False)
    pd.set_option('display.width', 220)
    print(s.pivot_table(index='model', columns='measurement', values='MAE').round(2).to_string())
    print(s.groupby('model').accuracy.mean().round(2).sort_values().to_string())


if __name__ == '__main__':
    main()
