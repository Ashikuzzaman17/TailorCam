"""
Every TailorCam model on its own, body part by body part, on the 24 real
tape-measured people (dataset/predictions/real24.csv), plus the combined model.

For each model and body part the prediction is "model + learned correction":
the correction is chosen and fitted on the training folds only (nested CV,
6 folds x 5 repeated splits), the same rule the combiner uses, so every model
is tested the same fair way. The raw model output is kept in the sheets too.

  accuracy (%) = 100 - mean absolute percentage error
  overall      = mean of the body-part accuracies the model can measure

Output:
  reports/TailorCam_Model_Data.xlsx           one data sheet per model (per person)
  reports/paper/tables/model_accuracy_*.csv   accuracy / MAE per model and body part

  python scripts/model_accuracy.py
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
from repeated_cv import first

TABLES = os.path.join(ROOT, 'reports', 'paper', 'tables')
XLSX = os.path.join(ROOT, 'reports', 'TailorCam_Model_Data.xlsx')
PARTS = {'chest': 'Chest', 'waist': 'Belly / waist', 'hip': 'Hip', 'shoulder_width': 'Shoulder',
         'sleeve_length': 'Arm (sleeve)', 'neck': 'Neck', 'thigh': 'Thigh'}
MODELS = {'HMR (2018)': 'hmr', 'SHAPY (2022)': 'shapy', 'CameraHMR (2025)': 'camerahmr',
          'Silhouette CNN x3': 'hybrid', 'BodyM prior': 'prior', 'ANSUR prior': 'ansur'}
COMBINED = 'Combined (TailorCam hybrid)'
SEEDS = range(5)
BODYM_NAMES = {'sleeve_length': 'arm-length', 'shoulder_width': 'shoulder-breadth'}


def col(m, t, d):
    c = f'{m}:{BODYM_NAMES.get(t, t)}' if m in ('hybrid', 'prior') else f'{m}:{t}'
    return c if c in d and d[c].notna().any() else None


def cv_predict(X, y):
    """Nested-CV predictions per person for each repeated split. Same rule as the app
    (calibration.fit_all keeps a correction only if it helps), decided inside the
    training people only (calibration.nested_cv_app), so test people decide nothing."""
    return np.array([C.nested_cv_app(X, y, 6, seed=s) for s in SEEDS])                     # splits x people


def evaluate(mode):
    d = pd.read_csv(data_path('predictions', 'real24.csv'))
    d = d[d['mode'] == mode].reset_index(drop=True)
    targets = C.TARGETS if mode == 'with_weight' else first('camerahmr', C.TARGETS)
    sheets, rows = {}, []
    for name, m in list(MODELS.items()) + [(COMBINED, None)]:
        sheet = d[['person_id', 'gender', 'height_cm', 'weight_kg']].copy()
        for t, label in PARTS.items():
            if f'true:{t}' not in d:
                continue
            if m is None and t == 'shoulder_width':
                cols = ['hmr:shoulder_width']        # the app has no combiner for shoulder: it shows HMR's value
            elif m is None:
                cols = [c for c in targets.get(t, []) if c in d and d[c].notna().any()]
            else:
                cols = [col(m, t, d)] if col(m, t, d) else []
            if not cols:
                continue
            ok = d[cols + [f'true:{t}']].notna().all(axis=1)
            X, y = d.loc[ok, cols].values.astype(float), d.loc[ok, f'true:{t}'].values.astype(float)
            P = cv_predict(X, y)
            p, e = P.mean(0), np.abs(P - y).mean(0)      # error averaged over the repeated splits
            sheet[f'{label} tape'] = d[f'true:{t}']
            if m is not None:
                sheet[f'{label} raw'] = d[cols[0]]
            sheet.loc[ok, f'{label} predicted'] = p.round(2)
            sheet.loc[ok, f'{label} abs error'] = e.round(2)
            sheet.loc[ok, f'{label} accuracy %'] = (100 - e / y * 100).round(2)
            rows.append({'model': name, 'part': label, 'n': int(ok.sum()), 'MAE': e.mean(),
                         'raw MAE': np.abs(X[:, 0] - y).mean() if m is not None else np.nan,
                         'accuracy': 100 - (e / y * 100).mean(),
                         'within 2 cm %': (e <= 2).mean() * 100})
        sheets[name] = sheet
    r = pd.DataFrame(rows)
    acc = r.pivot(index='model', columns='part', values='accuracy').reindex(
        index=list(MODELS) + [COMBINED], columns=list(PARTS.values()))
    acc['Overall (all parts it measures)'] = acc[list(PARTS.values())].mean(axis=1)
    acc['Parts measured'] = acc[list(PARTS.values())].notna().sum(axis=1)
    acc['Overall (chest, belly, hip)'] = acc[['Chest', 'Belly / waist', 'Hip']].mean(axis=1)
    mae = r.pivot(index='model', columns='part', values='MAE').reindex(
        index=list(MODELS) + [COMBINED], columns=list(PARTS.values()))
    raw = r.pivot(index='model', columns='part', values='raw MAE').reindex(
        index=list(MODELS), columns=list(PARTS.values()))
    return sheets, r, acc, mae, raw


def main():
    os.makedirs(TABLES, exist_ok=True)
    out = {}
    with pd.ExcelWriter(XLSX) as xw:
        for mode in ('with_weight', 'no_weight'):
            sheets, r, acc, mae, raw = evaluate(mode)
            out[mode] = (acc, mae)
            tag = 'W' if mode == 'with_weight' else 'noW'
            acc.round(2).to_csv(os.path.join(TABLES, f'model_accuracy_{mode}.csv'))
            mae.round(2).to_csv(os.path.join(TABLES, f'model_mae_{mode}.csv'))
            raw.round(2).to_csv(os.path.join(TABLES, f'model_raw_mae_{mode}.csv'))
            r.round(2).to_csv(os.path.join(TABLES, f'model_parts_long_{mode}.csv'), index=False)
            acc.round(2).to_excel(xw, sheet_name=f'Accuracy summary {tag}')
            mae.round(2).to_excel(xw, sheet_name=f'MAE cm summary {tag}')
            for name, s in sheets.items():
                short = name.split(' (')[0].replace('Silhouette ', '').replace('Combined', 'COMBINED')[:24]
                s.to_excel(xw, sheet_name=f'{short} {tag}'[:31], index=False)
            pd.set_option('display.width', 250)
            print(f'\n== {mode}: accuracy %'); print(acc.round(2).to_string())
            print(f'\n== {mode}: MAE cm'); print(mae.round(2).to_string())
    print('saved', XLSX)


if __name__ == '__main__':
    main()
