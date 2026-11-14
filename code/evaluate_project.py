"""
TailorCam — full-project evaluation.

Runs the exact measurement code the app uses and scores it against datasets
with real measurements. Writes a report folder with tables (CSV) and charts.

  Test 1  BodyM testA + testB (487 people, silhouettes, 3D-scan measurements)
          → the hybrid model the app uses for chest / waist / hip / thigh /
            calf / bicep / wrist / ankle (ensemble of all bodym_hybrid_s*.pt)
  Test 2  Real photos with tape measurements (Hugging Face sample, 6 people)
          → the WHOLE pipeline end to end, exactly as the app calls it:
            photo → MediaPipe → HMR 3D body (+weight) + hybrid → measurements

Metrics: MAE, RMSE, R², MAPE, % within ±1/2/3 cm, and size classification
(S…XXL from chest / waist): accuracy, precision, recall, F1, confusion matrix.

  python evaluate_project.py
"""

import os
import sys
import csv
import json
import time
import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)

ROOT = os.path.dirname(os.path.abspath(__file__))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
sys.path.insert(0, ROOT)

from model.bodym.bodym_net import MEASUREMENTS
from model.bodym.hybrid import HybridNet, member_paths, predict_split, DEV
from model.bodym.train_bodym import load_split

REAL_DIR = data_path('real_photos')
SIZES = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
SIZE_BINS = {'chest': [84, 92, 100, 108, 116], 'waist': [70, 78, 86, 94, 102]}


# ── Metrics ───────────────────────────────────────────────────────────────────
def regression_metrics(y, p):
    e = np.asarray(p, float) - np.asarray(y, float)
    a = np.abs(e)
    ss = ((y - np.mean(y)) ** 2).sum()
    return {'n': len(y), 'MAE (cm)': a.mean(), 'RMSE (cm)': np.sqrt((e ** 2).mean()),
            'bias (cm)': e.mean(), 'R²': 1 - (e ** 2).sum() / ss if ss > 0 else np.nan,
            'MAPE (%)': (a / y).mean() * 100,
            '≤1cm (%)': (a <= 1).mean() * 100, '≤2cm (%)': (a <= 2).mean() * 100,
            '≤3cm (%)': (a <= 3).mean() * 100}


def size_metrics(y, p, key):
    t, q = np.digitize(y, SIZE_BINS[key]), np.digitize(p, SIZE_BINS[key])
    pr, rc, f1, _ = precision_recall_fscore_support(t, q, average='macro', zero_division=0)
    _, _, f1w, _ = precision_recall_fscore_support(t, q, average='weighted', zero_division=0)
    return ({'accuracy (%)': accuracy_score(t, q) * 100,
             'within ±1 size (%)': (np.abs(t - q) <= 1).mean() * 100,
             'precision (macro)': pr, 'recall (macro)': rc,
             'F1 (macro)': f1, 'F1 (weighted)': f1w},
            confusion_matrix(t, q, labels=range(len(SIZES))))


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(6)); ax.set_xticklabels(SIZES)
    ax.set_yticks(range(6)); ax.set_yticklabels(SIZES)
    for a in range(6):
        for b in range(6):
            if cm[a, b]:
                ax.text(b, a, cm[a, b], ha='center', va='center', fontsize=8)
    ax.set_xlabel('predicted size'); ax.set_ylabel('true size'); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ── Test 1: BodyM ─────────────────────────────────────────────────────────────
def test_bodym(out):
    nets = []
    for p in member_paths():
        n = HybridNet().to(DEV)
        n.load_state_dict(torch.load(p, map_location=DEV, weights_only=True))
        nets.append(n)
    print(f'\n=== TEST 1: BodyM — hybrid ensemble of {len(nets)} model(s) ===')
    summary, cls_rows = [], []
    for split in ('testA', 'testB'):
        D = load_split(split)
        runs = {
            'TailorCam (hybrid)': {},
            'height+weight only': {'use_images': False},
            'TailorCam, no weight': {'use_weight': False},
        }
        for name, kw in runs.items():
            if name == 'height+weight only':
                p = np.mean([predict_split(n, D)['prior'] for n in nets], 0)
            else:
                p = np.mean([predict_split(n, D, use_weight=kw.get('use_weight', True))['image']
                             for n in nets], 0)
            rows = [{'measurement': m, **regression_metrics(D['y'][:, j], p[:, j])}
                    for j, m in enumerate(MEASUREMENTS)]
            t = pd.DataFrame(rows).set_index('measurement').round(2)
            t.to_csv(os.path.join(out, f'bodym_{split}_{name.split()[0].lower()}'
                                       f'{"_noweight" if "no weight" in name else ""}'
                                       f'{"_hw" if "height+weight" in name else ""}.csv'))
            for m in ('chest', 'waist', 'hip', 'thigh', 'shoulder-breadth', 'arm-length'):
                summary.append({'test set': split, 'method': name, 'measurement': m,
                                'MAE (cm)': t.loc[m, 'MAE (cm)'], '≤2cm (%)': t.loc[m, '≤2cm (%)']})
            if name == 'TailorCam (hybrid)':
                print(f'\n{split}: {len(set(D["subj"]))} people, {len(D["y"])} photo pairs')
                print(t.drop(columns=['n']).to_string())
                for key in ('chest', 'waist'):
                    j = MEASUREMENTS.index(key)
                    sm, cm = size_metrics(D['y'][:, j], p[:, j], key)
                    cls_rows.append({'test': f'BodyM {split}', 'size from': key, **sm})
                    plot_confusion(cm, f'BodyM {split} — {key} size',
                                   os.path.join(out, f'confusion_bodym_{split}_{key}.png'))
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                for ax, m in zip(axes, ('chest', 'waist', 'hip', 'thigh')):
                    j = MEASUREMENTS.index(m)
                    ax.scatter(D['y'][:, j], p[:, j], s=5, alpha=.4)
                    lo, hi = D['y'][:, j].min(), D['y'][:, j].max()
                    ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
                    ax.set_title(f'{split} {m}  MAE {t.loc[m, "MAE (cm)"]:.2f} cm')
                    ax.set_xlabel('true (cm)'); ax.set_ylabel('predicted (cm)')
                fig.tight_layout(); fig.savefig(os.path.join(out, f'scatter_bodym_{split}.png'), dpi=110)
                plt.close(fig)
    comp = pd.DataFrame(summary).pivot_table(index=['test set', 'measurement'],
                                             columns='method', values='MAE (cm)')
    comp.to_csv(os.path.join(out, 'bodym_method_comparison.csv'))
    print('\nMAE (cm) by method — does the camera help?')
    print(comp.round(2).to_string())
    return cls_rows


# ── Test 2: real photos, whole pipeline ───────────────────────────────────────
# merged dataset column → TailorCam key, and which model gives that key
REAL_KEYS = {'chest': ('chest', 'hybrid'), 'waist': ('waist', 'hybrid'),
             'hip': ('hip', 'hybrid'), 'thigh': ('thigh', 'hybrid'),
             'calf': ('calf', 'hybrid'), 'bicep': ('bicep', 'hybrid'),
             'neck': ('neck', '3D body'), 'shoulder_width': ('shoulder_width', '3D body'),
             'arm_length': ('sleeve_length', '3D body')}


def test_real(out):
    from model.hmr.hmr_body import HMRBody
    from model.hmr.hmr_worker import run_request, _load_hybrid
    from model.hmr.body_points import load_rings
    table = os.path.join(REAL_DIR, 'people.csv')
    if not os.path.exists(table):
        import subprocess
        subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'build_real_photo_dataset.py')],
                       check=True)
    people = list(csv.DictReader(open(table, encoding='utf-8')))
    print(f'\n=== TEST 2: real colour photos ({len(people)} people) - full TailorCam pipeline ===')
    print('(values the provider flagged "to be reviewed" are not scored)')
    hb, rings, hy = HMRBody(), load_rings(), _load_hybrid()
    per_person, cls_rows = [], []
    for mode in ('with weight', 'without weight'):
        y, p = {k: [] for k in REAL_KEYS}, {k: [] for k in REAL_KEYS}
        for r in people:
            pid = r['person_id']
            req = {'height_cm': float(r['height_cm']), 'gender': r['gender'], 'calibrate': False,
                   'weight_kg': float(r['weight_kg']) if mode == 'with weight' else None,
                   'frames': [{'path': os.path.join(REAL_DIR, pid, 'front.jpg'), 'view': 'FRONT'},
                              {'path': os.path.join(REAL_DIR, pid, 'side.jpg'), 'view': 'LEFT'}]}
            res = run_request(hb, rings, req, hy)
            if not res['ok']:
                print(f'  {pid}: {res["error"]}'); continue
            m, flagged = res['measurements'], set(filter(None, r['flagged'].split(';')))
            row = {'mode': mode, 'person': pid, 'gender': r['gender'], 'height': r['height_cm'],
                   'weight': r['weight_kg'], 'arms_away': res['arms_away'],
                   'silhouettes_used': res['image_used']}
            for col, (key, _) in REAL_KEYS.items():
                if r[col] == '':
                    continue
                row[f'{col} tape'] = float(r[col]); row[f'{col} TailorCam'] = m[key]
                row[f'{col} flagged'] = col in flagged
                if col not in flagged:
                    y[col].append(float(r[col])); p[col].append(m[key])
            per_person.append(row)
        rows = [{'measurement': col, 'model': src, **regression_metrics(np.array(y[col]), np.array(p[col]))}
                for col, (_, src) in REAL_KEYS.items() if len(y[col]) >= 3]
        t = pd.DataFrame(rows).set_index('measurement').round(2)
        print(f'\n[{mode}]  n = number of people scored for that measurement')
        print(t.to_string())
        t.to_csv(os.path.join(out, f'real_photos_{mode.replace(" ", "_")}.csv'))
        if mode == 'with weight':
            for key in ('chest', 'waist'):
                sm, cm = size_metrics(np.array(y[key]), np.array(p[key]), key)
                cls_rows.append({'test': f'real photos ({len(y[key])} people)', 'size from': key, **sm})
                plot_confusion(cm, f'real photos - {key} size',
                               os.path.join(out, f'confusion_real_{key}.png'))
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            for ax, k in zip(axes, ('chest', 'waist', 'hip', 'neck')):
                ax.scatter(y[k], p[k], s=25)
                lo, hi = min(y[k] + p[k]), max(y[k] + p[k])
                ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
                ax.set_title(f'real photos {k}  MAE {t.loc[k, "MAE (cm)"]:.1f} cm')
                ax.set_xlabel('tape (cm)'); ax.set_ylabel('TailorCam (cm)')
            fig.tight_layout(); fig.savefig(os.path.join(out, 'scatter_real_photos.png'), dpi=110)
            plt.close(fig)
    pd.DataFrame(per_person).to_csv(os.path.join(out, 'real_photos_per_person.csv'), index=False)
    hb.close()
    return cls_rows


def main():
    plt.switch_backend('Agg')          # files only (keeps notebooks that import us inline)
    out = os.path.join(ROOT, 'reports', 'evaluation_' + datetime.datetime.now().strftime('%Y%m%d_%H%M'))
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    cls = test_bodym(out) + test_real(out)
    c = pd.DataFrame(cls).round(3)
    c.to_csv(os.path.join(out, 'size_classification.csv'), index=False)
    print('\n=== SIZE CLASSIFICATION (accuracy / precision / recall / F1) ===')
    print(c.to_string(index=False))
    print(f'\nReport saved to {out}  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
