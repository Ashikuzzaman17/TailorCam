"""
NOMO-3D (Tampere University, Zenodo 3735905, CC BY 4.0) → an outside test set:
356 real 3D body scans (repetitions removed) with TC2 scanner measurements.

Each raw scan is rendered to a front and a left-side silhouette exactly like
BODY-fit (scripts/build_bodyfit_dataset.py), so the BODY-fit network can be
tested on bodies it never saw. NOMO and BODY-fit come from the same group, so
people that appear in both (identical TC2 measurements) are listed and removed
from BODY-fit training (split 3 in dataset/bodyfit/cache.npz).

  python scripts/build_nomo_test.py
"""

import os
import sys
import glob
import zipfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
sys.path.insert(0, ROOT)
from model.bodym.prep import prep_mask
from model.bodyfit.train_bodyfit import KEYS, DATA as BODYFIT
from build_bodyfit_dataset import FIELDS, DENSITY, render, volume, read_meas

NOMO = data_path('nomo3d', 'NOMO-3d-400-scans_and_tc2_measurements', 'nomo-scans(repetitions-removed)')
OUT = data_path('nomo3d', 'cache.npz')
MATCH = ('Head_Top_Height', 'HIP_Circ', 'Thigh_Circ', 'Shoulder_to_Wrist')


def read_txt(path):
    m = {}
    for line in open(path, errors='ignore'):
        if line.startswith('MEASURE') and '=' in line:
            k, v = line.split(None, 1)[1].split('=', 1)
            try:
                m[k.strip()] = float(v)
            except ValueError:
                pass
    return m                                       # cm


def read_obj(data):
    V, F = [], []
    for line in data.decode('utf-8', 'ignore').splitlines():
        if line.startswith('v '):
            V.append(line.split()[1:4])
        elif line.startswith('f '):
            idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
            F += [[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)]
    V, F = np.array(V, float), np.array(F, np.int64)
    h = V[:, 1].max() - V[:, 1].min()
    if h < 0.5 * (V.max(0) - V.min(0)).max():      # scanner Z-up → Y-up
        V = V[:, [0, 2, 1]] * np.array([1, 1, -1])
        h = V[:, 1].max() - V[:, 1].min()
    return V / (1000.0 if h > 100 else 100.0 if h > 10 else 1.0), F


def main():
    meas = {os.path.splitext(os.path.basename(p))[0]: read_txt(p)
            for p in glob.glob(os.path.join(NOMO, 'meas', '**', '*.txt'), recursive=True)}
    # duplicates with BODY-fit: same TC2 values (mm there, cm here)
    bf = {os.path.splitext(os.path.basename(p))[0]: read_meas(p) for p in
          glob.glob(data_path('bodyfit', 'meas', 'bodymeasurement_mat', '*', '*.mat'))}
    key = lambda m, s: tuple(round(m.get(k, np.nan) * s, 0) for k in MATCH)
    bf_keys = {}
    for n, m in bf.items():
        bf_keys.setdefault(key(m, 1.0), []).append(n)
    dup_bf = set()
    fr, sd, Y, h, w, g, names = [], [], [], [], [], [], []
    for zname, sex in (('male_obj.zip', 1), ('female_obj.zip', 0)):
        with zipfile.ZipFile(os.path.join(NOMO, zname)) as z:
            for info in sorted(z.infolist(), key=lambda i: i.filename):
                if not info.filename.lower().endswith('.obj'):
                    continue
                pid = os.path.splitext(os.path.basename(info.filename))[0]
                m = meas.get(pid)
                if m is None:
                    continue
                dup_bf.update(bf_keys.get(key(m, 10.0), []))
                V, F = read_obj(z.read(info))
                a, b = prep_mask(render(V, F, 'z')), prep_mask(render(V, F, 'x'))
                if a is None or b is None or not 120 < m.get('Head_Top_Height', 0) < 220:
                    continue
                Y.append([next((m[f] for f in FIELDS[k] if f in m and m[f] > 0), np.nan) for k in KEYS])
                fr.append(a.astype(np.uint8)); sd.append(b.astype(np.uint8))
                h.append(m['Head_Top_Height']); w.append(volume(V, F) * DENSITY); g.append(sex); names.append(pid)
    np.savez_compressed(OUT, front=np.array(fr), side=np.array(sd), y=np.array(Y, np.float32),
                        h=np.array(h, np.float32), w=np.array(w, np.float32), g=np.array(g, np.int8),
                        split=np.full(len(names), 2, np.int8), names=np.array(names, dtype='U16'))
    print(f'saved {OUT}: {len(names)} people; weight mean {np.mean(w):.1f} kg (range {np.min(w):.0f}-{np.max(w):.0f})')
    cv2.imwrite(data_path('nomo3d', 'example_silhouettes.png'),
                np.hstack([fr[0], sd[0], fr[-1], sd[-1]]) * 255)
    # keep NOMO a clean test: drop duplicates from BODY-fit training
    D = dict(np.load(BODYFIT))
    hit = np.isin(D['names'], sorted(dup_bf))
    D['split'][hit] = 3
    np.savez_compressed(BODYFIT, **D)
    print(f'{hit.sum()} BODY-fit people are also in NOMO -> removed from BODY-fit training/testing')


if __name__ == '__main__':
    main()
