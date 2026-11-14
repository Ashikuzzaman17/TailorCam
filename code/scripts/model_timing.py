"""
Time of each TailorCam model on one customer (front + side photo), on this PC.

Each model is timed on its own: the shared photo step (reading the photos and
finding the body with MediaPipe) is timed once and listed separately. One
warm-up run, then the mean of REPEATS runs over several real24 people.

Output: reports/paper/tables/model_timing.csv

  python scripts/model_timing.py
"""

import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, 'reports', 'paper', 'tables', 'model_timing.csv')
REPEATS = 5
PEOPLE = 4

from model.hmr.hmr_body import HMRBody
from model.hmr.body_points import measure, load_rings
from model.hmr.hmr_worker import _load_hybrid, _load_shapy, _load_camerahmr, _ansur_prior
from model.keypoints import mediapipe_to_body25
from model.calibration import apply_calibration


def sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def clock(fn):
    sync(); t = time.perf_counter(); out = fn(); sync()
    return time.perf_counter() - t, out


def main():
    t0 = time.perf_counter()
    hb, rings = HMRBody(), load_rings()
    hy, sh, cam, ap = _load_hybrid(), _load_shapy(), _load_camerahmr(), _ansur_prior()
    load_s = time.perf_counter() - t0
    people = pd.read_csv(data_path('real_photos', 'people.csv')).head(PEOPLE)
    T = {k: [] for k in ('Photo step (MediaPipe, shared)', 'HMR (2018)', 'SHAPY (2022)', 'CameraHMR (2025)',
                         'Silhouette CNN x3 (ours)', 'BodyM prior (ours)', 'ANSUR prior (ours)', 'Combiner (ours)')}
    for rep in range(REPEATS + 1):                    # run 0 = warm-up, not counted
        for _, r in people.iterrows():
            h, w = float(r.height_cm), float(r.weight_kg)
            g = r.gender if r.gender in ('male', 'female') else 'unknown'
            paths = [data_path('real_photos', r.person_id, f) for f in ('front.jpg', 'side.jpg')]

            def photos():
                res = []
                for p in paths:
                    img = cv2.imread(p)
                    kps, mask = hb.analyze(img)
                    res.append((img, kps, mask))
                return res
            dt, ph = clock(photos)
            t = {'Photo step (MediaPipe, shared)': dt}
            ph = [x for x in ph if x[1] is not None]

            def hmr():
                b = np.median([hb.predict(img, k)['betas'] for img, k, _ in ph], axis=0)
                b = hb.fit_weight(b, h, w)
                return measure(hb.rest_mesh(b), hb.faces, h, rings, hb.arms_down_mesh(b))
            t['HMR (2018)'], comps = clock(hmr)
            comps = {f'hmr:{k}': v for k, v in comps.items() if k != 'height'}

            def shapy():
                items = [(f'f{i:03d}', img, mediapipe_to_body25(k)) for i, (img, k, _) in enumerate(ph)]
                return sh.measure_person(items, h, w)[0]
            t['SHAPY (2022)'], m = clock(shapy)
            comps.update({f'shapy:{k}': v for k, v in (m or {}).items() if k in ('chest', 'waist', 'hip')})

            def camerahmr():
                bl = [c['betas'] for c in (cam.predict(img, k) for img, k, _ in ph) if c is not None]
                return cam.measure_person(bl, h, w, rings)[0]
            t['CameraHMR (2025)'], m = clock(camerahmr)
            comps.update({f'camerahmr:{k}': v for k, v in m.items()})

            views = [(ph[0][2], ph[1][2])] if len(ph) == 2 and ph[0][2] is not None and ph[1][2] is not None else []
            t['Silhouette CNN x3 (ours)'], m = clock(lambda: hy.predict(h, w, g, views))
            comps.update({f'hybrid:{k}': v[0] for k, v in m.items()})
            t['BodyM prior (ours)'], m = clock(lambda: hy.predict(h, w, g, []))
            comps.update({f'prior:{k}': v[0] for k, v in m.items()})
            t['ANSUR prior (ours)'], m = clock(lambda: ap.predict(h, w, g, r.age))
            comps.update({f'ansur:{k}': v for k, v in m.items()})
            t['Combiner (ours)'], _ = clock(lambda: apply_calibration(comps, weight_known=True))
            if rep:
                for k, v in t.items():
                    T[k].append(v)
        print(f'run {rep}/{REPEATS} done', flush=True)
    hb.close()
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    d = pd.DataFrame([{'part': k, 'mean_s': np.mean(v), 'sd_s': np.std(v), 'runs': len(v)} for k, v in T.items()])
    d.loc[len(d)] = ['Model loading (once, at app start)', load_s, 0.0, 1]
    d['device'] = gpu
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    d.round(4).to_csv(OUT, index=False)
    print(d.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
