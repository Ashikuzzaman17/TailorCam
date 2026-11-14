"""
Does fitting one body to all photos' outlines (model/silhouette_fit.py) beat
CameraHMR's own guess? Runs CameraHMR once per photo (cached), then tries fit
settings and prints the average error per dataset.

  python scripts/silhouette_fit_experiment.py            # cache + compare
"""

import os
import sys
import pickle
import time

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path, DATASETS  # noqa: E402  (dataset/, assets/, app_data/)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from run_models_on_datasets import DATASETS
from model.hmr.hmr_body import HMRBody
from model.hmr.body_points import measure, load_rings
from model.camerahmr_runner import CameraHMRRunner
from model.silhouette_fit import fit_shape

CACHE = data_path('predictions', 'fit_cache')
MAX_SIDE = 900          # masks are stored at this size (keypoints / camera scaled to match)
SETS = ('real24', 'hbw_val', 'bedlam')


def build_cache(hb, cam, name, max_frames=8):
    path = os.path.join(CACHE, f'{name}.pkl')
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    os.makedirs(CACHE, exist_ok=True)
    people = {}
    for p in DATASETS[name]():
        if p['person_id'] in people:
            continue
        frames = p['frames']
        if len(frames) > max_frames:
            frames = [frames[j] for j in np.linspace(0, len(frames) - 1, max_frames).round().astype(int)]
        fl = []
        for fr in frames:
            img = cv2.imread(fr['path'])
            if img is None:
                continue
            kps, mask = hb.analyze(img)
            if kps is None or mask is None:
                continue
            r = cam.predict(img, kps)
            if r is None:
                continue
            s = min(1.0, MAX_SIDE / max(img.shape[:2]))
            m = np.squeeze(mask)
            m = cv2.resize(m, (int(m.shape[1] * s), int(m.shape[0] * s)))
            k = np.array(kps, np.float32); k[:, :2] *= s
            K = r['cam_int'].copy(); K[:2] *= s
            fl.append({**{x: r[x] for x in ('betas', 'global_orient', 'body_pose', 'cam_t')},
                       'cam_int': K, 'kps': k, 'mask': (m > 0.5).astype(np.uint8), 'view': fr['view']})
        people[p['person_id']] = {'frames': fl, 'height_cm': float(p['height_cm']), 'gender': p['gender'],
                                  'weight_kg': None, 'truth': p['truth']}
        if p['mode'] == 'with_weight':
            people[p['person_id']]['weight_kg'] = float(p['weight_kg'])
    pickle.dump(people, open(path, 'wb'))
    print(f'{name}: cached {len(people)} people', flush=True)
    return people


def evaluate(cam, rings, people, settings, use_weight):
    errs = {}
    for pid, p in people.items():
        if not p['frames']:
            continue
        w = p['weight_kg'] if use_weight else None
        if settings is None:                     # CameraHMR alone (median shape)
            b = np.median([f['betas'] for f in p['frames']], axis=0)
            if w:
                b = cam.fit_weight(b, p['height_cm'], w)
        else:
            b = fit_shape(cam, p['frames'], p['height_cm'], w, **settings)
        m = measure(cam.rest_mesh(b), cam.faces, p['height_cm'], rings, cam.arms_down_mesh(b))
        for t, v in p['truth'].items():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                errs.setdefault(t, []).append(abs(m[t] - v))
    return {t: round(float(np.mean(e)), 2) for t, e in errs.items()}


def main():
    hb, rings, cam = HMRBody(), load_rings(), CameraHMRRunner()
    data = {n: build_cache(hb, cam, n) for n in SETS}
    hb.close()
    variants = {'CameraHMR alone': None,
                'fit: inside only': {},
                'fit: inside + light cover': {'cover': 0.3},
                'fit: inside + cover': {'cover': 1.0}}
    rows = []
    for n, people in data.items():
        for use_w in (True, False):
            for vname, st in variants.items():
                t = time.time()
                e = evaluate(cam, rings, people, st, use_w)
                rows.append({'dataset': n, 'weight': use_w, 'method': vname, **e,
                             's/person': round((time.time() - t) / max(len(people), 1), 2)})
                print(rows[-1], flush=True)
    cols = ['dataset', 'weight', 'method', 'chest', 'waist', 'hip', 'thigh', 'neck', 'shoulder_width', 's/person']
    print(pd.DataFrame(rows)[[c for c in cols if c in pd.DataFrame(rows)]].to_string())


if __name__ == '__main__':
    main()
