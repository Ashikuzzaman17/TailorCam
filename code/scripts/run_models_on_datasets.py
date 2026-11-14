"""
Run every TailorCam model on every real-photo dataset and store each model's
answer per person ("components"), for training and testing the combiner.

Datasets (all real colour photos with true measurements):
  real24   dataset/real_photos           24 people, front + side, tape, with weight
  agency   dataset/shapy/model_agency_images   ~715 fashion models, several photos,
           agency-listed height/chest/waist/hip (no weight), official SHAPY split
  hbw_val  dataset/shapy/HBW_low_resolution    10 people, lab + everyday photos,
           truth from 3D body scans (height, chest, waist, hip, weight)

Output: dataset/predictions/<dataset>.csv — one row per person and weight mode,
with columns 'hybrid:chest', 'prior:chest', 'hmr:chest', 'shapy:chest', …
and the truth as 'true:chest', … . Resumable: finished people are skipped.

  python scripts/run_models_on_datasets.py [real24 agency hbw_val]
"""

import os
import sys
import csv
import glob
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
OUT = data_path('predictions')

from model.hmr.hmr_body import HMRBody
from model.hmr.hmr_worker import run_request, _load_hybrid, _load_shapy, _load_segmenter, _load_camerahmr
from model.hmr.body_points import load_rings


def people_real24():
    d = data_path('real_photos')
    for _, r in pd.read_csv(os.path.join(d, 'people.csv')).iterrows():
        flagged = set(str(r.flagged).split(';'))
        truth = {t: (float(r[c]) if pd.notna(r[c]) and c not in flagged else None)
                 for t, c in [('chest', 'chest'), ('waist', 'waist'), ('hip', 'hip'),
                              ('thigh', 'thigh'), ('neck', 'neck'),
                              ('shoulder_width', 'shoulder_width'), ('sleeve_length', 'arm_length')]}
        frames = [{'path': os.path.join(d, r.person_id, 'front.jpg'), 'view': 'FRONT'},
                  {'path': os.path.join(d, r.person_id, 'side.jpg'), 'view': 'LEFT'}]
        for mode in ('with_weight', 'no_weight'):
            yield {'person_id': r.person_id, 'split': 'real24', 'mode': mode,
                   'height_cm': r.height_cm, 'weight_kg': r.weight_kg if mode == 'with_weight' else None,
                   'gender': r.gender, 'frames': frames, 'truth': truth}


def people_agency():
    d = data_path('shapy', 'model_agency_images')
    for _, r in pd.read_csv(os.path.join(d, 'index.csv')).iterrows():
        imgs = sorted(glob.glob(os.path.join(d, r.agency, str(r.model_id), '*.jpg')))
        if not imgs:
            continue
        yield {'person_id': f'{r.agency}/{r.model_id}', 'split': r.split, 'mode': 'no_weight',
               'height_cm': r.height_cm, 'weight_kg': None, 'gender': r.gender,
               'frames': [{'path': p, 'view': 'FRONT'} for p in imgs],
               'truth': {'chest': r.chest_cm, 'waist': r.waist_cm, 'hip': r.hip_cm}}


def people_hbw_val():
    from model.shapy_measure import load_obj, measure_smplx
    d = data_path('shapy', 'HBW_low_resolution')
    for obj in sorted(glob.glob(os.path.join(d, 'smplx', 'val', '*.obj'))):
        sid = os.path.basename(obj)[:-4]
        V, F = load_obj(obj)
        gt = measure_smplx(V, F)
        folder = glob.glob(os.path.join(d, 'images', 'val_small_resolution', f'{sid}_*'))[0]
        imgs = sorted(glob.glob(os.path.join(folder, '*', '*.png')))
        for mode in ('with_weight', 'no_weight'):
            yield {'person_id': f'hbw_{sid}', 'split': 'hbw_val', 'mode': mode,
                   'height_cm': gt['height'], 'weight_kg': gt['mass_kg'] if mode == 'with_weight' else None,
                   'gender': 'unspecified', 'frames': [{'path': p, 'view': 'FRONT'} for p in imgs],
                   'truth': {'chest': gt['chest'], 'waist': gt['waist'], 'hip': gt['hip']}}


def people_bedlam():
    """BEDLAM synthetic clothed people, truth from the exact body (scripts/build_bedlam_dataset.py)."""
    p = data_path('bedlam', 'people.csv')
    if not os.path.exists(p):
        return
    keys = ['chest', 'waist', 'hip', 'thigh', 'neck', 'shoulder_width', 'sleeve_length']
    for _, r in pd.read_csv(p).iterrows():
        for mode in ('with_weight', 'no_weight'):
            yield {'person_id': r.person_id, 'split': 'bedlam', 'mode': mode,
                   'height_cm': r.height_cm, 'weight_kg': r.weight_kg if mode == 'with_weight' else None,
                   'gender': r.gender, 'frames': [{'path': f, 'view': 'FRONT'} for f in r.frames.split(';')],
                   'truth': {k: r[k] for k in keys}}


DATASETS = {'real24': people_real24, 'agency': people_agency, 'hbw_val': people_hbw_val,
            'bedlam': people_bedlam}


def main(names, suffix='', sam2=False):
    os.makedirs(OUT, exist_ok=True)
    hb, rings, hy, sh = HMRBody(), load_rings(), _load_hybrid(), _load_shapy()
    seg = _load_segmenter() if sam2 else None
    cam = _load_camerahmr()
    print(f'models: HMR yes, hybrid {"yes" if hy else "no"}, SHAPY {"yes" if sh else "no"}, '
          f'CameraHMR {"yes" if cam else "no"}', flush=True)
    for name in names:
        path = os.path.join(OUT, f'{name}{suffix}.csv')
        done = set()
        if os.path.exists(path):
            old = pd.read_csv(path)
            done = set(zip(old.person_id, old['mode']))
        people = [p for p in DATASETS[name]() if (p['person_id'], p['mode']) not in done]
        print(f'== {name}: {len(people)} person-runs to do ({len(done)} already done)', flush=True)
        t0 = time.time()
        for i, p in enumerate(people):
            try:
                res = run_request(hb, rings, {
                    'height_cm': float(p['height_cm']), 'weight_kg': p['weight_kg'],
                    'gender': p['gender'], 'frames': p['frames'], 'calibrate': False,
                    'sigma': False, 'max_frames': 12}, hy, sh, seg, cam)
            except Exception as exc:               # one bad photo set must not stop the run
                res = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
            if not res['ok']:
                print(f'  {p["person_id"]}: {res["error"]}', flush=True)
                continue
            row = {'person_id': p['person_id'], 'split': p['split'], 'mode': p['mode'],
                   'height_cm': p['height_cm'], 'weight_kg': p['weight_kg'], 'gender': p['gender'],
                   'n_frames': res['n_frames'], 'arms_away': res['arms_away'],
                   'silhouettes_used': res['image_used'], **res['components'],
                   **{f'true:{k}': v for k, v in p['truth'].items()}}
            new = not os.path.exists(path)
            # append so an interruption keeps finished people
            pd.DataFrame([row]).to_csv(path, mode='a', header=new, index=False)
            if (i + 1) % 25 == 0 or i + 1 == len(people):
                rate = (time.time() - t0) / (i + 1)
                print(f'  {i + 1}/{len(people)}  ({rate:.1f}s per person, '
                      f'~{rate * (len(people) - i - 1) / 60:.0f} min left)', flush=True)
    hb.close()


def add_ansur(names, prefix='ansur'):
    """Add / refresh the ANSUR II (or NHANES) prior columns (needs no photos — fast)."""
    if prefix == 'nhanes':
        from model.nhanes_prior import NhanesPrior as AnsurPrior
    else:
        from model.ansur_prior import AnsurPrior
    ap = AnsurPrior()
    ages = pd.read_csv(data_path('real_photos', 'people.csv')).set_index('person_id')['age']
    for name in names:
        path = os.path.join(OUT, f'{name}.csv')
        d = pd.read_csv(path)
        rows = []
        for _, r in d.iterrows():
            age = ages.get(r.person_id) if name == 'real24' else None
            g = r.gender if r.gender in ('male', 'female') else 'unknown'
            w = r.weight_kg if pd.notna(r.weight_kg) else None
            rows.append({f'{prefix}:{k}': v for k, v in ap.predict(r.height_cm, w, g, age).items()})
        a = pd.DataFrame(rows, index=d.index)
        d = pd.concat([d.drop(columns=[c for c in a.columns if c in d]), a], axis=1)
        d.to_csv(path, index=False)
        print(f'{name}: {prefix} columns added for {len(d)} rows')


def add_camerahmr(names, max_frames=12, prefix='camerahmr'):
    """Add / refresh the CameraHMR columns only (the other models are not re-run)."""
    hb, rings, cam = HMRBody(), load_rings(), _load_camerahmr()
    for name in names:
        path = os.path.join(OUT, f'{name}.csv')
        d = pd.read_csv(path)
        people = {p['person_id']: p for p in DATASETS[name]()}   # same frames as the full run
        betas, cols, t0 = {}, [], time.time()
        for i, (_, r) in enumerate(d.iterrows()):
            p = people.get(r.person_id)
            if p is None:
                cols.append({}); continue
            if r.person_id not in betas:                  # photos → shape, once per person
                frames = p['frames']
                if len(frames) > max_frames:
                    frames = [frames[j] for j in np.linspace(0, len(frames) - 1, max_frames).round().astype(int)]
                bl = []
                for fr in frames:
                    img = cv2.imread(fr['path'])
                    if img is None:
                        continue
                    kps, _ = hb.analyze(img)
                    if kps is not None:
                        c = cam.predict(img, kps)
                        if c is not None:
                            bl.append(c['betas'])
                betas[r.person_id] = bl
            bl = betas[r.person_id]
            w = r.weight_kg if pd.notna(r.weight_kg) else None
            cols.append({f'{prefix}:{k}': round(float(v), 2) for k, v in
                         cam.measure_person(bl, float(r.height_cm), w, rings)[0].items()} if bl else {})
            if (i + 1) % 50 == 0:
                print(f'  {name}: {i + 1}/{len(d)}  ({(time.time() - t0) / (i + 1):.1f}s per row)', flush=True)
        a = pd.DataFrame(cols, index=d.index)
        d = pd.concat([d.drop(columns=[c for c in a.columns if c in d]), a], axis=1)
        d.to_csv(path, index=False)
        print(f'{name}: {prefix} columns added for {a.notna().any(axis=1).sum()}/{len(d)} rows', flush=True)
    hb.close()


if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] == '--add-ansur':
        add_ansur(args[1:] or list(DATASETS))
    elif args and args[0] == '--add-nhanes':
        add_ansur(args[1:] or list(DATASETS), prefix='nhanes')
    elif args and args[0] == '--add-camerahmr':
        add_camerahmr(args[1:] or list(DATASETS))
    elif args and args[0] == '--sam2':       # same run, SAM 2 silhouettes -> <name>_sam2.csv
        main(args[1:] or list(DATASETS), suffix='_sam2', sam2=True)
    else:
        main(args or list(DATASETS))
