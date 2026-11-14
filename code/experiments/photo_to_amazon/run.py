"""
Experiment (separate from the app): turn real RGB photos into Amazon BodyM-style
silhouettes in several ways, measure them with the BodyM CNN ensemble, and
compare accuracy on real tape-measured people. Nothing in the app is changed.

Variants (front + side silhouettes given to the same CNN):
  V0 prior            no photos, height / weight / gender only
  V1 mp_always        MediaPipe masks, used even with the arms down
  V2 mp_armsaway      MediaPipe masks only when the arms are away (the app's rule)
  V3 sam2             SAM 2 masks (if SAM 2 loads)
  V4 camerahmr_apose  CameraHMR 3D body (weight-fitted if weight known), re-posed
                      with the arms away like BodyM, rendered front + side
  V5 hmr_apose        the same with HMR

Datasets: real24 (front + side photos, tape) and HBW (photos, 3D-scan truth; a
front and a side photo are picked with the app's view guess).

  python experiments/photo_to_amazon/run.py           # → predictions.csv
  python experiments/photo_to_amazon/evaluate.py      # → accuracy tables
"""

import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from model.hmr.hmr_body import HMRBody
from model.hmr.hmr_worker import (arms_away, ARMS_AWAY_MIN, photo_check, _load_hybrid, _load_camerahmr,
                                  _load_segmenter)
from build_bodyfit_dataset import render

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'predictions.csv')
APOSE_DEG = 50.0          # lower the T-pose arms by 50 deg → arms ~40 deg from the body, like BodyM
KEYS = ('chest', 'waist', 'hip', 'thigh', 'arm-length', 'shoulder-breadth')


def people_real24():
    d = pd.read_csv(data_path('real_photos', 'people.csv'))
    for r in d.itertuples():
        p = data_path('real_photos', r.person_id)
        truth = {'chest': r.chest, 'waist': r.waist, 'hip': r.hip, 'thigh': r.thigh,
                 'arm-length': r.arm_length, 'shoulder-breadth': r.shoulder_width}
        flagged = set(str(r.flagged).split(';'))
        truth = {k: (v if pd.notna(v) and {'arm-length': 'arm_length', 'shoulder-breadth': 'shoulder_width'}.get(k, k)
                     not in flagged else np.nan) for k, v in truth.items()}
        yield 'real24', r.person_id, r.height_cm, r.weight_kg, r.gender, [os.path.join(p, 'front.jpg')], \
            [os.path.join(p, 'side.jpg')], truth


def people_hbw(hb):
    from model.shapy_measure import load_obj, measure_smplx
    d = data_path('shapy', 'HBW_low_resolution')
    for obj in sorted(glob.glob(os.path.join(d, 'smplx', 'val', '*.obj'))):
        sid = os.path.basename(obj)[:-4]
        V, F = load_obj(obj)
        gt = measure_smplx(V, F)
        imgs = sorted(glob.glob(os.path.join(d, 'images', 'val_small_resolution', f'{sid}_*', '*', '*.png')))
        fronts, sides = [], []
        for p in imgs:
            img = cv2.imread(p)
            k, _ = hb.analyze(img)
            if k is None:
                continue
            (fronts if photo_check(k, img.shape)['view_guess'] == 'FRONT' else sides).append(p)
        if fronts and sides:
            yield 'hbw', f'hbw_{sid}', gt['height'], gt['mass_kg'], 'unknown', fronts[:3], sides[:3], \
                {'chest': gt['chest'], 'waist': gt['waist'], 'hip': gt['hip']}


def main():
    hb, hy, cam = HMRBody(), _load_hybrid(), _load_camerahmr()
    try:
        seg = _load_segmenter()
    except Exception as exc:
        print('SAM 2 not available:', exc)
        seg = None
    rows = []
    people = list(people_real24()) + list(people_hbw(hb))
    for n, (ds, pid, h, w, g, fpaths, spaths) in enumerate([p[:7] for p in people]):
        truth = people[n][7]
        g = g if g in ('male', 'female') else 'unknown'
        data = {}
        for view, paths in (('front', fpaths), ('side', spaths)):
            for p in paths:
                img = cv2.imread(p)
                k, m = hb.analyze(img)
                if k is None:
                    continue
                sm = seg.mask(img, k) if seg is not None else None
                data.setdefault(view, []).append((img, k, m, sm))
        if 'front' not in data or 'side' not in data:
            continue
        fr, sd = data['front'], data['side']
        aa = np.median([arms_away(k) or 0 for _, k, _, _ in fr])
        mp_views = [(f[2], s[2]) for f in fr for s in sd if f[2] is not None and s[2] is not None]
        sam_views = [(f[3], s[3]) for f in fr for s in sd if f[3] is not None and s[3] is not None]
        all_imgs = fr + sd
        cb = [c['betas'] for c in (cam.predict(img, k) for img, k, _, _ in all_imgs) if c is not None]
        hbetas = [r['betas'] for r in (hb.predict(img, k) for img, k, _, _ in all_imgs) if r is not None]

        weights = [w, None] if pd.notna(w) else [None]
        apose = {}                                   # (variant, weight given?) → [(front, side)]
        for name, model, betas, faces in (('V4 camerahmr_apose', cam, cb, cam.faces),
                                          ('V5 hmr_apose', hb, hbetas, hb.faces)):
            if not betas:
                continue
            b = np.median(betas, 0)
            for weight in weights:
                bb = model.fit_weight(b, h, weight) if weight else b
                V = model.arms_down_mesh(bb, APOSE_DEG)
                apose[(name, weight is not None)] = [(render(V, faces, 'z'), render(V, faces, 'x'))]
        if n == 0:                                   # example of every kind of silhouette
            from model.bodym.prep import prep_mask
            ex = [mp_views[0][0], mp_views[0][1]] + ([sam_views[0][0], sam_views[0][1]] if sam_views else []) + \
                 [x for k in (('V4 camerahmr_apose', True), ('V5 hmr_apose', True)) if k in apose for x in apose[k][0]]
            cv2.imwrite(os.path.join(HERE, 'example_silhouettes.png'),
                        np.hstack([(prep_mask(x) * 255).astype(np.uint8) for x in ex]))

        for weight in weights:
            mode = 'with_weight' if weight else 'no_weight'
            variants = {'V0 prior': [], 'V1 mp_always': mp_views,
                        'V2 mp_armsaway': mp_views if aa >= ARMS_AWAY_MIN else []}
            if seg is not None:
                variants['V3 sam2'] = sam_views
            for name in ('V4 camerahmr_apose', 'V5 hmr_apose'):
                if (name, weight is not None) in apose:
                    variants[name] = apose[(name, weight is not None)]
            for name, views in variants.items():
                if name != 'V0 prior' and not views:
                    continue
                p = hy.predict(float(h), float(weight) if weight else None, g, views)
                rows.append({'dataset': ds, 'person_id': pid, 'mode': mode, 'variant': name,
                             'arms_away': round(float(aa), 2),
                             **{f'pred:{k}': float(p[k][0]) for k in KEYS},
                             **{f'true:{k}': truth.get(k, np.nan) for k in KEYS}})
        print(f'{n + 1}/{len(people)} {pid} done (arms_away {aa:.2f})', flush=True)
    hb.close()
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print('saved', OUT)


if __name__ == '__main__':
    main()
