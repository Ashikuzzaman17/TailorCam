"""
BEDLAM (Black et al., CVPR 2023) → a TailorCam test set with exact truth.

BEDLAM renders realistic clothed people (simulated clothes, real skin textures,
hair) in 3D scenes; for every image the exact body (SMPL shape) is known. The
true measurements below are taken from that exact body with the same tape
code the app uses (model/hmr/body_points.py) — so they have no tape error, but
they follow TailorCam's own definition of chest / waist / hip.

Needs, in dataset/bedlam (from bedlam.is.tue.mpg.de, non-commercial licence):
  images/<scene>/png/seq_*/...png       an extracted "6fps" image tar
  labels/bedlam-labels/<scene>.npz      from bedlam-labels-smpl.zip

Output: dataset/bedlam/people.csv — one row per person (sequence): gender, height,
weight (mesh volume × 1010 kg/m³), true measurements, and up to 4 frames in
which head, shoulders, hips and knees are in view.

NOTE: CameraHMR was trained on BEDLAM's training scenes, so its BEDLAM score is
not a fair test; HMR, SHAPY, the hybrid model and the priors never saw BEDLAM.

  python scripts/build_bedlam_dataset.py
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
sys.path.insert(0, ROOT)
from model.hmr.body_points import measure, load_rings

D = data_path('bedlam')
SMPL = os.path.join(ROOT, 'model', 'smpl_weights', 'smpl_nochumpy', 'SMPL_NEUTRAL.pkl')
FRAMES_PER_PERSON = 4


# OpenPose BODY_25 joints that must be in the picture: head, shoulders, hips, knees
# (zoom/close-up scenes rarely show the feet, which no measurement here needs)
NEEDED_JOINTS = [0, 1, 2, 5, 8, 9, 10, 12, 13]


def full_body_frames(kps, w, h, margin=5):
    """Indices of frames in which head, shoulders, hips and knees are all inside the image."""
    k = kps[:, NEEDED_JOINTS, :2]           # BEDLAM gtkps: first 25 follow OpenPose BODY_25
    inside = ((k[..., 0] > margin) & (k[..., 0] < w - margin) &
              (k[..., 1] > margin) & (k[..., 1] < h - margin)).all(1)
    return np.flatnonzero(inside)


def main():
    import smplx
    body = smplx.SMPL(model_path=SMPL, num_betas=11)
    F = body.faces.astype(np.int64)
    rings = load_rings()
    rows = []
    for lab in sorted(glob.glob(os.path.join(D, 'labels', 'bedlam-labels', '*.npz'))):
        scene = os.path.basename(lab)[:-4]
        img_dir = os.path.join(D, 'images', scene, 'png')
        if not os.path.isdir(img_dir):
            continue                         # labels for scenes whose images were not downloaded
        d = np.load(lab, allow_pickle=True)
        names = d['imgname']
        seq = np.array([n.split('/')[0] for n in names])
        for s in np.unique(seq):
            idx = np.flatnonzero(seq == s)
            cx, cy = d['cam_int'][idx[0]][:2, 2]
            ok = idx[full_body_frames(d['gtkps'][idx], 2 * cx, 2 * cy)]
            ok = [i for i in ok if os.path.exists(os.path.join(img_dir, names[i]))]
            if not ok:
                continue
            pick = [ok[j] for j in np.linspace(0, len(ok) - 1, min(FRAMES_PER_PERSON, len(ok))).round().astype(int)]
            b = torch.tensor(d['shape'][idx[0]][None], dtype=torch.float32)
            with torch.no_grad():
                V = body(betas=b).vertices[0].numpy()
                pose = torch.zeros(1, 69)
                pose[0, 15 * 3 + 2], pose[0, 16 * 3 + 2] = -np.radians(70), np.radians(70)
                Vd = body(betas=b, body_pose=pose).vertices[0].numpy()
            height = float(V[:, 1].max() - V[:, 1].min()) * 100
            a, bb, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
            mass = 1010.0 * abs(np.einsum('ij,ij->i', a, np.cross(bb, c)).sum()) / 6.0
            m = measure(V, F, height, rings, Vd)
            rows.append({'person_id': f'{scene}/{s}', 'scene': scene, 'gender': str(d['gender'][idx[0]]),
                         'height_cm': round(height, 1), 'weight_kg': round(float(mass), 1),
                         **{k: round(float(v), 2) for k, v in m.items() if k != 'height'},
                         'frames': ';'.join(os.path.join(img_dir, names[i]) for i in pick)})
        print(f'{scene}: {sum(r["scene"] == scene for r in rows)} people', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(D, 'people.csv'), index=False)
    print(f'{len(out)} people -> dataset/bedlam/people.csv')
    print(out[['height_cm', 'weight_kg', 'chest', 'waist', 'hip']].describe().round(1))


if __name__ == '__main__':
    main()
