"""
BODY-fit (Tampere University, Zenodo 5172126, CC BY 4.0) → training cache for
model/bodyfit/train_bodyfit.py.

  4,149 real people (1,474 male, 2,675 female): SMPL fits (6,890 vertices) of
  their 3D body scans, plus about 50 body measurements each (mm).

For each person: render a front and a left-side silhouette of the fitted body
(perspective camera 3 m away, like a photo), pass both through the same
prep_mask the app uses, and take
  height = Head_Top_Height, weight = mesh volume x 1,010 kg/m3,
  labels = chest/bust, waist, hip, neck, thigh, shoulder, sleeve, ...
Split 80 / 5 / 15 % (train / val / test) by person, fixed seed.

Expects dataset/bodyfit/meas/bodymeasurement_mat/{male,female}/*.mat and the
unzipped fits under dataset/bodyfit/fits/.

  python scripts/build_bodyfit_dataset.py
"""

import os
import sys
import glob

import cv2
import numpy as np
import scipy.io as sio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
sys.path.insert(0, ROOT)
from model.bodym.prep import prep_mask
from model.bodyfit.train_bodyfit import KEYS, DATA

BF = data_path('bodyfit')
DENSITY = 1010.0
# TailorCam key → BODY-fit field(s), first one present wins (male / female names differ)
FIELDS = {'chest': ['CHEST_Circ', 'BUST_Circ'], 'waist': ['MaxWAIST_Circ', 'NaturalWAIST_Circ'],
          'hip': ['HIP_Circ'], 'neck': ['Neck_Circ', 'NeckBase_Circ'], 'thigh': ['Thigh_Circ'],
          'shoulder_width': ['Shoulder_to_Shoulder'], 'sleeve_length': ['Shoulder_to_Wrist'],
          'bicep': ['Bicep_Circ'], 'wrist': ['Wrist_Circ'], 'calf': ['Calf_Circ'], 'ankle': ['Ankle_Circ']}


def smpl_faces():
    return np.load(os.path.join(ROOT, 'model', 'hmr', 'smpl_hmr.npz'))['faces'].astype(np.int64)


def read_meas(path):
    s = sio.loadmat(path)['s'][0, 0]
    val = {}
    for n in s.dtype.names:
        v = np.squeeze(s[n])
        if v.size == 1:
            try:
                val[n] = float(v)
            except (TypeError, ValueError):
                pass
    return val


def read_mesh(path):
    """Vertices (6890 x 3, metres) from .obj / .ply (ascii) / .mat / .npy."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.obj':
        V = np.array([l.split()[1:4] for l in open(path) if l.startswith('v ')], float)
    elif ext == '.ply':
        lines = open(path, errors='ignore').read().splitlines()
        n = int(next(l for l in lines if l.startswith('element vertex')).split()[-1])
        start = lines.index('end_header') + 1
        V = np.array([l.split()[:3] for l in lines[start:start + n]], float)
    elif ext == '.mat':
        m = sio.loadmat(path)
        k = next(k for k in m if not k.startswith('__') and np.asarray(m[k]).ndim == 2 and 6890 in np.asarray(m[k]).shape)
        V = np.asarray(m[k], float)
        V = V if V.shape[1] == 3 else V.T
    else:
        V = np.load(path, allow_pickle=False).astype(float)
    h = V[:, 1].max() - V[:, 1].min()
    return V / (1000.0 if h > 100 else 100.0 if h > 10 else 1.0)


def volume(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return abs(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0)


def render(V, F, axis):
    """Binary silhouette, camera 3 m in front (axis='z') or at the left side (axis='x')."""
    P = V - V.mean(0)
    u = P[:, 0] if axis == 'z' else -P[:, 2]
    depth = 3.0 - (P[:, 2] if axis == 'z' else P[:, 0])
    f = 700.0
    x, y = f * u / depth, -f * P[:, 1] / depth
    W, H = 600, 900
    pts = np.stack([x + W / 2, y + H / 2], 1)
    img = np.zeros((H, W), np.uint8)
    cv2.fillPoly(img, list(np.round(pts[F] * 4).astype(np.int32)), 255, shift=2)   # sub-pixel edges
    # fill the pin-holes between tiny triangles: everything not reachable from the border is body
    bg = img.copy()
    cv2.floodFill(bg, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 255)
    return img | cv2.bitwise_not(bg)


def main():
    F = smpl_faces()
    fits = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(BF, 'fits', '**', '*.*'), recursive=True)
            if os.path.splitext(p)[1].lower() in ('.obj', '.ply', '.mat', '.npy')}
    meas = sorted(glob.glob(os.path.join(BF, 'meas', 'bodymeasurement_mat', '*', '*.mat')))
    print(f'{len(meas)} measurement files, {len(fits)} mesh files', flush=True)
    fr, sd, Y, h, w, g, names = [], [], [], [], [], [], []
    for i, mp in enumerate(meas):
        pid = os.path.splitext(os.path.basename(mp))[0]
        if pid not in fits:
            continue
        m = read_meas(mp)
        V = read_mesh(fits[pid])
        if V.shape[0] != 6890:
            continue
        H_cm = m.get('Head_Top_Height', np.nan) / 10.0
        if not 120 < H_cm < 220:
            continue
        a, b = prep_mask(render(V, F, 'z')), prep_mask(render(V, F, 'x'))
        if a is None or b is None:
            continue
        y = []
        for k in KEYS:
            v = next((m[f] for f in FIELDS[k] if f in m and np.isfinite(m[f]) and m[f] > 0), np.nan)
            y.append(v / 10.0)
        fr.append(a.astype(np.uint8)); sd.append(b.astype(np.uint8)); Y.append(y)
        h.append(H_cm); w.append(volume(V, F) * DENSITY); g.append(1 if pid.startswith('male') else 0); names.append(pid)
        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{len(meas)}', flush=True)
    n = len(names)
    rng = np.random.default_rng(0)
    r = rng.random(n)
    split = np.where(r < 0.80, 0, np.where(r < 0.85, 1, 2))
    np.savez_compressed(DATA, front=np.array(fr), side=np.array(sd), y=np.array(Y, np.float32),
                        h=np.array(h, np.float32), w=np.array(w, np.float32), g=np.array(g, np.int8),
                        split=split.astype(np.int8), names=np.array(names, dtype='U16'))
    Yn = np.array(Y)
    print(f'saved {DATA}: {n} people (train {np.sum(split == 0)}, val {np.sum(split == 1)}, test {np.sum(split == 2)})')
    print('weight kg: mean %.1f, range %.0f-%.0f' % (np.mean(w), np.min(w), np.max(w)))
    for j, k in enumerate(KEYS):
        print(f'  {k:15s} n={np.isfinite(Yn[:, j]).sum():5d}  mean {np.nanmean(Yn[:, j]):6.1f} cm')
    cv2.imwrite(os.path.join(BF, 'example_silhouettes.png'),
                np.hstack([fr[0], sd[0], fr[-1], sd[-1]]) * 255)


if __name__ == '__main__':
    main()
