"""
Measurements on SMPL-X meshes (10,475 vertices) — the body format used by
SHAPY and by the HBW ground-truth scans.

Uses SHAPY's own landmark definitions (assets/shapy/shapy_data/utility_files/
measurements/smplx_measurements.yaml):
    chest = horizontal girth at the right nipple
    waist = horizontal girth at the belly button
    hip   = horizontal girth at crotch height (around both legs)
    height = head top ↔ left heel
Each girth is the convex-hull perimeter of the horizontal slice (a tape
bridging hollows), exactly like SHAPY's BodyMeasurements — but computed with
TailorCam's mesh slicer instead of SHAPY's CUDA extension.
"""

import os
import numpy as np
import yaml

from model.hmr.body_points import _slice, _hull_perimeter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
_LANDMARKS = data_path('shapy', 'shapy_data', 'utility_files',
                          'measurements', 'smplx_measurements.yaml')
_UP = np.array([0.0, 1.0, 0.0])
_lm_cache = {}


def landmarks():
    if not _lm_cache:
        _lm_cache.update(yaml.safe_load(open(_LANDMARKS)))
    return _lm_cache


def landmark(V, F, name):
    d = landmarks()[name]
    return (V[F[d['face_idx']]] * np.array(d['bc'])[:, None]).sum(0)


def girth_at(V, F, y):
    comps = _slice(V, F, np.array([0.0, y, 0.0]), _UP)
    return _hull_perimeter(np.vstack(comps), _UP) if comps else 0.0


def measure_smplx(V, F, height_cm=None):
    """
    V: 10475 x 3 T-pose SMPL-X vertices (metres), F: faces.
    height_cm: scale the body to this height (None = keep the mesh's size).
    Returns {'height', 'chest', 'waist', 'hip', 'mass_kg'} in cm / kg.
    """
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    top, heel = landmark(V, F, 'HeadTop'), landmark(V, F, 'HeelLeft')
    mesh_h = abs(top[1] - heel[1])
    k = (height_cm / 100.0) / mesh_h if height_cm else 1.0
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    volume = abs((a * np.cross(b, c)).sum()) / 6.0 * k ** 3
    out = {'height': mesh_h * k * 100}
    for key, lm in (('chest', 'NippleRight'), ('waist', 'BellyButton'), ('hip', 'Crotch')):
        out[key] = girth_at(V, F, landmark(V, F, lm)[1]) * k * 100
    out['mass_kg'] = volume * 985.0          # SHAPY's body density
    return {k_: round(float(v), 1) for k_, v in out.items()}


class SMPLXShape:
    """Neutral SMPL-X shape space (10 betas) — rebuilds SHAPY's T-pose bodies
    exactly and adjusts a shape to a known body weight."""

    DENSITY = 985.0

    def __init__(self, device=None):
        import torch
        import smplx
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        m = smplx.create(data_path('shapy', 'smplx', 'models'),
                         model_type='smplx', gender='neutral', num_betas=10,
                         ext='npz', use_pca=False)
        self.v_template = m.v_template.to(self.device)
        self.shapedirs = m.shapedirs[:, :, :10].to(self.device)
        self.faces = np.asarray(m.faces, np.int64)
        self._F = torch.as_tensor(self.faces, device=self.device)
        top, heel = landmarks()['HeadTop'], landmarks()['HeelLeft']
        self._h_idx = [(torch.as_tensor(self.faces[d['face_idx']], device=self.device),
                        torch.tensor(d['bc'], device=self.device, dtype=torch.float32))
                       for d in (top, heel)]

    def shaped(self, betas):
        b = self.torch.as_tensor(np.asarray(betas, np.float32), device=self.device)
        return self.v_template + self.torch.einsum('b,vcb->vc', b, self.shapedirs)

    def _mass(self, V, height_cm):
        (fa, ba), (fb, bb) = self._h_idx
        h = (V[fa] * ba[:, None]).sum(0)[1] - (V[fb] * bb[:, None]).sum(0)[1]
        k = (height_cm / 100.0) / h.abs()
        a, b, c = V[self._F[:, 0]], V[self._F[:, 1]], V[self._F[:, 2]]
        return self.DENSITY * (a * self.torch.cross(b, c, dim=1)).sum().abs() / 6.0 * k ** 3

    def fit_weight(self, betas, height_cm, weight_kg, steps=200):
        """Smallest shape change that gives the customer's real weight."""
        torch = self.torch
        b0 = torch.as_tensor(np.asarray(betas, np.float32), device=self.device)
        b = b0.clone().requires_grad_(True)
        opt = torch.optim.Adam([b], lr=0.05)
        for _ in range(steps):
            V = self.v_template + torch.einsum('b,vcb->vc', b, self.shapedirs)
            loss = 100 * ((self._mass(V, height_cm) - weight_kg) / weight_kg) ** 2 \
                + 0.01 * ((b - b0) ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        return b.detach().cpu().numpy()

    def measure(self, betas, height_cm, weight_kg=None):
        """Median/any betas → tape measurements at the true height (and weight)."""
        if weight_kg:
            betas = self.fit_weight(betas, height_cm, weight_kg)
        V = self.shaped(betas).detach().cpu().numpy()
        return measure_smplx(V, self.faces, height_cm)


def load_obj(path):
    V, F = [], []
    for line in open(path):
        if line.startswith('v '):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            F.append([int(x.split('/')[0]) - 1 for x in line.split()[1:4]])
    return np.array(V), np.array(F)
