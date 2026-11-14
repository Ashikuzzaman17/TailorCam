"""
CameraHMR (Patel & Black, 3DV 2025) inside TailorCam.

A 3D body model trained largely on BEDLAM; predicts SMPL body shape plus the
camera's field of view from one photo. The original demo finds people with
Detectron2 (hard to install on Windows) — here MediaPipe's keypoints give the
person box instead; everything after that follows the authors' code exactly.

Files (assets/camerahmr, from camerahmr.is.tue.mpg.de):
  camerahmr_checkpoint_cleaned.ckpt   CameraHMR model
  cam_model_cleaned.ckpt              HumanFoV model
  code/data/smpl_mean_params.npz      SMPL mean params
  code/                               github.com/pixelite1201/CameraHMR
SMPL body model: model/smpl_weights/smpl/SMPL_NEUTRAL.pkl (linked into code/data/models/SMPL)
Licence: CameraHMR / SMPL are for non-commercial research only.
"""

import os
import sys
import contextlib

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
CODE = data_path('camerahmr', 'code')
_NEEDED = [data_path('camerahmr', 'camerahmr_checkpoint_cleaned.ckpt'),
           data_path('camerahmr', 'cam_model_cleaned.ckpt'),
           os.path.join(CODE, 'data', 'smpl_mean_params.npz'),
           os.path.join(CODE, 'data', 'models', 'SMPL', 'SMPL_NEUTRAL.pkl')]


def is_available():
    return all(os.path.exists(p) for p in _NEEDED)


def _resize_image(img, target_size):
    """mesh_estimator.resize_image (copied: that module imports Detectron2)."""
    import cv2
    h, w = img.shape[:2]
    if w / h > 1:
        nw, nh = target_size, int(target_size / (w / h))
    else:
        nw, nh = int(target_size * (w / h)), target_size
    out = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    x0, y0 = (target_size - nw) // 2, (target_size - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return w / h, out


@contextlib.contextmanager
def _cwd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class CameraHMRRunner:
    def __init__(self, device=None):
        import smplx
        from torchvision.transforms import Normalize
        if CODE not in sys.path:
            sys.path.insert(0, CODE)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        with _cwd(CODE):                       # the authors' paths are relative to their repo
            from core.camerahmr_model import CameraHMR
            from core.cam_model.fl_net import FLNet
            from core.constants import (CHECKPOINT_PATH, CAM_MODEL_CKPT, SMPL_MODEL_PATH,
                                        IMAGE_MEAN, IMAGE_STD, NUM_BETAS)
            self.model = CameraHMR.load_from_checkpoint(
                CHECKPOINT_PATH, strict=False, model_type='smpl',
                map_location=self.device, weights_only=False)   # official MPI file, has a DictConfig.to(self.device).eval()
            fl = FLNet()
            fl.load_state_dict(torch.load(CAM_MODEL_CKPT, map_location='cpu',
                                          weights_only=False)['state_dict'])
            self.cam_model = fl.to(self.device).eval()
            self.body = smplx.SMPL(model_path=SMPL_MODEL_PATH, num_betas=NUM_BETAS).to(self.device)
        self.normalize = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        self.faces = np.asarray(self.body.faces, np.int64)
        self._faces_t = torch.as_tensor(self.faces, device=self.device)
        self.density = 1010.0            # kg/m³, whole body (used by fit_weight)

    @torch.no_grad()
    def _cam_intrinsics(self, img_rgb):
        from core.constants import IMAGE_SIZE
        h, w = img_rgb.shape[:2]
        _, full = _resize_image(img_rgb, IMAGE_SIZE)
        t = torch.from_numpy(np.transpose(full.astype('float32'), (2, 0, 1)) / 255.0)
        fov, _ = self.cam_model(self.normalize(t).unsqueeze(0).to(self.device))
        f = (h / (2 * torch.tan(fov[0, 1] / 2))).item()
        return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)

    @torch.no_grad()
    def predict(self, img_bgr, kps33, min_vis=0.3, pad=(0.10, 0.12, 0.10, 0.06), tta=True):
        """
        One photo + MediaPipe keypoints → {'betas': (10,), 'v_shaped': (6890, 3), pose,
        camera} in the SMPL neutral body space, or None.
        tta: also run the mirrored photo and average the two body shapes (tested
        2026-10-20: small, consistent gain on real24 and HBW, e.g. HBW waist without
        weight 9.6 → 9.2 cm). Pose and camera always come from the original photo.
        """
        r = self._predict_once(img_bgr, kps33, min_vis, pad, mirror=False)
        if r is not None and tta:
            m = self._predict_once(img_bgr, kps33, min_vis, pad, mirror=True)
            if m is not None:
                r['betas'] = (r['betas'] + m['betas']) / 2
                r['v_shaped'] = self.rest_mesh(r['betas'])
        return r

    @torch.no_grad()
    def _predict_once(self, img_bgr, kps33, min_vis, pad, mirror):
        import cv2
        from core.datasets.dataset import Dataset
        from core.utils import recursive_to
        k = np.asarray(kps33, np.float32)
        vis = k[k[:, 2] > min_vis]
        if len(vis) < 6:
            return None
        lo, hi = vis[:, :2].min(0), vis[:, :2].max(0)
        span = hi - lo
        # keypoints stop short of skin / hair: pad = (left/right, top, -, bottom) fractions
        lo, hi = lo - span * [pad[0], pad[1]], hi + span * [pad[2], pad[3]]
        if mirror:                          # same person, mirrored photo (test-time augmentation)
            img_bgr = img_bgr[:, ::-1].copy()
            w_ = img_bgr.shape[1]
            lo, hi = np.array([w_ - 1 - hi[0], lo[1]]), np.array([w_ - 1 - lo[0], hi[1]])
        center = ((lo + hi) / 2)[None]
        scale = ((hi - lo) / 200.0)[None]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        cam_int = self._cam_intrinsics(img_rgb)
        with _cwd(CODE):
            ds = Dataset(img_rgb, center, scale, cam_int, False, 'frame')
            batch = torch.utils.data.default_collate([ds[0]])
            batch = recursive_to(batch, self.device)
            params, pred_cam, _ = self.model(batch)
        betas = params['betas'][0].float()
        v = self.body(betas=betas[None]).vertices        # posed at zero pose = T-pose
        # camera translation in full-image coordinates (authors' convert_to_full_img_cam)
        s, tx, ty = pred_cam[0, 0], pred_cam[0, 1], pred_cam[0, 2]
        bh, (bx, by) = batch['box_size'][0], batch['box_center'][0]
        h, w = img_bgr.shape[:2]
        cam_t = torch.stack([tx + 2 * (bx - w / 2) / (s * bh), ty + 2 * (by - h / 2) / (s * bh),
                             2 * float(cam_int[0, 0]) / (bh * s)])
        return {'betas': betas.cpu().numpy(), 'v_shaped': v[0].cpu().numpy(),
                'global_orient': params['global_orient'][0].float().cpu().numpy(),   # (1, 3, 3)
                'body_pose': params['body_pose'][0].float().cpu().numpy(),           # (23, 3, 3)
                'cam_t': cam_t.float().cpu().numpy(), 'cam_int': cam_int}

    def posed_vertices(self, betas, global_orient, body_pose):
        """Vertices in the camera's frame (before translation), differentiable in betas."""
        return self.body(betas=betas, global_orient=global_orient, body_pose=body_pose,
                         pose2rot=False).vertices

    # ── same tools as HMRBody, on CameraHMR's own SMPL model ──────────────────
    def rest_mesh(self, betas):
        b = torch.as_tensor(np.asarray(betas, np.float32), device=self.device)[None]
        with torch.no_grad():
            return self.body(betas=b).vertices[0].cpu().numpy()

    def arms_down_mesh(self, betas, angle_deg=70.0):
        b = torch.as_tensor(np.asarray(betas, np.float32), device=self.device)[None]
        pose = torch.zeros(1, 69, device=self.device)        # body pose (no root)
        a = np.radians(angle_deg)
        pose[0, 15 * 3 + 2] = -a                              # joint 16 = left shoulder
        pose[0, 16 * 3 + 2] = a                               # joint 17 = right shoulder
        with torch.no_grad():
            return self.body(betas=b, body_pose=pose).vertices[0].cpu().numpy()

    def mass_kg(self, b, height_cm):
        V = self.body(betas=b[None]).vertices[0]
        k = (height_cm / 100.0) / (V[:, 1].max() - V[:, 1].min())
        F = self._faces_t
        a, bb, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        return self.density * (a * torch.cross(bb, c, dim=1)).sum().abs() / 6.0 * k ** 3

    def fit_weight(self, betas, height_cm, weight_kg, steps=200):
        """Smallest shape change that gives the real body weight (as HMRBody.fit_weight)."""
        b0 = torch.as_tensor(np.asarray(betas, np.float32), device=self.device)
        b = b0.clone().requires_grad_(True)
        opt = torch.optim.Adam([b], lr=0.05)
        for _ in range(steps):
            m = self.mass_kg(b, height_cm)
            loss = 100 * ((m - weight_kg) / weight_kg) ** 2 + 0.01 * ((b - b0) ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        return b.detach().cpu().numpy()

    def measure_person(self, betas_list, height_cm, weight_kg=None, rings=None):
        """Median shape over all photos (+ weight fit) → TailorCam measurements."""
        from model.hmr.body_points import measure
        b = np.median(np.asarray(betas_list), axis=0)
        if weight_kg:
            b = self.fit_weight(b, height_cm, weight_kg)
        m = measure(self.rest_mesh(b), self.faces, height_cm, rings, self.arms_down_mesh(b))
        return {k: v for k, v in m.items() if k != 'height'}, b
