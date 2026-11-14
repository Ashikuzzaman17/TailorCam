"""
HMR body-shape estimation for TailorCam.

  photo(s) + height  →  HMR (ResNet-50 regressor)  →  SMPL betas
                     →  T-pose mesh scaled to the true height
                     →  tape-style measurements (see body_points.py)

Why T-pose: the original HMR measurement script measured the *posed* mesh,
so bent arms / turned torso changed the numbers. Measuring the neutral
rest pose of the predicted shape removes that error source entirely.
"""

import os
import numpy as np
import cv2
import torch

from .hmr_net import HMR
from .smpl_hmr import SMPL

_HERE = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS = os.path.join(_HERE, 'hmr_weights.pt')
_SMPL_NPZ = os.path.join(_HERE, 'smpl_hmr.npz')
_POSE_TASK = os.path.join(os.path.dirname(os.path.dirname(_HERE)), 'pose_landmarker.task')

IMG_SIZE = 224
_PERSON_PX = 150.0          # HMR was trained with the person ~150 px tall in a 224 crop
_MIN_VIS = 0.3


def is_available():
    return os.path.exists(_WEIGHTS) and os.path.exists(_SMPL_NPZ)


class HMRBody:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = HMR()
        self.net.load_state_dict(torch.load(_WEIGHTS, map_location='cpu', weights_only=True))
        self.net.eval().to(self.device)
        self.smpl = SMPL(_SMPL_NPZ).to(self.device)
        self.faces = self.smpl.faces
        self._faces_t = torch.as_tensor(self.faces, device=self.device)
        self._landmarker = None

    # ── Person box ────────────────────────────────────────────────────────────
    def analyze(self, img_bgr):
        """
        MediaPipe on one photo → (keypoints, mask). keypoints: list of 33
        (x_px, y_px, visibility) or None; mask: HxW float person mask or None.
        """
        import mediapipe as mp
        if self._landmarker is None:
            opts = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=_POSE_TASK),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_poses=1, output_segmentation_masks=True)
            self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(opts)
        h, w = img_bgr.shape[:2]
        res = self._landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
        if not res.pose_landmarks:
            return None, None
        kps = [(l.x * w, l.y * h, getattr(l, 'visibility', 1.0))
               for l in res.pose_landmarks[0]]
        mask = (res.segmentation_masks[0].numpy_view().copy()
                if res.segmentation_masks else None)
        return kps, mask

    def _keypoints(self, img_bgr):
        return self.analyze(img_bgr)[0]

    @staticmethod
    def _box_from_keypoints(kps):
        """Same rule as HMR's openpose bbox: scale so the person is ~150 px."""
        pts = np.array([(x, y) for x, y, v in kps if v > _MIN_VIS])
        if len(pts) < 6:
            return None
        lo, hi = pts.min(0), pts.max(0)
        # MediaPipe stops at the heel / forehead; pad ~6% so head & feet fit
        span = hi - lo
        lo, hi = lo - span * [0.0, 0.06], hi + span * [0.0, 0.03]
        person = float(np.max(hi - lo))
        return _PERSON_PX / max(person, 1.0), (lo + hi) / 2.0

    # ── Pre-processing (matches src/util/image.py scale_and_crop) ─────────────
    @staticmethod
    def _crop(img_rgb, scale, center):
        h, w = img_rgb.shape[:2]
        nh, nw = int(np.floor(h * scale)), int(np.floor(w * scale))
        scaled = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA
                            if scale < 1 else cv2.INTER_LINEAR)
        sf = np.array([nw / w, nh / h])
        c = np.round(center * sf).astype(int)
        m = IMG_SIZE // 2
        pad = np.pad(scaled, ((m, m), (m, m), (0, 0)), mode='edge')
        start = c  # (c + m) - m
        crop = pad[max(start[1], 0):start[1] + IMG_SIZE, max(start[0], 0):start[0] + IMG_SIZE]
        if crop.shape[:2] != (IMG_SIZE, IMG_SIZE):     # person box ran off the photo
            ph, pw = IMG_SIZE - crop.shape[0], IMG_SIZE - crop.shape[1]
            crop = np.pad(crop, ((0, max(ph, 0)), (0, max(pw, 0)), (0, 0)), mode='edge') \
                if crop.size else np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
            crop = crop[:IMG_SIZE, :IMG_SIZE]
        return crop, {'sf': sf, 'start': start - m}

    # ── Inference ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, img_bgr, keypoints=None):
        """
        Run HMR on one photo. keypoints: optional list of 33 (x,y,vis) in px.
        Returns dict with theta, betas, verts (posed, crop space), joints2d (px
        in the original image) and the keypoints used, or None if no person.
        """
        kps = keypoints if keypoints is not None else self._keypoints(img_bgr)
        if kps is None:
            return None
        box = self._box_from_keypoints(kps)
        if box is None:
            return None
        scale, center = box
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        crop, proc = self._crop(rgb, scale, center)
        x = torch.from_numpy(crop).float().permute(2, 0, 1)[None] / 255.0
        x = (2.0 * (x - 0.5)).to(self.device)

        theta = self.net(x)[0]
        cam, pose, betas = theta[:3], theta[3:75], theta[75:]
        verts, joints = self.smpl(betas[None], pose[None])
        # Orthographic projection → crop px → original px
        j2 = cam[0] * (joints[0, :, :2] + cam[1:])
        j2 = ((j2 + 1) * 0.5 * IMG_SIZE).cpu().numpy()
        j2 = (j2 + proc['start']) / proc['sf']
        return {
            'theta': theta.cpu().numpy(),
            'betas': betas.cpu().numpy(),
            'pose': pose.cpu().numpy(),
            'verts': verts[0].cpu().numpy(),
            'joints2d': j2,
            'keypoints': kps,
            'crop': crop,
        }

    @torch.no_grad()
    def rest_mesh(self, betas):
        """T-pose vertices (metres, Y up) for the given betas."""
        b = torch.as_tensor(np.asarray(betas, np.float32), device=self.device)[None]
        return self.smpl.shaped(b)[0].cpu().numpy()

    def mass_kg(self, betas, height_cm):
        """Body mass of the mesh scaled to height (whole-body density 1010 kg/m³)."""
        V = self.smpl.shaped(betas[None])[0]
        k = (height_cm / 100.0) / (V[:, 1].max() - V[:, 1].min())
        F = self._faces_t
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        vol = (a * torch.cross(b, c, dim=1)).sum().abs() / 6.0
        return 1010.0 * vol * k ** 3

    def fit_weight(self, betas, height_cm, weight_kg, steps=200):
        """
        Smallest change to the HMR shape that gives the customer's real body
        weight. HMR alone pulls every body toward average; the known weight
        fixes the overall volume (halved girth errors in testing).
        """
        b0 =torch.as_tensor(np.asarray(betas, np.float32), device=self.device)
        b = b0.clone().requires_grad_(True)
        opt = torch.optim.Adam([b], lr=0.05)
        for _ in range(steps):
            m = self.mass_kg(b, height_cm)
            loss = 100 * ((m - weight_kg) / weight_kg) ** 2 + 0.01 * ((b - b0) ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        return b.detach().cpu().numpy()

    @torch.no_grad()
    def arms_down_mesh(self, betas, angle_deg=70.0):
        """Same body with both arms lowered (how a tailor measures the sleeve)."""
        b = torch.as_tensor(np.asarray(betas, np.float32), device=self.device)[None]
        pose = torch.zeros(1, 72, device=self.device)
        a = np.radians(angle_deg)
        pose[0, 16 * 3 + 2] = -a          # left shoulder: +X arm rotates down
        pose[0, 17 * 3 + 2] = a           # right shoulder
        return self.smpl(b, pose)[0][0].cpu().numpy()

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
