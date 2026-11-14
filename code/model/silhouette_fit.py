"""
Multi-photo silhouette fitting of the CameraHMR body.

CameraHMR guesses a body shape from each photo separately. Here ONE body shape
is fitted to ALL photos of the customer at once (front, side, back...):

  * the body must stay INSIDE the person's outline in every photo — clothes
    can only make the outline bigger than the body, never smaller, so this is
    a safe constraint even with loose clothes;
  * (optional, weight `cover`) the body should also FILL the outline — right
    for fitted clothes, wrong for loose ones;
  * the joints must stay on MediaPipe's keypoints (fixes position and size
    in each photo, so the outline can only change the body's proportions);
  * the shape stays close to CameraHMR's own guess, and — when known — the
    body's volume must match the customer's weight.

Pose and camera come from CameraHMR per photo; only the shared shape and a
small per-photo camera shift are optimised.

Tested 2026-10-19 (scripts/silhouette_fit_experiment.py, real24 / HBW / BEDLAM):
no gain, so the app does NOT use it. CameraHMR's body already lies inside the
outline in every photo (the "inside" rule never fires), and "fill the outline"
fits the clothes, not the body (real24 chest without weight 3.6 → 6.3 cm).
Worth re-testing only with photos of customers in tight clothes.
"""

import numpy as np
import torch
import torch.nn.functional as Fn
import cv2

# SMPL joint ↔ MediaPipe landmark: shoulders, elbows, wrists, knees, ankles
_SMPL_J = [16, 17, 18, 19, 20, 21, 4, 5, 7, 8]
_MP_J = [11, 12, 13, 14, 15, 16, 25, 26, 27, 28]


def _outside_map(mask, tol_px):
    """Distance (px) from each pixel to the person mask, minus a tolerance; 0 inside."""
    m = (np.squeeze(mask) > 0.5).astype(np.uint8)
    d = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    return np.maximum(d - tol_px, 0).astype(np.float32)


def _contour_points(mask, n=400):
    m = (np.squeeze(mask) > 0.5).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=len)[:, 0, :].astype(np.float32)
    return c[np.linspace(0, len(c) - 1, min(n, len(c))).astype(int)]


def fit_shape(runner, frames, height_cm, weight_kg=None, steps=150, w_out=1.0, cover=0.0,
              w_kp=1.0, w_prior=0.02, tol=0.005, lr=0.02):
    """
    runner: CameraHMRRunner; frames: list of dicts with CameraHMR output
    ('betas', 'global_orient', 'body_pose', 'cam_t', 'cam_int') plus 'mask' (HxW)
    and 'kps' (33 MediaPipe keypoints). Returns fitted betas (10,).
    """
    dev = runner.device
    T = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=dev)
    b0 = T(np.median([f['betas'] for f in frames], axis=0))
    if weight_kg:
        b0 = T(runner.fit_weight(b0.cpu().numpy(), height_cm, weight_kg))
    go = torch.stack([T(f['global_orient']) for f in frames])
    bp = torch.stack([T(f['body_pose']) for f in frames])
    t0 = torch.stack([T(f['cam_t']) for f in frames])
    K = torch.stack([T(f['cam_int']) for f in frames])
    n = len(frames)

    # per-photo maps, normalised by the person's height in pixels
    out_maps, scales, conts, kps, kvis = [], [], [], [], []
    for f in frames:
        k = np.asarray(f['kps'], np.float32)
        hpx = max(k[:, 1].max() - k[:, 1].min(), 50.0)
        scales.append(hpx)
        out_maps.append(T(_outside_map(f['mask'], tol * hpx)))
        conts.append(_contour_points(f['mask']))
        kps.append(k[_MP_J, :2]); kvis.append((k[_MP_J, 2] > 0.5).astype(np.float32))
    kps, kvis, scales = T(np.stack(kps)), T(np.stack(kvis)), T(scales)

    b = b0.clone().requires_grad_(True)
    dt = torch.zeros(n, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{'params': [b], 'lr': lr}, {'params': [dt], 'lr': lr * 0.5}])
    for it in range(steps):
        out = runner.body(betas=b[None].expand(n, -1), global_orient=go, body_pose=bp, pose2rot=False)
        V = out.vertices + (t0 + dt)[:, None]
        J = out.joints[:, _SMPL_J] + (t0 + dt)[:, None]
        px = lambda P: torch.stack([P[..., 0] / P[..., 2] * K[:, None, 0, 0] + K[:, None, 0, 2],
                                    P[..., 1] / P[..., 2] * K[:, None, 1, 1] + K[:, None, 1, 2]], -1)
        pv, pj = px(V), px(J)
        loss_kp = ((((pj - kps) ** 2).sum(-1) * kvis).sum(1) / kvis.sum(1).clamp(min=1) / scales ** 2).mean()
        loss = w_kp * loss_kp
        if it >= steps // 3:                 # align first, then let the outline shape the body
            lo = 0
            for i in range(n):
                H, W = out_maps[i].shape
                g = torch.stack([pv[i, :, 0] / (W - 1) * 2 - 1, pv[i, :, 1] / (H - 1) * 2 - 1], -1)
                d = Fn.grid_sample(out_maps[i][None, None], g[None, None], align_corners=True)[0, 0, 0]
                lo = lo + ((d / scales[i]) ** 2).mean()
                if cover and conts[i] is not None:
                    c = T(conts[i])
                    near = torch.cdist(c, pv[i, ::4]).min(1).values / scales[i]
                    lo = lo + cover * (near ** 2).mean()
            loss = loss + w_out * 100 * lo / n
            loss = loss + w_prior * ((b - b0) ** 2).sum()
            if weight_kg:
                loss = loss + 100 * ((runner.mass_kg(b, height_cm) - weight_kg) / weight_kg) ** 2
        opt.zero_grad(); loss.backward(); opt.step()
        if it < steps // 3:
            b.grad = None
            with torch.no_grad():
                b.copy_(b0)
    return b.detach().cpu().numpy()
