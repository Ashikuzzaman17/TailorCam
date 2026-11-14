"""
Synthetic training data generator for BodyMeasureNet.
No SMPL, no rendering — pure parametric anthropometry.

How it works
------------
1. Sample diverse body shapes: height, BMI, sex.
2. Compute ground-truth measurements using ANSUR II regression equations.
3. Build a simple articulated skeleton at those dimensions.
4. Simulate three-view world landmarks as MediaPipe would see them,
   including realistic position/depth noise.
5. Save (features, targets) to model/train_data.npz.

Run: py -3.13 model/generate_data.py
     (uses numpy only — no GPU required, ~2 min for 100k samples)
"""

import os
import sys
import numpy as np

N_SAMPLES = 120_000
SAVE_PATH = os.path.join(os.path.dirname(__file__), 'train_data.npz')

# Landmark indices (same as body_net.py)
_L_SH, _R_SH = 11, 12
_L_EL, _R_EL = 13, 14
_L_WR, _R_WR = 15, 16
_L_HI, _R_HI = 23, 24
_NOSE         = 0
_L_ANKLE      = 27

N_LM    = 33
N_VIEWS = 3


# ─── ANSUR II regression equations ───────────────────────────────────────────
# Source: ANSUR II report (2012) + published anthropometric surveys.
# Predicts body measurements (cm) from height (cm), BMI (kg/m²), sex (0=M, 1=F).
def _measurements(h, bmi, female):
    w = bmi * (h / 100) ** 2   # weight in kg

    if female:
        # Female averages calibrated to ANSUR II female dataset (n = 1986)
        sh  = 36.8 + (h - 162.0) * 0.214 + (bmi - 26.1) * 0.19
        ch  = 88.2 + (h - 162.0) * 0.350 + (bmi - 26.1) * 1.38
        wa  = 74.6 + (h - 162.0) * 0.160 + (bmi - 26.1) * 2.10
        hip = 95.8 + (h - 162.0) * 0.220 + (bmi - 26.1) * 1.82
        nk  = 32.4 + (h - 162.0) * 0.080 + (bmi - 26.1) * 0.42
    else:
        # Male averages calibrated to ANSUR II male dataset (n = 4082)
        sh  = 40.6 + (h - 175.6) * 0.218 + (bmi - 25.0) * 0.16
        ch  = 97.0 + (h - 175.6) * 0.390 + (bmi - 25.0) * 1.15
        wa  = 80.4 + (h - 175.6) * 0.175 + (bmi - 25.0) * 2.20
        hip = 95.2 + (h - 175.6) * 0.215 + (bmi - 25.0) * 1.48
        nk  = 37.2 + (h - 175.6) * 0.095 + (bmi - 25.0) * 0.31

    bk = h * 0.296
    sr = h * 0.330
    sl = h * 0.355

    return np.array([sh, ch, wa, hip, bk, sr, sl, nk], dtype=np.float32)


# ─── Skeleton builder ─────────────────────────────────────────────────────────
def _skeleton(h, sh_w, hip_w, sl, bk):
    """
    Build 33-landmark skeleton in the person's body frame (metres, Y-up,
    origin = hip midpoint), matching MediaPipe landmark numbering.

    Only the 8 anatomically informative landmarks are fully modelled;
    the rest are proxied to nearby positions (they'll be zeroed with noise).
    """
    h /= 100.0          # cm → m
    sh_w /= 200.0       # half-width in m
    hip_w /= 200.0
    bk /= 100.0
    sl /= 100.0

    lms = np.zeros((33, 3), dtype=np.float32)

    # Hip joints (origin ±)
    lms[_L_HI] = [-hip_w, 0.0,       0.0]
    lms[_R_HI] = [+hip_w, 0.0,       0.0]

    # Shoulder joints
    lms[_L_SH] = [-sh_w,  bk,        0.02]
    lms[_R_SH] = [+sh_w,  bk,        0.02]

    # Elbow: shoulder + arm_upper along a slightly downward and outward direction
    arm_upper = sl * 0.46
    lms[_L_EL] = [lms[_L_SH][0] - arm_upper * 0.82,
                  bk - arm_upper * 0.55, 0.0]
    lms[_R_EL] = [lms[_R_SH][0] + arm_upper * 0.82,
                  bk - arm_upper * 0.55, 0.0]

    # Wrist
    arm_lower = sl * 0.54
    for side in (-1, +1):
        el = lms[_L_EL if side < 0 else _R_EL]
        sh = lms[_L_SH if side < 0 else _R_SH]
        dir_ = (el - sh) / (np.linalg.norm(el - sh) + 1e-8)
        wr_pos = el + dir_ * arm_lower
        lms[_L_WR if side < 0 else _R_WR] = wr_pos

    # Nose: above shoulders
    lms[_NOSE] = [0.0, bk + h * 0.09, 0.02]

    # Ankle: below hip by ~54% of height
    lms[_L_ANKLE] = [-hip_w * 0.9, -h * 0.54, 0.0]

    return lms    # (33, 3) in metres, body frame


# ─── View simulation ──────────────────────────────────────────────────────────
def _Ry(deg):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s,0, c]], dtype=np.float32)


VIEW_ROTS = [_Ry(0), _Ry(-90), _Ry(90)]   # FRONT, LEFT, RIGHT


def _simulate_view(lms_body, R, height_m, depth_noise_sigma=0.025):
    """
    Apply view rotation + realistic MediaPipe-like noise.

    MediaPipe world landmark errors (rough calibration):
      xy position: σ ≈ 1-2 cm
      z (depth)  : σ ≈ 2-4 cm  (depth estimation is noisier)
    Plus a small global scale jitter (σ ≈ 2%).
    """
    rotated = (R @ lms_body.T).T          # (33, 3) in camera world frame

    # Normalise by height (same as body_net.features_from_views)
    norm = rotated / height_m

    # Add noise (normalised units)
    xy_sigma = 0.015 / height_m
    z_sigma  = depth_noise_sigma / height_m
    noise = np.random.randn(33, 3).astype(np.float32)
    noise[:, :2] *= xy_sigma
    noise[:, 2]  *= z_sigma

    # Global scale jitter ±3%
    scale_jitter = 1.0 + np.random.randn() * 0.02
    return norm * scale_jitter + noise


def _build_feature(h_cm, sh_w, hip_w, sl, bk, rng):
    """Build 298-dim feature vector for one synthetic person."""
    h_m  = h_cm / 100.0
    lms  = _skeleton(h_cm, sh_w, hip_w, sl, bk)
    parts = []
    for R in VIEW_ROTS:
        parts.append(_simulate_view(lms, R, h_m).flatten())
    parts.append(np.array([1.0], dtype=np.float32))   # normalised height
    return np.concatenate(parts)   # (298,)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(42)
    print(f'Generating {N_SAMPLES:,} synthetic body samples …')

    feats   = np.zeros((N_SAMPLES, 298), dtype=np.float32)
    targets = np.zeros((N_SAMPLES,   8), dtype=np.float32)

    for i in range(N_SAMPLES):
        female = rng.random() < 0.50
        h   = rng.uniform(152, 198)           # height cm
        bmi = rng.uniform(17,  36)            # BMI

        m = _measurements(h, bmi, female)
        # m = [sh, ch, wa, hip, bk, sr, sl, nk]  all in cm

        sh_w = m[0]   # shoulder_width
        hip_w = m[3]  # hip (used as proxy for hip_width; full circ from model)
        sl    = m[6]  # sleeve_length
        bk    = m[4]  # back_length

        feats[i]   = _build_feature(h, sh_w, hip_w, sl, bk, rng)
        targets[i] = m / h      # normalise by height

        if (i + 1) % 10_000 == 0:
            print(f'  {i+1:,} / {N_SAMPLES:,}', flush=True)

    np.savez_compressed(SAVE_PATH, features=feats, targets=targets)
    print(f'\nSaved {N_SAMPLES:,} samples -> {SAVE_PATH}')
    print('Feature shape:', feats.shape, '  Target shape:', targets.shape)
    print('Example targets (fractions of height):')
    for j, k in enumerate(['shoulder', 'chest', 'waist', 'hip',
                            'back', 'shirt', 'sleeve', 'neck']):
        print(f'  {k:12s}: {targets[:,j].mean():.3f} ± {targets[:,j].std():.3f}')


if __name__ == '__main__':
    main()
