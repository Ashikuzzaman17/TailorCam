"""
Geometric body measurement from MediaPipe 3D world landmarks.

Combines front + side world landmarks to get both widths (front) and depths
(side) then computes circumferences with the Ramanujan ellipse formula.

Runs natively in Python 3.14 — no subprocess, no SMPL, results in <5 ms.
After training body_net.py the neural worker refines these values further.
"""

import math
import numpy as np

# ---------- MediaPipe landmark indices ----------------------------------------
_NOSE       = 0
_L_SHOULDER = 11; _R_SHOULDER = 12
_L_ELBOW    = 13; _R_ELBOW    = 14
_L_WRIST    = 15; _R_WRIST    = 16
_L_HIP      = 23; _R_HIP      = 24

# ---------- ANSUR II anthropometric constants ----------------------------------
# MediaPipe hip/shoulder landmarks sit at *joints*, not widest body points.
# These correction factors scale joint-to-joint distances to anatomical widths.
_TORSO_OF_H      = 0.293   # shoulder_mid->hip_mid / height (ANSUR II avg)
_SH_JOINT_SCALE  = 1.185   # shoulder landmark span -> true anatomical shoulder width
_CHEST_W_OF_SH   = 0.895   # chest_width  / shoulder_width (ANSUR II ~89.5%)
_HIP_JOINT_SCALE = 1.435   # hip landmark span -> true hip width (joints sit inside pelvis)
_WAIST_W_OF_HIP  = 0.755   # waist_width  / hip_width
_CHEST_D_RATIO   = 0.545   # chest AP depth / chest_width  (fallback — no side view)
_WAIST_D_RATIO   = 0.690   # waist AP depth / waist_width  (fallback)
_HIP_D_RATIO     = 0.775   # hip AP depth   / hip_width    (fallback)
_NECK_OF_H       = 0.202   # neck circumference / height (ANSUR II: 34.6/171 = 0.202)
_SLEEVE_3D       = 1.10    # shoulder-elbow-wrist chain (2-D) -> actual sleeve length

# Cross-view consistency threshold: if L/R depths diverge more than this
# fraction, flag asymmetry and average rather than picking one side.
_CONSISTENCY_THRESHOLD = 0.12


def _pt(world_lms, idx: int) -> np.ndarray:
    """Extract (x, y, z) from a world landmark at index idx."""
    lm = world_lms[idx]
    if hasattr(lm, 'x'):
        return np.array([lm.x, lm.y, lm.z], dtype=np.float64)
    if isinstance(lm, dict):
        return np.array([lm.get('x', 0.0), lm.get('y', 0.0), lm.get('z', 0.0)])
    return np.asarray(lm[:3], dtype=np.float64)


def _d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _ellipse(a: float, b: float) -> float:
    """Ramanujan approximation of ellipse circumference; a, b = semi-axes."""
    if a <= 0 or b <= 0:
        return 0.0
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + math.sqrt(4.0 - 3.0 * h)))


def _scale(world_lms, height_cm: float) -> float:
    """
    Calibrate cm-per-world-unit using the torso (shoulder_mid -> hip_mid).
    Uses the vertical component preferentially to avoid lateral artefacts.
    """
    l_sh = _pt(world_lms, _L_SHOULDER); r_sh = _pt(world_lms, _R_SHOULDER)
    l_hi = _pt(world_lms, _L_HIP);     r_hi = _pt(world_lms, _R_HIP)
    sh   = (l_sh + r_sh) / 2.0
    hi   = (l_hi + r_hi) / 2.0

    torso_y = abs(sh[1] - hi[1])
    torso   = torso_y if torso_y > 0.02 else _d(sh, hi)

    if torso < 0.01:
        return 100.0   # metre -> cm fallback
    return (height_cm * _TORSO_OF_H) / torso


def _front_widths(world_lms, sc: float) -> dict:
    """Measure widths and linear distances from the FRONT view."""
    l_sh = _pt(world_lms, _L_SHOULDER); r_sh = _pt(world_lms, _R_SHOULDER)
    l_hi = _pt(world_lms, _L_HIP);     r_hi = _pt(world_lms, _R_HIP)
    l_el = _pt(world_lms, _L_ELBOW);   r_el = _pt(world_lms, _R_ELBOW)
    l_wr = _pt(world_lms, _L_WRIST);   r_wr = _pt(world_lms, _R_WRIST)

    sh_mid = (l_sh + r_sh) / 2.0
    hi_mid = (l_hi + r_hi) / 2.0

    shoulder_w = _d(l_sh, r_sh) * sc * _SH_JOINT_SCALE
    hip_w      = _d(l_hi, r_hi) * sc * _HIP_JOINT_SCALE
    chest_w    = shoulder_w * _CHEST_W_OF_SH
    waist_w    = hip_w      * _WAIST_W_OF_HIP

    back  = _d(sh_mid, hi_mid) * sc
    shirt = back * 1.12

    l_sl = (_d(l_sh, l_el) + _d(l_el, l_wr)) * sc * _SLEEVE_3D
    r_sl = (_d(r_sh, r_el) + _d(r_el, r_wr)) * sc * _SLEEVE_3D
    sleeve = (l_sl + r_sl) / 2.0

    return {
        'shoulder_w': shoulder_w,
        'chest_w':    chest_w,
        'waist_w':    waist_w,
        'hip_w':      hip_w,
        'back_len':   back,
        'shirt_len':  shirt,
        'sleeve_raw': sleeve,
    }


def _side_depths(side_lms, height_cm: float, sc: float, widths: dict) -> dict:
    """
    Estimate front-to-back body depths from a SIDE view.

    When the person turns 90 degrees, L_SHOULDER and R_SHOULDER end up at
    different depths (front vs. back of body), so their 3-D distance in world
    coordinates approximates the chest anterior-posterior diameter.
    We recalibrate the scale using height_cm, then clamp to anatomical limits.
    """
    if side_lms is None:
        return {
            'chest_d': widths['chest_w'] * _CHEST_D_RATIO,
            'waist_d': widths['waist_w'] * _WAIST_D_RATIO,
            'hip_d':   widths['hip_w']   * _HIP_D_RATIO,
        }

    l_sh = _pt(side_lms, _L_SHOULDER); r_sh = _pt(side_lms, _R_SHOULDER)
    l_hi = _pt(side_lms, _L_HIP);     r_hi = _pt(side_lms, _R_HIP)

    sc_side = _scale(side_lms, height_cm)

    raw_chest = _d(l_sh, r_sh) * sc_side
    raw_hip   = _d(l_hi, r_hi) * sc_side

    cw = widths['chest_w']
    hw = widths['hip_w']
    ww = widths['waist_w']

    # Clamp to physically plausible anatomical range
    chest_d = max(cw * 0.42, min(cw * 0.70, raw_chest))
    hip_d   = max(hw * 0.56, min(hw * 0.90, raw_hip))
    waist_d = chest_d * 0.82
    waist_d = max(ww * 0.55, min(ww * 0.80, waist_d))

    return {'chest_d': chest_d, 'waist_d': waist_d, 'hip_d': hip_d}


def _back_widths(back_lms, height_cm: float) -> dict | None:
    """
    Extract shoulder + hip widths from the BACK view.
    When person faces away, shoulder & hip landmark span is ARM-FREE on this view
    (arms hang at front, not visible on back silhouette), so we get the cleanest
    possible width measurement — often more accurate than FRONT which can include
    slightly displaced arm reflections.
    Returns None if landmarks are missing.
    """
    if back_lms is None:
        return None
    try:
        sc   = _scale(back_lms, height_cm)
        l_sh = _pt(back_lms, _L_SHOULDER);  r_sh = _pt(back_lms, _R_SHOULDER)
        l_hi = _pt(back_lms, _L_HIP);       r_hi = _pt(back_lms, _R_HIP)
        sh_w = _d(l_sh, r_sh) * sc * _SH_JOINT_SCALE
        hi_w = _d(l_hi, r_hi) * sc * _HIP_JOINT_SCALE
        if 28 < sh_w < 62 and 26 < hi_w < 70:
            return {'shoulder_w': sh_w, 'hip_w': hi_w}
    except Exception:
        pass
    return None


def _fuse_widths(front_w: dict, back_w: dict | None) -> dict:
    """
    Fuse FRONT and BACK width estimates.
    Weights: 55% BACK (cleaner, arm-free) + 45% FRONT when both available.
    Falls back to FRONT only when BACK is missing.
    """
    if back_w is None:
        return front_w

    fused = dict(front_w)  # copy all fields from front

    # Shoulder: back view is more reliable (no arm overhang)
    sh_fused = front_w['shoulder_w'] * 0.45 + back_w['shoulder_w'] * 0.55
    fused['shoulder_w'] = sh_fused
    # Recompute dependent widths from fused shoulder
    fused['chest_w'] = sh_fused * _CHEST_W_OF_SH
    fused['waist_w'] = fused['waist_w']  # waist keeps front-derived value (hip-based)

    # Hip: back view captures full pelvis span more accurately
    hi_fused = front_w['hip_w'] * 0.45 + back_w['hip_w'] * 0.55
    fused['hip_w'] = hi_fused

    delta_sh = back_w['shoulder_w'] - front_w['shoulder_w']
    delta_hi = back_w['hip_w'] - front_w['hip_w']
    print(f'[BackFuse] Δshoulder={delta_sh:+.1f}cm  Δhip={delta_hi:+.1f}cm  '
          f'→ sh={sh_fused:.1f}  hip={hi_fused:.1f}')
    return fused


def _compute_core(height_cm: float, front, side_left, side_right,
                  back=None) -> tuple[dict, dict, bool, str]:
    """
    Shared geometry logic used by both compute() and compute_with_geo_features().
    Returns (widths, depths, consistency_ok, consistency_msg).
    """
    sc       = _scale(front, height_cm)
    widths_f = _front_widths(front, sc)
    back_w   = _back_widths(back, height_cm)
    widths   = _fuse_widths(widths_f, back_w)

    consistency_ok  = True
    consistency_msg = ''

    if side_left is not None and side_right is not None:
        d_l = _side_depths(side_left,  height_cm, sc, widths)
        d_r = _side_depths(side_right, height_cm, sc, widths)

        for key in ('chest_d', 'waist_d', 'hip_d'):
            avg = (d_l[key] + d_r[key]) / 2.0
            if avg > 0 and abs(d_l[key] - d_r[key]) / avg > _CONSISTENCY_THRESHOLD:
                consistency_ok  = False
                consistency_msg = 'Side-view asymmetry — stand straight'
                break

        depths = {k: (d_l[k] + d_r[k]) / 2.0 for k in d_l}
    else:
        side   = side_left or side_right
        depths = _side_depths(side, height_cm, sc, widths)

    return widths, depths, consistency_ok, consistency_msg


def compute(height_cm: float, views: dict) -> dict:
    """
    Main entry point — geometric measurement.

    Parameters
    ----------
    height_cm : calibrated standing height
    views     : {view_name: world_landmarks_list}
                Keys: 'FRONT', 'LEFT', 'RIGHT', optionally 'BACK'
                BACK view improves shoulder + hip accuracy (arm-free widths).

    Returns
    -------
    dict with keys: shoulder_width, chest, waist, hip,
                    back_length, shirt_length, sleeve_length, neck
    """
    front = views.get('FRONT')
    if front is None:
        from measurements import compute_from_height
        return compute_from_height(height_cm)

    widths, depths, ok, msg = _compute_core(
        height_cm, front,
        views.get('LEFT'), views.get('RIGHT'),
        views.get('BACK')
    )

    neck  = height_cm * _NECK_OF_H
    chest = _ellipse(widths['chest_w'] / 2, depths['chest_d'] / 2)
    waist = _ellipse(widths['waist_w'] / 2, depths['waist_d'] / 2)
    hip   = _ellipse(widths['hip_w']   / 2, depths['hip_d']   / 2)

    result = {
        'shoulder_width': round(widths['shoulder_w'], 1),
        'chest':          round(chest,                1),
        'waist':          round(waist,                1),
        'hip':            round(hip,                  1),
        'back_length':    round(widths['back_len'],   1),
        'shirt_length':   round(widths['shirt_len'],  1),
        'sleeve_length':  round(widths['sleeve_raw'], 1),
        'neck':           round(neck,                 1),
    }

    if not ok:
        print(f'[GeoMeasure] WARNING: {msg}')

    views_used = [v for v in ('FRONT','LEFT','RIGHT','BACK') if views.get(v) is not None]
    print(f'[GeoMeasure] views={views_used}  h={height_cm}cm  '
          f'sh={result["shoulder_width"]}  chest={result["chest"]}  '
          f'waist={result["waist"]}  hip={result["hip"]}'
          + (f'  ⚠ {msg}' if not ok else ''))

    return result


def compute_with_geo_features(height_cm: float, views: dict) -> tuple[dict, dict]:
    """
    Like compute(), but also returns the geometric intermediate features dict
    needed for the BayesianBodyMeasure model.
    Now uses BACK view when available for improved width accuracy.
    Returns (measurements, geo_features).
    """
    front = views.get('FRONT')
    if front is None:
        return {}, {}

    widths, depths, ok, msg = _compute_core(
        height_cm, front,
        views.get('LEFT'), views.get('RIGHT'),
        views.get('BACK')
    )

    if not ok:
        print(f'[GeoMeasure] WARNING: {msg}')

    neck  = height_cm * _NECK_OF_H
    chest = _ellipse(widths['chest_w'] / 2, depths['chest_d'] / 2)
    waist = _ellipse(widths['waist_w'] / 2, depths['waist_d'] / 2)
    hip   = _ellipse(widths['hip_w']   / 2, depths['hip_d']   / 2)

    measurements = {
        'shoulder_width': round(widths['shoulder_w'], 1),
        'chest':          round(chest,                1),
        'waist':          round(waist,                1),
        'hip':            round(hip,                  1),
        'back_length':    round(widths['back_len'],   1),
        'shirt_length':   round(widths['shirt_len'],  1),
        'sleeve_length':  round(widths['sleeve_raw'], 1),
        'neck':           round(neck,                 1),
    }

    geo_features = {
        'shoulder_w': widths['shoulder_w'],
        'chest_w':    widths['chest_w'],
        'waist_w':    widths['waist_w'],
        'hip_w':      widths['hip_w'],
        'back_len':   widths['back_len'],
        'chest_d':    depths['chest_d'],
        'waist_d':    depths['waist_d'],
        'hip_d':      depths['hip_d'],
        'sleeve_raw': widths['sleeve_raw'],
        'height_cm':  height_cm,
        'has_back':   float(views.get('BACK') is not None),  # extra feature
    }

    return measurements, geo_features
