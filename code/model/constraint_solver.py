"""
Anthropometric constraint solver.

After geometric + neural measurement, adjusts all 8 values minimally so that
they satisfy known ANSUR II proportional relationships.

The optimisation is weighted: measurements flagged as high-uncertainty are
pulled harder toward the population mean; confident measurements resist change.
"""

import numpy as np
import scipy.optimize as opt


# ── ANSUR II mean measurement fractions of standing height ────────────────────
# Source: ANSUR II 2012 report (combined male + female weighted means)
_FRAC = {
    'shoulder_width': 0.239,
    'chest':          0.528,
    'waist':          0.455,
    'hip':            0.540,
    'back_length':    0.296,
    'shirt_length':   0.330,
    'sleeve_length':  0.355,
    'neck':           0.198,
}

# ── Population standard deviations as fraction of height ─────────────────────
# (higher = more variable = less aggressive constraint)
_STD = {
    'shoulder_width': 0.020,
    'chest':          0.060,   # high BMI variation
    'waist':          0.080,   # very high BMI variation
    'hip':            0.055,
    'back_length':    0.020,
    'shirt_length':   0.022,
    'sleeve_length':  0.022,
    'neck':           0.018,
}

KEYS = list(_FRAC.keys())


def solve(raw_m: dict, height_cm: float,
          uncertainties: dict | None = None) -> dict:
    """
    Adjust raw_m minimally so measurements satisfy ANSUR II proportions.

    Parameters
    ----------
    raw_m         : dict[str, float] — raw measurements in cm
    height_cm     : calibrated standing height in cm
    uncertainties : dict[str, float] — per-key standard deviation (cm)
                    from Bayesian model, or None (equal weights applied)

    Returns
    -------
    dict[str, float] — corrected measurements (same keys as raw_m)
    """
    if not raw_m or height_cm < 100:
        return raw_m

    # Only work on measurements we have a prior for
    present = [k for k in KEYS if k in raw_m]
    if not present:
        return raw_m

    x0       = np.array([raw_m[k] for k in present], dtype=np.float64)
    expected = np.array([_FRAC[k] * height_cm  for k in present])
    pop_std  = np.array([_STD[k]  * height_cm  for k in present])

    # Inverse-uncertainty weights: uncertain measurement → smaller weight
    # (less resistant to correction toward population mean)
    if uncertainties:
        obs_std = np.array([uncertainties.get(k, pop_std[i])
                             for i, k in enumerate(present)])
    else:
        obs_std = pop_std * 0.5   # default: treat geometric as moderately confident

    weights = 1.0 / (obs_std + 0.1)   # +0.1 avoids div-by-zero

    # ------------------------------------------------------------------
    def objective(x: np.ndarray) -> float:
        # Term 1: pull toward ANSUR II expected (scaled by population std)
        ratio_pen = weights * ((x - expected) / (pop_std + 0.1)) ** 2
        # Term 2: stay close to raw observation (scaled by obs uncertainty)
        raw_pen   = ((x - x0) / (obs_std + 0.1)) ** 2
        return float(np.sum(ratio_pen + raw_pen))

    # Hard bounds: never move a measurement by more than 30% from raw value
    bounds = [(x0[i] * 0.70, x0[i] * 1.30) for i in range(len(present))]

    result = opt.minimize(
        objective, x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-7},
    )

    out = dict(raw_m)
    for i, k in enumerate(present):
        out[k] = round(float(result.x[i]), 1)

    return out


def cross_view_consistency(depth_left: dict, depth_right: dict,
                           threshold: float = 0.12) -> tuple[bool, str]:
    """
    Check whether left-view and right-view depth estimates agree.

    Returns (ok: bool, warning_message: str).
    If ok=False, the UI should warn the user to stand straight.
    """
    if not depth_left or not depth_right:
        return True, ''

    warnings = []
    for key in ('chest_depth', 'waist_depth', 'hip_depth'):
        l = depth_left.get(key, 0)
        r = depth_right.get(key, 0)
        if l <= 0 or r <= 0:
            continue
        avg = (l + r) / 2.0
        if avg > 0 and abs(l - r) / avg > threshold:
            warnings.append(key.split('_')[0])

    if warnings:
        parts = ', '.join(warnings)
        return False, f'Asymmetry detected ({parts}) — stand straight, shoulders parallel'
    return True, ''
