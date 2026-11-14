import os
import math
import joblib


MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'correction_model.pkl')

# ── Shoulder correction ────────────────────────────────────────────────────────
SHOULDER_ACROMION    = 1.05   # glenohumeral joint → acromion tip
SLEEVE_3D_CORRECTION = 1.15   # 2-D landmark chain underestimates 3-D arm length
NECK_RATIO           = 0.88   # neck circumference / acromion shoulder width

# ── ANSUR II + CAESAR adult averages ─────────────────────────────────────────
# WHY mask-based silhouette width is NOT used for chest/waist:
#   MediaPipe's person mask includes the arms at all torso heights.
#   At chest level the arms add 12-20 cm to each side, producing absurd values
#   like chest=125 cm on a slim person.  Shoulder 3-D distance is arm-free.
#
# These ratios are calibrated so an average adult (acromion ≈ 41 cm) gets:
#   chest ≈ 96 cm  |  waist ≈ 85 cm  |  hip ≈ 94 cm  |  neck ≈ 36 cm
#
CHEST_HW  = 0.43   # chest half-width  / acromion_shoulder
CHEST_HD  = 0.28   # chest half-depth  / acromion_shoulder  (depth ≈ width × 0.65)
WAIST_HW  = 0.38   # waist half-width  / acromion_shoulder
WAIST_HD  = 0.24   # waist half-depth  / acromion_shoulder
HIP_HW    = 0.44   # hip half-width    / acromion_shoulder (fallback only)

# Depth-to-width ratios for when side-mask depth is unavailable
CHEST_DEPTH_RATIO = 0.62
WAIST_DEPTH_RATIO = 0.65
HIP_DEPTH_RATIO   = 0.72


def _ellipse_circ(a, b):
    """Ramanujan approximation of ellipse perimeter (semi-axes a and b)."""
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))


def compute_measurements(_world_lms, _screen_lms, tracker, _LM):
    scale = tracker.scale_px_to_cm
    if scale is None or scale <= 0:
        return {'error': 'Calibration failed - face not visible. Press R to retry.'}

    m = {}

    # ── 1. Shoulder width (acromion-to-acromion) ──────────────────────────────
    if tracker.shoulder_3d_cm and 25 < tracker.shoulder_3d_cm < 65:
        shoulder_width = tracker.shoulder_3d_cm * SHOULDER_ACROMION
        src = '3D'
    else:
        shoulder_width = tracker.max_shoulder_px * scale * SHOULDER_ACROMION
        src = '2D'
    m['shoulder_width'] = round(shoulder_width, 1)

    # ── 2. Back length ─────────────────────────────────────────────────────────
    back_raw = abs(tracker.front_sh_y - tracker.front_hip_y) * scale
    if tracker.height_cm and not (tracker.height_cm * 0.24 < back_raw < tracker.height_cm * 0.40):
        back_raw = tracker.height_cm * 0.30
    m['back_length'] = round(back_raw, 1)

    # ── 3. Shirt length ────────────────────────────────────────────────────────
    neck_offset_px = tracker.max_shoulder_px * 0.08
    shirt_raw = abs((tracker.front_sh_y - neck_offset_px) - tracker.front_hip_y) * scale
    if tracker.height_cm and not (tracker.height_cm * 0.27 < shirt_raw < tracker.height_cm * 0.44):
        shirt_raw = tracker.height_cm * 0.33
    m['shirt_length'] = round(shirt_raw, 1)

    # ── 4. Sleeve length ──────────────────────────────────────────────────────
    max_sleeve_px = (tracker.height_cm * 0.75 / scale) if tracker.height_cm else float('inf')
    valid = [px for px in [tracker.sleeve_l_px, tracker.sleeve_r_px]
             if 0 < px <= max_sleeve_px]
    sleeve_px = sum(valid) / len(valid) if valid else 0
    m['sleeve_length'] = round(sleeve_px * scale * SLEEVE_3D_CORRECTION, 1)

    # ── 5. Chest circumference ─────────────────────────────────────────────────
    # Width source: shoulder 3-D distance (arm-free — the ONLY reliable
    # width proxy from a single camera with arms at sides).
    # Depth source: real front-to-back depth from side-view mask when available.
    chest_d = tracker.chest_depth_cm

    chest_hw = shoulder_width * CHEST_HW
    if chest_d and 14 < chest_d < 40:
        chest_hd  = chest_d / 2
        depth_src = 'mask-depth'
    else:
        chest_hd  = chest_hw * CHEST_DEPTH_RATIO
        depth_src = 'depth-ratio'

    m['chest'] = round(_ellipse_circ(chest_hw, chest_hd), 1)
    print(f'[Chest] sh={shoulder_width:.1f}cm  hw={chest_hw:.1f} hd={chest_hd:.1f} '
          f'-> {m["chest"]}cm ({src}+{depth_src})')

    # ── 6. Waist circumference ─────────────────────────────────────────────────
    waist_hw = shoulder_width * WAIST_HW
    if chest_d and 14 < chest_d < 40:
        waist_hd = (chest_d * 0.88) / 2   # waist depth ~88% of chest depth
    else:
        waist_hd = shoulder_width * WAIST_HD

    m['waist'] = round(_ellipse_circ(waist_hw, waist_hd), 1)
    print(f'[Waist] hw={waist_hw:.1f} hd={waist_hd:.1f} -> {m["waist"]}cm')

    # ── 7. Hip circumference ───────────────────────────────────────────────────
    # Hip landmark sits 30% of height below shoulder — arms reach wrist level
    # there, so contamination is smaller. Still cross-check against shoulder.
    hip_w = tracker.hip_width_cm
    hip_hw_expected = shoulder_width * HIP_HW
    used_mask = False
    if hip_w and 28 < hip_w < 58:
        hip_hw = hip_w / 2
        # Reject if the mask measurement is >35% away from the shoulder-based
        # estimate — almost certainly arm contamination
        if 0.65 < hip_hw / hip_hw_expected < 1.35:
            hip_hd = hip_hw * HIP_DEPTH_RATIO
            m['hip'] = round(_ellipse_circ(hip_hw, hip_hd), 1)
            print(f'[Hip] hw={hip_hw:.1f} hd={hip_hd:.1f} -> {m["hip"]}cm (mask)')
            used_mask = True
        else:
            print(f'[Hip] mask {hip_hw:.1f}cm rejected '
                  f'(expected ≈{hip_hw_expected:.1f}cm — arm contamination)')

    if not used_mask:
        hip_hd = hip_hw_expected * HIP_DEPTH_RATIO
        m['hip'] = round(_ellipse_circ(hip_hw_expected, hip_hd), 1)
        print(f'[Hip] hw={hip_hw_expected:.1f} hd={hip_hd:.1f} '
              f'-> {m["hip"]}cm (shoulder-ratio fallback)')

    # ── 8. Neck circumference ──────────────────────────────────────────────────
    m['neck'] = round(shoulder_width * NECK_RATIO, 1)

    print(f'[Measurements] src={src}  shoulder={m["shoulder_width"]}cm  '
          f'chest={m["chest"]}cm  waist={m["waist"]}cm  '
          f'hip={m.get("hip","?")}cm  sleeve={m["sleeve_length"]}cm  '
          f'scale={scale:.4f} cm/px')

    return _apply_correction(m)


def compute_from_height(height_cm):
    """
    Height-based anthropometric proportions (ANSUR II averages).
    Used as fallback when SMPL fitting fails or is unavailable.
    Results are marked with '_estimated': True so the UI can warn the user.
    """
    h = height_cm
    m = {
        'shoulder_width': round(h * 0.239, 1),
        'chest':          round(h * 0.528, 1),
        'waist':          round(h * 0.455, 1),
        'back_length':    round(h * 0.296, 1),
        'shirt_length':   round(h * 0.330, 1),
        'sleeve_length':  round(h * 0.355, 1),
        'neck':           round(h * 0.198, 1),
        '_estimated':     True,
    }
    print(f'[Fallback] height-based measurements for h={h}cm: '
          f'chest={m["chest"]} waist={m["waist"]}')
    return m


def _apply_correction(raw):
    if not os.path.exists(MODEL_PATH):
        return raw
    try:
        bundle = joblib.load(MODEL_PATH)
        corrected = dict(raw)
        for key, model in bundle.items():
            if key in raw and isinstance(raw[key], (int, float)):
                corrected[key] = round(float(model.predict([[raw[key]]])[0]), 1)
        return corrected
    except Exception:
        return raw
