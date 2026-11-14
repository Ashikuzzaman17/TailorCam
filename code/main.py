import os
import sys
import json
import math
import time
import threading
import subprocess
import cv2
import numpy as np
from camera import Camera
from pose_detector import PoseDetector
from measurements import compute_from_height

# ── Optional: voice guidance ───────────────────────────────────────────────────
_tts_engine = None
_tts_lock   = threading.Lock()
def _init_tts():
    global _tts_engine
    try:
        import pyttsx3
        _tts_engine = pyttsx3.init()
        _tts_engine.setProperty('rate', 165)
        _tts_engine.setProperty('volume', 0.85)
        print('[Voice] pyttsx3 ready')
    except Exception as _e:
        print(f'[Voice] pyttsx3 not available ({_e}) — silent mode')

threading.Thread(target=_init_tts, daemon=True).start()

def _speak(text):
    """Non-blocking TTS — speaks in a background thread."""
    def _run():
        with _tts_lock:
            try:
                if _tts_engine:
                    _tts_engine.say(text)
                    _tts_engine.runAndWait()
            except Exception:
                pass
    threading.Thread(target=_run, daemon=True).start()
try:
    from model import body_measurer as _body_measurer
except ImportError:
    _body_measurer = None
try:
    from model.hmr.service import HMRService as _HMRService
except Exception as _hmr_ie:
    _HMRService = None
    print(f'[HMR] not available: {_hmr_ie}')

HMR_SAMPLE_EVERY = 15   # countdown ticks between HMR snapshots (3 per view)
from output import (draw_results_overlay,
                    draw_height_input_overlay,
                    draw_name_input_overlay,
                    draw_measurement_edit_overlay,
                    draw_photo_guide,
                    draw_turn_instruction,
                    draw_toast,
                    launch_folder_picker, pick_photos, load_photo,
                    save_pdf,
                    draw_home_screen,
                    draw_splash_screen,
                    draw_history_screen,
                    save_record,
                    load_records,
                    draw_weight_age_overlay,
                    draw_customer_select_screen,
                    draw_order_screen,
                    draw_reports_screen,
                    draw_settings_screen,
                    print_order_label,
                    classify_body_type,
                    draw_inventory_screen,
                    draw_pose_guidance_arrows,
                    draw_style_tip_overlay,
                    draw_loyalty_badge,
                    render_smpl_mesh,
                    draw_processing_screen,
                    COUNTRY_KEYS)
import output as _output_mod   # for USE_INCHES toggle
from output import GOLD, RED, GREEN, TEAL   # colour constants for toasts
from output import DARK as GOLD_BG
from ui_text import put_text as _put_text, text_size as _text_size
import ui_text as _ui


def _show(img):
    """Show a screen at the window's real size with sharp text (ui_text.present)."""
    try:
        _, _, ww, wh = cv2.getWindowImageRect('TailorCam')
    except cv2.error:
        ww, wh = img.shape[1], img.shape[0]
    cv2.imshow('TailorCam', _ui.present(img, ww, wh, GOLD_BG))


def _dpi_aware():
    """Stop Windows from bitmap-stretching (blurring) the window at 125 % / 150 % display scaling."""
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
        u = ctypes.windll.user32
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        return None

# ── Optional: SQLite customer database ────────────────────────────────────────
try:
    from model.customer_db import (add_customer, save_measurement, update_customer,
                                    create_order, search_customers,
                                    get_today_stats,
                                    get_inventory, get_low_stock_items,
                                    get_staff_list, get_customer_loyalty,
                                    save_feedback, get_feedback_corrections,
                                    get_feedback_stats,
                                    save_customer_photo, get_customer_photo,
                                    get_measurements)
    _have_db = True
    print('[DB] Customer database ready')
except Exception as _db_e:
    _have_db = False
    print(f'[DB] Not available: {_db_e}')
from model.data_collector import DataCollector
try:
    from model.kalman_landmarks import LandmarkKalmanFilter as _KF
    from model.constraint_solver import solve as _constraint_solve
    from model.bayesian_body import BayesianBodyMeasure as _BayesianBody
    _have_new_modules = True
except ImportError as _ie:
    _have_new_modules = False
    print(f'[Info] New accuracy modules not loaded: {_ie}')

CONFIG_FILE    = os.path.join(os.path.dirname(__file__), 'tailorcam_config.json')
DEFAULT_FOLDER = os.path.join(os.path.expanduser('~'), 'Documents', 'TailorCam Measurements')

# ── Photo capture: front + one side ───────────────────────────────────────────
# Every accuracy test (real24, HBW) used a front and a side photo; the second
# side and the back view added scan time without a measured gain.
PHOTO_VIEWS = [
    # (view_name, instruction_text, smpl_rotation_deg)
    ('FRONT', 'Face the camera  --  arms slightly away from the body',   0),
    ('LEFT',  'Turn to your left  --  arms relaxed by your sides',     -90),
]

# Timers are in 1/30 s "ticks" of REAL time (see _tick in the main loop), so a slow
# camera / busy computer no longer stretches them.
MATCH_FRAMES  = 15   # 0.5 s with a good pose before the countdown starts
COUNTDOWN_SEC = 1.5  # seconds of countdown
FLASH_FRAMES  = 8    # brief capture flash
TURN_FRAMES   = 45   # 1.5 s "turn to your left" card between views

AUTO_ADVANCE_SECS  = 15   # auto-advance results screen
GUIDE_TIMEOUT_SECS = 15   # auto-force-capture if pose never locks (back view esp.)
RETAKE_THRESHOLD   = 0.45 # if capture-time score below this → auto-retake once

# Fields visible in the tailor feedback overlay (same order as results display)
_FEEDBACK_KEYS = [
    'chest', 'waist', 'hip', 'shoulder_width',
    'sleeve_length', 'back_length', 'shirt_length', 'neck',
]


def _load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {'save_folder': DEFAULT_FOLDER}


def _save_config(cfg):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


def _no_camera_frame(w=1280, h=720):
    frame = np.zeros((h, w, 3), dtype='uint8')
    msg = 'Camera unavailable -- reconnecting...'
    (tw, _), _ = _text_size(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    _put_text(frame, msg, ((w - tw) // 2, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (60, 60, 80), 2)
    return frame


def _estimate_height_cm(screen_lms, LM):
    """
    Auto-detect standing height from MediaPipe Pose landmarks.

    Front/Back scale refs:
      - Ear-to-ear  ≈ 14.5 cm (bizygomatic width)
      - Eye-to-eye  ≈  6.3 cm (interpupillary)
      - Shoulder span ≈ 40 cm average adult (fallback for back view)
    Body extent: head-top → lower ankle.
    Accuracy: ±4-7 cm (SMPL shape betas compensate).
    """
    try:
        l_ear   = screen_lms[LM.LEFT_EAR]
        r_ear   = screen_lms[LM.RIGHT_EAR]
        l_eye   = screen_lms[LM.LEFT_EYE]
        r_eye   = screen_lms[LM.RIGHT_EYE]
        l_ankle = screen_lms[LM.LEFT_ANKLE]
        r_ankle = screen_lms[LM.RIGHT_ANKLE]
        l_sh    = screen_lms[LM.LEFT_SHOULDER]
        r_sh    = screen_lms[LM.RIGHT_SHOULDER]

        scales = []

        # ── Face landmarks (front view) ──────────────────────────────────
        ear_px = abs(r_ear['x'] - l_ear['x'])
        if (ear_px > 12
                and l_ear.get('visibility', 0) > 0.45
                and r_ear.get('visibility', 0) > 0.45):
            scales.append(14.5 / ear_px)

        eye_px = abs(r_eye['x'] - l_eye['x'])
        if (eye_px > 6
                and l_eye.get('visibility', 0) > 0.45
                and r_eye.get('visibility', 0) > 0.45):
            scales.append(6.3 / eye_px)

        # ── Shoulder span fallback (works for BACK view too) ─────────────
        sh_px = abs(r_sh['x'] - l_sh['x'])
        sh_vis = min(l_sh.get('visibility', 0), r_sh.get('visibility', 0))
        if sh_px > 20 and sh_vis > 0.5 and not scales:
            # Average adult shoulder span ≈ 40 cm; use only if face not available
            scales.append(40.0 / sh_px)

        if not scales:
            return None

        cm_per_px = sum(scales) / len(scales)

        # Foot reference: lower of two ankles
        la_vis = l_ankle.get('visibility', 0)
        ra_vis = r_ankle.get('visibility', 0)
        if la_vis < 0.25 and ra_vis < 0.25:
            return None
        foot_y = max(l_ankle['y'], r_ankle['y'])

        # Head-top proxy
        if ear_px > 12 and l_ear.get('visibility', 0) > 0.3:
            ear_mid_y  = (l_ear['y'] + r_ear['y']) / 2
            head_top_y = ear_mid_y - ear_px * 0.38
        else:
            # Back view: estimate head top from shoulder midpoint minus ~18% height
            sh_mid_y   = (l_sh['y'] + r_sh['y']) / 2
            # Head is roughly 13% of body height above shoulders
            head_top_y = sh_mid_y - sh_px * 0.55

        body_px   = abs(foot_y - head_top_y)
        height_cm = body_px * cm_per_px + 3.5

        if 130 < height_cm < 225:
            return round(height_cm, 1)
        return None
    except Exception:
        return None


ARMS_AWAY_MIN = 0.45   # same rule as model/hmr/hmr_worker.py: silhouettes only help above this


def _arms_away(screen_lms, LM):
    """Wrist–hip horizontal gap / shoulder width (A-pose ≈ 0.5+, arms down ≈ 0.1–0.4)."""
    try:
        sw = abs(screen_lms[LM.LEFT_SHOULDER]['x'] - screen_lms[LM.RIGHT_SHOULDER]['x'])
        if sw < 1:
            return None
        gap_l = abs(screen_lms[LM.LEFT_WRIST]['x'] - screen_lms[LM.LEFT_HIP]['x'])
        gap_r = abs(screen_lms[LM.RIGHT_WRIST]['x'] - screen_lms[LM.RIGHT_HIP]['x'])
        return (gap_l + gap_r) / 2 / sw
    except (KeyError, TypeError, IndexError):
        return None


def _pose_score(screen_lms, LM, view_name, img_w, img_h):
    """
    Returns 0.0-1.0 quality score for how well the person matches the
    expected pose for this view. Front / back views are capped below the
    capture threshold until the arms are held away from the body.
    """
    score = _pose_score_base(screen_lms, LM, view_name, img_w, img_h)
    if view_name in ('FRONT', 'BACK'):
        aa = _arms_away(screen_lms, LM)
        if aa is not None and aa < ARMS_AWAY_MIN:
            score = min(score, 0.6)
    return score


def _pose_score_base(screen_lms, LM, view_name, img_w, img_h):
    try:
        L_sh = screen_lms[LM.LEFT_SHOULDER]
        R_sh = screen_lms[LM.RIGHT_SHOULDER]
        nose = screen_lms[LM.NOSE]
        L_hi = screen_lms[LM.LEFT_HIP]

        lx = L_sh['x'];  rx = R_sh['x']
        sh_w    = abs(rx - lx)
        body_cx = (lx + rx) / 2
        vis     = min(L_sh.get('visibility', 1.0), R_sh.get('visibility', 1.0))

        if vis < 0.25:
            return 0.0

        center_dev = abs(body_cx - img_w * 0.5) / (img_w * 0.30)
        center_ok  = max(0.0, 1.0 - center_dev)

        hip_y    = L_hi['y']
        body_ok  = 1.0 if 0 < hip_y < img_h * 0.95 else 0.0
        nose_vis = nose.get('visibility', 0.5)

        if view_name == 'FRONT':
            wide = min(1.0, sh_w / (img_w * 0.14))
            return min(1.0, vis * (wide * 0.45 + nose_vis * 0.30
                                   + center_ok * 0.15 + body_ok * 0.10))

        elif view_name in ('LEFT', 'RIGHT'):
            narrow = max(0.0, 1.0 - sh_w / (img_w * 0.12))
            return min(1.0, vis * (narrow * 0.55 + center_ok * 0.25
                                   + body_ok * 0.20))

        elif view_name == 'BACK':
            # BACK view: MediaPipe assigns lower visibility to back-facing landmarks.
            # Use raw shoulder visibility (not nose) and a much lower vis floor.
            # nose NOT visible is GOOD for back view.
            wide     = min(1.0, sh_w / (img_w * 0.14))
            no_face  = max(0.0, 1.0 - nose_vis)          # 1.0 when face away
            # vis for back can be 0.5-0.75 which is fine — don't penalise it heavily
            vis_adj  = min(1.0, vis * 1.35)              # boost back-view visibility
            return min(1.0, vis_adj * (wide * 0.45 + no_face * 0.30
                                       + center_ok * 0.15 + body_ok * 0.10))

    except Exception:
        return 0.0
    return 0.0


# ── Garment thickness correction (cm subtracted from circumferences) ───────────
GARMENT_TYPES = ['tight_fit', 't_shirt', 'shirt', 'sweater', 'jacket']
GARMENT_LABELS = ['Tight fit', 'T-Shirt', 'Shirt', 'Sweater', 'Jacket']
GARMENT_OFFSETS = {
    'tight_fit': {'chest': 0.0, 'waist': 0.0, 'hip': 0.0},
    't_shirt':   {'chest': 1.5, 'waist': 1.0, 'hip': 1.5},
    'shirt':     {'chest': 2.5, 'waist': 1.5, 'hip': 2.0},
    'sweater':   {'chest': 4.0, 'waist': 2.5, 'hip': 3.5},
    'jacket':    {'chest': 6.0, 'waist': 3.5, 'hip': 4.5},
}


def _apply_garment_correction(measurements: dict, garment_type: str) -> dict:
    """Subtract estimated garment fabric thickness from circumference measurements."""
    offsets = GARMENT_OFFSETS.get(garment_type, {})
    if not offsets:
        return measurements
    out = dict(measurements)
    for key, off in offsets.items():
        if key in out and out[key] is not None:
            out[key] = round(max(0.0, out[key] - off), 1)
    return out


def _mrs_factors(world_lms_dicts, view_name: str) -> dict:
    """
    Compute Measurement Readiness Score factors from 3-D world landmarks.

    Returns dict with keys: symmetry, verticality, arm_clearance, overall (0-1 each).
    'overall' is a composite that can be added to pose_score as a bonus.
    """
    result = {'symmetry': 1.0, 'verticality': 1.0, 'arm_clearance': 1.0, 'overall': 0.0}
    if not world_lms_dicts or len(world_lms_dicts) < 33:
        return result
    try:
        def pt(i):
            lm = world_lms_dicts[i]
            return (lm.get('x', 0.0), lm.get('y', 0.0), lm.get('z', 0.0))

        l_sh = pt(11); r_sh = pt(12)
        l_hi = pt(23); r_hi = pt(24)
        l_wr = pt(15); r_wr = pt(16)

        if view_name in ('FRONT', 'BACK'):
            # Shoulder symmetry: L and R shoulder Y should be within 2% of torso height
            torso_h = abs(((l_sh[1] + r_sh[1]) / 2) - ((l_hi[1] + r_hi[1]) / 2))
            sh_diff  = abs(l_sh[1] - r_sh[1])
            symmetry = max(0.0, 1.0 - sh_diff / (torso_h * 0.04 + 1e-6))
            result['symmetry'] = min(1.0, symmetry)

            # Arm clearance: wrist X should be outside hip X by at least 8%
            left_clear  = (l_hi[0] - l_wr[0]) / (abs(l_hi[0]) + 1e-6)
            right_clear = (r_wr[0] - r_hi[0]) / (abs(r_hi[0]) + 1e-6)
            arm_score   = min(1.0, max(0.0, (left_clear + right_clear) / 0.16))
            result['arm_clearance'] = arm_score

        # Torso verticality: spine should be near-vertical (Y dominant)
        sh_mid = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2, (l_sh[2] + r_sh[2]) / 2)
        hi_mid = ((l_hi[0] + r_hi[0]) / 2, (l_hi[1] + r_hi[1]) / 2, (l_hi[2] + r_hi[2]) / 2)
        spine_len = math.sqrt(sum((sh_mid[k] - hi_mid[k]) ** 2 for k in range(3))) + 1e-8
        vert_component = abs(sh_mid[1] - hi_mid[1]) / spine_len
        result['verticality'] = min(1.0, max(0.0, (vert_component - 0.88) / 0.10))

        # Overall bonus (0-0.15 range, added to pose_score)
        result['overall'] = (
            result['symmetry']    * 0.05 +
            result['verticality'] * 0.05 +
            result['arm_clearance'] * 0.05
        )
    except Exception:
        pass
    return result


def _screen_lms_to_list(screen_lms, img_w, img_h):
    """Convert screen_lms dict to list of [x_px, y_px, conf] for SMPL."""
    kps = []
    for i in range(33):
        try:
            lm = screen_lms[i]
            kps.append([float(lm['x']), float(lm['y']),
                        float(lm.get('visibility', 1.0))])
        except (KeyError, TypeError, IndexError):
            kps.append([img_w / 2, img_h / 2, 0.0])
    return kps



def _launch_body_worker(height_cm, world_views, show_toast_fn):
    """
    Start body_worker.py in a daemon thread (Python 3.13 + PyTorch/CUDA).
    Returns (proc_box, True).  proc_box[0] is filled by the thread.
    """
    if not world_views:
        return None, False

    # Convert MediaPipe landmark objects → plain [x, y, z] lists for JSON
    def _lms_to_list(lms):
        if lms is None:
            return None
        out = []
        for lm in lms:
            if hasattr(lm, 'x'):
                out.append([lm.x, lm.y, lm.z])
            elif isinstance(lm, (list, tuple)):
                out.append(list(lm[:3]))
            else:
                out.append([0.0, 0.0, 0.0])
        return out

    payload = json.dumps({
        'height_cm': round(height_cm, 1),
        'views': {k: _lms_to_list(v) for k, v in world_views.items()
                  if v is not None},
    })

    tmp_path = os.path.join(os.path.dirname(__file__), 'model', '_body_input.json')
    try:
        with open(tmp_path, 'w') as fh:
            fh.write(payload)
    except Exception as e:
        print(f'[BodyWorker] Cannot write input file: {e}')
        return None, False

    worker = os.path.join(os.path.dirname(__file__), 'model', 'body_worker.py')
    py_cmd = (sys.executable.replace('python3.14', 'python3.13')
               if '3.14' in sys.executable else 'py')

    print(f'[BodyWorker] Launching  height={height_cm}cm  '
          f'views={list(world_views.keys())}')

    proc_box = [None]

    def _spawn():
        try:
            proc = subprocess.Popen(
                [py_cmd, '-3.13', worker, tmp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            proc_box[0] = proc
        except FileNotFoundError:
            print('[BodyWorker] py -3.13 not found')
            proc_box[0] = 'error'
        except Exception as exc:
            print(f'[BodyWorker] Launch error: {exc}')
            proc_box[0] = 'error'

    threading.Thread(target=_spawn, daemon=True).start()
    show_toast_fn('Refining with neural model…', (28, 145, 215), 6.0)
    return proc_box, True


def _photo_messages(checks, uploaded):
    """Short, specific advice from the worker's per-photo checks ('' if all fine)."""
    msgs = []
    for c in checks:
        n = c.get('photo')
        if not c.get('person'):
            msgs.append(f"Photo {n}: {c.get('problem', 'no person found')}")
        elif not c.get('full_body'):
            msgs.append(f'Photo {n}: whole body not visible (head to feet)')
        elif c.get('person_height_frac', 1) < 0.35:
            msgs.append(f'Photo {n}: person too small -- stand closer')
    if uploaded:
        views = [c.get('view_guess') for c in checks if c.get('person')]
        if views and 'SIDE' not in views:
            msgs.append('add a SIDE photo for better accuracy')
        elif views and 'FRONT' not in views:
            msgs.append('add a FRONT photo for better accuracy')
    return '; '.join(msgs)


def main():
    print('=' * 52)
    print('  TailorCam  —  3-View Photo Body Scanner')
    print('=' * 52)
    print('\nHow to use:')
    print('  1. Stand in front of camera (height auto-detected, or press T to type)')
    print('  2. App counts down and captures FRONT view automatically')
    print('  3. Turn LEFT side to camera -> captured')
    print('  4. Turn RIGHT side to camera -> captured')
    print('  5. Measurements shown instantly + neural model refines in background')
    print('\nControls:  T type-height   R restart   S save-PDF   F folder   N next   Q quit\n')

    cfg         = _load_config()
    save_folder = cfg.get('save_folder', DEFAULT_FOLDER)

    try:
        camera = Camera(index=0)
    except RuntimeError as e:
        print(f'[Error] {e}')
        return

    detector  = PoseDetector()
    collector = DataCollector()

    # ── Optional: MediaPipe selfie segmentation for background removal ─────────
    _segmentor = None
    try:
        import mediapipe as mp
        _seg_mp = mp.solutions.selfie_segmentation
        _segmentor = _seg_mp.SelfieSegmentation(model_selection=0)
        print('[Segment] MediaPipe selfie segmentation ready')
    except Exception as _se:
        print(f'[Segment] Not available ({_se}) — full frame mode')

    def _apply_bg_removal(frame_bgr):
        """Replace background with dark navy for cleaner pose detection."""
        if _segmentor is None:
            return frame_bgr
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = _segmentor.process(rgb)
            mask = result.segmentation_mask   # float 0-1, 1=person
            mask3 = np.stack([mask, mask, mask], axis=-1)
            bg = np.full_like(frame_bgr, (42, 22, 13), dtype=np.uint8)
            out = (frame_bgr * mask3 + bg * (1 - mask3)).astype(np.uint8)
            return out
        except Exception:
            return frame_bgr

    # ── Session state ──────────────────────────────────────────────────────────
    height_cm    = None   # auto-detected from landmarks; None until first estimate
    _height_ests = []     # rolling buffer of recent per-frame estimates

    # Manual height override (press T during scanning to type)
    _manual_mode  = False
    _manual_input = ''

    # ── Per-customer state ─────────────────────────────────────────────────────
    photo_state    = 'GUIDE'   # GUIDE|COUNTDOWN|FLASH|TURN|PROCESSING|RESULTS
    view_idx       = 0
    captured_views = []        # list of {keypoints_2d, rotation_deg, img_w, img_h}
    world_views    = {}        # {view_name: world_landmarks} for body_measurer
    match_frames   = 0         # consecutive frames with good pose score
    countdown_f    = 0         # countdown frames remaining
    flash_f        = 0         # flash frames remaining
    turn_f         = 0         # turn instruction frames remaining
    final_m        = None      # final measurement dict
    _guide_start   = time.time()  # when current GUIDE view began (for timeout)
    _retake_count  = 0            # how many auto-retakes have happened this view
    showing_results = False
    advance_start  = None
    countdown_secs = None
    _last_voice_state = None   # track voice triggers to avoid repetition

    # ── Multi-frame landmark buffers (accuracy improvement) ────────────────────
    _lm_buffers  = {'FRONT': [], 'LEFT': [], 'RIGHT': [], 'BACK': []}
    _LM_BUF_MAX  = 30   # keep at most N frames per view
    _LM_BUF_USE  = 18   # median-smooth the last N frames at capture time

    # ── Kalman filters — one per view for smooth landmark tracking ─────────────
    if _have_new_modules:
        _kalmans = {'FRONT': _KF(), 'LEFT': _KF(), 'RIGHT': _KF(), 'BACK': _KF()}
    else:
        _kalmans = {}

    # ── Bayesian body measurer (trains on synthetic data on first use) ─────────
    _bayesian_m     = None
    _uncertainties  = {}   # last prediction uncertainties {key: sigma}
    _tape_status    = {}   # {key: camera/tape verdict} from model.calibration.tape_check
    if _have_new_modules:
        def _init_bayesian():
            global _bayesian_m
            try:
                bm = _BayesianBody()
                bm.ensure_trained()
                _bayesian_m = bm
            except Exception as e:
                print(f'[BayesianBody] Init failed: {e}')
        threading.Thread(target=_init_bayesian, daemon=True).start()

    # ── Garment type (affects circumference correction) ────────────────────────
    _garment_type  = 't_shirt'   # default; user can cycle with G key

    # SMPL background state
    _smpl_proc    = None
    _smpl_running = False
    _smpl_mesh_img = None

    # HMR 3D body model: loads in its own process while the user gets ready
    _hmr        = _HMRService().start() if _HMRService is not None else None
    _hmr_frames = []      # [(view_name, clean_bgr_frame, keypoints)] for this scan
    _raw_frame  = None

    # UI / naming state
    naming_mode  = False
    name_input   = ''
    _picker_proc = None
    _toast_msg   = ''
    _toast_color = (40, 182, 28)
    _toast_until = 0.0

    def show_toast(msg, color=(40, 182, 28), duration=3.0):
        nonlocal _toast_msg, _toast_color, _toast_until
        _toast_msg, _toast_color, _toast_until = msg, color, time.time() + duration

    def _save_profile():
        """Create or update the customer's profile with the details just entered."""
        nonlocal _current_customer_id
        if not _have_db or not _current_customer_name:
            return
        try:
            info = dict(age=_age_years, weight_kg=_weight_kg, height_cm=height_cm, gender=_gender)
            if _current_customer_id is None:
                _current_customer_id = add_customer(_current_customer_name, **info)
            else:
                update_customer(_current_customer_id, name=_current_customer_name,
                                **{k: v for k, v in info.items() if v is not None})
            print(f'[DB] Profile saved: {_current_customer_name} (id {_current_customer_id})')
        except Exception as exc:
            print(f'[DB] Profile not saved: {exc}')

    def _reset_customer(same_person=False, profile=None):
        """
        Start a new scan. A new person clears weight / age / gender (they used to
        carry over to the next customer) and opens the details screen; profile =
        a saved customer record pre-fills it. same_person=True (Restart) keeps them.
        """
        nonlocal _weight_kg, _age_years, _gender, _wa_skip_warned
        nonlocal photo_state, view_idx, captured_views, world_views, match_frames
        nonlocal countdown_f, flash_f, turn_f, final_m, showing_results
        nonlocal _guide_start, _retake_count
        nonlocal advance_start, countdown_secs, naming_mode, name_input
        nonlocal _smpl_proc, _smpl_running, _smpl_mesh_img, _hmr_frames
        nonlocal height_cm, _height_ests, _manual_mode, _manual_input
        nonlocal _lm_buffers, _uncertainties, _last_voice_state
        nonlocal _wa_input_mode, _wa_weight_txt, _wa_age_txt, _wa_active
        nonlocal _wa_height_txt, _height_locked, _upload_mode, _upload_result, _wa_name_txt
        nonlocal _order_mode, _order_field, _order_vals
        nonlocal _current_customer_id, _current_customer_name, _last_measurement_id
        nonlocal _results_tab
        nonlocal _feedback_mode, _feedback_field_idx, _feedback_vals
        nonlocal _profile_photo
        nonlocal _edit_field, _edit_input_txt, _results_card_rects, _hist_measurements
        photo_state  = 'GUIDE';  view_idx = 0;  captured_views = [];  world_views = {}
        match_frames = 0;  countdown_f = 0;  flash_f = 0;  turn_f = 0
        final_m = None;  showing_results = False
        _guide_start = time.time();  _retake_count = 0
        advance_start = None;  countdown_secs = None
        naming_mode = False;  name_input = ''
        _smpl_mesh_img = None;  _hmr_frames = [];  _upload_mode = False;  _upload_result = None
        if not same_person or not _height_locked:
            height_cm = None;  _height_locked = False
        _height_ests = []
        _manual_mode = False;  _manual_input = ''
        _lm_buffers    = {'FRONT': [], 'LEFT': [], 'RIGHT': [], 'BACK': []}
        _uncertainties = {}
        _last_voice_state = None
        # Reset weight/age + order state
        _wa_input_mode = False;  _wa_weight_txt = '';  _wa_age_txt = '';  _wa_active = 'height'
        _wa_height_txt = f'{height_cm:g}' if height_cm else ''
        _wa_skip_warned = False
        _arms_spoken.clear()
        if not same_person:
            p = profile or {}
            _weight_kg = p.get('weight_kg') or None
            _age_years = p.get('age') or None
            _gender = p.get('gender') if p.get('gender') in ('male', 'female') else 'unspecified'
            _wa_weight_txt = f'{_weight_kg:g}' if _weight_kg else ''
            _wa_age_txt = str(_age_years) if _age_years else ''
            _wa_height_txt = f"{p['height_cm']:g}" if p.get('height_cm') else ''
            _wa_name_txt = p.get('name') or _current_customer_name or ''
            _wa_active = 'height' if _wa_name_txt else 'name'
            _wa_input_mode = True          # ask for the details before scanning
        _order_mode = False;  _order_field = 'garment'
        _order_vals = {k: '' for k in _order_field_seq}
        # Pre-fill staff assignment with first staff name
        if _staff_names:
            _order_vals['assigned_to'] = _staff_names[0]
        # Keep customer selected but clear measurement link and tab
        _last_measurement_id = None
        _results_tab = 0
        _feedback_mode = False;  _feedback_field_idx = 0;  _feedback_vals = {}
        _profile_photo = None
        _edit_field = None;  _edit_input_txt = '';  _results_card_rects = {}
        _hist_measurements = []
        for kf in _kalmans.values():
            kf.reset()
        if _smpl_proc is not None:
            try:
                p = _smpl_proc[0]
                if p and p != 'error':
                    p.kill()
            except Exception:
                pass
        _smpl_proc = None;  _smpl_running = False

    # ── Weight + age + gender (optional, improve Bayesian accuracy) ───────────
    _weight_kg     = None
    _age_years     = None
    _gender        = 'unspecified'   # 'male' | 'female' | 'unspecified'
    _wa_input_mode = False     # True = show weight/age input overlay
    _wa_weight_txt = ''
    _wa_age_txt    = ''
    _wa_height_txt = ''
    _wa_name_txt   = ''        # customer / profile name the scan is saved under
    _upload_pending = False    # True = open the photo picker once the details are entered
    _wa_active     = 'height'  # which field is focused: name | height | weight | age
    _height_locked = False     # True = height typed by the tailor (never auto-guessed)
    _upload_mode   = False     # True = this result comes from uploaded photos (O key)
    _upload_result = None      # worker reply for the uploaded photos (measured at once)
    _wa_skip_warned = False    # Esc pressed once without a weight
    _arms_hint_until = 0.0     # next time the "arms away" hint may show
    _arms_spoken     = set()   # views where the arms hint was already spoken

    # ── Tailor feedback state ──────────────────────────────────────────────────
    _feedback_mode         = False   # True = showing feedback entry overlay
    _feedback_field_idx    = 0       # which field is selected (0-7)
    _feedback_vals         = {}      # {field: actual_cm_str}  user-typed corrections

    # ── Customer state ─────────────────────────────────────────────────────────
    _current_customer_id   = None
    _current_customer_name = ''
    _cust_select_btns      = {}
    _cust_list_cache       = []
    _cust_search_query     = ''
    _cust_scroll           = 0
    _last_measurement_id   = None
    _profile_photo         = None   # BGR numpy array of current customer's photo
    _photo_cache           = {}     # {customer_id: BGR array} for select screen avatars

    # ── Order creation state ───────────────────────────────────────────────────
    _order_mode      = False
    _order_field     = 'garment'
    _order_vals      = {'garment':'', 'notes':'', 'fabric':'', 'colour':'',
                        'price':'', 'delivery':'', 'assigned_to':''}
    _order_field_seq = ['garment', 'notes', 'fabric', 'colour', 'price',
                        'delivery', 'assigned_to']

    # ── Results tab (0=Measurements, 1=Size Guide, 2=History) ────────────────
    _results_tab    = 0
    _size_country      = 'BD'   # active country for size suggestions
    _edit_field        = None   # field being manually edited (or None)
    _edit_input_txt    = ''     # text typed for the edit
    _results_card_rects = {}    # {field: (x0,y0,x1,y1)} from last draw_results_overlay
    _hist_measurements  = []    # past scans for current customer (from DB)

    # ── Settings state ─────────────────────────────────────────────────────────
    _shop_name   = 'TailorCam Shop'
    _currency    = 'BDT'
    _settings_btns = {}
    # Load persisted settings from config
    try:
        with open(CONFIG_FILE) as _cf:
            _cfg = json.load(_cf)
            _shop_name   = _cfg.get('shop_name', _shop_name)
            _currency    = _cfg.get('currency',  _currency)
            _output_mod.USE_INCHES = _cfg.get('use_inches', False)
    except Exception:
        pass

    # ── DB stats cache (refresh on home screen) ────────────────────────────────
    _db_stats = {}
    def _refresh_db_stats():
        nonlocal _db_stats
        if _have_db:
            try:
                _db_stats = get_today_stats()
            except Exception:
                pass

    _refresh_db_stats()

    # ── Inventory state ────────────────────────────────────────────────────────
    _inventory_items    = []
    _inventory_low_ids  = set()
    _inventory_category = 'all'
    _inventory_btns     = {}

    def _refresh_inventory():
        nonlocal _inventory_items, _inventory_low_ids
        if _have_db:
            try:
                _inventory_items   = get_inventory()
                low_items          = get_low_stock_items()
                _inventory_low_ids = {it['id'] for it in low_items}
            except Exception:
                pass

    _refresh_inventory()

    # ── Staff state ────────────────────────────────────────────────────────────
    _staff_names = ['Main Tailor']
    try:
        if _have_db:
            _staff_names = [s['name'] for s in get_staff_list()] or ['Main Tailor']
    except Exception:
        pass

    # ── Loyalty cache for customer select ─────────────────────────────────────
    _loyalty_cache = {}   # {customer_id: {'points': int, 'tier': str}}

    def _refresh_loyalty(customers):
        """Pre-fetch loyalty points for the current customer list."""
        nonlocal _loyalty_cache
        if not _have_db:
            return
        for c in customers:
            cid = c.get('id')
            if cid and cid not in _loyalty_cache:
                try:
                    _loyalty_cache[cid] = get_customer_loyalty(cid)
                except Exception:
                    pass

    # ── App screen ─────────────────────────────────────────────────────────────
    app_screen    = 'SPLASH'
    splash_start  = time.time()
    _home_btns    = {}
    _hist_btns    = {}
    _hist_scroll  = 0
    _hist_records = load_records(save_folder)  # pre-load for dashboard

    # ── Previous scan comparison ───────────────────────────────────────────────
    def _get_prev_measurements():
        """Return the most recent saved scan's measurements for comparison."""
        recs = load_records(save_folder)
        if not recs:
            return None
        latest = recs[-1]
        return {k: latest.get(k) for k in
                ('shoulder_width','chest','waist','hip','back_length',
                 'shirt_length','sleeve_length','neck')
                if latest.get(k) is not None} or None
    _click_buf    = [None]
    _mouse_pos    = [0, 0]

    def _on_mouse(event, x, y, *_):
        x, y = _ui.to_design(x, y)            # window pixels → screen layout pixels
        _mouse_pos[0] = x
        _mouse_pos[1] = y
        if event == cv2.EVENT_LBUTTONDOWN:
            _click_buf[0] = (x, y)

    def _wait_key():
        """waitKey that also turns a click on a shortcut button into that key."""
        k = cv2.waitKey(1) & 0xFF
        click = _click_buf[0]
        if k == 255 and click is not None:
            for rect, code in reversed(_output_mod.CLICK_TARGETS):
                if _in_rect(click[0], click[1], rect):
                    _click_buf[0] = None
                    k = code
                    break
        _output_mod.CLICK_TARGETS.clear()
        return k

    _ui.DEFER = True                          # text is drawn sharply by _show()
    _scr = _dpi_aware()
    cv2.namedWindow('TailorCam', cv2.WINDOW_NORMAL)
    if _scr:                                  # open at ~85 % of the screen, camera aspect ratio
        _ww = int(min(_scr[0] * 0.85, _scr[1] * 0.85 * camera.width / camera.height))
        cv2.resizeWindow('TailorCam', _ww, int(_ww * camera.height / camera.width))
    else:
        cv2.resizeWindow('TailorCam', camera.width, camera.height)
    cv2.setMouseCallback('TailorCam', _on_mouse)
    _fullscreen = False   # tracks current fullscreen state

    print(f'Save folder: {save_folder}')

    _last_now = time.time()
    while True:
        now   = time.time()
        _tick = max(1, min(10, int(round((now - _last_now) * 30))))   # real 1/30 s ticks since last frame
        _last_now = now
        frame = camera.read()

        if frame is None:
            if camera.needs_reopen:
                if not camera.reopen():
                    pass
            frame = _no_camera_frame(camera.width, camera.height)

        # ── Splash / intro screen ──────────────────────────────────────────────
        if app_screen == 'SPLASH':
            elapsed = now - splash_start
            splash_frame, splash_done = draw_splash_screen(frame.copy(), elapsed)
            _show(splash_frame)
            key = _wait_key()
            if splash_done or (key not in (0xFF, 255)):
                app_screen = 'HOME'
            continue

        # ── Home screen ────────────────────────────────────────────────────────
        if app_screen == 'HOME':
            _refresh_db_stats()   # refresh revenue/order counts every home render
            home_frame, _home_btns = draw_home_screen(
                frame.copy(), records=_hist_records,
                t=now, mouse_xy=(_mouse_pos[0], _mouse_pos[1]),
                stats=_db_stats)
            _show(home_frame)
            click = _click_buf[0]
            if click:
                _click_buf[0] = None
                cx, cy = click
                new_btn = (_in_rect(cx, cy, _home_btns.get('new')) or
                           _in_rect(cx, cy, _home_btns.get('new_cta')) or
                           _in_rect(cx, cy, _home_btns.get('scan')))
                if new_btn:
                    _current_customer_id = None; _current_customer_name = ''
                    _reset_customer()
                    app_screen = 'SCANNING'
                elif (_in_rect(cx, cy, _home_btns.get('history')) or
                      _in_rect(cx, cy, _home_btns.get('home'))):
                    _hist_records = load_records(save_folder)
                    _hist_scroll  = 0
                    app_screen    = 'HISTORY'
                elif _in_rect(cx, cy, _home_btns.get('upload')):
                    _current_customer_id = None; _current_customer_name = ''
                    _reset_customer()
                    _upload_pending = True
                    app_screen = 'SCANNING'
                elif _in_rect(cx, cy, _home_btns.get('custs')):
                    _cust_search_query = ''; _cust_scroll = 0
                    if _have_db:
                        _cust_list_cache = search_customers('', limit=50)
                    app_screen = 'CUSTOMER_SELECT'
                elif _in_rect(cx, cy, _home_btns.get('reports')):
                    _hist_records = load_records(save_folder)
                    app_screen = 'REPORTS'
                elif _in_rect(cx, cy, _home_btns.get('settings')):
                    app_screen = 'SETTINGS'
                elif _in_rect(cx, cy, _home_btns.get('exit')):
                    break
            key = _wait_key()
            if key in (ord('q'), 27):
                break
            elif key == ord('n'):
                _current_customer_id = None; _current_customer_name = ''
                _reset_customer()
                app_screen = 'SCANNING'
            elif key == ord('o'):
                _current_customer_id = None; _current_customer_name = ''
                _reset_customer()
                _upload_pending = True
                app_screen = 'SCANNING'
            elif key == ord('h'):
                _hist_records = load_records(save_folder)
                _hist_scroll  = 0
                app_screen    = 'HISTORY'
            elif key == ord('c'):
                # C = customer select
                _cust_search_query = ''; _cust_scroll = 0
                if _have_db:
                    _cust_list_cache = search_customers('', limit=50)
                app_screen = 'CUSTOMER_SELECT'
            elif key == ord('p'):
                # P = reports / analytics
                _hist_records = load_records(save_folder)
                app_screen = 'REPORTS'
            elif key == ord('v'):
                # V = inventory
                _refresh_inventory()
                app_screen = 'INVENTORY'
            elif key == ord('e'):
                # E = settings
                app_screen = 'SETTINGS'
            elif key == ord('a'):
                # A = weight/age calibration
                _wa_input_mode = True
                _wa_weight_txt = str(int(_weight_kg)) if _weight_kg else ''
                _wa_age_txt    = str(_age_years) if _age_years else ''
                _wa_height_txt = f'{height_cm:g}' if height_cm else ''
                _wa_active = 'height'
            continue

        # ── History screen ─────────────────────────────────────────────────────
        if app_screen == 'HISTORY':
            hist_frame, _hist_btns = draw_history_screen(
                frame.copy(), _hist_records, _hist_scroll)
            _show(hist_frame)
            click = _click_buf[0]
            if click:
                _click_buf[0] = None
                cx, cy = click
                if _in_rect(cx, cy, _hist_btns.get('back')):
                    app_screen = 'HOME'
                elif _in_rect(cx, cy, _hist_btns.get('exit')):
                    break
                elif _in_rect(cx, cy, _hist_btns.get('new')):
                    _reset_customer()
                    app_screen = 'SCANNING'
            key = _wait_key()
            if key == ord('q'):
                break
            elif key in (27, ord('b')):
                app_screen = 'HOME'
            elif key == ord('n'):
                _reset_customer()
                app_screen = 'SCANNING'
            elif key == 82:
                _hist_scroll = max(0, _hist_scroll - 1)
            elif key == 84:
                _hist_scroll += 1
            continue

        # ── Customer select screen ─────────────────────────────────────────────
        if app_screen == 'CUSTOMER_SELECT':
            if _have_db:
                _cust_list_cache = search_customers(_cust_search_query, limit=50)
                _refresh_loyalty(_cust_list_cache)
            else:
                _cust_list_cache = [{'name': r.get('name','?'),
                                      'phone': '', 'updated_at': r.get('date','')}
                                     for r in _hist_records]
            # Populate photo cache for visible customers
            if _have_db:
                for _c in _cust_list_cache:
                    _cid = _c.get('id')
                    if _cid and _cid not in _photo_cache:
                        try:
                            _ph = get_customer_photo(_cid)
                            if _ph and os.path.isfile(_ph):
                                _img = cv2.imread(_ph)
                                if _img is not None:
                                    _photo_cache[_cid] = _img
                        except Exception:
                            pass
            cs_frame, _cust_select_btns = draw_customer_select_screen(
                frame.copy(), _cust_list_cache, query=_cust_search_query,
                scroll=_cust_scroll, loyalty_data=_loyalty_cache,
                photo_cache=_photo_cache)
            _show(cs_frame)
            click = _click_buf[0]
            if click:
                _click_buf[0] = None
                cx, cy = click
                if _in_rect(cx, cy, _cust_select_btns.get('back')):
                    app_screen = 'HOME'
                elif _in_rect(cx, cy, _cust_select_btns.get('new_customer')):
                    _current_customer_id   = None
                    _current_customer_name = ''
                    _reset_customer()
                    app_screen = 'SCANNING'
                else:
                    for bi in range(len(_cust_list_cache)):
                        if _in_rect(cx, cy, _cust_select_btns.get(f'cust_{bi}')):
                            cust = _cust_list_cache[bi]
                            _current_customer_name = cust.get('name', '')
                            if _have_db:
                                _current_customer_id = cust.get('id')
                            _reset_customer(profile=cust)
                            # Load this customer's profile photo
                            if _have_db and _current_customer_id:
                                _ph_path = get_customer_photo(_current_customer_id)
                                if _ph_path and os.path.isfile(_ph_path):
                                    try:
                                        _profile_photo = cv2.imread(_ph_path)
                                    except Exception:
                                        _profile_photo = None
                            app_screen = 'SCANNING'
                            show_toast(f'Customer: {_current_customer_name}',
                                       (28, 145, 215))
                            break
            key = _wait_key()
            if key in (27, ord('b')):
                app_screen = 'HOME'
            elif key == ord('n'):
                _current_customer_id = None; _current_customer_name = ''
                _reset_customer(); app_screen = 'SCANNING'
            elif key == 82:
                _cust_scroll = max(0, _cust_scroll - 1)
            elif key == 84:
                _cust_scroll += 1
            elif key in (8, 127):
                _cust_search_query = _cust_search_query[:-1]
            elif 32 <= key <= 126:
                _cust_search_query += chr(key)
                _cust_scroll = 0
            continue

        # ── Inventory screen ───────────────────────────────────────────────────
        if app_screen == 'INVENTORY':
            _refresh_inventory()
            inv_frame, _inventory_btns = draw_inventory_screen(
                frame.copy(),
                items=_inventory_items,
                low_stock_ids=_inventory_low_ids,
                mouse_xy=(_mouse_pos[0], _mouse_pos[1]),
                selected_category=_inventory_category)
            _show(inv_frame)
            click = _click_buf[0]
            if click:
                _click_buf[0] = None
                cx, cy = click
                if _in_rect(cx, cy, _inventory_btns.get('back')):
                    app_screen = 'HOME'
                else:
                    # Category filter tabs
                    for cat in ('all','fabric','thread','button','other'):
                        if _in_rect(cx, cy, _inventory_btns.get(f'cat_{cat}')):
                            _inventory_category = cat
                            break
                    # Add item — open a quick toast for now (future: modal input)
                    if _in_rect(cx, cy, _inventory_btns.get('add_item')):
                        show_toast('Press A in inventory screen to add items (keyboard)',
                                   TEAL, 4.0)
            key = _wait_key()
            if key in (27, ord('h'), ord('b')):
                app_screen = 'HOME'
            elif key == ord('q'):
                break
            continue

        # ── Reports screen ─────────────────────────────────────────────────────
        if app_screen == 'REPORTS':
            rp_frame = draw_reports_screen(frame.copy(),
                                           records=_hist_records, t=now)
            _show(rp_frame)
            key = _wait_key()
            if key in (27, ord('b'), ord('h')):
                app_screen = 'HOME'
            elif key == ord('q'):
                break
            continue

        # ── Settings screen ────────────────────────────────────────────────────
        if app_screen == 'SETTINGS':
            st_frame, _settings_btns = draw_settings_screen(
                frame.copy(),
                shop_name=_shop_name,
                currency=_currency,
                use_inches=_output_mod.USE_INCHES,
                camera_idx=0,
                mouse_xy=(_mouse_pos[0], _mouse_pos[1]))
            _show(st_frame)
            click = _click_buf[0]
            if click:
                _click_buf[0] = None
                cx, cy = click
                if _in_rect(cx, cy, _settings_btns.get('back')):
                    app_screen = 'HOME'
                elif _in_rect(cx, cy, _settings_btns.get('units')):
                    _output_mod.USE_INCHES = not _output_mod.USE_INCHES
                    show_toast('Units: ' + ('Inches' if _output_mod.USE_INCHES else 'cm'), GOLD)
                elif _in_rect(cx, cy, _settings_btns.get('currency')):
                    _currencies = ['BDT', 'USD', 'INR', 'EUR', 'GBP']
                    _currency = _currencies[(_currencies.index(_currency) + 1) % len(_currencies)]
                    show_toast(f'Currency: {_currency}', GOLD)
            key = _wait_key()
            if key in (27, ord('h')):
                app_screen = 'HOME'
            elif key == ord('i'):
                _output_mod.USE_INCHES = not _output_mod.USE_INCHES
                show_toast('Units: ' + ('Inches' if _output_mod.USE_INCHES else 'cm'), GOLD)
            elif key == ord('q'):
                break
            continue

        # ── Weight/age input (overlay before scan) ─────────────────────────────
        if _wa_input_mode:
            frame = draw_weight_age_overlay(
                frame, _wa_weight_txt, _wa_age_txt, _wa_active, gender=_gender,
                warning=('No weight: measurements will be about 2x less accurate. '
                         'Esc again = skip anyway' if _wa_skip_warned else ''),
                height_text=_wa_height_txt, name_text=_wa_name_txt)
            _show(frame)
            key = _wait_key()
            try:
                _h_typed = float(_wa_height_txt) if _wa_height_txt else None
            except ValueError:
                _h_typed = None
            _h_ok = bool(_h_typed and 100 <= _h_typed <= 230)
            if 200 <= key <= 203:   # a click on an input box focuses it
                _wa_active = ['name', 'height', 'weight', 'age'][key - 200]
            elif key == 9:   # Tab = next field
                _seq = ['name', 'height', 'weight', 'age']
                _wa_active = _seq[(_seq.index(_wa_active) + 1) % 4]
            elif _wa_active == 'name' and key in (8, 127):
                _wa_name_txt = _wa_name_txt[:-1]
            elif _wa_active == 'name' and 32 <= key <= 126 and len(_wa_name_txt) < 32:
                _wa_name_txt += chr(key)
            elif key in (ord('m'), ord('M')):
                _gender = 'male'
            elif key in (ord('f'), ord('F')):
                _gender = 'female'
            elif key in (ord('u'), ord('U')):
                _gender = 'unspecified'
            elif key == 13:  # Enter = confirm (height + weight required)
                try:
                    w_ = float(_wa_weight_txt) if _wa_weight_txt else None
                except ValueError:
                    w_ = None
                if not _wa_name_txt.strip():
                    _wa_active = 'name'
                    show_toast("Please type the customer's name", RED, 2.5)
                elif not _h_ok:
                    _wa_active = 'height'
                    show_toast('Please enter the height in cm (tape or ID card)', RED, 2.5)
                elif not w_ or not (25 <= w_ <= 250):
                    _wa_skip_warned = True
                    _wa_active = 'weight'
                    show_toast('Please enter the weight (kg)', RED, 2.5)
                else:
                    _weight_kg = w_
                    height_cm, _height_locked = _h_typed, True
                    _current_customer_name = _wa_name_txt.strip()
                    try:
                        _age_years = int(_wa_age_txt) if _wa_age_txt else None
                    except ValueError:
                        _age_years = None
                    _wa_input_mode = False
                    _save_profile()
                    show_toast(f'{_current_customer_name}: {height_cm:g} cm, {_weight_kg:g} kg -- '
                               'ready to scan (O = use photos instead)', GOLD, 4.0)
            elif key == 27:  # Esc = skip weight (needs a second press); height is always needed
                if not _h_ok:
                    _wa_active = 'height'
                    show_toast('Height is required -- measurements scale with it', RED, 2.5)
                elif _wa_skip_warned or _weight_kg:
                    height_cm, _height_locked = _h_typed, True
                    _current_customer_name = _wa_name_txt.strip() or _current_customer_name
                    _save_profile()
                    _wa_input_mode = False
                    if not _weight_kg:
                        show_toast('Scanning without weight -- lower accuracy', RED, 4.0)
                else:
                    _wa_skip_warned = True
            elif key in (8, 127):
                if _wa_active == 'weight':
                    _wa_weight_txt = _wa_weight_txt[:-1]
                elif _wa_active == 'height':
                    _wa_height_txt = _wa_height_txt[:-1]
                else:
                    _wa_age_txt = _wa_age_txt[:-1]
            elif 48 <= key <= 57 and key != 0xFF:
                if _wa_active == 'weight' and len(_wa_weight_txt) < 5:
                    _wa_weight_txt += chr(key)
                elif _wa_active == 'height' and len(_wa_height_txt) < 5:
                    _wa_height_txt += chr(key)
                elif _wa_active == 'age' and len(_wa_age_txt) < 3:
                    _wa_age_txt += chr(key)
            elif key == ord('.') and _wa_active == 'weight' and '.' not in _wa_weight_txt:
                _wa_weight_txt += '.'
            elif key == ord('.') and _wa_active == 'height' and '.' not in _wa_height_txt:
                _wa_height_txt += '.'
            continue

        # ── Tailor feedback entry overlay ──────────────────────────────────────
        if _feedback_mode and showing_results and final_m:
            frame = _draw_feedback_overlay(
                frame, final_m, _feedback_keys_list=_FEEDBACK_KEYS,
                field_idx=_feedback_field_idx, vals=_feedback_vals)
            _show(frame)
            key = _wait_key()
            if key == 27 or key == ord('q'):    # Esc / Q = cancel
                _feedback_mode = False
            elif key == 13:                     # Enter = save & close
                try:
                    corrections = {}
                    for fk in _FEEDBACK_KEYS:
                        txt = _feedback_vals.get(fk, '').strip()
                        if txt and final_m.get(fk) is not None:
                            try:
                                actual_v = float(txt)
                                if 10.0 < actual_v < 200.0:
                                    corrections[fk] = (final_m[fk], actual_v)
                            except ValueError:
                                pass
                    if corrections and _have_db and _last_measurement_id is not None:
                        save_feedback(
                            _last_measurement_id,
                            _current_customer_id or 0,
                            corrections,
                            tailor_note='')
                        show_toast(
                            f'Feedback saved — {len(corrections)} fields updated',
                            (80, 205, 60), 3.0)
                        print(f'[Feedback] saved {len(corrections)} corrections '
                              f'for measurement {_last_measurement_id}')
                    elif not corrections:
                        show_toast('No changes entered', (82, 78, 72), 2.0)
                    else:
                        show_toast('No measurement ID — rescan first', RED, 3.0)
                except Exception as _fbe:
                    print(f'[Feedback] save error: {_fbe}')
                _feedback_mode = False
            elif key == 9 or key == 84:         # Tab or Down = next field
                _feedback_field_idx = (_feedback_field_idx + 1) % len(_FEEDBACK_KEYS)
            elif key == 353 or key == ord('k'): # Shift-Tab or K = prev field
                _feedback_field_idx = (_feedback_field_idx - 1) % len(_FEEDBACK_KEYS)
            elif key in (8, 127):               # Backspace
                fk = _FEEDBACK_KEYS[_feedback_field_idx]
                _feedback_vals[fk] = _feedback_vals.get(fk, '')[:-1]
            elif key == ord('.') or (48 <= key <= 57):
                fk = _FEEDBACK_KEYS[_feedback_field_idx]
                cur = _feedback_vals.get(fk, '')
                if key == ord('.') and '.' not in cur and len(cur) < 6:
                    _feedback_vals[fk] = cur + '.'
                elif key != ord('.') and len(cur) < 6:
                    _feedback_vals[fk] = cur + chr(key)
            continue

        # ── Order creation mode (after saving a scan) ──────────────────────────
        if _order_mode:
            frame = draw_order_screen(
                frame.copy(),
                customer_name=_current_customer_name or 'Customer',
                measurements=final_m,
                garment_type=_garment_type,
                style_notes=_order_vals.get('notes', ''),
                fabric=_order_vals.get('fabric', ''),
                colour=_order_vals.get('colour', ''),
                price=_order_vals.get('price', ''),
                delivery=_order_vals.get('delivery', ''),
                active_field=_order_field,
                assigned_to=_order_vals.get('assigned_to', ''),
                staff_list=_staff_names,
            )
            _show(frame)
            key = _wait_key()
            if key == 9:   # Tab = next field
                idx_f = _order_field_seq.index(_order_field)
                _order_field = _order_field_seq[(idx_f + 1) % len(_order_field_seq)]
            elif key == 13:  # Enter = confirm order
                if _have_db and _current_customer_id is not None:
                    try:
                        _pr = float(_order_vals.get('price', '0') or '0')
                    except ValueError:
                        _pr = 0.0
                    create_order(
                        customer_id=_current_customer_id,
                        garment_type=_garment_type,
                        style_notes=_order_vals.get('notes', ''),
                        fabric=_order_vals.get('fabric', ''),
                        colour=_order_vals.get('colour', ''),
                        price=_pr,
                        delivery_date=_order_vals.get('delivery', ''),
                        measurement_id=_last_measurement_id,
                        assigned_to=_order_vals.get('assigned_to', ''),
                    )
                    show_toast('Order created!', (40, 182, 28))
                _order_mode = False
                _order_vals = {k: '' for k in _order_field_seq}
                _order_field = 'garment'
            elif key == 27:
                _order_mode = False
            elif key in (8, 127):
                _order_vals[_order_field] = _order_vals.get(_order_field, '')[:-1]
            elif 32 <= key <= 126:
                cur = _order_vals.get(_order_field, '')
                _order_vals[_order_field] = cur + chr(key)
            continue

        # ── Poll folder-picker subprocess ──────────────────────────────────────
        if _picker_proc is not None:
            ret = _picker_proc.poll()
            if ret is not None:
                chosen = _picker_proc.stdout.read().strip()
                _picker_proc = None
                if chosen:
                    os.makedirs(chosen, exist_ok=True)
                    save_folder = chosen
                    cfg['save_folder'] = chosen
                    _save_config(cfg)
                    show_toast(f'Folder: {os.path.basename(save_folder)}',
                               (28, 145, 215))

        # ── Poll SMPL worker subprocess ────────────────────────────────────────
        if _smpl_proc is not None and _smpl_running:
            proc = _smpl_proc[0]   # None = thread not ready; 'error' = failed
            if proc == 'error':
                _smpl_proc = None;  _smpl_running = False
                show_toast('3D fit failed -- worker could not start',
                           (80, 80, 180), 4.0)
                advance_start = time.time()
            elif proc is not None:
                ret = proc.poll()
                if ret is not None:
                    raw_out    = proc.stdout.read().strip()
                    stderr_out = proc.stderr.read().strip()
                    if stderr_out:
                        print(f'[BodyWorker stderr]\n{stderr_out}')
                    _smpl_proc = None;  _smpl_running = False

                    if raw_out and final_m is not None:
                        try:
                            worker_data = json.loads(raw_out)
                            # Hard bounds per measurement (cm)
                            _BOUNDS = {
                                'shoulder_width': (28, 62),
                                'chest':          (66, 155),
                                'waist':          (54, 145),
                                'hip':            (68, 155),
                                'back_length':    (34, 58),
                                'shirt_length':   (40, 76),
                                'sleeve_length':  (44, 82),
                                'neck':           (27, 52),
                            }
                            applied = {}
                            skipped = {}
                            for k, val in worker_data.items():
                                if not isinstance(val, (int, float)):
                                    continue
                                lo, hi = _BOUNDS.get(k, (20, 250))
                                geo = final_m.get(k)
                                # Reject if outside hard bounds
                                if not (lo <= val <= hi):
                                    skipped[k] = f'{val:.1f} (out of {lo}-{hi})'
                                    continue
                                # Reject if more than 15% away from geometric estimate
                                if geo and abs(val - geo) / geo > 0.15:
                                    skipped[k] = f'{val:.1f} (>25% from geo {geo})'
                                    continue
                                print(f'[BodyNet] {k}: {val:.1f}cm '
                                      f'(was: {geo}cm)')
                                final_m[k] = round(val, 1)
                                applied[k] = round(val, 1)
                            if skipped:
                                print(f'[BodyNet] Skipped {len(skipped)} outliers: '
                                      + ', '.join(f'{k}={v}' for k, v in skipped.items()))
                            final_m.pop('_estimated', None)
                            print(f'[BodyNet] Applied {len(applied)} refined measurements')
                            collector.save_measurements(final_m)
                            show_toast('Neural model refined measurements',
                                       (30, 200, 248), 4.0)
                            advance_start = time.time()
                        except json.JSONDecodeError:
                            print(f'[BodyNet] Bad JSON: {raw_out[:120]}')
                            advance_start = time.time()
                    else:
                        advance_start = time.time()

        # ── Manual height override (T key during scanning) ────────────────────
        if _manual_mode:
            frame = draw_height_input_overlay(frame, _manual_input)
            _show(frame)
            key = _wait_key()
            if key == 13:
                val = _parse_height(_manual_input)
                if val:
                    height_cm, _height_locked = val, True
                    _height_ests.clear()
                    show_toast(f'Height set: {val} cm', (40, 182, 28))
                    print(f'[Height] manual override -> {val} cm')
                _manual_mode = False;  _manual_input = ''
            elif key == 27:
                _manual_mode = False;  _manual_input = ''
            elif key in (8, 127):
                _manual_input = _manual_input[:-1]
            elif 48 <= key <= 57 and len(_manual_input) < 3:
                _manual_input += chr(key)
            continue

        # ── Results display ────────────────────────────────────────────────────
        if showing_results and final_m:
            # reveal animation timer (no auto-advance; user exits manually)
            reveal_elapsed = (now - advance_start) if advance_start else 9999.0
            # Posture quality: derive from MRS factors if available
            _mrs_overall = None
            if world_views.get('FRONT'):
                try:
                    _mrs = _mrs_factors(world_views['FRONT'], 'FRONT')
                    _mrs_overall = min(1.0, _mrs.get('overall', 0.0) / 0.15)
                except Exception:
                    pass
            frame, _results_card_rects = draw_results_overlay(
                                         frame, final_m,
                                         mesh_img=_smpl_mesh_img,
                                         refining=_smpl_running,
                                         reveal_elapsed=reveal_elapsed,
                                         uncertainties=_uncertainties,
                                         garment_type=_garment_type,
                                         posture_score=_mrs_overall,
                                         prev_measurements=_get_prev_measurements(),
                                         active_tab=_results_tab,
                                         customer_name=_current_customer_name,
                                         profile_photo=_profile_photo,
                                         country=_size_country,
                                         edit_field=_edit_field,
                                         history_records=_hist_measurements or None,
                                         tape_status=_tape_status or None)
            # ── Style tip based on body type ──────────────────────────────
            if final_m and _results_tab == 0:
                _body_type_res = classify_body_type(final_m)
                draw_style_tip_overlay(frame, _body_type_res)

            # ── Measurement edit overlay ───────────────────────────────────
            if _edit_field is not None and final_m is not None:
                _edit_lbl = next((lbl for k, lbl in [
                    ('shoulder_width','Shoulder Width'),('chest','Chest Circ.'),
                    ('waist','Waist Circ.'),('back_length','Back Length'),
                    ('shirt_length','Shirt Length'),('sleeve_length','Sleeve Length'),
                    ('neck','Neck Circ.'),('hip','Hip Circ.'),
                ] if k == _edit_field), _edit_field)
                frame = draw_measurement_edit_overlay(
                    frame, _edit_field, _edit_lbl,
                    final_m.get(_edit_field), _edit_input_txt)

            if naming_mode:
                frame = draw_name_input_overlay(frame, name_input)

        # ── Photo capture state machine ────────────────────────────────────────
        else:
            # Apply background removal for cleaner guide display
            if photo_state in ('GUIDE', 'COUNTDOWN') and _segmentor is not None:
                frame = _apply_bg_removal(frame)

            # Only run MediaPipe when landmarks are actually needed
            if photo_state in ('GUIDE', 'COUNTDOWN'):
                results    = detector.process(frame)
                screen_lms = detector.get_screen_landmarks(
                    results, camera.width, camera.height)
                # Clean copy (no overlays) for the HMR body model
                _raw_frame = frame.copy() if photo_state == 'COUNTDOWN' else None
                frame = detector.draw_landmarks(frame, results)
            else:
                results    = None
                screen_lms = None

            if photo_state == 'GUIDE':
                view_name = PHOTO_VIEWS[view_idx][0]

                # Voice guidance: announce each new view once
                _voice_key = f'guide_{view_name}'
                if _last_voice_state != _voice_key:
                    _last_voice_state = _voice_key
                    _guide_msgs = {
                        'FRONT': 'Please face the camera with your arms away from your body. '
                                 'Loose clothes reduce accuracy.',
                        'LEFT':  'Now turn to your left. Show your left side to the camera.',
                        'RIGHT': 'Keep turning right. Show your right side to the camera.',
                        'BACK':  'Now turn your back to the camera. Stand straight with arms slightly away.',
                    }
                    msg = _guide_msgs.get(view_name)
                    if msg:
                        _speak(msg)

                score = (_pose_score(screen_lms, detector.LM,
                                     view_name,
                                     camera.width, camera.height)
                         if screen_lms else 0.0)

                # ── Arms must be away from the body (front / back) ─────────
                if view_name in ('FRONT', 'BACK') and screen_lms:
                    _aa = _arms_away(screen_lms, detector.LM)
                    if _aa is not None and _aa < ARMS_AWAY_MIN and now > _arms_hint_until:
                        show_toast('Lift your arms a little away from your body',
                                   (28, 145, 215), 2.5)
                        _arms_hint_until = now + 4.0
                        if view_name not in _arms_spoken:
                            _arms_spoken.add(view_name)
                            _speak('Please hold your arms a little away from your body.')

                # ── Accumulate frames + Kalman filter update ──────────────
                if results:
                    _wlms = detector.get_world_landmarks(results)
                    if _wlms is not None:
                        _wlms_d = _world_lms_to_dicts(_wlms)
                        if view_name in _kalmans:
                            _kalmans[view_name].update(_wlms_d)
                            # MRS bonus improves score when body geometry is good
                            mrs = _mrs_factors(_kalmans[view_name].get_state(), view_name)
                            score = min(1.0, score + mrs['overall'])
                        if score >= 0.65:
                            _lm_buffers[view_name].append(_wlms_d)
                            if len(_lm_buffers[view_name]) > _LM_BUF_MAX:
                                _lm_buffers[view_name].pop(0)

                # ── Auto-detect height from FRONT/BACK view landmarks ─────
                if (view_name in ('FRONT', 'BACK') and screen_lms and not _manual_mode
                        and not _height_locked):
                    est = _estimate_height_cm(screen_lms, detector.LM)
                    if est is not None:
                        _height_ests.append(est)
                        if len(_height_ests) > 90:
                            _height_ests.pop(0)
                        # Lock in height once we have 20+ stable readings
                        if len(_height_ests) >= 20:
                            recent   = _height_ests[-20:]
                            variance = max(recent) - min(recent)
                            if variance < 10:
                                new_h = round(sum(recent) / len(recent), 1)
                                if height_cm != new_h:
                                    height_cm = new_h
                                    print(f'[Height] Auto-detected: {height_cm} cm '
                                          f'(variance {variance:.1f}cm)')

                # ── Scan quality checks ───────────────────────────────────
                _mean_brightness = float(np.mean(frame)) if frame is not None else 128
                _lighting_ok  = _mean_brightness > 55
                # Simple distance heuristic: if pose detected, shoulder span
                # should be >12% of frame width for good distance
                _dist_ok = True
                if screen_lms:
                    try:
                        _lsh = screen_lms[detector.LM.LEFT_SHOULDER]
                        _rsh = screen_lms[detector.LM.RIGHT_SHOULDER]
                        _span = abs(_lsh.x - _rsh.x)
                        _dist_ok = 0.10 < _span < 0.70
                    except Exception:
                        pass

                frame = draw_photo_guide(
                    frame,
                    view_name,
                    PHOTO_VIEWS[view_idx][1],
                    score,
                    view_idx=view_idx,
                    height_est=height_cm,
                    total_views=len(PHOTO_VIEWS),
                    lighting_ok=_lighting_ok,
                    distance_ok=_dist_ok,
                    guide_elapsed=now - _guide_start,
                    guide_timeout=GUIDE_TIMEOUT_SECS)

                # ── Pose guidance arrows (draw on top of guide overlay) ────
                if screen_lms and score < 0.75:
                    _lms_dict = {i: lm for i, lm in enumerate(screen_lms)}
                    draw_pose_guidance_arrows(
                        frame, _lms_dict, view_name,
                        camera.width, camera.height)

                # ── Guide timeout: auto-force-capture if stuck too long ──────
                _guide_elapsed = now - _guide_start
                _is_back_view  = (view_name == 'BACK')

                # Back view: lower threshold (MediaPipe inherently gives lower
                # visibility scores to back-facing landmarks)
                _score_thresh = 0.72 if _is_back_view else 0.85

                if score >= _score_thresh:
                    match_frames += _tick
                    if match_frames >= MATCH_FRAMES:
                        if height_cm is None and not _is_back_view:
                            # Height not detected yet — show toast and keep waiting
                            match_frames = 0
                            show_toast('Height not detected -- press T to enter manually',
                                       (28, 145, 215), 3.0)
                            print('[Height] Pose good but height not detected yet -- waiting')
                        else:
                            print(f'[View {view_idx+1}] Pose matched — '
                                  f'height={height_cm}cm — starting countdown')
                            photo_state  = 'COUNTDOWN'
                            countdown_f  = int(COUNTDOWN_SEC * 30)
                            match_frames = 0
                            _speak('Hold still.')
                else:
                    match_frames = max(0, match_frames - 2)

                # ── 15-second guide timeout → force-start countdown ──────────
                if (_guide_elapsed >= GUIDE_TIMEOUT_SECS
                        and photo_state == 'GUIDE'
                        and score > 0.20):   # at least a person is visible
                    print(f'[View {view_idx+1}] Guide timeout after '
                          f'{_guide_elapsed:.0f}s — force-capturing {view_name}')
                    show_toast(f'Auto-capturing {view_name} view — hold still!',
                               (195, 155, 20), 2.5)
                    photo_state  = 'COUNTDOWN'
                    countdown_f  = int(COUNTDOWN_SEC * 30)
                    match_frames = 0
                    _speak('Auto capture. Hold still.')

            elif photo_state == 'COUNTDOWN':
                score = (_pose_score(screen_lms, detector.LM,
                                     PHOTO_VIEWS[view_idx][0],
                                     camera.width, camera.height)
                         if screen_lms else 0.0)

                if score < 0.60:
                    print('[View] Pose lost — resetting')
                    photo_state  = 'GUIDE'
                    match_frames = 0
                else:
                    # Continue accumulating + Kalman update during countdown
                    _cur_vname = PHOTO_VIEWS[view_idx][0]
                    _wlms = detector.get_world_landmarks(results)
                    if _wlms is not None:
                        _wlms_d2 = _world_lms_to_dicts(_wlms)
                        if _cur_vname in _kalmans:
                            _kalmans[_cur_vname].update(_wlms_d2)
                        _lm_buffers[_cur_vname].append(_wlms_d2)
                        if len(_lm_buffers[_cur_vname]) > _LM_BUF_MAX:
                            _lm_buffers[_cur_vname].pop(0)

                    # Snapshot for HMR (person is holding the pose now)
                    if (_hmr is not None and _raw_frame is not None and screen_lms
                            and (countdown_f // HMR_SAMPLE_EVERY) != ((countdown_f - _tick) // HMR_SAMPLE_EVERY)):
                        _hmr_frames.append((
                            _cur_vname, _raw_frame,
                            _screen_lms_to_list(screen_lms, camera.width, camera.height)))

                    countdown_f -= _tick
                    cnt_val = max(1, math.ceil(countdown_f / 30))
                    frame = draw_photo_guide(
                        frame,
                        PHOTO_VIEWS[view_idx][0],
                        PHOTO_VIEWS[view_idx][1],
                        score,
                        countdown_val=cnt_val,
                        view_idx=view_idx,
                        height_est=height_cm,
                        total_views=len(PHOTO_VIEWS))

                    if countdown_f <= 0:
                        # ── CAPTURE — prefer Kalman state, fall back to average ──
                        view_name = PHOTO_VIEWS[view_idx][0]
                        kps = (_screen_lms_to_list(screen_lms,
                                                   camera.width, camera.height)
                               if screen_lms else None)

                        # Kalman smoothed state is the most accurate estimate
                        kalman_state = (_kalmans[view_name].get_state()
                                        if view_name in _kalmans else None)
                        if kalman_state is not None:
                            best_lms = kalman_state
                            lms_src  = 'kalman'
                        else:
                            buf      = _lm_buffers.get(view_name, [])
                            use      = buf[-_LM_BUF_USE:] if len(buf) >= _LM_BUF_USE else buf
                            best_lms = _avg_world_lms(use)
                            lms_src  = f'avg({len(use)} frames)' if best_lms else 'none'

                        # ── Auto-retake if capture quality is too poor ─────────
                        # Use the current pose score at capture time
                        _cap_score = (_pose_score(screen_lms, detector.LM,
                                                   view_name,
                                                   camera.width, camera.height)
                                      if screen_lms else 0.0)
                        _back_retake_thresh = RETAKE_THRESHOLD * 0.75  # back view is harder
                        _thresh = _back_retake_thresh if view_name == 'BACK' else RETAKE_THRESHOLD

                        if _cap_score < _thresh and _retake_count < 1:
                            # Bad capture — discard, flash red, go back to GUIDE
                            _retake_count += 1
                            _guide_start   = time.time()  # reset timeout
                            match_frames   = 0
                            _hmr_frames    = [f for f in _hmr_frames if f[0] != view_name]
                            photo_state    = 'GUIDE'
                            show_toast(f'Poor pose — retaking {view_name} view automatically',
                                       (58, 52, 222), 3.0)
                            print(f'[AutoRetake] {view_name} score={_cap_score:.2f} < {_thresh:.2f} '
                                  f'— discarding and retrying')
                            _speak('Poor pose. Please try again.')
                        else:
                            if kps:
                                captured_views.append({
                                    'keypoints_2d': kps,
                                    'rotation_deg': PHOTO_VIEWS[view_idx][2],
                                    'img_w':        camera.width,
                                    'img_h':        camera.height,
                                })
                            if best_lms is not None:
                                world_views[view_name] = best_lms
                            print(f'[View {view_idx+1}/{len(PHOTO_VIEWS)}] '
                                  f'{view_name} captured  '
                                  f'score={_cap_score:.2f}  '
                                  f'kps={len(kps) if kps else 0}  '
                                  f'world_lms={lms_src}')
                            collector.save_photo_view(view_idx, kps)
                            # Save FRONT view as customer profile photo
                            if view_name == 'FRONT' and _have_db and _current_customer_id:
                                try:
                                    _profiles_dir = os.path.join(
                                        os.path.dirname(__file__), 'profiles')
                                    os.makedirs(_profiles_dir, exist_ok=True)
                                    _ph_path = os.path.join(
                                        _profiles_dir, f'{_current_customer_id}.jpg')
                                    cv2.imwrite(_ph_path, frame)
                                    save_customer_photo(_current_customer_id, _ph_path)
                                    _profile_photo = frame.copy()
                                    _photo_cache[_current_customer_id] = _profile_photo
                                except Exception as _pe:
                                    print(f'[Profile] Failed to save photo: {_pe}')
                            photo_state = 'FLASH'
                            flash_f     = FLASH_FRAMES

            elif photo_state == 'FLASH':
                # Brief white flash
                alpha = flash_f / FLASH_FRAMES
                bright = np.ones_like(frame) * 255
                cv2.addWeighted(bright, alpha, frame, 1 - alpha, 0, frame)
                flash_f -= _tick
                if flash_f <= 0:
                    view_idx += 1
                    if view_idx >= len(PHOTO_VIEWS):
                        # All views done → HMR 3D body first, geometric as fallback
                        _hmr_used = False
                        _tape_status = {}
                        final_m   = None
                        geo_feat  = {}
                        if _hmr is not None and _hmr.ready and _hmr_frames and height_cm:
                            _show(draw_processing_screen(frame.copy()))
                            cv2.waitKey(1)
                            _hres = (_upload_result if _upload_mode else
                                     _hmr.measure(height_cm, _hmr_frames,
                                                  weight_kg=_weight_kg, gender=_gender,
                                                  age=_age_years))
                            if _hres:
                                _hmr_used = True
                                final_m = {k: v for k, v in _hres['measurements'].items()
                                           if k != 'height'}
                                _uncertainties = dict(_hres.get('sigma') or {})
                                try:
                                    from model.calibration import tape_check
                                    _tape_status = tape_check(bool(_hres.get('weight_used')))
                                    _need = [k for k, v in _tape_status.items() if v['status'] == 'tape']
                                    print('[Measure] tape these: ' + (', '.join(_need) or 'none'))
                                except Exception as _te:
                                    _tape_status = {}
                                    print(f'[Measure] tape check skipped: {_te}')
                                try:
                                    _smpl_mesh_img = render_smpl_mesh(
                                        np.asarray(_hres['vertices']),
                                        np.asarray(_hres['faces']), 360, 540)
                                except Exception as _me:
                                    print(f'[HMR] mesh render skipped: {_me}')
                                print(f"[Measure] {_hres['n_frames']} frames "
                                      f"({', '.join(sorted(set(_hres['views'])))})  "
                                      f"weight={'yes' if _hres.get('weight_used') else 'NO'}  "
                                      f"arms_away={_hres.get('arms_away')}  "
                                      f"silhouettes={'used' if _hres.get('image_used') else 'not used'}")
                                _pmsg = _photo_messages(_hres.get('photo_checks') or [],
                                                        _upload_mode)
                                if _pmsg:
                                    show_toast(_pmsg, RED, 8.0)
                                    print('[Upload] ' + _pmsg)
                                elif not _hres.get('weight_used'):
                                    show_toast('Tip: enter weight for much better accuracy',
                                               (28, 145, 215), 5.0)
                                elif not _hres.get('image_used'):
                                    show_toast('Arms were close to body -- '
                                               'measured from height & weight only',
                                               (28, 145, 215), 5.0)
                        elif _hmr is not None and not _hmr.ready:
                            print(f'[HMR] not used: {_hmr.failed or "still loading"}')

                        if _hmr_used:
                            pass
                        elif _body_measurer is not None and world_views:
                            try:
                                geo_m, geo_feat = _body_measurer.compute_with_geo_features(
                                    height_cm, world_views)
                                final_m = geo_m if geo_m else compute_from_height(height_cm)
                            except Exception as _e:
                                print(f'[GeoMeasure] fallback: {_e}')
                                final_m  = compute_from_height(height_cm)
                                geo_feat = {}
                        else:
                            final_m  = compute_from_height(height_cm)
                            geo_feat = {}

                        # ── Inject age + has_back into geo_feat ───────────
                        if geo_feat:
                            geo_feat['age']      = float(_age_years or 30)
                            # has_back is already set by compute_with_geo_features
                            # but ensure it's present even if body_measurer is old
                            if 'has_back' not in geo_feat:
                                geo_feat['has_back'] = float('BACK' in world_views)

                        # ── Bayesian Ridge refinement + uncertainty ────────
                        if _have_new_modules and _bayesian_m is not None and geo_feat:
                            try:
                                bay_pred = _bayesian_m.predict(geo_feat)
                                if bay_pred:
                                    _uncertainties = {k: v['sigma'] for k, v in bay_pred.items()}
                                    # Blend: 60% geometric + 40% Bayesian
                                    for k, v in bay_pred.items():
                                        if k in final_m and final_m[k] is not None:
                                            final_m[k] = round(
                                                final_m[k] * 0.60 + v['value'] * 0.40, 1)
                                    print(f'[Bayesian] uncertainties: '
                                          + '  '.join(f'{k}±{v["sigma"]:.1f}'
                                                       for k, v in bay_pred.items()))
                            except Exception as _be:
                                print(f'[Bayesian] skipped: {_be}')

                        # ── Tailor feedback corrections ────────────────────
                        # Apply mean systematic error learned from past feedback
                        if _have_db and final_m:
                            try:
                                _fb_corrections = get_feedback_corrections(min_samples=3)
                                if _fb_corrections:
                                    for _fc_k, _fc_v in _fb_corrections.items():
                                        if _fc_k in final_m and final_m[_fc_k] is not None:
                                            final_m[_fc_k] = round(final_m[_fc_k] + _fc_v, 1)
                                    print(f'[Feedback] applied corrections: '
                                          + '  '.join(f'{k}:{v:+.1f}'
                                                       for k, v in _fb_corrections.items()))
                            except Exception as _fbe:
                                print(f'[Feedback] corrections skipped: {_fbe}')

                        # ── Anthropometric constraint solver ───────────────
                        # (HMR girths come from one consistent 3D body — the
                        #  2D-landmark correction steps below don't apply to them)
                        if _have_new_modules and final_m and height_cm and not _hmr_used:
                            try:
                                final_m = _constraint_solve(
                                    final_m, height_cm, _uncertainties or None)
                            except Exception as _ce:
                                print(f'[Constraint] skipped: {_ce}')

                        # ── Plausibility check 1: height-ratio bounds ────────
                        if final_m and height_cm:
                            _RATIO_BOUNDS = {
                                'shoulder_width': (0.20, 0.32),
                                'chest':          (0.44, 0.70),
                                'waist':          (0.36, 0.62),
                                'back_length':    (0.22, 0.36),
                                'shirt_length':   (0.36, 0.54),
                                'sleeve_length':  (0.28, 0.46),
                                'neck':           (0.18, 0.30),
                            }
                            for _k, (_lo, _hi) in _RATIO_BOUNDS.items():
                                _v = final_m.get(_k)
                                if _v is not None:
                                    _r = _v / height_cm
                                    if not (_lo <= _r <= _hi):
                                        _fallback = round(height_cm * (_lo + _hi) / 2, 1)
                                        print(f'[Plausibility] {_k}={_v:.1f} ratio={_r:.3f} '
                                              f'outside [{_lo},{_hi}] → {_fallback}')
                                        final_m[_k] = _fallback

                        # ── Plausibility check 2: anatomical relationships ────
                        if final_m:
                            chest = final_m.get('chest')
                            waist = final_m.get('waist')
                            hip   = final_m.get('hip')
                            sh_w  = final_m.get('shoulder_width')
                            # waist should be less than chest
                            if chest and waist and waist >= chest:
                                old_w = waist
                                final_m['waist'] = round(chest * 0.84, 1)
                                print(f'[Plausibility] waist({old_w}) >= chest({chest}) → {final_m["waist"]}')
                            # hip should be >= waist
                            if hip and waist and hip < waist:
                                final_m['hip'] = round(waist * 1.06, 1)
                                print(f'[Plausibility] hip < waist → corrected to {final_m["hip"]}')
                            # shoulder_width should be < chest/2 + margin
                            if chest and sh_w and sh_w > chest * 0.55:
                                final_m['shoulder_width'] = round(chest * 0.47, 1)
                                print(f'[Plausibility] shoulder_width too wide → corrected')

                        # ── Confidence gating: warn if high uncertainty ───────
                        if _uncertainties:
                            _high_sigma = {k: v for k, v in _uncertainties.items() if v > 2.5}
                            if _high_sigma:
                                worst = max(_high_sigma, key=_high_sigma.get)
                                sigma = _high_sigma[worst]
                                show_toast(
                                    f'Low confidence on {worst} (±{sigma:.1f}cm) — consider re-scan',
                                    (45, 130, 215), 5.0)
                                print(f'[Confidence] High uncertainty: '
                                      + ', '.join(f'{k}±{v:.1f}' for k, v in _high_sigma.items()))

                        # ── Elliptical cross-section scaling ──────────────
                        # Research: width measurement * pi/2 gives better circumference
                        # than simple width * 2.  Correction factors from 3DLOOK study:
                        # chest/waist multiply by 2.4, hip by 2.5 (vs naive 2.0).
                        # We blend 70% original + 30% elliptical correction.
                        if final_m and not _hmr_used:
                            _ELLIP = {'chest': 2.4, 'waist': 2.4, 'hip': 2.5}
                            for _ek, _ef in _ELLIP.items():
                                _orig = final_m.get(_ek)
                                if _orig is not None:
                                    # back-compute apparent half-width
                                    _hw = _orig / 2.0
                                    _ellip_circ = _hw * _ef
                                    final_m[_ek] = round(_orig * 0.70 + _ellip_circ * 0.30, 1)

                        # ── Posture lean correction (from side views) ────────
                        # If spine appears tilted in side view, chest/back are
                        # projected smaller than reality — apply a correction factor.
                        if (final_m and not _hmr_used
                                and (world_views.get('LEFT') or world_views.get('RIGHT'))):
                            try:
                                side_lms = world_views.get('LEFT') or world_views.get('RIGHT')
                                l_sh3 = side_lms[11];  l_hi3 = side_lms[23]
                                # Spine lean angle from vertical (in degrees)
                                dx = l_sh3['x'] - l_hi3['x']
                                dy = abs(l_sh3['y'] - l_hi3['y']) + 1e-6
                                lean_angle = abs(math.degrees(math.atan(dx / dy)))
                                # >5 degrees lean → scale circumferences up by cos(lean)
                                if lean_angle > 5.0:
                                    cos_factor = 1.0 / max(0.85, math.cos(math.radians(lean_angle)))
                                    for _pk in ('chest', 'waist', 'hip'):
                                        if final_m.get(_pk):
                                            final_m[_pk] = round(final_m[_pk] * cos_factor, 1)
                                    print(f'[Posture] lean={lean_angle:.1f}° correction factor={cos_factor:.3f}')
                            except Exception:
                                pass

                        # ── Gender-aware proportional adjustment ─────────────
                        if final_m and _gender != 'unspecified' and not _hmr_used:
                            # Female: narrower shoulders (~5%), wider hip (~4%)
                            # Male: wider shoulders (~3%), narrower hip (~3%)
                            if _gender == 'female':
                                if final_m.get('shoulder_width'):
                                    final_m['shoulder_width'] = round(
                                        final_m['shoulder_width'] * 0.96, 1)
                                if final_m.get('hip'):
                                    final_m['hip'] = round(final_m['hip'] * 1.04, 1)
                            elif _gender == 'male':
                                if final_m.get('shoulder_width'):
                                    final_m['shoulder_width'] = round(
                                        final_m['shoulder_width'] * 1.03, 1)
                                if final_m.get('hip'):
                                    final_m['hip'] = round(final_m['hip'] * 0.97, 1)
                            print(f'[Gender] {_gender} proportional correction applied')

                        # ── Garment thickness correction ───────────────────
                        # (skipped for the new models: they estimate the body itself)
                        if final_m and not _hmr_used:
                            final_m = _apply_garment_correction(final_m, _garment_type)
                            if _garment_type != 'tight_fit':
                                print(f'[Garment] applied {_garment_type} offset')

                        # ── Auto-save to SQLite DB ──────────────────────────
                        if _have_db and final_m:
                            try:
                                if _current_customer_id is None:
                                    _current_customer_id = add_customer(
                                        _current_customer_name or 'Walk-in Customer',
                                        age=_age_years,
                                        weight_kg=_weight_kg,
                                        height_cm=height_cm,
                                        gender=_gender)
                                _last_measurement_id = save_measurement(
                                    _current_customer_id, final_m,
                                    height_cm=height_cm,
                                    garment_type=_garment_type)
                                print(f'[DB] Saved measurement id={_last_measurement_id}')
                            except Exception as _dbe:
                                print(f'[DB] Save failed: {_dbe}')

                        # ── Keep the scan for training on this shop's customers ──
                        # (tailor's tape values come later via Feedback; then run
                        #  python scripts/retrain_from_shop.py)
                        if _hmr_used and _hres:
                            try:
                                from model.shop_data import save_scan
                                save_scan(_hres, _hmr_frames, measurement_id=_last_measurement_id,
                                          customer_id=_current_customer_id, height_cm=height_cm,
                                          weight_kg=_weight_kg, gender=_gender, age=_age_years,
                                          garment=_garment_type)
                            except Exception as _sse:
                                print(f'[ShopData] scan not saved: {_sse}')

                        showing_results = True
                        advance_start   = time.time()
                        if not _hmr_used:
                            _smpl_proc, _smpl_running = _launch_body_worker(
                                height_cm, world_views, show_toast)
                        _print_results(final_m)
                        _speak('Scan complete. Your measurements are ready.')
                        # Load scan history for history tab
                        if _have_db and _current_customer_id:
                            try:
                                _hist_measurements = get_measurements(
                                    _current_customer_id, limit=8)
                            except Exception:
                                _hist_measurements = []
                    else:
                        photo_state = 'TURN'
                        turn_f      = TURN_FRAMES

            elif photo_state == 'TURN':
                next_view = PHOTO_VIEWS[view_idx][0]
                frame = draw_turn_instruction(frame, next_view, view_idx, len(PHOTO_VIEWS))
                turn_f -= _tick
                if turn_f <= 0:
                    photo_state   = 'GUIDE'
                    match_frames  = 0
                    _guide_start  = time.time()   # reset timeout for each new view
                    _retake_count = 0
                    print(f'[View {view_idx+1}] Ready — waiting for {next_view} pose')

        # ── Toast overlay ──────────────────────────────────────────────────────
        if _toast_msg and now < _toast_until:
            frame = draw_toast(frame, _toast_msg, _toast_color)

        _show(frame)
        key = _wait_key()

        # ── Name input ────────────────────────────────────────────────────────
        if naming_mode:
            if key == 13:
                name = name_input.strip() or 'Customer'
                path = save_pdf(final_m, customer_name=name, save_dir=save_folder)
                save_record(final_m, name, save_folder, os.path.basename(path))
                print(f'[PDF saved] {path}')
                show_toast(f'Saved: {os.path.basename(path)}', (28, 145, 215))
                naming_mode = False;  name_input = ''
            elif key == 27:
                naming_mode = False;  name_input = ''
            elif key in (8, 127):
                name_input = name_input[:-1]
            elif 32 <= key <= 126:
                name_input += chr(key)
            continue

        if _upload_pending and not _wa_input_mode and not showing_results and app_screen == 'SCANNING':
            _upload_pending = False
            key = ord('o')                     # continue the "Upload photos" flow

        # ── Global hotkeys ─────────────────────────────────────────────────────

        # Shift+F = toggle fullscreen  (F11 loses its code after & 0xFF masking)
        if key == ord('F'):
            _fullscreen = not _fullscreen
            cv2.setWindowProperty(
                'TailorCam',
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if _fullscreen else cv2.WINDOW_NORMAL)
            show_toast('Fullscreen ON  (Shift+F to exit)' if _fullscreen
                       else 'Windowed mode', (195, 155, 20), 1.5)

        if key == ord('t') and not showing_results and not _manual_mode:
            _manual_mode = True;  _manual_input = ''   # T = type height manually
        if key == ord('g'):
            # G = cycle garment type for thickness correction
            idx = GARMENT_TYPES.index(_garment_type)
            _garment_type = GARMENT_TYPES[(idx + 1) % len(GARMENT_TYPES)]
            lbl = GARMENT_LABELS[GARMENT_TYPES.index(_garment_type)]
            show_toast(f'Garment: {lbl}', (28, 145, 215))
            print(f'[Garment] {_garment_type}')
        if key == ord('m') and not showing_results:
            # M = cycle gender (improves proportional accuracy)
            _gender_cycle = ['unspecified', 'male', 'female']
            _gender = _gender_cycle[(_gender_cycle.index(_gender) + 1) % len(_gender_cycle)]
            show_toast(f'Gender: {_gender.title()}', GOLD, 2.0)
            print(f'[Gender] set to {_gender}')
        if key in (ord('q'), ord('h')):
            _reset_customer()
            _hist_records = load_records(save_folder)
            app_screen = 'HOME'
        elif key == ord('r'):
            _reset_customer(same_person=True)
            print('\n[Restart]')
        elif key == ord('n') and showing_results:
            _reset_customer()
            print('\n[Next customer]')
        elif key == ord('p') and not showing_results:
            # P = reports screen
            _hist_records = load_records(save_folder)
            app_screen = 'REPORTS'
        elif key == ord('e') and not showing_results:
            # E = settings screen
            app_screen = 'SETTINGS'
        elif key == ord('i'):
            # I = toggle inches / cm
            _output_mod.USE_INCHES = not _output_mod.USE_INCHES
            unit_lbl = 'Inches' if _output_mod.USE_INCHES else 'Centimetres'
            show_toast(f'Units: {unit_lbl}', GOLD, 2.0)
            # Persist to config
            try:
                cfg_data = {}
                if os.path.exists(CONFIG_FILE):
                    with open(CONFIG_FILE) as _cf2:
                        cfg_data = json.load(_cf2)
                cfg_data['use_inches'] = _output_mod.USE_INCHES
                with open(CONFIG_FILE, 'w') as _cf3:
                    json.dump(cfg_data, _cf3, indent=2)
            except Exception:
                pass
        elif key == ord('l') and showing_results and final_m:
            # L = print order label
            try:
                lbl_path = print_order_label(
                    _current_customer_name or 'Customer',
                    _garment_type,
                    delivery_date=_order_vals.get('delivery', ''),
                    style_notes=_order_vals.get('notes', ''),
                    price=_order_vals.get('price', ''),
                    measurement_id=_last_measurement_id,
                    save_dir=save_folder)
                show_toast(f'Label: {os.path.basename(lbl_path)}', GOLD, 4.0)
                print(f'[Label] {lbl_path}')
                import subprocess as _sp2
                _sp2.Popen(['start', '', lbl_path], shell=True)
            except Exception as _le:
                show_toast(f'Label error: {_le}', RED, 3.0)
        elif key == ord('a') and not showing_results:
            # A = customer details (height / weight / age / gender)
            _wa_input_mode = True
            _wa_weight_txt = f'{_weight_kg:g}' if _weight_kg else ''
            _wa_age_txt    = str(_age_years) if _age_years else ''
            _wa_height_txt = f'{height_cm:g}' if (height_cm and _height_locked) else ''
            _wa_active = 'height'
        elif key == ord('o') and not showing_results:
            # O = measure from photos (front + side) instead of the live camera
            if not (height_cm and _height_locked):
                show_toast('Enter the real height first (A = details)', RED, 3.0)
            elif _hmr is None or not _hmr.ready:
                show_toast('3D body models are still loading -- try again shortly', RED, 3.0)
            else:
                _ups = []
                for _vn, _pth in zip(('FRONT', 'LEFT'), pick_photos()):
                    _img = load_photo(_pth)
                    if _img is not None:
                        _ups.append((_vn, _img, None))   # keypoints found by the worker
                if _ups:
                    print(f'[Upload] measuring {len(_ups)} photo(s)')
                    _show(draw_processing_screen(frame.copy()))
                    cv2.waitKey(1)
                    _res = _hmr.measure(height_cm, _ups, weight_kg=_weight_kg, gender=_gender,
                                        age=_age_years, timeout=90.0, auto_view=True)
                    if _res is None:
                        _err = getattr(_hmr, 'last_error', None) or {}
                        _pmsg = _photo_messages(_err.get('photo_checks') or [], True)
                        show_toast(_pmsg or 'Could not measure these photos -- use a clear, '
                                   'full-body photo with one person', RED, 8.0)
                        print(f'[Upload] failed: {_pmsg or _err.get("error")}')
                    else:
                        # finish exactly like a live scan, with the uploaded photos
                        _hmr_frames = _ups;  _upload_mode = True;  _upload_result = _res
                        world_views = {};  captured_views = []
                        view_idx    = len(PHOTO_VIEWS) - 1
                        photo_state = 'FLASH';  flash_f = 1
                        print(f"[Upload] views read from the photos: {', '.join(_res.get('views', []))}")
                else:
                    show_toast('No photo chosen', (82, 78, 72), 2.0)
        elif key == ord('s') and showing_results and final_m:
            # S = save PDF + record (and DB if available)
            naming_mode = True;  name_input = _current_customer_name or ''
        elif key == ord('w') and showing_results and final_m:
            # W = WhatsApp share — open wa.me URL with measurement summary
            lines = []
            for k, lbl in [('chest','Chest'),('waist','Waist'),
                            ('shoulder_width','Shoulder'),('sleeve_length','Sleeve'),
                            ('neck','Neck'),('shirt_length','Shirt Len')]:
                v = final_m.get(k)
                if v is not None:
                    lines.append(f'{lbl}: {v}cm')
            msg = 'TailorCam Measurements%0A' + '%0A'.join(lines)
            wa_url = f'https://wa.me/?text={msg}'
            try:
                import webbrowser
                webbrowser.open(wa_url)
                show_toast('Opening WhatsApp...', (40, 182, 28))
            except Exception as _we:
                show_toast(f'WhatsApp: {_we}', (80, 80, 200), 3.0)
        elif key in (ord('1'), ord('2'), ord('3')) and showing_results:
            # 1/2/3 = switch results tab
            _results_tab = key - ord('1')
        elif key == ord('o') and showing_results and final_m:
            # O = create order
            _order_mode  = True
            _order_field = 'garment'
            _order_vals  = {k: '' for k in _order_field_seq}
        elif key == ord('f') and showing_results:
            if _picker_proc is None:
                _picker_proc = launch_folder_picker(save_folder)
                show_toast('Opening folder picker...', (28, 145, 215))

        # ── T = Tailor feedback entry (after garment fitting) ──────────────────
        elif key == ord('t') and showing_results and _have_db:
            _feedback_mode = True
            _feedback_field_idx = 0
            _feedback_vals = {k: '' for k in _FEEDBACK_KEYS}

        # ── C = cycle country for size suggestions ─────────────────────────────
        elif key == ord('c') and showing_results:
            _ci = COUNTRY_KEYS.index(_size_country) if _size_country in COUNTRY_KEYS else 0
            _size_country = COUNTRY_KEYS[(_ci + 1) % len(COUNTRY_KEYS)]
            show_toast(f'Size standard: {_size_country}', TEAL, 2.0)

        # ── Measurement edit mode: click on a card → open edit overlay ─────────
        # Handle click on measurement card in tab 0
        if showing_results and _results_tab == 0 and _edit_field is None:
            click = _click_buf[0]
            if click:
                cx2, cy2 = click
                for _fk, (rx0, ry0, rx1, ry1) in _results_card_rects.items():
                    if isinstance((rx0, ry0, rx1, ry1), tuple) and rx0 <= cx2 <= rx1 and ry0 <= cy2 <= ry1:
                        _click_buf[0] = None
                        _edit_field = _fk
                        _edit_input_txt = ''
                        break

        # ── History tab: click Load button to restore a past scan ──────────────
        if showing_results and _results_tab == 2 and _edit_field is None:
            click = _click_buf[0]
            if click:
                cx2, cy2 = click
                hist_recs = _results_card_rects.get('_hist_records', [])
                for _hk, rect in _results_card_rects.items():
                    if not isinstance(rect, tuple) or len(rect) != 4:
                        continue
                    if not _hk.startswith('hist_'):
                        continue
                    rx0, ry0, rx1, ry1 = rect
                    if rx0 <= cx2 <= rx1 and ry0 <= cy2 <= ry1:
                        _click_buf[0] = None
                        try:
                            idx = int(_hk.split('_')[1])
                        except (IndexError, ValueError):
                            break
                        if hist_recs and idx < len(hist_recs):
                            rec = hist_recs[idx]
                            if final_m is None:
                                final_m = {}
                            for fk in ('chest', 'waist', 'hip', 'shoulder_width',
                                       'back_length', 'shirt_length', 'sleeve_length', 'neck'):
                                if rec.get(fk) is not None:
                                    final_m[fk] = rec[fk]
                            _results_tab = 0
                            scan_date = str(rec.get('scan_date', ''))[:10]
                            show_toast(f'Loaded scan: {scan_date}', TEAL, 3.0)
                        break

        # Edit overlay key handling
        if _edit_field is not None:
            if key == 27:                        # Esc = cancel
                _edit_field = None;  _edit_input_txt = ''
            elif key == 13:                      # Enter = confirm
                try:
                    _new_v = float(_edit_input_txt.strip())
                    if 10.0 < _new_v < 300.0 and final_m is not None:
                        final_m[_edit_field] = round(_new_v, 1)
                        show_toast(f'{_edit_field.replace("_"," ").title()} → {_new_v:.1f} cm',
                                   GREEN, 2.5)
                        print(f'[Edit] {_edit_field} manually set to {_new_v}')
                except ValueError:
                    show_toast('Invalid number', RED, 1.5)
                _edit_field = None;  _edit_input_txt = ''
            elif key in (8, 127):               # Backspace
                _edit_input_txt = _edit_input_txt[:-1]
            elif key == ord('.') and '.' not in _edit_input_txt:
                _edit_input_txt += '.'
            elif 48 <= key <= 57 and len(_edit_input_txt) < 6:
                _edit_input_txt += chr(key)

    camera.release()
    detector.close()
    if _hmr is not None:
        _hmr.close()
    if _segmentor is not None:
        try:
            _segmentor.close()
        except Exception:
            pass
    cv2.destroyAllWindows()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _draw_feedback_overlay(frame, final_m: dict, _feedback_keys_list: list,
                           field_idx: int, vals: dict):
    """Tailor feedback overlay (drawn by output.draw_feedback_overlay)."""
    from output import draw_feedback_overlay
    return draw_feedback_overlay(frame, final_m, _feedback_keys_list, field_idx, vals)


def _world_lms_to_dicts(world_lms):
    """Convert MediaPipe world landmarks to plain list-of-dicts for buffering."""
    if world_lms is None:
        return None
    out = []
    for lm in world_lms:
        if hasattr(lm, 'x'):
            out.append({'x': lm.x, 'y': lm.y, 'z': lm.z})
        elif isinstance(lm, dict):
            out.append({'x': lm.get('x', 0.0), 'y': lm.get('y', 0.0),
                        'z': lm.get('z', 0.0)})
        else:
            out.append({'x': float(lm[0]), 'y': float(lm[1]), 'z': float(lm[2])})
    return out


def _avg_world_lms(lm_list):
    """
    Median-smooth a list of world-landmark frames (better than average:
    outlier frames from bad posture snapshots are ignored).
    Each frame is a list of {'x','y','z'} dicts with 33 entries.
    """
    if not lm_list:
        return None
    n = len(lm_list)
    if n == 1:
        return lm_list[0]
    n_lms = len(lm_list[0])
    result = []
    for i in range(n_lms):
        xs = sorted(lm_list[j][i]['x'] for j in range(n))
        ys = sorted(lm_list[j][i]['y'] for j in range(n))
        zs = sorted(lm_list[j][i]['z'] for j in range(n))
        mid = n // 2
        if n % 2 == 1:
            result.append({'x': xs[mid], 'y': ys[mid], 'z': zs[mid]})
        else:
            result.append({
                'x': (xs[mid - 1] + xs[mid]) / 2,
                'y': (ys[mid - 1] + ys[mid]) / 2,
                'z': (zs[mid - 1] + zs[mid]) / 2,
            })
    return result


def _parse_height(text):
    try:
        val = float(text.strip())
        return val if 100 < val < 230 else None
    except ValueError:
        return None


def _print_results(m):
    estimated = m.get('_estimated', False)
    print('\n' + '=' * 48)
    print('  MEASUREMENTS' + ('  (estimated — SMPL processing…)' if estimated else ''))
    print('=' * 48)
    for key, label in [
        ('shoulder_width', 'Shoulder Width'),
        ('chest',          'Chest         '),
        ('waist',          'Waist         '),
        ('hip',            'Hip           '),
        ('back_length',    'Back Length   '),
        ('shirt_length',   'Shirt Length  '),
        ('sleeve_length',  'Sleeve Length '),
        ('neck',           'Neck          '),
    ]:
        val = m.get(key)
        if val is not None:
            print(f'  {label}: {val} cm')
    print('=' * 48 + '\n')


def _in_rect(x, y, rect):
    if rect is None:
        return False
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


if __name__ == '__main__':
    main()
