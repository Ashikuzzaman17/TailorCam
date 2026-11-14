"""
Long-running measurement worker (HMR 3D body + BodyM hybrid model).
Loads the networks once, then answers JSON-line requests on stdin with
JSON-line replies on stdout (logs go to stderr).

Request : {"id": 1, "height_cm": 175.0, "weight_kg": 70 | null,
           "gender": "male" | "female" | "unspecified",
           "frames": [{"path": "...jpg", "view": "FRONT|LEFT|RIGHT|BACK",
                       "keypoints": [[x, y, vis], ... 33] | null}, ...]}
Reply   : {"id": 1, "ok": true, "measurements": {...}, "sigma": {...},
           "source": {key: "hybrid" | "3d"}, "image_used": bool,
           "arms_away": float, "n_frames": 8, "vertices": [...], "faces": [...]}

How the models are combined (tested on BodyM + real photos):
  * chest / waist / hip / thigh / calf / bicep / wrist / ankle come from the
    BodyM hybrid model: a height+weight prior corrected by the front + side
    silhouettes. The silhouettes are only trusted when the arms are clearly
    away from the body (arms_away ≥ ARMS_AWAY_MIN); otherwise the prior is used.
  * shoulder width, neck, back length, shirt length and sleeve come from the
    HMR 3D body, whose shape is adjusted to the customer's real weight.
  * SHAPY (CVPR 2022) gives a second 3D body (chest / waist / hip), also
    adjusted to the real weight, when assets/shapy is installed.
  * CameraHMR (3DV 2025, trained on BEDLAM) gives a third 3D body with all
    measurements, also adjusted to the real weight, when assets/camerahmr is installed.
  * The final combiner (model/calibration.py, trained on real photos) blends
    these per measurement — separately for "weight known" and "weight unknown".

  python -m model.hmr.hmr_worker --image front.jpg side.jpg --height 175 --weight 70
"""

import os
import sys
import json
import time
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import cv2

from model.hmr.hmr_body import HMRBody
from model.hmr.body_points import measure, load_rings

ARMS_AWAY_MIN = 0.45   # wrist–hip gap / shoulder width; A-pose ≈ 0.5+, arms down ≈ 0.1–0.4

# BodyM name → TailorCam key, for measurements taken from the hybrid model
_HYBRID_KEYS = {'chest': 'chest', 'waist': 'waist', 'hip': 'hip', 'thigh': 'thigh',
                'calf': 'calf', 'bicep': 'bicep', 'wrist': 'wrist', 'ankle': 'ankle'}


def _log(*a):
    print('[Measure]', *a, file=sys.stderr, flush=True)


def arms_away(kps):
    """Horizontal wrist–hip gap relative to shoulder width (front/back view)."""
    if not kps or len(kps) < 25:
        return None
    sw = abs(kps[11][0] - kps[12][0])
    if sw < 1:
        return None
    return (abs(kps[15][0] - kps[23][0]) + abs(kps[16][0] - kps[24][0])) / 2 / sw


FRONT_RATIO = 0.48   # shoulder width / torso length: front photos 0.61–0.78, side 0.01–0.37 (real24)


def photo_check(kps, shape):
    """Quality of one photo: which way the person faces and whether the whole body is in it."""
    k = np.asarray(kps, float)
    h, w = shape[:2]
    torso = abs((k[11, 1] + k[12, 1]) / 2 - (k[23, 1] + k[24, 1]) / 2)
    ratio = abs(k[11, 0] - k[12, 0]) / torso if torso > 1 else 0.0
    ankle = max((27, 28), key=lambda i: k[i, 2])           # side photos hide one ankle
    head_ok = k[0, 2] > 0.5 and k[0, 1] > 0.01 * h
    feet_ok = k[ankle, 2] > 0.5 and k[ankle, 1] < 0.995 * h
    person_h = (k[ankle, 1] - min(k[0, 1], k[11, 1])) / h
    return {'view_guess': 'FRONT' if ratio >= FRONT_RATIO else 'SIDE', 'shoulder_ratio': round(float(ratio), 2),
            'full_body': bool(head_ok and feet_ok), 'person_height_frac': round(float(person_h), 2)}


def _load_hybrid():
    try:
        from model.bodym.hybrid import HybridPredictor
        return HybridPredictor()
    except Exception as exc:
        _log(f'hybrid model unavailable: {exc}')
        return None


def _load_shapy():
    try:
        from model.shapy_runner import ShapyRunner, is_available
        if not is_available():
            _log('SHAPY files not found (assets/shapy) - running without SHAPY')
            return None
        return ShapyRunner()
    except Exception as exc:
        _log(f'SHAPY unavailable: {exc}')
        return None


_ansur = []


def _ansur_prior():
    """ANSUR II height/weight prior (model/ansur_prior.py), loaded once."""
    if not _ansur:
        try:
            from model.ansur_prior import AnsurPrior
            _ansur.append(AnsurPrior())
        except Exception as exc:
            _log(f'ANSUR prior unavailable: {exc}')
            _ansur.append(None)
    return _ansur[0]


def _load_camerahmr():
    try:
        from model.camerahmr_runner import CameraHMRRunner, is_available
        if not is_available():
            _log('CameraHMR files not found (assets/camerahmr) - running without CameraHMR')
            return None
        return CameraHMRRunner()
    except Exception as exc:
        _log(f'CameraHMR unavailable: {exc}')
        return None


def _load_segmenter():
    try:
        from model.segmenter import PersonSegmenter
        return PersonSegmenter()
    except Exception as exc:
        _log(f'SAM 2 segmenter unavailable: {exc}')
        return None


def run_request(hb, rings, req, hybrid=None, shapy=None, segmenter=None, camerahmr=None):
    height = float(req['height_cm'])
    weight = req.get('weight_kg') or None
    gender = req.get('gender') or 'unspecified'
    age = req.get('age') or None

    frames = req.get('frames', [])
    if req.get('max_frames') and len(frames) > req['max_frames']:
        pick = np.linspace(0, len(frames) - 1, req['max_frames']).round().astype(int)
        frames = [frames[i] for i in pick]
    betas, fronts, sides, aa, shapy_items, cam_betas = [], [], [], [], [], []
    checks = []                     # per photo: found? which way facing? whole body?
    for i, fr in enumerate(frames):
        img = cv2.imread(fr['path'])
        if img is None:
            checks.append({'photo': i + 1, 'person': False, 'problem': 'cannot read the file'})
            continue
        kps, mask = hb.analyze(img)
        if fr.get('keypoints'):
            kps = [tuple(k) for k in fr['keypoints']]
        if kps is None:
            checks.append({'photo': i + 1, 'person': False, 'problem': 'no person found'})
            continue
        q = photo_check(kps, img.shape)
        if req.get('auto_view'):    # uploaded photos: trust what the photo shows, not the order
            fr = dict(fr, view='FRONT' if q['view_guess'] == 'FRONT' else 'LEFT')
        checks.append({'photo': i + 1, 'person': True, 'view_used': fr.get('view', 'FRONT'), **q})
        if segmenter is not None:                     # optional; off in the app (see segmenter.py)
            sharp = segmenter.mask(img, kps)          # sharper silhouette than MediaPipe's
            if sharp is not None:
                mask = sharp
        view = fr.get('view', 'FRONT')
        if mask is not None:
            if view in ('FRONT', 'BACK'):
                fronts.append(mask)
                a = arms_away(kps)
                if a is not None:
                    aa.append(a)
            else:
                sides.append(mask)
        r = hb.predict(img, kps)
        if r is not None:
            betas.append(r['betas'])
            if shapy is not None:
                from model.keypoints import mediapipe_to_body25
                shapy_items.append((f'f{len(shapy_items):03d}', img, mediapipe_to_body25(kps)))
            if camerahmr is not None:
                c = camerahmr.predict(img, kps)
                if c is not None:
                    cam_betas.append(c['betas'])
    if not betas:
        return {'ok': False, 'error': 'no person found in any frame', 'photo_checks': checks}

    # ── 3D body: HMR shape, corrected to the real weight ──────────────────────
    def body(b):
        if weight:
            b = hb.fit_weight(b, height, weight)
        V = hb.rest_mesh(b)
        return V, measure(V, hb.faces, height, rings, hb.arms_down_mesh(b))

    betas = np.array(betas)
    b_med = np.median(betas, axis=0)
    V, m = body(b_med)
    # body shown in the app: same shape, arms relaxed down (looks natural, measuring unchanged)
    V_show = hb.arms_down_mesh(hb.fit_weight(b_med, height, weight) if weight else b_med, 65.0)
    sigma = {}
    if len(betas) >= 3 and req.get('sigma', True):
        per = [body(b)[1] for b in betas]
        for k in m:
            if k != 'height':
                sigma[k] = round(float(np.std([p[k] for p in per])), 1)
    source = {k: '3d' for k in m if k != 'height'}

    # ── Hybrid (height/weight prior + silhouettes) for girths ─────────────────
    arms = float(np.median(aa)) if aa else 0.0
    image_used = False
    components = {f'hmr:{k}': v for k, v in m.items() if k != 'height'}
    if hybrid is not None:
        views = []
        if fronts and sides and arms >= ARMS_AWAY_MIN:
            n = max(len(fronts), len(sides))
            views = [(fronts[i % len(fronts)], sides[i % len(sides)]) for i in range(n)]
            image_used = True
        g = gender if gender in ('male', 'female') else 'unknown'
        hp = hybrid.predict(height, weight, g, views)
        prior = hybrid.predict(height, weight, g, [])
        components.update({f'hybrid:{k}': v[0] for k, v in hp.items()})
        components.update({f'prior:{k}': v[0] for k, v in prior.items()})
        for bodym_key, key in _HYBRID_KEYS.items():
            m[key], sigma[key] = hp[bodym_key]
            source[key] = 'hybrid'

    # ── ANSUR II prior: 6,068 tape-measured people (height/weight/gender/age) ──
    ap = _ansur_prior()
    if ap is not None:
        components.update({f'ansur:{k}': v for k, v in ap.predict(
            height, weight, gender if gender in ('male', 'female') else 'unknown', age).items()})

    # ── SHAPY 3D body (CVPR 2022), adjusted to the real weight ────────────────
    if shapy is not None and shapy_items:
        sm, _ = shapy.measure_person(shapy_items, height, weight)
        if sm:
            components.update({f'shapy:{k}': v for k, v in sm.items()
                               if k in ('chest', 'waist', 'hip')})

    # ── CameraHMR 3D body (trained on BEDLAM), adjusted to the real weight ────
    if camerahmr is not None and cam_betas:
        cm, _ = camerahmr.measure_person(cam_betas, height, weight, rings)
        components.update({f'camerahmr:{k}': v for k, v in cm.items()})

    # ── Final combiner learned on real photos (model/calibration.py) ──────────
    if req.get('calibrate', True):
        from model.calibration import apply_calibration
        for key, val in apply_calibration(components, weight_known=bool(weight)).items():
            m[key] = val
            source[key] = source.get(key, '') + '+calibrated'
        # this shop's own tape style (scripts/retrain_from_shop.py); no file = no change
        from model.calibration import apply_shop_offset, SHOP_PATH
        for key, val in apply_shop_offset(m, bool(weight), req.get('shop_offset_path', SHOP_PATH)).items():
            m[key] = val
            source[key] = source.get(key, '') + '+shop'

    return {
        'ok': True,
        'measurements': m,
        'sigma': sigma,
        'source': source,
        'image_used': image_used,
        'arms_away': round(arms, 2),
        'weight_used': bool(weight),
        'components': {k: round(float(v), 2) for k, v in components.items()},
        'n_frames': len(betas),
        'views': [c.get('view_used', fr.get('view', '?')) for c, fr in zip(checks, frames)] if checks else
                 [fr.get('view', '?') for fr in frames],
        'photo_checks': checks,
        'vertices': V_show.round(4).tolist(),
        'faces': hb.faces.tolist(),
    }


def serve():
    t0 = time.time()
    hb = HMRBody()
    rings = load_rings()
    hybrid = _load_hybrid()
    shapy = _load_shapy()
    camerahmr = _load_camerahmr()
    _log(f'models ready on {hb.device} in {time.time() - t0:.1f}s '
         f'(hybrid: {"yes" if hybrid else "no"}, SHAPY: {"yes" if shapy else "no"}, '
         f'CameraHMR: {"yes" if camerahmr else "no"})')
    print(json.dumps({'ready': True, 'device': hb.device}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        if req.get('cmd') == 'quit':
            break
        t = time.time()
        try:
            out = run_request(hb, rings, req, hybrid, shapy, camerahmr=camerahmr)
        except Exception as exc:          # keep the worker alive for the next scan
            out = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
        out['id'] = req.get('id')
        _log(f"request {out['id']}: ok={out['ok']} frames={out.get('n_frames', 0)} "
             f"images={'used' if out.get('image_used') else 'not used'} "
             f'{time.time() - t:.2f}s')
        print(json.dumps(out), flush=True)
    hb.close()


def main():
    ap = argparse.ArgumentParser(description='TailorCam body measurement')
    ap.add_argument('--image', nargs='+',
                    help='photos of one person; name side photos with "side" in the file name')
    ap.add_argument('--height', type=float, help='true height in cm')
    ap.add_argument('--weight', type=float, help='body weight in kg (recommended)')
    ap.add_argument('--gender', default='unspecified')
    ap.add_argument('--serve', action='store_true', help='JSON-lines server mode')
    a = ap.parse_args()
    if a.serve or not a.image:
        serve()
        return
    hb = HMRBody()
    frames = [{'path': p, 'view': 'LEFT' if 'side' in os.path.basename(p).lower() else 'FRONT'}
              for p in a.image]
    out = run_request(hb, load_rings(),
                      {'height_cm': a.height, 'weight_kg': a.weight, 'gender': a.gender,
                       'frames': frames}, _load_hybrid(), _load_shapy(), camerahmr=_load_camerahmr())
    hb.close()
    if not out['ok']:
        sys.exit(out['error'])
    print(f"arms away from body: {out['arms_away']}  "
          f"silhouettes used: {out['image_used']}  weight used: {out['weight_used']}")
    for k, v in out['measurements'].items():
        s = out['sigma'].get(k)
        print(f'{k:15s} {v:6.1f} cm' + (f'  +/-{s}' if s is not None else '')
              + f"  [{out['source'].get(k, '')}]")


if __name__ == '__main__':
    main()
