"""
Your shop's own training data.

Every TailorCam scan stores, in app_data/shop_scans/<scan id>/:
  scan.json   each model's answer (the "components"), height / weight / gender /
              age, garment, and the DB measurement id
  NN_VIEW.jpg the camera frames used (so the models can be re-run later)

When the tailor tape-measures the customer and enters the real values with the
Feedback button, they are stored in tailorcam.db (table feedback). Joining the
two gives training rows exactly like the public datasets — but for your
customers, your camera and your way of measuring.

  python scripts/retrain_from_shop.py      # retrain the combiner with them
"""

import os
import json
import time

import cv2

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
SCANS = data_path('shop_scans')
DB = os.path.join(_ROOT, 'tailorcam.db')


def save_scan(result, frames, measurement_id=None, customer_id=None, height_cm=None,
              weight_kg=None, gender='unspecified', age=None, garment=''):
    """result: worker reply (needs 'components'); frames: [(view, bgr, keypoints)]."""
    sid = time.strftime('%Y%m%d_%H%M%S') + (f'_m{measurement_id}' if measurement_id else '')
    folder = os.path.join(SCANS, sid)
    os.makedirs(folder, exist_ok=True)
    for i, (view, img, _) in enumerate(frames):
        cv2.imwrite(os.path.join(folder, f'{i:02d}_{view}.jpg'), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(os.path.join(folder, 'scan.json'), 'w') as f:
        json.dump({'scan_id': sid, 'measurement_id': measurement_id, 'customer_id': customer_id,
                   'height_cm': height_cm, 'weight_kg': weight_kg, 'gender': gender, 'age': age,
                   'garment': garment, 'arms_away': result.get('arms_away'),
                   'silhouettes_used': result.get('image_used'),
                   'components': result.get('components', {}),
                   'measurements': result.get('measurements', {})}, f, indent=1)
    print(f'[ShopData] scan saved: {folder}')
    return folder


def load_rows():
    """
    Training rows (same format as dataset/predictions/*.csv) for every saved scan
    whose customer has tailor feedback: components + 'true:<measurement>'.
    """
    import sqlite3
    if not os.path.isdir(SCANS) or not os.path.exists(DB):
        return []
    truth = {}
    conn = sqlite3.connect(DB)
    try:
        for mid, field, actual in conn.execute(
                'SELECT measurement_id, field, actual_cm FROM feedback WHERE actual_cm IS NOT NULL'):
            truth.setdefault(mid, {})[field] = float(actual)
    finally:
        conn.close()
    rows = []
    for name in sorted(os.listdir(SCANS)):
        p = os.path.join(SCANS, name, 'scan.json')
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        t = truth.get(s.get('measurement_id'))
        if not t:
            continue
        rows.append({'person_id': s['scan_id'], 'split': 'shop',
                     'mode': 'with_weight' if s.get('weight_kg') else 'no_weight',
                     'height_cm': s.get('height_cm'), 'weight_kg': s.get('weight_kg'),
                     'gender': s.get('gender'), **s.get('components', {}),
                     **{f'true:{k}': v for k, v in t.items()}})
    return rows
