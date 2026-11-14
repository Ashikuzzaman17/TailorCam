"""
Build the 'odjag' test set: our own photos + hand-written tape measurements.

Source (copied, never edited): ~/Downloads/odjag, or the folder in TAILORCAM_ODJAG_SRC
  body_measurements.json   tape values for 10 people (inches; height in feet'inches")
  photos                   one front photo per person; the name is written INSIDE each
                           picture; dataset/odjag_photo_names.json maps every file to its person
                           (read from the pictures on 2026-11-13)

Output  dataset/odjag/
          raw/                 untouched copy of the source folder
          people.csv           one row per person, all values in cm
          <person_id>/front.jpg

Assumptions (see the printout): values are inches, no weight, gender not given.

  python scripts/build_odjag_dataset.py
"""

import json
import os
import re
import shutil

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
SRC = os.environ.get('TAILORCAM_ODJAG_SRC', os.path.join(os.path.expanduser('~'), 'Downloads', 'odjag'))
OUT = data_path('odjag')
IN2CM = 2.54
# photo file -> person, read from the name written inside each picture. Kept outside the code
# (dataset/odjag_photo_names.json) because it holds the participants' names.
_NAMES_FILE = data_path('odjag_photo_names.json')
PHOTO_NAMES = json.load(open(_NAMES_FILE, encoding='utf-8')) if os.path.exists(_NAMES_FILE) else {}
# JSON field → TailorCam key
FIELDS = {'chest': 'chest', 'waist': 'waist', 'belly': 'belly', 'hips': 'hip', 'neck': 'neck', 'thigh': 'thigh',
          'length_arm': 'sleeve_length', 'shoulder_length': 'shoulder_width', 'wrist': 'wrist', 'ankle': 'ankle'}


def height_cm(s):
    m = re.match(r"\s*(\d+)'\s*([\d.]*)", s)
    return round((int(m.group(1)) * 12 + float(m.group(2) or 0)) * IN2CM, 1)


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)                                 # rebuild from the source folder only
    os.makedirs(OUT)
    shutil.copytree(SRC, os.path.join(OUT, 'raw'))
    meas = json.load(open(os.path.join(SRC, 'body_measurements.json'), encoding='utf-8'))['measurements']
    photos = {PHOTO_NAMES.get(f, os.path.splitext(f)[0].lower()): f for f in os.listdir(SRC)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    unknown = [f for f in os.listdir(SRC) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
               and PHOTO_NAMES.get(f, os.path.splitext(f)[0].lower()) not in {n.lower() for n in meas}]
    if unknown:
        print('photos with no matching person in the JSON:', unknown)
    rows = []
    for name, v in meas.items():
        pid = name.lower()
        row = {'person_id': pid, 'name_in_json': name, 'photo_file': photos.get(pid, ''), 'height_cm': height_cm(v['height']), 'weight_kg': None,
               'gender': 'unknown', 'has_photo': pid in photos}
        row.update({k: round(float(v[f]) * IN2CM, 1) for f, k in FIELDS.items() if f in v})
        rows.append(row)
        if pid in photos:
            os.makedirs(os.path.join(OUT, pid), exist_ok=True)
            shutil.copy(os.path.join(SRC, photos[pid]), os.path.join(OUT, pid, 'front.jpg'))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, 'people.csv'), index=False)
    print(d[['person_id', 'has_photo', 'height_cm', 'chest', 'waist', 'belly', 'hip', 'neck', 'thigh', 'sleeve_length',
             'shoulder_width']].to_string(index=False))
    print(f'\nsaved {OUT}: {len(d)} people, {int(d.has_photo.sum())} with a photo')


if __name__ == '__main__':
    main()
