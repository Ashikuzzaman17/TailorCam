"""
Build one merged test set of REAL colour photos with real body measurements
from every free source found (October 2026):

  ud-biometrics/body-measurements-image-dataset   (Hugging Face, 6 people)
  UniqueData/body-measurements-dataset            (Hugging Face, 21 people)

Both are free samples from Unidata, licence CC BY-NC-ND 4.0: non-commercial
use, no redistribution of modified versions. Fine for testing; don't publish
the merged copy.

Output  dataset/real_photos/
          people.csv            one row per person, all measurements in cm
          <person_id>/front.jpg, side.jpg (and selfie.jpg when available)
          raw/<source>/...      original downloads

Values marked "_tbr" ("to be reviewed") by the provider are kept, but listed
in the `flagged` column so they can be excluded from scoring.

  python scripts/build_real_photo_dataset.py
"""

import os
import csv
import json
import shutil
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
OUT = data_path('real_photos')
RAW = os.path.join(OUT, 'raw')

HF = 'https://huggingface.co/datasets/{repo}/resolve/main/{path}'

# Unified column ← provider field (both providers use the same names)
FIELDS = {
    'chest': 'chest_circumference_cm',
    'waist': 'waist_circumference_cm',
    'hip': 'pelvis_circumference_cm',          # fullest girth over the seat
    'hip_upper': 'hips_circumference_cm',      # provider's "hips" (higher, at hip bone)
    'under_chest': 'under_chest_circumference_cm',
    'thigh': 'thigh_circumference_cm',
    'calf': 'calf_circumference_cm',
    'bicep': 'arm_circumference_cm',
    'neck': 'neck_circumference_cm',
    'shoulder_width': 'shoulder_width_cm',
    'arm_length': 'arm_length_cm',
    'upper_arm_length': 'upper_arm_length_cm',
    'leg_length': 'leg_length_cm',
    'front_build': 'front_build_cm',
    'neck_to_waist_front': 'neck_waist_length_front_cm',
}
COLUMNS = (['person_id', 'source', 'gender', 'age', 'race', 'height_cm', 'weight_kg']
           + list(FIELDS) + ['flagged'])


def fetch(repo, path, dest):
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(HF.format(repo=repo, path=path.replace(' ', '%20')), dest)
    return dest


def parse(raw):
    """Provider record → unified row (numbers as float, '_tbr' noted)."""
    row, flagged = {}, []
    for col, key in FIELDS.items():
        v = str(raw.get(key, '')).strip()
        if v.endswith('_tbr'):
            flagged.append(col)
            v = v[:-4]
        try:
            row[col] = float(v)
        except ValueError:
            row[col] = ''
    row['height_cm'] = float(raw['height'])
    row['weight_kg'] = float(raw['weight'])
    row['gender'], row['age'] = raw.get('gender', ''), raw.get('age', '')
    row['race'] = raw.get('race', '')
    row['flagged'] = ';'.join(flagged)
    return row


def source_ud_biometrics():
    repo = 'ud-biometrics/body-measurements-image-dataset'
    d = os.path.join(RAW, 'ud-biometrics')
    table = fetch(repo, 'Body Measurements Image Dataset.csv', os.path.join(d, 'measurements.csv'))
    for r in csv.DictReader(open(table, encoding='utf-8')):
        i = r['set_id']
        imgs = {v: fetch(repo, f'{i}/{v}_img.jpg', os.path.join(d, i, f'{v}_img.jpg'))
                for v in ('front', 'side')}
        yield f'udb_{i}', 'ud-biometrics (HF)', parse(r), imgs


def source_uniquedata():
    repo = 'UniqueData/body-measurements-dataset'
    d = os.path.join(RAW, 'uniquedata')
    table = fetch(repo, 'body.csv', os.path.join(d, 'body.csv'))
    for r in csv.DictReader(open(table, encoding='utf-8')):
        folder = r['front'].split('/')[1]
        meas = fetch(repo, r['measurements'], os.path.join(d, folder, 'measurements.json'))
        imgs = {v: fetch(repo, r[v], os.path.join(d, folder, f'{v}_img.jpg'))
                for v in ('front', 'side', 'selfie')}
        yield f'uniq_{folder}', 'UniqueData (HF)', parse(json.load(open(meas, encoding='utf-8'))), imgs


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, seen = [], {}
    for gen in (source_ud_biometrics, source_uniquedata):
        for pid, src, row, imgs in gen():
            # same person published twice? (identical body record)
            key = (row['gender'], row['age'], row['height_cm'], row['weight_kg'],
                   row['chest'], row['waist'])
            if key in seen:
                print(f'  duplicate: {pid} = {seen[key]} (skipped)')
                continue
            seen[key] = pid
            os.makedirs(os.path.join(OUT, pid), exist_ok=True)
            for v, path in imgs.items():
                shutil.copyfile(path, os.path.join(OUT, pid, f'{v}.jpg'))
            rows.append({'person_id': pid, 'source': src, **row})
    with open(os.path.join(OUT, 'people.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    men = sum(r['gender'] == 'male' for r in rows)
    print(f'\n{len(rows)} people ({men} men, {len(rows) - men} women) -> {OUT}')
    print('measurements available:', ', '.join(
        f"{c} {sum(r[c] != '' for r in rows)}" for c in FIELDS))


if __name__ == '__main__':
    main()
