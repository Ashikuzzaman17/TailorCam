"""
One combined dataset from real24 (Unidata samples) and odjag (own photos, local tailor):
dataset/combined/

  people.csv           one row per person, all values in cm, same columns for both
  <person_id>/front.jpg (and side.jpg when there is one)
  README.txt           where every column comes from

Waist and belly are kept separate. real24 has no belly value; odjag has both.
The 'source' column must be used when training (each source tapes a little
differently, see experiments/odjag/shop_fix.py). Not for publishing: the Unidata
part is CC BY-NC-ND 4.0 (no redistribution of modified copies).

  python scripts/build_combined_dataset.py
"""

import os
import shutil

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
OUT = data_path('combined')
COLS = ['person_id', 'source', 'measured_by', 'gender', 'age', 'height_cm', 'weight_kg', 'chest', 'waist', 'belly', 'hip',
        'neck', 'thigh', 'shoulder_width', 'sleeve_length', 'wrist', 'ankle', 'hip_upper', 'under_chest', 'calf', 'bicep',
        'has_front', 'has_side', 'has_weight', 'flagged']


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    r24 = pd.read_csv(data_path('real_photos', 'people.csv'))
    for r in r24.itertuples():
        src = data_path('real_photos', r.person_id)
        pid = f'r24_{r.person_id}'
        os.makedirs(os.path.join(OUT, pid), exist_ok=True)
        for f in ('front.jpg', 'side.jpg'):
            if os.path.exists(os.path.join(src, f)):
                shutil.copy(os.path.join(src, f), os.path.join(OUT, pid, f))
        rows.append({'person_id': pid, 'source': 'real24', 'measured_by': 'Unidata (data vendor)', 'gender': r.gender,
                     'age': r.age, 'height_cm': r.height_cm, 'weight_kg': r.weight_kg, 'chest': r.chest, 'waist': r.waist,
                     'belly': None, 'hip': r.hip, 'neck': r.neck, 'thigh': r.thigh, 'shoulder_width': r.shoulder_width,
                     'sleeve_length': r.arm_length, 'wrist': None, 'ankle': None, 'hip_upper': r.hip_upper,
                     'under_chest': r.under_chest, 'calf': r.calf, 'bicep': r.bicep,
                     'has_front': True, 'has_side': os.path.exists(os.path.join(src, 'side.jpg')),
                     'has_weight': pd.notna(r.weight_kg), 'flagged': r.flagged if pd.notna(r.flagged) else ''})
    od = pd.read_csv(data_path('odjag', 'people.csv'))
    for r in od.itertuples():
        pid = f'odj_{r.person_id}'
        src = data_path('odjag', r.person_id, 'front.jpg')
        if os.path.exists(src):
            os.makedirs(os.path.join(OUT, pid), exist_ok=True)
            shutil.copy(src, os.path.join(OUT, pid, 'front.jpg'))
        rows.append({'person_id': pid, 'source': 'odjag', 'measured_by': 'local tailor (handwritten, inches)',
                     'gender': None, 'age': None, 'height_cm': r.height_cm, 'weight_kg': None, 'chest': r.chest,
                     'waist': r.waist, 'belly': r.belly, 'hip': r.hip, 'neck': r.neck, 'thigh': r.thigh,
                     'shoulder_width': r.shoulder_width, 'sleeve_length': r.sleeve_length, 'wrist': r.wrist,
                     'ankle': r.ankle, 'hip_upper': None, 'under_chest': None, 'calf': None, 'bicep': None,
                     'has_front': bool(r.has_photo), 'has_side': False, 'has_weight': False,
                     'flagged': 'handwritten, some values hard to read'})
    d = pd.DataFrame(rows)[COLS]
    d.to_csv(os.path.join(OUT, 'people.csv'), index=False)
    with open(os.path.join(OUT, 'README.txt'), 'w', encoding='utf-8') as f:
        f.write(__doc__.strip() + '\n\nColumns (all cm / kg):\n'
                '  waist          real24: vendor waist; odjag: tailor waist (may be the trouser waist)\n'
                '  belly          odjag only: fullest part of the stomach\n'
                '  sleeve_length  real24: arm_length; odjag: length_arm\n'
                '  shoulder_width real24: shoulder_width; odjag: shoulder_length\n'
                '  flagged        real24: provider "to be reviewed" fields; odjag: handwriting note\n'
                f'\nPeople: {len(d)} ({(d.source == "real24").sum()} real24, {(d.source == "odjag").sum()} odjag); '
                f'with a front photo: {int(d.has_front.sum())}; with a side photo: {int(d.has_side.sum())}; '
                f'with weight: {int(d.has_weight.sum())}\n')
    print(d.groupby('source').agg(people=('person_id', 'count'), front=('has_front', 'sum'), side=('has_side', 'sum'),
                                  weight=('has_weight', 'sum'), belly=('belly', 'count'), waist=('waist', 'count')).to_string())
    print('saved', OUT)


if __name__ == '__main__':
    main()
