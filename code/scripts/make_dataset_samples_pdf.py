"""
How TailorCam is trained and tested, with 2-3 real samples from every dataset:
reports/TailorCam_Datasets_and_Training.pdf

  python scripts/make_dataset_samples_pdf.py
"""

import glob
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import DATASETS  # noqa: E402  (dataset/, assets/, app_data/)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from make_model_accuracy_pdf import S, table, figure, P, N   # same fonts / tables as the other reports

DATA = DATASETS
FIG = os.path.join(ROOT, 'reports', 'figures_datasets')
OUT = os.path.join(ROOT, 'reports', 'TailorCam_Datasets_and_Training.pdf')
W = A4[0] - 36 * mm


def rgb(path, h=360):
    img = cv2.imread(path)
    s = h / img.shape[0]
    return cv2.cvtColor(cv2.resize(img, (int(img.shape[1] * s), h)), cv2.COLOR_BGR2RGB)


def grid(rows, path, titles):
    """rows: list of lists of images (one row per sample); titles: one caption per row."""
    n, m = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(n, m, figsize=(1.9 * m, 2.3 * n), dpi=150, squeeze=False)
    for i, r in enumerate(rows):
        for j in range(m):
            ax = axes[i][j]
            ax.axis('off')
            if j < len(r):
                ax.imshow(r[j], cmap='gray' if r[j].ndim == 2 else None)
        axes[i][0].set_title(titles[i], fontsize=7, loc='left', color='#52514e')
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def real24():
    d = pd.read_csv(os.path.join(DATA, 'real_photos', 'people.csv'))
    s = d[d.chest.notna() & d.waist.notna() & d.hip.notna()].iloc[[0, 5, 12]]
    rows = [[rgb(os.path.join(DATA, 'real_photos', r.person_id, f)) for f in ('front.jpg', 'side.jpg')]
            for r in s.itertuples()]
    grid(rows, os.path.join(FIG, 'real24.png'), [f'{r.person_id}, {r.gender}' for r in s.itertuples()])
    tab = [['Person', 'Gender', 'Height', 'Weight', 'Chest', 'Waist', 'Hip', 'Neck', 'Thigh']] + \
          [[r.person_id, r.gender, f'{r.height_cm:.0f}', f'{r.weight_kg:.0f}', f'{r.chest:.0f}', f'{r.waist:.0f}',
            f'{r.hip:.0f}', '-' if pd.isna(r.neck) else f'{r.neck:.0f}', '-' if pd.isna(r.thigh) else f'{r.thigh:.0f}']
           for r in s.itertuples()]
    return tab


def bodym():
    d = os.path.join(DATA, 'bodym', 'train')
    meas = pd.read_csv(os.path.join(d, 'measurements.csv')).set_index('subject_id')
    meta = pd.read_csv(os.path.join(d, 'hwg_metadata.csv')).set_index('subject_id')
    mp = pd.read_csv(os.path.join(d, 'subject_to_photo_map.csv')).drop_duplicates('subject_id').iloc[[0, 40, 400]]
    rows, tab = [], [['Subject', 'Gender', 'Height', 'Weight', 'Chest', 'Waist', 'Hip', 'Arm length', 'Shoulder']]
    for r in mp.itertuples():
        pid = r.photo_id.strip()
        rows.append([cv2.imread(os.path.join(d, 'mask', pid + '.png'), 0), cv2.imread(os.path.join(d, 'mask_left', pid + '.png'), 0)])
        m, g = meas.loc[r.subject_id], meta.loc[r.subject_id]
        tab.append([r.subject_id[:8] + '...', g.gender, f'{g.height_cm:.0f}', f'{g.weight_kg:.0f}', f'{m.chest:.1f}',
                    f'{m.waist:.1f}', f'{m.hip:.1f}', f'{m["arm-length"]:.1f}', f'{m["shoulder-breadth"]:.1f}'])
    grid(rows, os.path.join(FIG, 'bodym.png'), [t[0] for t in tab[1:]])
    return tab


def ansur():
    tab = [['Gender', 'Age', 'Height', 'Weight', 'Chest', 'Waist', 'Hip (buttock)', 'Neck', 'Sleeve']]
    for f, i in (('ANSUR_II_MALE_Public.csv', 0), ('ANSUR_II_MALE_Public.csv', 100), ('ANSUR_II_FEMALE_Public.csv', 0)):
        r = pd.read_csv(os.path.join(DATA, 'ansur2', f), encoding='latin-1').iloc[i]
        tab.append([r.Gender, f'{r.Age}', f'{r.stature / 10:.0f}', f'{r.weightkg / 10:.0f}', f'{r.chestcircumference / 10:.1f}',
                    f'{r.waistcircumference / 10:.1f}', f'{r.buttockcircumference / 10:.1f}',
                    f'{r.neckcircumferencebase / 10:.1f}', f'{r.sleeveoutseam / 10:.1f}'])
    return tab


def hbw():
    d = pd.read_csv(os.path.join(DATA, 'predictions', 'hbw_val.csv'))
    d = d[d['mode'] == 'with_weight'].iloc[[0, 3, 6]]
    rows, tab = [], [['Person', 'Height', 'Weight (scan)', 'Chest', 'Waist', 'Hip']]
    for r in d.itertuples():
        sid = r.person_id.replace('hbw_', '')
        imgs = sorted(glob.glob(os.path.join(DATA, 'shapy', 'HBW_low_resolution', 'images', 'val_small_resolution', f'{sid}_*', '*', '*.png')))
        rows.append([rgb(p) for p in imgs[::max(1, len(imgs) // 3)][:3]])
    t = d[['person_id', 'height_cm', 'weight_kg', 'true:chest', 'true:waist', 'true:hip']].values
    tab = [tab[0]] + [[a, f'{b:.0f}', f'{c:.0f}', f'{x:.1f}', f'{y:.1f}', f'{z:.1f}'] for a, b, c, x, y, z in t]
    grid(rows, os.path.join(FIG, 'hbw.png'), [r[0] for r in tab[1:]])
    return tab


def agency():
    d = pd.read_csv(os.path.join(DATA, 'shapy', 'model_agency_images', 'index.csv'))
    d = d[d.split == 'test'].iloc[[0, 10, 30]]
    rows, tab = [], [['Model', 'Gender', 'Height', 'Chest', 'Waist', 'Hip']]
    for r in d.itertuples():
        imgs = sorted(glob.glob(os.path.join(DATA, 'shapy', 'model_agency_images', r.agency, str(r.model_id), '*.jpg')))[:3]
        rows.append([rgb(p) for p in imgs])
        tab.append([f'{r.agency} #{r.Index}', r.gender, f'{r.height_cm:.0f}', f'{r.chest_cm:.0f}', f'{r.waist_cm:.0f}', f'{r.hip_cm:.0f}'])
    grid(rows, os.path.join(FIG, 'agency.png'), [t[0] for t in tab[1:]])
    return tab


def bedlam():
    d = pd.read_csv(os.path.join(DATA, 'bedlam', 'people.csv')).iloc[[0, 10, 20]]
    rows, tab = [], [['Person', 'Gender', 'Height', 'Weight', 'Chest', 'Waist', 'Hip']]
    for r in d.itertuples():
        fr = [p for p in r.frames.split(';') if os.path.exists(p)][:3]
        rows.append([rgb(p) for p in fr])
        tab.append([r.person_id.split('/')[-1], r.gender, f'{r.height_cm:.0f}', f'{r.weight_kg:.0f}', f'{r.chest:.1f}',
                    f'{r.waist:.1f}', f'{r.hip:.1f}'])
    grid(rows, os.path.join(FIG, 'bedlam.png'), [t[0] for t in tab[1:]])
    return tab


def scans(cache, name, pick):
    D = dict(np.load(cache))
    rows, tab = [], [['Person', 'Gender', 'Height', 'Weight*', 'Chest', 'Waist', 'Hip', 'Shoulder', 'Arm']]
    for i in pick:
        rows.append([D['front'][i] * 255, D['side'][i] * 255])
        y = D['y'][i]
        tab.append([str(D['names'][i]), 'male' if D['g'][i] == 1 else 'female', f'{D["h"][i]:.0f}', f'{D["w"][i]:.0f}',
                    f'{y[0]:.1f}', f'{y[1]:.1f}', f'{y[2]:.1f}', f'{y[5]:.1f}', f'{y[6]:.1f}'])
    grid(rows, os.path.join(FIG, f'{name}.png'), [t[0] for t in tab[1:]])
    return tab


def build():
    os.makedirs(FIG, exist_ok=True)
    t_real, t_bm, t_an, t_hbw, t_ag, t_bed = real24(), bodym(), ansur(), hbw(), agency(), bedlam()
    t_bf = scans(os.path.join(DATA, 'bodyfit', 'cache.npz'), 'bodyfit', [0, 1500, 3500])
    t_nm = scans(os.path.join(DATA, 'nomo3d', 'cache.npz'), 'nomo', [0, 100, 300])
    cw = lambda n: [W / n] * n
    st = [Paragraph('TailorCam: How the Models Are Trained and Tested', S['title']), Spacer(1, 3),
          Paragraph('With 2 to 3 real samples from every dataset · Md Ashikuzzaman, ID 2022-3-60-101 · Supervisor: '
                    'Dr. Mohammad Mahdi Hasan · November 2026', S['sub']), Spacer(1, 6)]

    st += [Paragraph('1. The rule: train and test on different people', S['h']),
           P('A model is only useful if it works on people it has never seen. So every dataset has one job: it is used '
             'either to <b>train</b> a model or to <b>test</b> it, never both for the same people.'),
           table([['Dataset', 'People', 'What one sample is', 'Job'],
                  ['Real photos (real24)', '24', '2 colour photos + tape values', 'train combiner and test (cross-validation)'],
                  ['BodyM (Amazon)', '2,505', '2 silhouettes + scan values', 'train CNN (2,018) / test (400)'],
                  ['ANSUR II (US Army)', '6,068', 'a table row, no photo', 'train ANSUR prior'],
                  ['BODY-fit (Tampere)', '4,132', '3D body + scanner values', 'train new network / test (558)'],
                  ['HBW (SHAPY project)', '10', 'many photos + 3D scan', 'test only'],
                  ['Model agency (SHAPY)', '134 test', 'fashion photos + agency sizes', 'test only'],
                  ['BEDLAM (synthetic)', '42', 'video frames + exact body', 'test only'],
                  ['NOMO-3D (Tampere)', '356', 'real 3D scan + scanner values', 'test only']],
                 [W * 0.24, W * 0.1, W * 0.31, W * 0.35]),
           Paragraph('2. How training works', S['h']),
           P('<b>Silhouette CNN (3 copies).</b> Each training sample is a front and a side silhouette plus height, weight and '
             'gender, and the target is the true measurement. The network (ResNet-18) sees a batch of 64 people, predicts '
             'their measurements, compares with the truth, and adjusts its weights. This is repeated for 40 passes over '
             'the data (epochs) on the RTX 4070 GPU. Silhouettes are randomly rotated, shifted, blurred and given holes so '
             'the network also works with real, imperfect masks. Three copies with different random seeds are averaged.'),
           P('<b>BodyM and ANSUR priors.</b> Ridge regression: a formula from height, weight, gender (and age) to each '
             'measurement, fitted on all training people at once in a few seconds.'),
           P('<b>3D models (HMR, SHAPY, CameraHMR).</b> Not trained here; they come pre-trained from their authors. TailorCam '
             'adjusts each 3D body to the customer\'s height and weight and measures it with a virtual tape.'),
           P('<b>Combiner.</b> Learns, for each body part, which model to trust and how much to correct it, from the real '
             'tape-measured people.'),
           Paragraph('3. How testing works', S['h']),
           P('<b>Nested cross-validation on the 24 real people.</b> The 24 people are split into 6 groups of 4. Round 1: '
             'the combiner learns from groups 2 to 6 (20 people) and is tested on group 1 (4 people). Round 2: it learns '
             'from groups 1, 3, 4, 5, 6 and is tested on group 2, and so on, until every person has been tested once by a '
             'combiner that never saw them. Because 24 is small, the whole thing is repeated with 5 different random '
             'splits and the errors are averaged. Inside each round, the choice of rule is also made only on the 20 '
             'training people ("nested").'),
           P('<b>Outside test sets</b> (HBW, model agency, BEDLAM, NOMO-3D, BodyM test, BODY-fit test) are never used for '
             'training at all; the finished models are simply run on them.'),
           P('<b>Scores:</b> average error in cm (MAE), accuracy = 100 minus the average % error, right clothing size, F1.')]

    sets = [
        ('4. Real photos (real24): train + test', 'real24', t_real,
         'Front and side colour photos in normal clothes, with tape measurements (cm, kg). Source: Unidata free samples, '
         'Hugging Face (CC BY-NC-ND 4.0).'),
        ('5. BodyM (Amazon): training the silhouette CNN', 'bodym', t_bm,
         'Front and left-side silhouettes (no colour photo), with measurements from 3D scans (cm, kg). '
         'registry.opendata.aws/bodym.'),
        ('6. ANSUR II (US Army): training the ANSUR prior', None, t_an,
         'Tape measurements only, no photos (converted to cm, kg). openlab.psu.edu/ansur2.'),
        ('7. HBW (SHAPY project): test only', 'hbw', t_hbw,
         'Several everyday and lab photos of each person; truth measured on a 3D body scan. shapy.is.tue.mpg.de.'),
        ('8. Model agency (SHAPY project): test only', 'agency', t_ag,
         'Fashion photos; truth is the size list published by the model agency (cm), no weight.'),
        ('9. BEDLAM: test only (synthetic)', 'bedlam', t_bed,
         'Computer-rendered clothed people in video scenes; truth is measured on the exact 3D body. bedlam.is.tue.mpg.de.'),
        ('10. BODY-fit (Tampere University): training the new network', 'bodyfit', t_bf,
         'Real 3D body scans fitted with a body model, rendered as front and side silhouettes; scanner measurements. '
         '*Weight is computed from body volume. zenodo.org/records/5172126.'),
        ('11. NOMO-3D (Tampere University): test only', 'nomo', t_nm,
         'Raw real 3D scans rendered as silhouettes; scanner measurements. *Weight from volume is not reliable here '
         '(scans have holes), so NOMO is tested without weight. zenodo.org/records/3735905.'),
    ]
    for title, img, tab, note in sets:
        block = [Paragraph(title, S['h']), P(note)]
        if img:
            block.append(figure(os.path.join(FIG, f'{img}.png'), W * (0.55 if img in ('real24', 'bodym', 'bodyfit', 'nomo') else 0.8),
                                'Samples, one row per person.'))
        block.append(table(tab, cw(len(tab[0]))))
        st += [KeepTogether(block)]
    st += [Spacer(1, 6), N('All values in cm, weight in kg. Reproduce: scripts/make_dataset_samples_pdf.py. Datasets are '
                           'used under their research licences; photos are not redistributed.')]

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm,
                            bottomMargin=16 * mm, title='TailorCam Datasets and Training', author='Md Ashikuzzaman')
    doc.build(st)
    print('saved', OUT)


if __name__ == '__main__':
    build()
