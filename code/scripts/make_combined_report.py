"""
Chart-first report for the combined dataset tested as ONE dataset:
reports/TailorCam_Combined_Test.pdf

Needs: python scripts/build_combined_dataset.py ; python experiments/combined/test.py

  python scripts/make_combined_report.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from paths import dataset
from make_model_accuracy_pdf import S, P, N, table, figure, COL, GOOD, MUTED

RES = os.path.join(ROOT, 'experiments', 'combined')
FIG = os.path.join(ROOT, 'reports', 'figures_combined')
OUT = os.path.join(ROOT, 'reports', 'TailorCam_Combined_Test.pdf')
W = A4[0] - 36 * mm
HK = S['h'].clone('h_keep', keepWithNext=1)
KEYS = ['chest', 'waist', 'hip', 'neck', 'thigh', 'sleeve_length', 'shoulder_width']
LAB = dict(zip(KEYS, ['Chest', 'Waist', 'Hip', 'Neck', 'Thigh', 'Sleeve', 'Shoulder']))
TC = 'TailorCam (combined)'
MODELS = [TC, 'CameraHMR', 'SHAPY', 'HMR', 'Silhouette CNN', 'BodyM prior', 'ANSUR prior']
MCOL = {TC: '#2a78d6', 'HMR': COL['HMR (2018)'], 'SHAPY': COL['SHAPY (2022)'], 'CameraHMR': COL['CameraHMR (2025)'],
        'Silhouette CNN': COL['Silhouette CNN x3'], 'BodyM prior': COL['BodyM prior'], 'ANSUR prior': COL['ANSUR prior']}
SIZES, BINS = ['XS', 'S', 'M', 'L', 'XL', 'XXL'], {'chest': [84, 92, 100, 108, 116], 'waist': [70, 78, 86, 94, 102]}


def style(ax, grid='y'):
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    ax.grid(axis=grid, color='#e1e0d9', lw=0.6); ax.set_axisbelow(True); ax.tick_params(labelsize=7)


def val(s, m, k, c):
    q = s[(s.model == m) & (s.measurement == k)]
    return q[c].iloc[0] if len(q) else np.nan


def charts(pp, s, ov):
    os.makedirs(FIG, exist_ok=True)
    f = {}
    order = sorted(MODELS, key=lambda m: -ov[m])
    # 1 overall accuracy
    fig, ax = plt.subplots(figsize=(7.4, 2.4), dpi=200)
    ax.barh(range(len(order)), [ov[m] - 88 for m in order], left=88, color=[MCOL[m] for m in order], height=0.65)
    for i, m in enumerate(order):
        ax.text(ov[m] + 0.08, i, f'{ov[m]:.2f}%', va='center', fontsize=7)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7.5); ax.invert_yaxis(); ax.set_xlim(88, 96.5)
    ax.set_xlabel('overall accuracy (%), mean over the body parts each model measures', fontsize=7); style(ax, 'x')
    fig.tight_layout(); f['overall'] = os.path.join(FIG, 'overall.png'); fig.savefig(f['overall']); plt.close(fig)
    # 2 accuracy heat map
    A = np.array([[val(s, m, k, 'accuracy') for k in KEYS] for m in MODELS])
    fig, ax = plt.subplots(figsize=(7.4, 2.8), dpi=200)
    ax.imshow(A, cmap='Blues', vmin=86, vmax=98.5, aspect='auto')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if np.isnan(A[i, j]):
                ax.text(j, i, '-', ha='center', va='center', fontsize=7, color='#898781'); continue
            best = A[i, j] == np.nanmax(A[:, j])
            ax.text(j, i, f'{A[i, j]:.1f}', ha='center', va='center', fontsize=7.5, fontweight='bold' if best else 'normal',
                    color='white' if A[i, j] > 95 else '#0b0b0b')
    ax.set_xticks(range(len(KEYS))); ax.set_xticklabels([LAB[k] for k in KEYS], fontsize=7.5)
    ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(MODELS, fontsize=7.5); ax.tick_params(length=0)
    ax.set_title('Accuracy (%) per body part; bold = best in the column', fontsize=8.5, color='#52514e')
    for s_ in ax.spines.values():
        s_.set_visible(False)
    fig.tight_layout(); f['heat'] = os.path.join(FIG, 'heat.png'); fig.savefig(f['heat']); plt.close(fig)
    # 3 error per body part
    fig, ax = plt.subplots(figsize=(7.4, 2.9), dpi=200)
    x = np.arange(len(KEYS))
    for i, m in enumerate(MODELS):
        ax.bar(x + (i - 3) * 0.12, [val(s, m, k, 'MAE') for k in KEYS], 0.11, color=MCOL[m], label=m)
    ax.set_xticks(x); ax.set_xticklabels([LAB[k] for k in KEYS], fontsize=8); ax.set_ylabel('error (cm)', fontsize=7.5)
    ax.legend(fontsize=6.3, ncol=4, frameon=False, loc='upper right'); style(ax)
    fig.tight_layout(); f['mae'] = os.path.join(FIG, 'mae.png'); fig.savefig(f['mae']); plt.close(fig)
    # 4 tape vs TailorCam
    t = pp[pp.model == TC]
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.9), dpi=200)
    for ax, k in zip(axes.flat, KEYS):
        g = t[t.measurement == k]
        lo, hi = min(g.tape.min(), g.predicted.min()) - 3, max(g.tape.max(), g.predicted.max()) + 3
        ax.plot([lo, hi], [lo, hi], color='#898781', lw=0.8)
        ax.scatter(g.tape, g.predicted, s=12, color='#2a78d6', zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_title(f'{LAB[k]} ({len(g)})', fontsize=8); style(ax, 'both')
        ax.tick_params(labelsize=5.5)
    axes.flat[-1].axis('off')
    fig.supxlabel('tape (cm)', fontsize=7); fig.supylabel('TailorCam (cm)', fontsize=7)
    fig.tight_layout(); f['scatter'] = os.path.join(FIG, 'scatter.png'); fig.savefig(f['scatter']); plt.close(fig)
    # 5 signed error of every person
    fig, ax = plt.subplots(figsize=(7.4, 2.5), dpi=200)
    for i, k in enumerate(KEYS):
        g = t[t.measurement == k]
        e = (g.predicted - g.tape).values
        ax.scatter(np.full(len(e), i) + np.linspace(-0.2, 0.2, len(e)), e, s=10, zorder=3,
                   color=['#2a78d6' if v > 0 else '#eb6834' for v in e])
        ax.plot([i - 0.3, i + 0.3], [e.mean()] * 2, color='#0b0b0b', lw=1.4, zorder=4)
    ax.axhline(0, color='#898781', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([LAB[k] for k in KEYS], fontsize=8); ax.set_ylabel('TailorCam - tape (cm)', fontsize=7.5)
    ax.set_title('Each dot is one person: blue = too big, orange = too small, black line = bias', fontsize=8.5, color='#52514e')
    style(ax)
    fig.tight_layout(); f['signed'] = os.path.join(FIG, 'signed.png'); fig.savefig(f['signed']); plt.close(fig)
    # 6 per-person error heat map
    people = sorted(t.person.unique())
    E = np.array([[(lambda g: (g.predicted - g.tape).iloc[0] if len(g) else np.nan)(t[(t.person == p) & (t.measurement == k)])
                   for k in KEYS] for p in people])
    fig, ax = plt.subplots(figsize=(7.4, 6.2), dpi=200)
    ax.imshow(np.nan_to_num(E), cmap='RdBu_r', vmin=-12, vmax=12, aspect='auto')
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            ax.text(j, i, '-' if np.isnan(E[i, j]) else f'{E[i, j]:+.1f}', ha='center', va='center', fontsize=5.8,
                    color='white' if abs(np.nan_to_num(E[i, j])) > 7 else '#0b0b0b')
    ax.set_xticks(range(len(KEYS))); ax.set_xticklabels([LAB[k] for k in KEYS], fontsize=7)
    ax.set_yticks(range(len(people))); ax.set_yticklabels(people, fontsize=6); ax.tick_params(length=0)
    ax.set_title('Error of every person with TailorCam (cm): red = too big, blue = too small', fontsize=8.5, color='#52514e')
    for s_ in ax.spines.values():
        s_.set_visible(False)
    fig.tight_layout(); f['persons'] = os.path.join(FIG, 'persons.png'); fig.savefig(f['persons']); plt.close(fig)
    return f


def size_rows(pp):
    rows = [['Model', 'Chest: right size', 'Chest: within 1 size', 'Waist: right size', 'Waist: within 1 size']]
    for m in (TC, 'CameraHMR', 'ANSUR prior'):
        r = [m]
        for k in ('chest', 'waist'):
            g = pp[(pp.model == m) & (pp.measurement == k)]
            a, b = np.digitize(g.tape, BINS[k]), np.digitize(g.predicted, BINS[k])
            r += [f'{(a == b).mean() * 100:.0f}% ({int((a == b).sum())} of {len(g)})', f'{(abs(a - b) <= 1).mean() * 100:.0f}%']
        rows.append(r)
    return rows


def footer(c, doc):
    c.saveState(); c.setFont('UI', 7.5); c.setFillColor(MUTED)
    c.drawString(18 * mm, 10 * mm, 'TailorCam combined-dataset test · Md Ashikuzzaman (2022-3-60-101) · East West University')
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f'Page {doc.page}'); c.restoreState()


def build():
    pp = pd.read_csv(os.path.join(RES, 'people.csv'))
    s = pd.read_csv(os.path.join(RES, 'summary.csv'))
    ppl = pd.read_csv(dataset('combined', 'people.csv'))
    ppl = ppl[ppl.has_front]
    ov = s.groupby('model').accuracy.mean().to_dict()
    n = pp.person.nunique()
    f = charts(pp, s, ov)
    best_single = max((m for m in MODELS if m != TC), key=lambda m: ov[m])
    lab_ = lambda ks: ', '.join(LAB[k].lower() for k in ks) or 'none'
    best_k = [k for k in KEYS if val(s, TC, k, 'MAE') <= min(val(s, m, k, 'MAE') for m in MODELS if not np.isnan(val(s, m, k, 'MAE'))) + 1e-9]
    close_k = [k for k in KEYS if k not in best_k and val(s, TC, k, 'MAE') <= min(val(s, m, k, 'MAE') for m in MODELS
               if not np.isnan(val(s, m, k, 'MAE'))) + 0.35]
    best_parts, close_parts = lab_(best_k), lab_(close_k)
    st = [Paragraph('TailorCam: Test on the Combined Dataset', S['title']), Spacer(1, 3),
          Paragraph(f'{n} people, no weight, leak-free nested cross-validation · Md Ashikuzzaman, ID 2022-3-60-101', S['sub']),
          Spacer(1, 6)]
    box = Table([[Paragraph(
        f"<b><font color='#1f8a54'>Result.</font></b> On the combined dataset TailorCam reaches <b>{ov[TC]:.1f}%</b> overall "
        f"accuracy without the weight: chest {val(s, TC, 'chest', 'MAE'):.2f} cm, waist {val(s, TC, 'waist', 'MAE'):.2f} cm, "
        f"hip {val(s, TC, 'hip', 'MAE'):.2f} cm, neck {val(s, TC, 'neck', 'MAE'):.2f} cm. The best single model, "
        f"{best_single}, is {ov[best_single]:.1f}%; the other five single models are between "
        f"{min(ov[m] for m in MODELS if m not in (TC, best_single)):.1f}% and "
        f"{max(ov[m] for m in MODELS if m not in (TC, best_single)):.1f}%.", S['box'])]], colWidths=[W])
    box.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#e7f5ec')), ('LINEBEFORE', (0, 0), (0, -1), 3, GOOD),
                             ('LEFTPADDING', (0, 0), (-1, -1), 9), ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
    g = ppl.gender.fillna('not given').value_counts()
    st += [box, Paragraph('The dataset', HK),
           table([['People', 'Women / men / not given', 'Height', 'Chest', 'Waist', 'Hip', 'Photos'],
                  [f'{len(ppl)} ({n} tested)', f"{g.get('female', 0)} / {g.get('male', 0)} / {g.get('not given', 0)}",
                   f'{ppl.height_cm.min():.0f} to {ppl.height_cm.max():.0f} cm', f'{ppl.chest.min():.0f} to {ppl.chest.max():.0f} cm',
                   f'{ppl.waist.min():.0f} to {ppl.waist.max():.0f} cm', f'{ppl.hip.min():.0f} to {ppl.hip.max():.0f} cm',
                   'front for all, side for most']],
                 [W * 0.08, W * 0.19, W * 0.15, W * 0.14, W * 0.14, W * 0.14, W * 0.16]),
           N(f'Real people in normal clothes, measured with a tape. {len(ppl) - n} people are not tested because the data provider '
             'marked all their tape values as "to be reviewed". No weight is used, because part of the dataset has none.'),
           Paragraph('How it was tested', HK),
           P(f"The {n} people are split into 6 groups. Each group is measured once by models that learned only from the other "
             "groups, and this is repeated with 5 different random splits. Every single model learns one offset per body part "
             "(each model measures at a slightly different place); the combined model learns which models to trust for each "
             "body part. Accuracy = 100 minus the average percentage error."),
           figure(f['overall'], W, 'Figure 1. Overall accuracy of every model.')]
    rows = [['Model'] + [LAB[k] for k in KEYS] + ['Overall']]
    rows += [[m] + [('-' if np.isnan(val(s, m, k, 'MAE')) else f"{val(s, m, k, 'MAE'):.2f}") for k in KEYS] + [f'{ov[m]:.1f}%']
             for m in MODELS]
    cw = [W * 0.2] + [W * 0.095] * 7 + [W * 0.135]
    st += [Paragraph('Error of every model and body part (cm)', HK), table(rows, cw, hl=(1,)),
           N(f"People per body part: " + ', '.join(f"{LAB[k]} {int(val(s, TC, k, 'people'))}" for k in KEYS) + '.'),
           figure(f['heat'], W, 'Figure 2. Accuracy per body part and model.'),
           figure(f['mae'], W, 'Figure 3. Error per body part and model.'),
           Paragraph('Clothing size from the measurement', HK), table(size_rows(pp), [W * 0.24, W * 0.22, W * 0.16, W * 0.22, W * 0.16]),
           N('Sizes: chest under 84 cm XS, 84 to 92 S, 92 to 100 M, 100 to 108 L, 108 to 116 XL, above XXL; waist bands '
             '70, 78, 86, 94, 102 cm.'),
           KeepTogether([Paragraph('TailorCam person by person', HK),
                         figure(f['scatter'], W, 'Figure 4. Tape against TailorCam; dots on the grey line would be perfect.')]),
           figure(f['signed'], W, 'Figure 5. Too big or too small for every person.'),
           figure(f['persons'], W, 'Figure 6. Error of every person and body part.'),
           Paragraph('What the results show', HK),
           P(f"&bull; TailorCam is the best of all models on {best_parts} and close to the best on "
             f"{close_parts}.<br/>"
             f"&bull; {best_single} alone is {'slightly ahead' if ov[best_single] > ov[TC] else 'slightly behind'} overall "
             f"({ov[best_single]:.1f}% against {ov[TC]:.1f}%): without the weight, the 3D model that reads body shape from the "
             "photo carries most of the information. The difference is small for a dataset this size.<br/>"
             f"&bull; The shoulder pulls TailorCam down ({val(s, TC, 'shoulder_width', 'MAE'):.2f} cm against "
             f"{val(s, best_single, 'shoulder_width', 'MAE'):.2f} cm for {best_single}): the app has no learned rule for the "
             "shoulder yet, so it shows HMR's value directly.<br/>"
             "&bull; The tape values were taken by two different measurers, who place the tape a little differently on the "
             "waist, arm and shoulder; one set of combined rules cannot fit both at once, which is why the app's shop offset "
             "exists.<br/>"
             "&bull; The height-and-weight models are clearly weaker here, because without the weight they only know the "
             "height."),
           Paragraph('Limits', HK),
           P(f"&bull; {n} people is still a small dataset, so small differences between models may not hold on new people.<br/>"
             "&bull; No weight is used; with the weight, the 24 people who have it reach 2.62 cm on the chest (main report).<br/>"
             "&bull; The shoulder has only a few tape values."),
           N('Reproduce: scripts/build_combined_dataset.py, experiments/combined/test.py, scripts/make_combined_report.py.')]
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm,
                            bottomMargin=16 * mm, title='TailorCam combined test', author='Md Ashikuzzaman')
    doc.build(st, onFirstPage=footer, onLaterPages=footer)
    print('saved', OUT)


if __name__ == '__main__':
    build()
