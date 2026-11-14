"""
Model-by-model accuracy report, following the supervisor's 13 points:
reports/TailorCam_Model_Accuracy.pdf

Needs (run first):
  python scripts/model_accuracy.py      per model / body part accuracy + data sheets
  python scripts/model_timing.py        time per model
  python scripts/individual_models.py   other test sets (HBW, model agency)

  python scripts/make_model_accuracy_pdf.py
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, 'reports', 'paper', 'tables')
FIG = os.path.join(ROOT, 'reports', 'figures_model_accuracy')
OUT = os.path.join(ROOT, 'reports', 'TailorCam_Model_Accuracy.pdf')
FONTS = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts')
pdfmetrics.registerFont(TTFont('UI', os.path.join(FONTS, 'segoeui.ttf')))
pdfmetrics.registerFont(TTFont('UI-B', os.path.join(FONTS, 'segoeuib.ttf')))
pdfmetrics.registerFont(TTFont('UI-I', os.path.join(FONTS, 'segoeuii.ttf')))
pdfmetrics.registerFontFamily('UI', normal='UI', bold='UI-B', italic='UI-I')

INK, MUTED, LINE, GOOD = (colors.HexColor(c) for c in ('#18202b', '#5a6676', '#d7dde6', '#1f8a54'))
S = {
    'title': ParagraphStyle('t', fontName='UI-B', fontSize=19, leading=23, textColor=INK),
    'sub': ParagraphStyle('s', fontName='UI', fontSize=9.5, leading=13, textColor=MUTED),
    'h': ParagraphStyle('h', fontName='UI-B', fontSize=12.5, leading=16, textColor=INK, spaceBefore=10, spaceAfter=4),
    'p': ParagraphStyle('p', fontName='UI', fontSize=9.8, leading=13.8, textColor=INK, spaceAfter=5),
    'cell': ParagraphStyle('c', fontName='UI', fontSize=8, leading=10, textColor=INK),
    'small': ParagraphStyle('sm', fontName='UI', fontSize=8, leading=10.5, textColor=MUTED, spaceAfter=4),
    'cap': ParagraphStyle('cap', fontName='UI-I', fontSize=8, leading=10.5, textColor=MUTED, spaceBefore=2, spaceAfter=6),
    'box': ParagraphStyle('b', fontName='UI', fontSize=9.8, leading=13.8, textColor=INK),
}
P = lambda t: Paragraph(t, S['p'])
N = lambda t: Paragraph(t, S['small'])

PARTS = ['Chest', 'Belly / waist', 'Hip', 'Shoulder', 'Arm (sleeve)', 'Neck', 'Thigh']
COMB = 'Combined (TailorCam hybrid)'
MODELS = ['HMR (2018)', 'SHAPY (2022)', 'CameraHMR (2025)', 'Silhouette CNN x3', 'BodyM prior', 'ANSUR prior', COMB]
# fixed categorical order (dataviz default palette); colour follows the model everywhere
COL = dict(zip(MODELS, ['#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#2a78d6']))
TIME_KEY = {'HMR (2018)': 'HMR (2018)', 'SHAPY (2022)': 'SHAPY (2022)', 'CameraHMR (2025)': 'CameraHMR (2025)',
            'Silhouette CNN x3': 'Silhouette CNN x3 (ours)', 'BodyM prior': 'BodyM prior (ours)',
            'ANSUR prior': 'ANSUR prior (ours)'}
plt.rcParams.update({'font.family': 'Segoe UI', 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#52514e',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'text.color': '#0b0b0b'})


def load():
    acc = {m: pd.read_csv(os.path.join(TAB, f'model_accuracy_{m}.csv'), index_col=0) for m in ('with_weight', 'no_weight')}
    mae = {m: pd.read_csv(os.path.join(TAB, f'model_mae_{m}.csv'), index_col=0) for m in ('with_weight', 'no_weight')}
    n = pd.read_csv(os.path.join(TAB, 'model_parts_long_with_weight.csv')).groupby('part')['n'].max()
    t = pd.read_csv(os.path.join(TAB, 'model_timing.csv')).set_index('part')
    ext = pd.read_csv(os.path.join(TAB, 'individual_models_external.csv'))
    ext = ext[~ext.model.str.startswith('HMR 2.0')]
    return acc, mae, n, t, ext


def model_time(t):
    """Seconds per customer for each model on its own (+ the shared photo step)."""
    photo = t.loc['Photo step (MediaPipe, shared)', 'mean_s']
    own = {m: t.loc[k, 'mean_s'] for m, k in TIME_KEY.items()}
    # the hybrid runs every model once, then the combiner
    own[COMB] = sum(own.values()) + t.loc['Combiner (ours)', 'mean_s']
    sd = {m: t.loc[k, 'sd_s'] for m, k in TIME_KEY.items()}
    sd[COMB] = float(np.sqrt(sum(v ** 2 for v in sd.values())))
    needs_photo = {m: m not in ('BodyM prior', 'ANSUR prior') for m in MODELS}
    total = {m: own[m] + (photo if needs_photo[m] else 0) for m in MODELS}
    return photo, own, sd, total


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='x', color='#e1e0d9', lw=0.6); ax.set_axisbelow(True)


# ── charts ────────────────────────────────────────────────────────────────────
def chart_heatmap(acc, path, title):
    a = acc.loc[MODELS, PARTS]
    fig, ax = plt.subplots(figsize=(7.4, 3.0), dpi=200)
    v = a.values
    ax.imshow(np.where(np.isnan(v), np.nan, v), cmap='Blues', vmin=88, vmax=98.5, aspect='auto')
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isnan(v[i, j]):
                ax.text(j, i, 'n/a', ha='center', va='center', fontsize=7, color='#898781')
            else:
                best = v[i, j] == np.nanmax(v[:, j])
                ax.text(j, i, f'{v[i, j]:.1f}', ha='center', va='center', fontsize=7.5,
                        color='#ffffff' if v[i, j] > 95.5 else '#0b0b0b', fontweight='bold' if best else 'normal')
    ax.set_xticks(range(len(PARTS))); ax.set_xticklabels(PARTS, fontsize=7.5)
    ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(MODELS, fontsize=7.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=8.5, color='#52514e')
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def chart_parts(mae, path):
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.2), dpi=200)
    for ax, part in zip(axes.flat, PARTS):
        s = mae.loc[MODELS, part].dropna()
        y = range(len(s))
        ax.barh(list(y), s.values, color=[COL[m] for m in s.index], height=0.7, edgecolor='#fcfcfb', linewidth=1)
        for i, (m, val) in enumerate(s.items()):
            ax.text(val + 0.08, i, f'{val:.2f}', va='center', fontsize=6, color='#52514e',
                    fontweight='bold' if val == s.min() else 'normal')
        ax.set_yticks(list(y)); ax.set_yticklabels([m.replace(' (TailorCam hybrid)', '') for m in s.index], fontsize=6)
        ax.invert_yaxis(); ax.set_xlim(0, max(5.5, s.max() * 1.25)); ax.tick_params(axis='x', labelsize=6)
        ax.set_title(part, fontsize=8, color='#0b0b0b'); style(ax)
    ax = axes.flat[-1]; ax.axis('off')
    ax.text(0, 0.5, 'Average error in cm\n(lower is better).\nBold = best for\nthat body part.', fontsize=7,
            color='#52514e', va='center')
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def chart_overall(acc, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.5), dpi=200, sharey=True)
    for ax, mode, ttl in zip(axes, ('with_weight', 'no_weight'), ('Weight entered', 'No weight')):
        s = acc[mode].loc[MODELS, 'Overall (all parts it measures)']
        ax.barh(range(len(MODELS)), s.values - 85, left=85, color=[COL[m] for m in MODELS], height=0.68,
                edgecolor='#fcfcfb', linewidth=1)
        for i, v in enumerate(s.values):
            ax.text(v + 0.1, i, f'{v:.2f}%', va='center', fontsize=6.5, color='#52514e',
                    fontweight='bold' if v == s.max() else 'normal')
        ax.set_xlim(85, 98); ax.set_title(ttl, fontsize=8.5); ax.tick_params(labelsize=6.5); style(ax)
    axes[0].set_yticks(range(len(MODELS))); axes[0].set_yticklabels(MODELS, fontsize=6.8); axes[0].invert_yaxis()
    fig.supxlabel('Overall accuracy (%), mean of the body parts each model measures', fontsize=7, color='#52514e')
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def chart_time(total, own, photo, path):
    fig, ax = plt.subplots(figsize=(7.4, 2.4), dpi=200)
    for i, m in enumerate(MODELS):
        p = total[m] - own[m]
        if p:
            ax.barh(i, p, color='#c3c2b7', height=0.65, edgecolor='#fcfcfb', linewidth=1)
        ax.barh(i, own[m], left=p, color=COL[m], height=0.65, edgecolor='#fcfcfb', linewidth=1)
        ax.text(total[m] + 0.05, i, f'{total[m]:.2f} s', va='center', fontsize=6.5, color='#52514e')
    ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(MODELS, fontsize=6.8); ax.invert_yaxis()
    ax.set_xlabel('Seconds for one customer (2 photos). Grey = shared photo step (MediaPipe)', fontsize=7)
    ax.tick_params(labelsize=6.5); style(ax)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def chart_trade(acc, total, path):
    fig, ax = plt.subplots(figsize=(7.4, 2.7), dpi=200)
    for m in MODELS:
        a = acc.loc[m, 'Overall (chest, belly, hip)']
        ax.scatter(total[m], a, s=46, color=COL[m], edgecolor='#fcfcfb', linewidth=1.5, zorder=3)
        ax.annotate(m.replace(' (TailorCam hybrid)', ''), (total[m], a), xytext=(6, 3), textcoords='offset points',
                    fontsize=6.5, color='#52514e')
    ax.set_xscale('log'); ax.set_xlabel('Seconds per customer (log scale)', fontsize=7)
    ax.set_ylabel('Accuracy, chest/belly/hip (%)', fontsize=7); ax.tick_params(labelsize=6.5)
    ax.grid(color='#e1e0d9', lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_title('No weight entered: accuracy against time', fontsize=8.5, color='#52514e')
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def chart_vs(mae, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6), dpi=200, sharey=True)
    for ax, mode, ttl in zip(axes, ('with_weight', 'no_weight'), ('Weight entered', 'No weight')):
        single = mae[mode].loc[MODELS[:-1], PARTS]
        best = single.min()
        comb = mae[mode].loc[COMB, PARTS]
        y = np.arange(len(PARTS))
        for i in y:
            ax.plot([best.iloc[i], comb.iloc[i]], [i, i], color='#c3c2b7', lw=1.5, zorder=1)
        ax.scatter(best.values, y, s=34, color='#898781', label='Best single model', zorder=3,
                   edgecolor='#fcfcfb', linewidth=1.2)
        ax.scatter(comb.values, y, s=34, color=COL[COMB], label='Combined hybrid', zorder=3,
                   edgecolor='#fcfcfb', linewidth=1.2)
        ax.set_title(ttl, fontsize=8.5); ax.tick_params(labelsize=6.5); style(ax)
        ax.set_xlabel('Average error (cm)', fontsize=7)
    axes[0].set_yticks(range(len(PARTS))); axes[0].set_yticklabels(PARTS, fontsize=7); axes[0].invert_yaxis()
    axes[1].legend(fontsize=6.5, frameon=False, loc='lower right')
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ── tables ────────────────────────────────────────────────────────────────────
def table(rows, widths, hl=(), bold_best=None):
    data = [[Paragraph(f'<b>{c}</b>' if i == 0 else str(c), S['cell']) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign='LEFT', repeatRows=1)
    st = [('GRID', (0, 0), (-1, -1), 0.4, LINE), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
          ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f4f8')),
          ('TOPPADDING', (0, 0), (-1, -1), 2.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
          ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4)]
    st += [('BACKGROUND', (0, r), (-1, r), colors.HexColor('#eaf1fd')) for r in hl]
    t.setStyle(TableStyle(st))
    return t


def fmt(v, d=1, suf=''):
    return '-' if pd.isna(v) else f'{v:.{d}f}{suf}'


def grid(df, cols, best='max', d=1, suf=''):
    """Rows = models; the best value in each column is bold."""
    out = []
    for m in MODELS:
        r = [m]
        for c in cols:
            v = df.loc[m, c]
            b = df.loc[MODELS, c].max() if best == 'max' else df.loc[MODELS, c].min()
            s = fmt(v, d, suf)
            r.append(f'<b>{s}</b>' if not pd.isna(v) and np.isclose(v, b) else s)
        out.append(r)
    return out


def figure(path, w, cap):
    iw, ih = ImageReader(path).getSize()
    return KeepTogether([Image(path, width=w, height=w * ih / iw), Paragraph(cap, S['cap'])])


def footer(c, doc):
    c.saveState(); c.setFont('UI', 7.5); c.setFillColor(MUTED)
    c.drawString(18 * mm, 10 * mm, 'TailorCam model accuracy report · Md Ashikuzzaman (2022-3-60-101) · East West University')
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f'Page {doc.page}'); c.restoreState()


def build():
    os.makedirs(FIG, exist_ok=True)
    acc, mae, n, t, ext = load()
    photo, own, sd, total = model_time(t)
    W = A4[0] - 36 * mm
    f = {k: os.path.join(FIG, f'{k}.png') for k in ('heat_w', 'heat_nw', 'parts_w', 'overall', 'time', 'trade', 'vs')}
    chart_heatmap(acc['with_weight'], f['heat_w'], 'Accuracy (%) per body part, weight entered. Bold = best in the column')
    chart_heatmap(acc['no_weight'], f['heat_nw'], 'Accuracy (%) per body part, no weight. Bold = best in the column')
    chart_parts(mae['with_weight'], f['parts_w'])
    chart_overall(acc, f['overall'])
    chart_time(total, own, photo, f['time'])
    chart_trade(acc['no_weight'], total, f['trade'])
    chart_vs(mae, f['vs'])

    aw, anw = acc['with_weight'], acc['no_weight']
    six = aw.loc[MODELS[:-1]]
    best_single_w = six[six['Parts measured'] >= 6]['Overall (all parts it measures)'].idxmax()
    ov_w = aw.loc[MODELS, 'Overall (all parts it measures)']
    wins_nw = sum(round(mae['no_weight'].loc[COMB, p], 2) <= round(mae['no_weight'].loc[MODELS[:-1], p].min(), 2)
                  for p in PARTS)
    gpu = t['device'].iloc[0]

    st = [Paragraph('TailorCam: Accuracy of Every Model, Body Part by Body Part', S['title']), Spacer(1, 3),
          Paragraph('Answer to the supervisor\'s 13 points · Md Ashikuzzaman, ID 2022-3-60-101 · Supervisor: '
                    'Dr. Mohammad Mahdi Hasan · East West University · October 2026', S['sub']), Spacer(1, 7)]
    box = Table([[Paragraph(
        '<b><font color="#1f8a54">Short answer.</font></b> Six models were tested one by one on 24 real people measured '
        'with a tape, on 7 body parts, and then the combined hybrid was tested the same way. '
        f'<b>Without the weight</b> (the normal case for a walk-in customer) the hybrid is clearly best: overall '
        f'{anw.loc[COMB, "Overall (all parts it measures)"]:.2f}% against {anw.loc[MODELS[:-1], "Overall (all parts it measures)"].max():.2f}% '
        f'for the best single model (CameraHMR), and best or equal best on {wins_nw} of 7 body parts. '
        f'<b>With the weight</b> entered all models are close ({ov_w.min():.1f}% to {ov_w.max():.1f}% overall); the hybrid '
        f'({aw.loc[COMB, "Overall (all parts it measures)"]:.2f}%) is in the middle of the group: it is by far the best on the '
        'thigh, but single models are a little better on shoulder and arm. The hybrid takes the longest '
        f'(about {total[COMB]:.1f} s per customer on an {gpu.replace("NVIDIA GeForce ", "")}), which is still fast enough for a shop.',
        S['box'])]], colWidths=[W])
    box.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#e7f5ec')),
                             ('LINEBEFORE', (0, 0), (0, -1), 3, GOOD), ('LEFTPADDING', (0, 0), (-1, -1), 9),
                             ('RIGHTPADDING', (0, 0), (-1, -1), 9), ('TOPPADDING', (0, 0), (-1, -1), 6),
                             ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
    st += [box]

    # 1. data sheets
    st += [Paragraph('1. Data sheets for each model', S['h']),
           P('All data sheets are in one Excel file, <b>reports/TailorCam_Model_Data.xlsx</b>. There is one sheet per model '
             'and per mode (weight entered, "W"; no weight, "noW"), 14 sheets, plus summary sheets. Each sheet has one row '
             'per person: gender, height, weight, and for every body part the tape value, the model\'s raw value, the '
             'final predicted value, the error and the accuracy. The test people are the 24 real people from the Unidata '
             'samples (front and side photo, tape measurements), in their normal clothes.'),
           table([['Body part', 'Tape values available', 'Note'],
                  ['Chest', f'{n["Chest"]} people', ''], ['Belly / waist', f'{n["Belly / waist"]} people',
                                                          'the dataset measures the waist at the belly line'],
                  ['Hip', f'{n["Hip"]} people', ''],
                  ['Shoulder', f'{n["Shoulder"]} people', 'very few, so these results are only a rough guide'],
                  ['Arm (sleeve)', f'{n["Arm (sleeve)"]} people', 'shoulder to wrist'],
                  ['Neck', f'{n["Neck"]} people', ''], ['Thigh', f'{n["Thigh"]} people', '']],
                 [W * 0.22, W * 0.25, W * 0.53])]

    # 2. how each model measures
    st += [Paragraph('2. How each model calculates the body measurements', S['h']),
           table([['Model', 'Input it uses', 'How it gets the measurement', 'Body parts'],
                  ['HMR (2018)', 'photos, height, weight', '3D body from each photo; volume fitted to the weight; '
                   'virtual tape around the body', '7'],
                  ['SHAPY (2022)', 'photos, height, weight', '3D body (SMPL-X) from the photos; same virtual tape',
                   'chest, belly, hip'],
                  ['CameraHMR (2025)', 'photos, height, weight', 'newest 3D body model; same virtual tape', '7'],
                  ['Silhouette CNN x3', 'front + side outline, height, weight', '3 neural networks (ResNet-18) trained on '
                   'Amazon BodyM; average of the 3', '6 (no neck)'],
                  ['BodyM prior', 'height, weight, gender', 'regression trained on BodyM (2,505 people)', '6 (no neck)'],
                  ['ANSUR prior', 'height, weight, gender, age', 'regression trained on ANSUR II (6,068 soldiers)', '7'],
                  [COMB, 'all of the above', 'for each body part a learned rule picks and corrects or blends the models',
                   '7']],
                 [W * 0.2, W * 0.22, W * 0.43, W * 0.15], hl=(7,)),
           N('Every model gets the same fair treatment: a small correction (offset or scale) is learned on training '
             'people only and kept only if it lowers the error on test people, exactly the rule the app uses. Testing '
             'is nested cross-validation: 6 folds, repeated with 5 random splits, so nobody is tested by a model that '
             'learned from them. Shoulder: the app has no combiner (too few tape values), so the hybrid shows HMR\'s value.')]

    # 3-4. individual body part accuracy
    st += [Paragraph('3 and 4. Accuracy of each body part, for every model', S['h']),
           P('Accuracy = 100 minus the average percentage error. Average error (MAE) is in cm. Bold is the best model '
             'for that body part. "-" means the model does not measure that part.'),
           Paragraph('<b>Weight entered: accuracy (%)</b>', S['p']),
           table([['Model'] + PARTS] + grid(aw, PARTS, 'max', 1, '%'), [W * 0.23] + [W * 0.11] * 7, hl=(7,)),
           Spacer(1, 4), Paragraph('<b>Weight entered: average error (cm)</b>', S['p']),
           table([['Model'] + PARTS] + grid(mae['with_weight'], PARTS, 'min', 2), [W * 0.23] + [W * 0.11] * 7, hl=(7,)),
           Spacer(1, 4), Paragraph('<b>No weight: accuracy (%)</b>', S['p']),
           table([['Model'] + PARTS] + grid(anw, PARTS, 'max', 1, '%'), [W * 0.23] + [W * 0.11] * 7, hl=(7,)),
           Spacer(1, 4), Paragraph('<b>No weight: average error (cm)</b>', S['p']),
           table([['Model'] + PARTS] + grid(mae['no_weight'], PARTS, 'min', 2), [W * 0.23] + [W * 0.11] * 7, hl=(7,))]

    # 5. time
    trows = [['Model', 'Own time (s)', 'Spread, sd (s)', 'With photo step (s)', 'Customers per minute']]
    for m in MODELS:
        trows.append([m, f'{own[m]:.3f}', f'{sd[m]:.3f}', f'{total[m]:.2f}', f'{60 / total[m]:.0f}' if total[m] > 0.06 else 'over 1,000'])
    st += [Paragraph('5. Time of each model', S['h']),
           P(f'Time was measured on this PC ({gpu}) for one customer with a front and a side photo, after a warm-up run, '
             f'as the mean of {int(t["runs"].iloc[0])} runs (5 repeats on 4 people). The shared photo step (reading the '
             f'photos and finding the body with MediaPipe) takes {photo:.2f} s and is needed by every photo model. '
             f'Loading all models when the app starts takes {t.loc["Model loading (once, at app start)", "mean_s"]:.0f} s '
             'once. The small spread (sd) shows the times are stable.'),
           table(trows, [W * 0.3, W * 0.15, W * 0.15, W * 0.2, W * 0.2], hl=(7,)),
           figure(f['time'], W, 'Figure 1. Seconds per customer for each model. The two priors need no photos, so they '
                                'are almost instant. The hybrid runs every model once, then the combiner.')]

    # 6. overall
    orows = [['Model', 'Parts', 'Overall, weight', 'Overall, no weight', 'Chest+belly+hip, weight',
              'Chest+belly+hip, no weight']]
    for m in MODELS:
        orows.append([m, f'{int(aw.loc[m, "Parts measured"])}', fmt(aw.loc[m, 'Overall (all parts it measures)'], 2, '%'),
                      fmt(anw.loc[m, 'Overall (all parts it measures)'], 2, '%'),
                      fmt(aw.loc[m, 'Overall (chest, belly, hip)'], 2, '%'), fmt(anw.loc[m, 'Overall (chest, belly, hip)'], 2, '%')])
    st += [Paragraph('6. Overall accuracy of each model', S['h']),
           P('The overall accuracy is the mean of the body-part accuracies a model measures. Because SHAPY only measures '
             '3 parts, the last two columns compare every model on the same 3 parts (chest, belly, hip).'),
           table(orows, [W * 0.26, W * 0.08, W * 0.15, W * 0.17, W * 0.17, W * 0.17], hl=(7,)),
           figure(f['overall'], W, 'Figure 2. Overall accuracy per model. Bold label = highest.')]

    # 7. complete list
    lrows = [['Model', 'Time (s)'] + [p.split(' ')[0] for p in PARTS] + ['Overall']]
    for mode, lab in (('with_weight', 'weight'), ('no_weight', 'no weight')):
        for m in MODELS:
            lrows.append([f'{m}, {lab}', f'{total[m]:.2f}'] + [fmt(acc[mode].loc[m, p], 1) for p in PARTS] +
                         [fmt(acc[mode].loc[m, 'Overall (all parts it measures)'], 2)])
    st += [Paragraph('7. Complete list of results', S['h']),
           P('Time, accuracy (%) of every body part, and overall accuracy, for every model in both modes:'),
           table(lrows, [W * 0.28, W * 0.08] + [W * 0.077] * 7 + [W * 0.1], hl=(7, 14))]

    # 8. charts
    st += [Paragraph('8. Diagrams and charts', S['h']),
           figure(f['heat_w'], W, 'Figure 3. Accuracy heat map, weight entered. Darker = more accurate.'),
           figure(f['heat_nw'], W, 'Figure 4. Accuracy heat map, no weight.'),
           figure(f['parts_w'], W, 'Figure 5. Average error per body part for each model, weight entered.'),
           figure(f['trade'], W * 0.9, 'Figure 6. Accuracy against time, no weight. Up and to the left is better. '
                                       'The hybrid is the most accurate but also the slowest.')]

    # 9. compare per part
    prow = [['Body part', 'Best, weight entered', 'Worst, weight entered', 'Best, no weight', 'Worst, no weight']]
    for p in PARTS:
        r = [p]
        for mode in ('with_weight', 'no_weight'):
            s = mae[mode].loc[MODELS, p].dropna()
            r += [f'{s.idxmin()} ({s.min():.2f})', f'{s.idxmax()} ({s.max():.2f})']
        prow.append(r)
    st += [Paragraph('9. Comparing the models on each body part', S['h']),
           table(prow, [W * 0.13, W * 0.22, W * 0.22, W * 0.22, W * 0.21]),
           N('Average error in cm in brackets.'),
           P('<b>Differences.</b> With the weight entered, the gap between models is small for chest (2.25 to 2.90 cm) '
             'because the weight already tells a lot about the body size. The old HMR is the weakest on most parts. Each '
             'model has its own strength: SHAPY on the chest, the BodyM prior on the belly and arm, the silhouette CNN on the '
             'hip and shoulder, ANSUR on the neck. Without the weight the gaps become large: the models that rely on the '
             'weight (HMR, SHAPY, the CNN, the priors) lose about 2 to 5.5 cm on chest, belly and hip, while CameraHMR stays good because it '
             'reads the body shape from the photo itself.')]

    # 10. individual vs combined
    vrows = [['', 'Best single model', 'Combined hybrid', 'Better']]
    for mode, lab in (('with_weight', 'weight entered'), ('no_weight', 'no weight')):
        s = acc[mode].loc[MODELS[:-1], 'Overall (all parts it measures)']
        sev = acc[mode].loc[MODELS[:-1]]
        sev = sev[sev['Parts measured'] == 7]['Overall (all parts it measures)']
        c = acc[mode].loc[COMB, 'Overall (all parts it measures)']
        vrows.append([f'Overall, {lab} (models with all 7 parts)', f'{sev.idxmax()} {sev.max():.2f}%', f'{c:.2f}%',
                      'Hybrid' if c > sev.max() else 'Single'])
        wins = sum(mae[mode].loc[COMB, p] <= mae[mode].loc[MODELS[:-1], p].min() + 1e-9 for p in PARTS)
        vrows.append([f'Body parts where hybrid is best or equal, {lab}', '', f'{wins} of 7', ''])
    ex = ext.set_index(['dataset', 'model'])
    for ds, lab in (('hbw_val (with weight)', 'HBW 3D scans, weight'), ('hbw_val (no weight)', 'HBW 3D scans, no weight'),
                    ('agency (no weight)', 'Model agency 134 people, no weight')):
        e = ex.loc[ds]
        singles = e.drop(index='Combiner = TailorCam (ours)')[['chest', 'waist', 'hip']].mean(axis=1)
        c = e.loc['Combiner = TailorCam (ours)', ['chest', 'waist', 'hip']].mean()
        vrows.append([f'{lab}, mean error chest/belly/hip', f'{singles.idxmin().replace(" (ours)", "")} {singles.min():.2f} cm',
                      f'{c:.2f} cm', 'Hybrid' if c < singles.min() else 'Single'])
    st += [Paragraph('10. Individual models against the combined hybrid', S['h']),
           table(vrows, [W * 0.42, W * 0.25, W * 0.17, W * 0.16]),
           figure(f['vs'], W, 'Figure 7. For each body part: error of the best single model (grey) and of the hybrid '
                              '(blue). The best single model is a different model for each part.'),
           P('<b>What this shows.</b> No single model is best on every body part, and which one is best changes when the '
             'weight is missing. The hybrid does not always beat the best single model on one body part, because with '
             'only 15 to 20 tape-measured people per part it cannot always learn which model to trust. But it stays within about '
             '0.7 cm of the best, and it is the only approach that stays good in every situation: with weight, without weight, '
             'and on the model agency photos (where it has the lowest error of all). On the HBW scans with the weight '
             'entered, single models (ANSUR, SHAPY) did better than the hybrid; that set has only 10 people.'),
           P('<b>Limits found by this test.</b> Shoulder has only 6 tape values, and there the hybrid just shows HMR\'s value, '
             'which is not the best (the silhouette CNN is better). For the arm, the BodyM prior alone (2.11 cm) is better '
             'than the hybrid (2.72 cm); adding it to the hybrid\'s choices gave 2.61 cm, a small gain. Both will improve '
             'when more tape-measured customers are collected.')]

    # 11. best model
    st += [Paragraph('11. Which model is best', S['h']),
           P(f'<b>Best overall: the combined hybrid (TailorCam).</b> Without the weight it has the highest overall accuracy '
             f'({anw.loc[COMB, "Overall (all parts it measures)"]:.2f}%) and the lowest error on chest (equal with CameraHMR), belly, hip and thigh. '
             f'With the weight, all models are within about 2% and the hybrid is in the middle of the group '
             f'({aw.loc[COMB, "Overall (all parts it measures)"]:.2f}% against {aw.loc[best_single_w, "Overall (all parts it measures)"]:.2f}% '
             f'for {best_single_w}); it is by far the best on the thigh, and its shoulder and arm can still be improved. It is '
             'the most reliable across different test sets, which matters more than a small win on one set.'),
           P('<b>Best single model: CameraHMR (2025).</b> It measures all 7 parts, is the best single model without weight '
             f'and among the best with weight. It is also the slowest single model ({total["CameraHMR (2025)"]:.1f} s).'),
           P('<b>Fastest: the two priors</b> (a few milliseconds), but they only guess from height and weight, so they are '
             'weak without the weight and cannot see the body shape.'),
           P('<b>Weakest: HMR (2018)</b>, the oldest model, on most body parts.')]

    # 12. final per body part
    frows = [['Body part'] + [m.replace(' (TailorCam hybrid)', '').replace('Silhouette ', '').replace(' (20', ' (')
                              for m in MODELS]]
    for p in PARTS:
        r = [p]
        for m in MODELS:
            w_, nw_ = acc['with_weight'].loc[m, p], acc['no_weight'].loc[m, p]
            r.append('-' if pd.isna(w_) else f'{w_:.1f} / {nw_:.1f}')
        frows.append(r)
    r = ['Overall']
    for m in MODELS:
        r.append(f'{aw.loc[m, "Overall (all parts it measures)"]:.1f} / {anw.loc[m, "Overall (all parts it measures)"]:.1f}')
    frows.append(r)
    st += [KeepTogether([Paragraph('12. Final results for every body part', S['h']),
           P('Accuracy (%) of every body part for every model. Each cell: <b>weight entered / no weight</b>.'),
           table(frows, [W * 0.13] + [W * 0.124] * 7, hl=(8,))])]

    # 13. summary
    st += [Paragraph('13. Final summary', S['h']),
           P('1. Six models and one hybrid were tested on the same 24 real tape-measured people, on 7 body parts, with the '
             'same fair method (nested cross-validation), in two modes: with and without the customer\'s weight.'),
           P('2. Every model reaches 89% to 98% accuracy per body part. The body parts that are easiest are the chest and '
             'arm (about 96% to 97%); the hardest are the shoulder and, without weight, the belly.'),
           P('3. Each single model has a different strength, and the best single model changes with the body part and '
             'with the mode. This is why TailorCam combines them.'),
           P(f'4. The combined hybrid is the best overall approach. Without the weight it is clearly the most accurate '
             f'({anw.loc[COMB, "Overall (all parts it measures)"]:.2f}% overall, chest {mae["no_weight"].loc[COMB, "Chest"]:.2f} cm, '
             f'belly {mae["no_weight"].loc[COMB, "Belly / waist"]:.2f} cm, hip {mae["no_weight"].loc[COMB, "Hip"]:.2f} cm). With the '
             f'weight it is close to the best single models ({aw.loc[COMB, "Overall (all parts it measures)"]:.2f}%) and is much better '
             f'on the thigh ({mae["with_weight"].loc[COMB, "Thigh"]:.2f} cm). It is also the most stable on other test sets.'),
           P(f'5. The hybrid is the slowest ({total[COMB]:.1f} s per customer against {total["CameraHMR (2025)"]:.1f} s for '
             'CameraHMR alone), but a few seconds is fine in a tailor shop, so the extra accuracy is worth it.'),
           P('6. The weak points are the shoulder (only 6 tape values) and the arm. More tape-measured customers, which the '
             'app already saves, are the most useful next step.'),
           N('Reproduce: scripts/model_accuracy.py (data sheets and accuracy), scripts/model_timing.py (time), '
             'scripts/individual_models.py (HBW and model agency), scripts/make_model_accuracy_pdf.py (this report).')]

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm,
                            bottomMargin=16 * mm, title='TailorCam Model Accuracy', author='Md Ashikuzzaman')
    doc.build(st, onFirstPage=footer, onLaterPages=footer)
    print('saved', OUT)


if __name__ == '__main__':
    build()
