"""
Short, chart-first report for the odjag test (own people, one front photo each,
local tailor's tape): reports/TailorCam_Odjag_Test.pdf

All numbers come from ONE run of experiments/odjag/shop_offset_e2e.py:
  app_data/shop_extra/odjag.csv              every model's answer (components) + tape
  experiments/odjag/shop_offset_e2e.csv  app value, and app + shop offset (leave-one-out)

  python scripts/make_odjag_report.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage, ImageOps
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from make_model_accuracy_pdf import S, P, N, table, figure, COL, GOOD, MUTED

DATA = data_path('odjag')
FIG = os.path.join(ROOT, 'reports', 'figures_odjag')
OUT = os.path.join(ROOT, 'reports', 'TailorCam_Odjag_Test.pdf')
W = A4[0] - 36 * mm
HK = S['h'].clone('h_keep', keepWithNext=1)      # a heading never ends a page
KEYS = ['chest', 'waist', 'hip', 'neck', 'thigh', 'sleeve_length', 'shoulder_width']
LAB = dict(zip(KEYS, ['Chest', 'Waist', 'Hip', 'Neck', 'Thigh', 'Sleeve', 'Shoulder']))
MOD = {'TailorCam': None, 'HMR': 'hmr', 'SHAPY': 'shapy', 'CameraHMR': 'camerahmr', 'Silhouette CNN': 'hybrid',
       'BodyM prior': 'prior', 'ANSUR prior': 'ansur'}
MCOL = {'TailorCam': '#2a78d6', 'HMR': COL['HMR (2018)'], 'SHAPY': COL['SHAPY (2022)'], 'CameraHMR': COL['CameraHMR (2025)'],
        'Silhouette CNN': COL['Silhouette CNN x3'], 'BodyM prior': COL['BodyM prior'], 'ANSUR prior': COL['ANSUR prior']}
BODYM = {'sleeve_length': 'arm-length', 'shoulder_width': 'shoulder-breadth'}


def style(ax, grid='y'):
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    ax.grid(axis=grid, color='#e1e0d9', lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)


def load():
    comp = pd.read_csv(data_path('shop_extra', 'odjag.csv'))
    e2e = pd.read_csv(os.path.join(ROOT, 'experiments', 'odjag', 'shop_offset_e2e.csv'))
    rows = []
    for _, r in comp.iterrows():
        g = e2e[e2e.person_id == r.person_id].set_index('measurement')
        for k in KEYS:
            row = {'person_id': r.person_id, 'measurement': k, 'tape': r[f'true:{k}'], 'TailorCam': g.loc[k, 'app'],
                   'TailorCam + shop offset': g.loc[k, 'app + shop offset']}
            for name, m in MOD.items():
                if m:
                    c = f'{m}:{BODYM.get(k, k)}' if m in ('hybrid', 'prior') else f'{m}:{k}'
                    row[name] = r[c] if c in r and pd.notna(r[c]) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def loo_offset(y, p):
    return np.array([p[i] + np.mean(np.delete(y - p, i)) for i in range(len(y))])


def stats(d):
    """(model, key) -> MAE, accuracy, bias; for raw values and with a leave-one-out shop offset."""
    raw, off = {}, {}
    for k, g in d.groupby('measurement', sort=False):
        y = g.tape.values
        for m in MOD:
            p = g[m].values.astype(float)
            if np.isnan(p).any():
                continue
            e = p - y
            raw[(m, k)] = (np.abs(e).mean(), 100 - (np.abs(e) / y * 100).mean(), e.mean())
            q = g['TailorCam + shop offset'].values if m == 'TailorCam' else loo_offset(y, p)
            e2 = np.abs(q - y)
            off[(m, k)] = (e2.mean(), 100 - (e2 / y * 100).mean())
    return raw, off


def overall(sc):
    return {m: np.mean([sc[(m, k)][1] for k in KEYS if (m, k) in sc]) for m in MOD}


def charts(d, raw, off):
    os.makedirs(FIG, exist_ok=True)
    f = {}
    ims = []
    for pid in d.person_id.unique():
        im = ImageOps.exif_transpose(PILImage.open(os.path.join(DATA, pid, 'front.jpg'))); im.thumbnail((240, 420)); ims.append(im)
    per = 5
    strip = PILImage.new('RGB', (240 * min(per, len(ims)), 420 * ((len(ims) + per - 1) // per)), 'white')
    for i, im in enumerate(ims):
        strip.paste(im, ((i % per) * 240 + (240 - im.width) // 2, (i // per) * 420))
    f['photos'] = os.path.join(FIG, 'photos.png'); strip.save(f['photos'])
    # 1 overall accuracy per model, raw vs shop offset
    o1, o2 = overall(raw), overall(off)
    fig, ax = plt.subplots(figsize=(7.4, 2.7), dpi=200)
    y = np.arange(len(MOD))
    ax.barh(y - 0.2, [o1[m] for m in MOD], 0.38, color='#c3c2b7', label='app as it is')
    ax.barh(y + 0.2, [o2[m] for m in MOD], 0.38, color=[MCOL[m] for m in MOD], label='with shop offset')
    for i, m in enumerate(MOD):
        ax.text(o1[m] + 0.1, i - 0.2, f'{o1[m]:.1f}%', va='center', fontsize=6.5, color='#52514e')
        ax.text(o2[m] + 0.1, i + 0.2, f'{o2[m]:.1f}%', va='center', fontsize=6.5, color='#0b0b0b')
    ax.set_yticks(y); ax.set_yticklabels(list(MOD), fontsize=7.5); ax.invert_yaxis(); ax.set_xlim(85, 99.5)
    ax.set_xlabel('Overall accuracy (%), mean over the body parts each model measures', fontsize=7)
    ax.legend(fontsize=7, frameon=False, loc='lower right'); style(ax, 'x')
    fig.tight_layout(); f['overall'] = os.path.join(FIG, 'overall.png'); fig.savefig(f['overall']); plt.close(fig)
    # 2 error per body part per model
    fig, ax = plt.subplots(figsize=(7.4, 2.9), dpi=200)
    x = np.arange(len(KEYS))
    for i, m in enumerate(MOD):
        ax.bar(x + (i - 3) * 0.12, [raw.get((m, k), (np.nan,))[0] for k in KEYS], 0.11, color=MCOL[m], label=m)
    ax.set_xticks(x); ax.set_xticklabels([LAB[k] for k in KEYS], fontsize=8); ax.set_ylabel('error (cm)', fontsize=7.5)
    ax.legend(fontsize=6.5, ncol=4, frameon=False, loc='upper left'); style(ax)
    ax.set_title('Average error per body part, app as it is (lower is better)', fontsize=8.5, color='#52514e')
    fig.tight_layout(); f['mae'] = os.path.join(FIG, 'mae.png'); fig.savefig(f['mae']); plt.close(fig)
    # 3 bias heat map
    B = np.array([[raw.get((m, k), (np.nan,) * 3)[2] for k in KEYS] for m in MOD])
    fig, ax = plt.subplots(figsize=(7.4, 2.8), dpi=200)
    ax.imshow(B, cmap='RdBu_r', vmin=-12, vmax=12, aspect='auto')
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            ax.text(j, i, '-' if np.isnan(B[i, j]) else f'{B[i, j]:+.1f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(np.nan_to_num(B[i, j])) > 7 else '#0b0b0b')
    ax.set_xticks(range(len(KEYS))); ax.set_xticklabels([LAB[k] for k in KEYS], fontsize=7.5)
    ax.set_yticks(range(len(MOD))); ax.set_yticklabels(list(MOD), fontsize=7.5); ax.tick_params(length=0)
    ax.set_title('Bias (cm): red = model measures too big, blue = too small', fontsize=8.5, color='#52514e')
    for s_ in ax.spines.values():
        s_.set_visible(False)
    fig.tight_layout(); f['bias'] = os.path.join(FIG, 'bias.png'); fig.savefig(f['bias']); plt.close(fig)
    # 4 tape vs TailorCam
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.9), dpi=200)
    for ax, k in zip(axes.flat, KEYS):
        g = d[d.measurement == k]
        lo = min(g.tape.min(), g.TailorCam.min(), g['TailorCam + shop offset'].min()) - 3
        hi = max(g.tape.max(), g.TailorCam.max(), g['TailorCam + shop offset'].max()) + 3
        ax.plot([lo, hi], [lo, hi], color='#898781', lw=0.8)
        ax.scatter(g.tape, g.TailorCam, s=16, color='#2a78d6', zorder=3, label='app as it is')
        ax.scatter(g.tape, g['TailorCam + shop offset'], s=16, color='#1baf7a', marker='s', zorder=3, label='with shop offset')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_title(LAB[k], fontsize=8); style(ax, 'both')
        ax.tick_params(labelsize=5.5)
    axes.flat[-1].axis('off')
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), fontsize=7, frameon=False, loc='center')
    fig.supxlabel('tape (cm)', fontsize=7); fig.supylabel('TailorCam (cm)', fontsize=7)
    fig.tight_layout(); f['scatter'] = os.path.join(FIG, 'scatter.png'); fig.savefig(f['scatter']); plt.close(fig)
    # 5 before / after shop offset
    fig, ax = plt.subplots(figsize=(7.4, 2.5), dpi=200)
    a1 = [raw[('TailorCam', k)][0] for k in KEYS]; a2 = [off[('TailorCam', k)][0] for k in KEYS]
    ax.bar(x - 0.2, a1, 0.38, color='#c3c2b7', label='app as it is'); ax.bar(x + 0.2, a2, 0.38, color='#2a78d6', label='with shop offset')
    for i in range(len(KEYS)):
        ax.text(i - 0.2, a1[i] + 0.15, f'{a1[i]:.1f}', ha='center', fontsize=6.5)
        ax.text(i + 0.2, a2[i] + 0.15, f'{a2[i]:.1f}', ha='center', fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([LAB[k] for k in KEYS], fontsize=8); ax.set_ylabel('error (cm)', fontsize=7.5)
    ax.legend(fontsize=7, frameon=False); style(ax)
    fig.tight_layout(); f['before_after'] = os.path.join(FIG, 'before_after.png'); fig.savefig(f['before_after']); plt.close(fig)
    # 6 signed error of every person
    fig, ax = plt.subplots(figsize=(7.4, 2.5), dpi=200)
    for i, k in enumerate(KEYS):
        g = d[d.measurement == k]
        e = (g.TailorCam - g.tape).values
        ax.scatter(np.full(len(e), i) + np.linspace(-0.15, 0.15, len(e)), e, s=16, zorder=3,
                   color=['#2a78d6' if v > 0 else '#eb6834' for v in e])
    ax.axhline(0, color='#898781', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([LAB[k] for k in KEYS], fontsize=8); ax.set_ylabel('TailorCam - tape (cm)', fontsize=7.5)
    ax.set_title('Each dot is one person: blue = too big, orange = too small', fontsize=8.5, color='#52514e'); style(ax)
    fig.tight_layout(); f['signed'] = os.path.join(FIG, 'signed.png'); fig.savefig(f['signed']); plt.close(fig)
    # 7 per-person error heat map
    pids = list(d.person_id.unique())
    E = np.array([[float(d[(d.person_id == p) & (d.measurement == k)].eval('TailorCam - tape').iloc[0]) for k in KEYS] for p in pids])
    fig, ax = plt.subplots(figsize=(7.4, 2.4), dpi=200)
    ax.imshow(E, cmap='RdBu_r', vmin=-15, vmax=15, aspect='auto')
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            ax.text(j, i, f'{E[i, j]:+.1f}', ha='center', va='center', fontsize=7, color='white' if abs(E[i, j]) > 9 else '#0b0b0b')
    ax.set_xticks(range(len(KEYS))); ax.set_xticklabels([LAB[k] for k in KEYS], fontsize=7.5)
    ax.set_yticks(range(len(pids))); ax.set_yticklabels(pids, fontsize=7.5); ax.tick_params(length=0)
    ax.set_title('Error of every person (cm), app as it is: red = too big, blue = too small', fontsize=8.5, color='#52514e')
    for s_ in ax.spines.values():
        s_.set_visible(False)
    fig.tight_layout(); f['persons'] = os.path.join(FIG, 'persons.png'); fig.savefig(f['persons']); plt.close(fig)
    return f, o1, o2


def footer(c, doc):
    c.saveState(); c.setFont('UI', 7.5); c.setFillColor(MUTED)
    c.drawString(18 * mm, 10 * mm, 'TailorCam odjag test · Md Ashikuzzaman (2022-3-60-101) · East West University')
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f'Page {doc.page}'); c.restoreState()


def build():
    d = load()
    raw, off = stats(d)
    f, o1, o2 = charts(d, raw, off)
    n = d.person_id.nunique()
    best_raw = max((m for m in MOD if m != 'TailorCam'), key=lambda m: o1[m])
    best_off = max(MOD, key=lambda m: o2[m])
    tc = lambda k: raw[('TailorCam', k)][0]
    b = lambda m, k: raw[(m, k)][2]
    st = [Paragraph('TailorCam: Test on Our Own Photos', S['title']), Spacer(1, 3),
          Paragraph(f'{n} people, one front photo each, tape by a local tailor, no weight · Md Ashikuzzaman, ID 2022-3-60-101', S['sub']),
          Spacer(1, 6)]
    box = Table([[Paragraph(
        f"<b><font color='#1f8a54'>Result.</font></b> App as it is: <b>{o1['TailorCam']:.1f}%</b> overall. With a shop offset "
        f"learned from the other customers: <b>{o2['TailorCam']:.1f}%</b>. Chest, hip and neck are good straight away "
        f"({tc('chest'):.1f}, {tc('hip'):.1f} and {tc('neck'):.1f} cm error); waist, thigh, sleeve and shoulder are off by a fixed "
        "amount because this tailor tapes them differently.", S['box'])]], colWidths=[W])
    box.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#e7f5ec')), ('LINEBEFORE', (0, 0), (0, -1), 3, GOOD),
                             ('LEFTPADDING', (0, 0), (-1, -1), 9), ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
    st += [box, figure(f['photos'], W, f'The {n} test people.'),
           figure(f['overall'], W, 'Figure 1. Overall accuracy of every model: app as it is (grey) and with a shop offset (colour).')]
    cw = [W * 0.17] + [W * 0.1] * 7 + [W * 0.13]
    rows = [['Model'] + [LAB[k] for k in KEYS] + ['Overall']]
    rows2 = [['Model'] + [LAB[k] for k in KEYS] + ['Overall']]
    for m in MOD:
        rows.append([m] + [f"{raw[(m, k)][0]:.2f}" if (m, k) in raw else '-' for k in KEYS] + [f"{o1[m]:.1f}%"])
        rows2.append([m] + [f"{off[(m, k)][0]:.2f}" if (m, k) in off else '-' for k in KEYS] + [f"{o2[m]:.1f}%"])
    st += [Paragraph('Error of every model (cm), app as it is', HK), table(rows, cw, hl=(1,)),
           figure(f['mae'], W, 'Figure 2. Average error per body part and model.'),
           Paragraph('Error of every model (cm), with a shop offset', HK), table(rows2, cw, hl=(1,)),
           figure(f['before_after'], W, f'Figure 3. TailorCam before and after the shop offset (learned on the other {n - 1} people).')]
    st += [KeepTogether([Paragraph('Why the combined result is lower than some single models', HK),
                         figure(f['bias'], W, f'Figure 4. Bias of every model on these {n} people.')]),
           P(f"&bull; <b>Sleeve:</b> the combiner's rule adds 5.48 cm to HMR, because HMR measured short on the 24 people it "
             f"learned from. Here HMR is already {b('HMR', 'sleeve_length'):+.1f} cm, so TailorCam ends at "
             f"{b('TailorCam', 'sleeve_length'):+.1f} cm.<br/>"
             f"&bull; <b>Thigh:</b> the rule leans partly on the ANSUR prior, which is {b('ANSUR prior', 'thigh'):+.1f} cm here, so "
             f"TailorCam ends at {b('TailorCam', 'thigh'):+.1f} cm while HMR alone is {b('HMR', 'thigh'):+.1f} cm.<br/>"
             f"&bull; <b>Waist:</b> the no-weight rule is built around CameraHMR, which is {b('CameraHMR', 'waist'):+.1f} cm here.<br/>"
             f"&bull; Single models have no such learned corrections, so on these body parts they happen to sit closer to this "
             f"tailor. Best single model as it is: {best_raw} ({o1[best_raw]:.1f}%). With a shop offset the models land between "
             f"{min(o2.values()):.1f}% and {max(o2.values()):.1f}% (best: {best_off}, {o2[best_off]:.1f}%).")]
    st += [KeepTogether([Paragraph('Why the accuracy is only this much', HK),
                         figure(f['signed'], W, 'Figure 5. Error of every person: waist, thigh, sleeve and shoulder go one way for nearly everyone.')]),
           figure(f['scatter'], W, 'Figure 6. Tape against TailorCam; dots on the grey line would be perfect.'),
           figure(f['persons'], W, 'Figure 7. Error of every person and body part.'),
           P("&bull; <b>Different tape style:</b> most of the error is a fixed shift, not random noise (Figures 4 and 5); the "
             "shop offset removes it.<br/>"
             "&bull; <b>No weight:</b> the app used its weaker no-weight rules. On the 24-person test the weight lowered the chest "
             "error from 3.5 to 2.6 cm.<br/>"
             "&bull; <b>Front photo only:</b> the silhouette CNN needs a front and a side outline, so here it gives the same value "
             "as the height-and-weight prior.<br/>"
             "&bull; <b>Normal clothes and arms down</b> hide the waist and hip line.<br/>"
             f"&bull; <b>Only {n} people</b>, and some handwritten values were hard to read, so one unusual person moves the "
             "average a lot.")]
    st += [N(f'All values in cm (inches x 2.54). Shop offset: learned on {n - 1} people, tested on the remaining one, repeated for all {n}. '
             'Reproduce: experiments/odjag/shop_offset_e2e.py, scripts/make_odjag_report.py.')]
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm,
                            bottomMargin=16 * mm, title='TailorCam odjag test', author='Md Ashikuzzaman')
    doc.build(st, onFirstPage=footer, onLaterPages=footer)
    print('saved', OUT)


if __name__ == '__main__':
    build()
