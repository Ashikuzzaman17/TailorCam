"""
One complete review report for the faculty: reports/TailorCam_Faculty_Report.pdf

Merges the project report, the dataset list, the per-model accuracy report
(supervisor's 13 points), the comparison with papers and software, the phone
version test and the Q&A into one document with a contents page.

Needs (run first):
  python scripts/model_accuracy.py
  python scripts/model_timing.py
  python scripts/individual_models.py

  python scripts/make_faculty_report.py
"""

import os
import sys

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import make_model_accuracy_pdf as M          # charts, styles, tables, timing helpers
from make_model_accuracy_pdf import COMB, MODELS, PARTS, S, fmt, grid, table, figure

OUT = os.path.join(ROOT, 'reports', 'TailorCam_Faculty_Report.pdf')
TAB = M.TAB
FIGP = os.path.join(ROOT, 'reports', 'paper', 'figures')
UI = os.path.join(ROOT, 'reports', 'ui')
INK, MUTED, GOOD = M.INK, M.MUTED, M.GOOD
S = dict(S)
S['h1'] = ParagraphStyle('h1', parent=S['h'], fontSize=14, leading=18, spaceBefore=12, spaceAfter=5)
S['h2'] = ParagraphStyle('h2', parent=S['h'], fontSize=11, leading=14, spaceBefore=8, spaceAfter=3)
S['q'] = ParagraphStyle('q', parent=S['h'], fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=3,
                        textColor=colors.HexColor('#1c5cab'))
for k in ('h1', 'h2', 'q'):
    S[k].keepWithNext = 1                    # a heading never sits alone at the bottom of a page
S['cover_t'] = ParagraphStyle('ct', fontName='UI-B', fontSize=26, leading=31, textColor=INK, alignment=1)
S['cover_s'] = ParagraphStyle('cs', fontName='UI', fontSize=12, leading=17, textColor=MUTED, alignment=1)
S['cover_m'] = ParagraphStyle('cm', fontName='UI', fontSize=11, leading=16, textColor=INK, alignment=1)
P, N = M.P, M.N
LITE = ('hybrid:', 'prior:', 'ansur:')


def H1(t):
    return Paragraph(t, S['h1'])


def H2(t):
    return Paragraph(t, S['h2'])


def box(html, W, bg='#e7f5ec', line=GOOD):
    b = Table([[Paragraph(html, S['box'])]], colWidths=[W])
    b.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(bg)), ('LINEBEFORE', (0, 0), (0, -1), 3, line),
                           ('LEFTPADDING', (0, 0), (-1, -1), 9), ('RIGHTPADDING', (0, 0), (-1, -1), 9),
                           ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
    return b


class Doc(BaseDocTemplate):
    """A4 document whose H1/H2 headings feed the contents page."""

    def __init__(self, path, **kw):
        super().__init__(path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm,
                         bottomMargin=16 * mm, **kw)
        fr = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id='f')
        self.addPageTemplates([PageTemplate('cover', [fr]), PageTemplate('body', [fr], onPage=self.footer)])

    @staticmethod
    def footer(c, doc):
        c.saveState(); c.setFont('UI', 7.5); c.setFillColor(MUTED)
        c.drawString(18 * mm, 10 * mm, 'TailorCam · Md Ashikuzzaman (2022-3-60-101) · East West University')
        c.drawRightString(A4[0] - 18 * mm, 10 * mm, f'Page {doc.page}'); c.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ('h1', 'h2'):
            level = 0 if f.style.name == 'h1' else 1
            text = f.getPlainText()
            key = f'h{self.seq.nextf("toc")}'
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            self.notify('TOCEntry', (level, text, self.page, key))


def lite_results():
    """Phone version: combiner using only the models that fit on a phone (CNN, priors)."""
    import model.calibration as C
    from model_accuracy import cv_predict
    path = os.path.join(TAB, 'phone_lite.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    d0 = pd.read_csv(data_path('predictions', 'real24.csv'))
    names = {'chest': 'Chest', 'waist': 'Belly / waist', 'hip': 'Hip', 'shoulder_width': 'Shoulder',
             'sleeve_length': 'Arm (sleeve)', 'neck': 'Neck', 'thigh': 'Thigh'}
    rows = []
    for mode in ('with_weight', 'no_weight'):
        d = d0[d0['mode'] == mode]
        for t, part in names.items():
            c = [x for x in C.TARGETS[t] if x.startswith(LITE) and x in d and d[x].notna().any()]
            ok = d[c + [f'true:{t}']].notna().all(axis=1)
            X, y = d.loc[ok, c].values.astype(float), d.loc[ok, f'true:{t}'].values.astype(float)
            e = np.abs(cv_predict(X, y) - y).mean(0)
            rows.append({'mode': mode, 'part': part, 'MAE': e.mean(), 'accuracy': 100 - (e / y * 100).mean()})
    r = pd.DataFrame(rows)
    r.round(3).to_csv(path, index=False)
    return r


def build():
    acc, mae, n, t, ext = M.load()
    photo, own, sd, total = M.model_time(t)
    aw, anw = acc['with_weight'], acc['no_weight']
    mw, mnw = mae['with_weight'], mae['no_weight']
    lite = lite_results()
    gpu = t['device'].iloc[0]
    ksum = pd.read_csv(os.path.join(TAB, 'kfold_summary.csv'))
    size = pd.read_csv(os.path.join(TAB, 'size_classification.csv'))
    W = A4[0] - 36 * mm
    os.makedirs(M.FIG, exist_ok=True)
    f = {k: os.path.join(M.FIG, f'{k}.png') for k in ('heat_w', 'heat_nw', 'parts_w', 'overall', 'time', 'trade', 'vs')}
    M.chart_heatmap(aw, f['heat_w'], 'Accuracy (%) per body part, weight entered. Bold = best in the column')
    M.chart_heatmap(anw, f['heat_nw'], 'Accuracy (%) per body part, no weight. Bold = best in the column')
    M.chart_parts(mw, f['parts_w'])
    M.chart_overall(acc, f['overall'])
    M.chart_time(total, own, photo, f['time'])
    M.chart_trade(anw, total, f['trade'])
    M.chart_vs(mae, f['vs'])
    ov = 'Overall (all parts it measures)'
    st = []

    # ── cover ────────────────────────────────────────────────────────────────
    st += [Spacer(1, 40 * mm), Paragraph('TailorCam', S['cover_t']), Spacer(1, 4),
           Paragraph('Body Measurement from Two Photos with a Hybrid of Deep Learning Models', S['cover_s']),
           Spacer(1, 18 * mm),
           Paragraph('Capstone Project Review Report', S['cover_m']), Spacer(1, 10 * mm),
           Paragraph('<b>Md Ashikuzzaman</b><br/>ID 2022-3-60-101<br/>B.Sc. in Computer Science and Engineering '
                     '(Intelligent Systems and Data Science)', S['cover_m']), Spacer(1, 8 * mm),
           Paragraph('Supervisor: <b>Dr. Mohammad Mahdi Hasan</b>', S['cover_m']), Spacer(1, 8 * mm),
           Paragraph('Department of Computer Science and Engineering<br/>East West University, Dhaka, Bangladesh',
                     S['cover_m']), Spacer(1, 8 * mm),
           Paragraph('November 2026', S['cover_m']), Spacer(1, 22 * mm),
           box('<b><font color="#1f8a54">Abstract.</font></b> TailorCam is a desktop application for tailor shops that '
               'measures a customer\'s body from a front and a side photo plus height and weight. It runs six measurement '
               'models (three 3D body models, a silhouette neural network and two statistical models) and combines them '
               'with a learned combiner. On 24 real people measured with a tape, tested with nested cross-validation, the '
               f'final hybrid has an average error of {mw.loc[COMB, "Chest"]:.2f} cm on the chest, '
               f'{mw.loc[COMB, "Belly / waist"]:.2f} cm on the waist and {mw.loc[COMB, "Hip"]:.2f} cm on the hip '
               f'(97.3%, 95.6% and 96.8% accuracy) and picks the right shirt size for 83% of people. Without the weight it '
               f'is the most accurate approach tested ({anw.loc[COMB, ov]:.1f}% overall). Run on the same people, it is more '
               'accurate than the published methods HMR, SHAPY and CameraHMR in 4 of 5 tests.', W),
           NextPageTemplate('body'), PageBreak()]

    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle('t0', fontName='UI-B', fontSize=10, leading=15, leftIndent=0),
                       ParagraphStyle('t1', fontName='UI', fontSize=9, leading=12.5, leftIndent=14, textColor=MUTED)]
    st += [Paragraph('Contents', S['h1'].clone('toc_h', spaceBefore=0)), toc, PageBreak()]

    # ── 1 introduction ───────────────────────────────────────────────────────
    st += [H1('1. Introduction'),
           P('In a tailor shop every customer is measured by hand with a tape. It takes time, and two tailors often get '
             'slightly different numbers for the same person. The goal of this project is to find out how close an '
             'ordinary camera can get to a tape, and to build a working application around the best method.'),
           H2('1.1 Objectives'),
           P('&bull; Measure chest, waist (belly), hip, neck, thigh, shoulder and arm length from a front and a side '
             'photo, plus the customer\'s height and (optionally) weight.<br/>'
             '&bull; Combine several state-of-the-art models into one hybrid that is more reliable than any single model.<br/>'
             '&bull; Test every model and the hybrid fairly on real tape-measured people, and report accuracy, error, '
             'clothing-size accuracy, F1 score and time.<br/>'
             '&bull; Run published methods on the same people and compare them with TailorCam.<br/>'
             '&bull; Deliver a complete shop application: customers, scans, orders, PDF reports.'),
           H2('1.2 Contributions'),
           P('&bull; A hybrid of six measurement models with a per-measurement learned combiner.<br/>'
             '&bull; Weight fitting: every 3D body is adjusted so its volume matches the real weight (density 1,010 kg/m3).<br/>'
             '&bull; A fair evaluation with nested cross-validation, repeated over 5 random splits, on every model separately.<br/>'
             '&bull; A working desktop application with webcam and photo-upload modes and a tape-check label on every value.')]

    # ── 2 system ─────────────────────────────────────────────────────────────
    st += [H1('2. System overview'),
           P('The customer stands in front of a webcam (or uploads two photos), enters height and weight, and the app does '
             'the rest. MediaPipe finds the body in each photo; then six models each produce a set of measurements, and the '
             'combiner gives the final values.'),
           figure(os.path.join(FIGP, 'pipeline.png'), W, 'Figure 1. The TailorCam pipeline.'),
           H2('2.1 The six models and the combiner'),
           table([['#', 'Model', 'Who trained it', 'Input', 'What I did'],
                  ['1', 'HMR (2018)', 'Pre-trained (Kanazawa et al.)', 'photos', 'ported to PyTorch, weight fitting, measuring code'],
                  ['2', 'SHAPY (2022)', 'Pre-trained (Max Planck Inst.)', 'photos', 'made it run on Windows, weight fitting, measuring'],
                  ['3', 'CameraHMR (2025)', 'Pre-trained (Max Planck Inst.)', 'photos', 'MediaPipe person box, mirror-photo averaging, weight fitting'],
                  ['4', 'Silhouette CNN x3', 'Trained by me (BodyM, 2,018 people)', 'front + side outline', '3 ResNet-18 copies, different seeds, averaged'],
                  ['5', 'BodyM prior', 'Trained by me (BodyM)', 'height, weight, gender', 'ridge regression'],
                  ['6', 'ANSUR prior', 'Trained by me (6,068 people)', 'height, weight, gender, age', 'ridge regression'],
                  ['7', 'Combiner', 'Trained by me (24 tape people)', 'outputs of 1 to 6', 'learned rule per measurement']],
                 [W * 0.04, W * 0.17, W * 0.27, W * 0.18, W * 0.34], hl=(7,)),
           P('Training a 3D body model from scratch needs millions of images and weeks of GPU time, so the three 3D models '
             'are pre-trained by their authors and adapted here, which is the normal approach in research.'),
           H2('2.2 How the combiner decides'),
           P('It is not a simple average. For each measurement it tries three kinds of rules on the training people and '
             'keeps the one that predicts best for people it has not seen: (a) keep one model as it is, (b) one model plus '
             'a learned offset, or (c) a ridge-regression blend of several models. A rule is only used if it beats rule (a) '
             'on held-out people. The final rules with the weight entered are:'),
           table([['Measurement', 'Rule chosen', 'Formula'],
                  ['Chest', 'One model + offset', 'SHAPY - 3.55 cm'],
                  ['Waist', 'One model + offset', 'BodyM prior - 4.01 cm'],
                  ['Hip', 'Keep one model', 'Silhouette CNN'],
                  ['Neck', 'One model + offset', 'ANSUR - 2.55 cm'],
                  ['Sleeve', 'One model + offset', 'ANSUR + 1.25 cm'],
                  ['Thigh', 'Blend of 5 models', '0.89 CNN + 0.96 ANSUR + 0.25 BodyM - 0.77 HMR - 0.34 CameraHMR - 0.30 cm']],
                 [W * 0.16, W * 0.24, W * 0.60]),
           N('Without the weight a second set of rules is used, mostly built around CameraHMR. Six ways of combining were '
             'compared (best rule per measurement, positive-only blend, best single model, accuracy weighting, mean, median); '
             'the current method had the lowest error: 2.44 cm average with weight and 3.02 cm without, against 2.62 and '
             '4.32 cm for a simple average.'),
           H2('2.3 The application'),
           Table([[Image(os.path.join(UI, '5_scan_guide.png'), width=W * 0.49, height=W * 0.49 * 720 / 1280),
                   Image(os.path.join(UI, '8_results.png'), width=W * 0.49, height=W * 0.49 * 720 / 1280)]],
                 colWidths=[W * 0.5, W * 0.5], style=[('LEFTPADDING', (0, 0), (-1, -1), 0)]),
           Paragraph('Figure 2. Left: the scan screen guides the customer. Right: results with the 3D body, measurements, '
                     'clothing sizes and the tape-check labels.', S['cap']),
           P('Written in Python with OpenCV and MediaPipe. It saves customers, orders and PDF reports, accepts uploaded '
             'photos, and marks every value "Camera OK" or "Check with tape". One customer takes about '
             f'{total[COMB]:.0f} seconds on an {gpu.replace("NVIDIA GeForce ", "")}.')]

    # ── 3 datasets ───────────────────────────────────────────────────────────
    st += [H1('3. Datasets'),
           P('Very few public datasets have both photos and real tape measurements. Every free dataset found was used, '
             'each for a clearly separate purpose so that no person is ever tested by a model that learned from them.'),
           H2('3.1 Training only'),
           table([['Dataset', 'People', 'Used to train', 'Link'],
                  ['BodyM (Amazon, 2022)', '2,018 (train split)', 'silhouette CNN x3, BodyM prior', 'registry.opendata.aws/bodym'],
                  ['ANSUR II (US Army, 2012)', '6,068', 'ANSUR prior', 'openlab.psu.edu/ansur2'],
                  ['BODY-fit (Tampere Univ., 2021)', '3,013 (train split)', 'BODY-fit network (chapter 9)',
                   'zenodo.org/records/5172126']],
                 [W * 0.26, W * 0.17, W * 0.27, W * 0.30]),
           H2('3.2 Training and testing (nested cross-validation)'),
           table([['Dataset', 'People', 'Used for', 'Link'],
                  ['Unidata real photos, part 1', '6', 'combiner training and main test',
                   'huggingface.co/datasets/ud-biometrics/body-measurements-image-dataset'],
                  ['Unidata real photos, part 2', '21', 'combiner training and main test',
                   'huggingface.co/datasets/UniqueData/body-measurements-dataset']],
                 [W * 0.24, W * 0.08, W * 0.22, W * 0.46]),
           N('Together the "real24" set: 27 downloaded, 24 with usable front and side photos and tape values, normal clothes.'),
           H2('3.3 Testing only (never used for training)'),
           table([['Dataset', 'People', 'Truth from', 'Link'],
                  ['HBW (SHAPY project, Max Planck)', '10', '3D body scans', 'shapy.is.tue.mpg.de'],
                  ['Model agency (SHAPY project)', '134 test', 'agency-listed sizes', 'shapy.is.tue.mpg.de'],
                  ['BodyM test split B', '400', '3D scans', 'registry.opendata.aws/bodym'],
                  ['BEDLAM (synthetic, CVPR 2023)', '42', 'exact 3D body', 'bedlam.is.tue.mpg.de'],
                  ['BODY-fit held-out split', '558', 'scanner measurements', 'zenodo.org/records/5172126'],
                  ['NOMO-3D (Tampere Univ., 2020)', '356', 'real 3D scans + scanner', 'zenodo.org/records/3735905']],
                 [W * 0.34, W * 0.13, W * 0.23, W * 0.30]),
           N('CameraHMR was trained on BEDLAM by its authors, so its BEDLAM score is not a fair test. NHANES (about 28,000 '
             'people, wwwn.cdc.gov/nchs/nhanes) was tested as an extra prior and rejected because it did not help. BODY-rgb '
             '(Tampere University, 194 people) and Fashion-body (Fudan University, 30 people) have been requested.')]

    # ── 4 method ─────────────────────────────────────────────────────────────
    st += [H1('4. Evaluation method'),
           P('<b>Nested cross-validation.</b> The 24 real people are split into 6 groups. Each group is tested once while '
             'the system learns only from the other 5, and the choice of rule is also made inside those 5. Because 24 '
             'people is a small number, this is repeated with 5 different random splits and the errors are averaged. A '
             'standard 5-fold version is also reported fold by fold. Every single model gets the same treatment: a small '
             'correction is learned on the training people and kept only if it lowers the test error, exactly as the app '
             'does.'),
           P('<b>Metrics.</b> Mean absolute error (MAE) in cm; accuracy = 100 minus the mean absolute percentage error; '
             'share of people within 2 cm; clothing-size accuracy with precision, recall and F1; Wilcoxon signed-rank tests; '
             'Bland-Altman limits of agreement; time per customer.')]

    # ── 5 results of the final hybrid ────────────────────────────────────────
    rows = [['Measurement', 'MAE, weight', 'Accuracy, weight', 'MAE, no weight', 'Accuracy, no weight']]
    for p in PARTS:
        rows.append([p, f'{mw.loc[COMB, p]:.2f} cm', f'{aw.loc[COMB, p]:.1f}%', f'{mnw.loc[COMB, p]:.2f} cm',
                     f'{anw.loc[COMB, p]:.1f}%'])
    rows.append(['Overall', '', f'{aw.loc[COMB, ov]:.2f}%', '', f'{anw.loc[COMB, ov]:.2f}%'])
    kf = ksum[ksum.weight == 'known'].set_index('measurement')
    sz = size[size.dataset.str.startswith('real24')]
    srows = [['Size from', 'Weight', 'Right size', 'Within 1 size', 'Precision', 'Recall', 'F1']]
    for _, r in sz.iterrows():
        srows.append([r['size from'].title(), 'entered' if r.weight == 'weight known' else 'not entered',
                      f'{r["accuracy (%)"]:.0f}%', f'{r["within ±1 size (%)"]:.0f}%', f'{r["precision (macro)"]:.2f}',
                      f'{r["recall (macro)"]:.2f}', f'{r["F1 (macro)"]:.2f}'])
    st += [H1('5. Results of the final hybrid'),
           P('All numbers are on people the system never learned from (24 real tape-measured people, nested CV).'),
           table(rows, [W * 0.24, W * 0.19, W * 0.19, W * 0.19, W * 0.19], hl=(8,)),
           N('Shoulder has only 6 tape values; the app has no combiner for it and shows HMR\'s value.'),
           H2('5.1 Fold by fold (standard 5-fold, weight entered)'),
           table([['Measurement', 'Accuracy mean', 'Spread (sd)', 'MAE mean', 'Within 2 cm']] +
                 [[m, f'{r["accuracy (%) mean"]:.1f}%', f'{r["accuracy (%) std"]:.1f}%', f'{r["MAE (cm) mean"]:.2f} cm',
                   f'{r["within ±2 cm (%) mean"]:.0f}%'] for m, r in kf.iterrows()],
                 [W * 0.24, W * 0.19, W * 0.19, W * 0.19, W * 0.19]),
           H2('5.2 Clothing size'),
           table(srows, [W * 0.13, W * 0.15, W * 0.14, W * 0.15, W * 0.14, W * 0.14, W * 0.15]),
           P('With the weight entered the chest gives the right shirt size for 83% of people and is never more than one '
             'size off. Entering the weight matters: without it chest error rises from 2.5 to 3.4 cm and hip from 3.2 to '
             '4.5 cm.'),
           figure(os.path.join(FIGP, 'bland_altman.png'), W, 'Figure 3. Bland-Altman plots: each dot is one person; dashed '
                                                            'lines show where 95% of the differences from the tape fall.'),
           H2('5.3 Other test sets'),
           table([['Test set', 'Chest', 'Waist', 'Hip'],
                  ['HBW 3D scans, weight entered (10)', '4.20 cm', '5.47 cm', '5.09 cm'],
                  ['HBW 3D scans, no weight (10)', '4.77', '9.36', '3.34'],
                  ['Model agency, no weight (134)', '6.84', '6.48', '8.55'],
                  ['BodyM test B, silhouette CNN (400)', '2.47', '2.75', '2.04']],
                 [W * 0.46, W * 0.18, W * 0.18, W * 0.18]),
           N('Errors are larger on the model agency photos because they are fashion poses and the truth is an agency size '
             'list, not a tape.')]

    # ── 6 individual models (supervisor's 13 points) ─────────────────────────
    trows = [['Model', 'Own time (s)', 'sd (s)', 'With photo step (s)', 'Customers / min']]
    for m in MODELS:
        trows.append([m, f'{own[m]:.3f}', f'{sd[m]:.3f}', f'{total[m]:.2f}',
                      f'{60 / total[m]:.0f}' if total[m] > 0.06 else 'over 1,000'])
    orows = [['Model', 'Parts', 'Overall, weight', 'Overall, no weight', 'Chest+belly+hip, weight',
              'Chest+belly+hip, no weight']]
    for m in MODELS:
        orows.append([m, f'{int(aw.loc[m, "Parts measured"])}', fmt(aw.loc[m, ov], 2, '%'), fmt(anw.loc[m, ov], 2, '%'),
                      fmt(aw.loc[m, 'Overall (chest, belly, hip)'], 2, '%'),
                      fmt(anw.loc[m, 'Overall (chest, belly, hip)'], 2, '%')])
    prow = [['Body part', 'Best, weight', 'Worst, weight', 'Best, no weight', 'Worst, no weight']]
    for p in PARTS:
        r = [p]
        for mode in ('with_weight', 'no_weight'):
            s = mae[mode].loc[MODELS, p].dropna()
            r += [f'{s.idxmin()} ({s.min():.2f})', f'{s.idxmax()} ({s.max():.2f})']
        prow.append(r)
    st += [H1('6. Evaluation of every model separately'),
           P('This chapter answers the supervisor\'s request to test every model on its own, body part by body part, '
             'with time, and to compare them with the hybrid. The per-person data sheet of every model is in '
             '<b>reports/TailorCam_Model_Data.xlsx</b> (one sheet per model and mode: tape value, raw value, predicted '
             'value, error and accuracy for each body part).'),
           H2('6.1 Accuracy of each body part'),
           Paragraph('<b>Weight entered: accuracy (%)</b>', S['p']),
           table([['Model'] + PARTS] + grid(aw, PARTS, 'max', 1, '%'), [W * 0.23] + [W * 0.11] * 7, hl=(7,)),
           Spacer(1, 4), Paragraph('<b>Weight entered: average error (cm)</b>', S['p']),
           table([['Model'] + PARTS] + grid(mw, PARTS, 'min', 2), [W * 0.23] + [W * 0.11] * 7, hl=(7,)),
           Spacer(1, 4), Paragraph('<b>No weight: accuracy (%)</b>', S['p']),
           table([['Model'] + PARTS] + grid(anw, PARTS, 'max', 1, '%'), [W * 0.23] + [W * 0.11] * 7, hl=(7,)),
           Spacer(1, 4), Paragraph('<b>No weight: average error (cm)</b>', S['p']),
           table([['Model'] + PARTS] + grid(mnw, PARTS, 'min', 2), [W * 0.23] + [W * 0.11] * 7, hl=(7,)),
           N('Bold = best for that body part. "-" = the model does not measure that part.'),
           figure(f['heat_w'], W, 'Figure 4. Accuracy heat map, weight entered. Darker = more accurate.'),
           figure(f['heat_nw'], W, 'Figure 5. Accuracy heat map, no weight.'),
           figure(f['parts_w'], W, 'Figure 6. Average error per body part for each model, weight entered.'),
           H2('6.2 Time of each model'),
           P(f'Measured on {gpu} for one customer with two photos, after a warm-up run, mean of '
             f'{int(t["runs"].iloc[0])} runs. The shared photo step (MediaPipe) takes {photo:.2f} s. Loading all models at '
             f'app start takes {t.loc["Model loading (once, at app start)", "mean_s"]:.0f} s once.'),
           table(trows, [W * 0.3, W * 0.15, W * 0.12, W * 0.2, W * 0.23], hl=(7,)),
           figure(f['time'], W, 'Figure 7. Seconds per customer. Grey = shared photo step.'),
           H2('6.3 Overall accuracy of each model'),
           P('Mean of the body-part accuracies each model measures. SHAPY measures only 3 parts, so the last two columns '
             'compare all models on the same 3 parts.'),
           table(orows, [W * 0.26, W * 0.08, W * 0.15, W * 0.17, W * 0.17, W * 0.17], hl=(7,)),
           figure(f['overall'], W, 'Figure 8. Overall accuracy per model. Bold label = highest.'),
           figure(f['trade'], W * 0.9, 'Figure 9. Accuracy against time, no weight. Up and to the left is better.'),
           H2('6.4 Comparing the models on each body part'),
           table(prow, [W * 0.13, W * 0.22, W * 0.22, W * 0.22, W * 0.21]),
           N('Average error in cm in brackets.'),
           P('With the weight entered the gap between models is small (chest 2.25 to 2.90 cm), because weight already '
             'tells a lot about body size. Each model has its own strength: SHAPY on the chest, the BodyM prior on the belly '
             'and arm, the silhouette CNN on the hip and shoulder, ANSUR on the neck. Without the weight the gaps become '
             'large: HMR, SHAPY, the CNN and the priors lose about 2 to 5.5 cm on chest, belly and hip, while CameraHMR stays '
             'good because it reads the body shape from the photo itself.'),
           H2('6.5 Individual models against the hybrid'),
           figure(f['vs'], W, 'Figure 10. Error of the best single model (grey) and of the hybrid (blue) for each body part.'),
           P(f'<b>Without the weight</b> the hybrid is clearly best: {anw.loc[COMB, ov]:.2f}% overall against '
             f'{anw.loc[MODELS[:-1], ov].max():.2f}% for the best single model (CameraHMR), and the lowest error on chest '
             '(equal with CameraHMR), belly, hip and thigh. '
             f'<b>With the weight</b> all models are within about 2% ({aw.loc[MODELS, ov].min():.1f}% to '
             f'{aw.loc[MODELS, ov].max():.1f}%) and the hybrid ({aw.loc[COMB, ov]:.2f}%) is in the middle of the group: by far '
             'the best on the thigh, but single models are better on shoulder and arm. On the model agency photos the hybrid '
             'has the lowest error of all; on the 10 HBW scans with weight, SHAPY alone did better.'),
           P('No single model is best on every body part, and the best one changes when the weight is missing. With only '
             '15 to 20 tape-measured people per part the combiner cannot always learn which model to trust, but it stays '
             'within about 0.7 cm of the best and is the only approach that is good in every situation.'),
           H2('6.6 Which model is best'),
           P('<b>Best overall: the hybrid (TailorCam)</b>, because it is the most accurate without weight and the most '
             'reliable across test sets. <b>Best single model: CameraHMR (2025)</b>: all 7 parts, best single model without '
             f'weight, but the slowest ({total["CameraHMR (2025)"]:.1f} s). <b>Fastest: the two priors</b> (milliseconds), but '
             'they cannot see body shape. <b>Weakest: HMR (2018)</b>.')]
    frows = [['Body part'] + [m.replace(' (TailorCam hybrid)', '').replace('Silhouette ', '').replace(' (20', ' (')
                              for m in MODELS]]
    for p in PARTS + [ov]:
        r = ['Overall' if p == ov else p]
        for m in MODELS:
            a, b = aw.loc[m, p], anw.loc[m, p]
            r.append('-' if pd.isna(a) else f'{a:.1f} / {b:.1f}')
        frows.append(r)
    st += [KeepTogether([H2('6.7 Final results for every body part'),
                         P('Accuracy (%) per body part. Each cell: <b>weight entered / no weight</b>.'),
                         table(frows, [W * 0.13] + [W * 0.124] * 7, hl=(8,))])]

    # ── 7 comparison with published work and software ────────────────────────
    # every number below comes from my own runs: the published methods' code and
    # weights run as released, on the same people as TailorCam
    rawp = {m: pd.read_csv(os.path.join(TAB, f'model_raw_mae_{m}.csv'), index_col=0) for m in ('with_weight', 'no_weight')}
    exm = ext.set_index(['dataset', 'model'])
    PUB = [('HMR', 'CVPR 2018', 'HMR (2018)'), ('SHAPY', 'CVPR 2022', 'SHAPY (2022)'),
           ('CameraHMR', '3DV 2025', 'CameraHMR (2025)')]
    CWH = ['Chest', 'Belly / waist', 'Hip']

    def cmp_rows(get, ours):
        rows, avgs = [['Method', 'Published', 'Chest', 'Waist', 'Hip', 'Average', 'TailorCam better?']], {}
        o = np.mean(ours)
        for name, venue, key in PUB:
            v = get(key)
            avgs[name] = np.mean(v)
            rows.append([name, venue] + [f'{x:.2f}' for x in v] + [f'{np.mean(v):.2f}', 'Yes' if o < np.mean(v) else 'No'])
        rows.append(['<b>TailorCam (hybrid)</b>', '2026'] + [f'<b>{x:.2f}</b>' for x in ours] + [f'<b>{o:.2f}</b>', ''])
        return rows, o, avgs

    sets = []
    for mode, lab in (('with_weight', 'weight entered'), ('no_weight', 'no weight')):
        r, o, a = cmp_rows(lambda k: [rawp[mode].loc[k, c] for c in CWH], [mae[mode].loc[COMB, c] for c in CWH])
        sets.append((f'24 real people, tape, {lab}', r, o, a))
    for ds, lab in (('hbw_val (with weight)', 'HBW 3D scans (10 people), weight entered'),
                    ('hbw_val (no weight)', 'HBW 3D scans (10 people), no weight'),
                    ('agency (no weight)', 'Model agency photos (134 people), no weight')):
        e = exm.loc[ds]
        r, o, a = cmp_rows(lambda k: list(e.loc[k, ['chest', 'waist', 'hip']].astype(float)),
                           list(e.loc['Combiner = TailorCam (ours)', ['chest', 'waist', 'hip']].astype(float)))
        sets.append((lab, r, o, a))
    wins = sum(o < min(a.values()) for _, _, o, a in sets)
    summ = [['Test set', 'Best other method (average)', 'TailorCam (average)', 'TailorCam better?']]
    for lab, _, o, a in sets:
        bn = min(a, key=a.get)
        summ.append([lab, f'{bn} {a[bn]:.2f} cm', f'{o:.2f} cm', 'Yes' if o < a[bn] else 'No'])

    st += [H1('7. Comparison with other published methods'),
           P('To compare fairly, I ran the other methods myself, with their authors\' released code and trained weights, '
             'on exactly the same people as TailorCam. The three methods are well-known published 3D body models: HMR '
             '(CVPR 2018), SHAPY (CVPR 2022) and CameraHMR (2025). Each was given the same photos, height and (where '
             'stated) weight, and measured with the same virtual tape. All errors are mean absolute errors in cm, lower '
             'is better, on chest, waist and hip, the three parts every method measures. Small changes needed to run them '
             'here: HMR was converted to PyTorch with its original weights, and CameraHMR uses MediaPipe to find the person.')]
    for i, (lab, r, o, a) in enumerate(sets):
        st += [KeepTogether([H2(f'7.{i + 1} {lab}'),
                             table(r, [W * 0.22, W * 0.13, W * 0.1, W * 0.1, W * 0.1, W * 0.13, W * 0.22], hl=(4,))])]
    st += [KeepTogether([H2(f'7.{len(sets) + 1} Result: is TailorCam better?'),
           table(summ, [W * 0.4, W * 0.24, W * 0.18, W * 0.18]),
           P(f'<b>TailorCam is better than all three published methods in {wins} of {len(sets)} tests.</b> On the 24 real '
             'tape-measured people it has the lowest average error both with and without the weight, and on the model '
             'agency photos it is clearly better than SHAPY and CameraHMR on the waist. The one test where it is not better '
             'is HBW with the weight entered: there SHAPY alone is more accurate. HBW has only 10 people, and its truth comes '
             'from 3D scans rather than a tape, which the combiner was not trained on.')]),
           N('Not compared: commercial apps (3DLOOK, Bodygram, MTailor) are closed and paid, and the code of other papers '
             '(for example Amazon\'s MeasureNet) is not public, so they could not be run on the same people.')]

    # ── 8 phone version ──────────────────────────────────────────────────────
    lw = lite[lite['mode'] == 'with_weight'].set_index('part')
    lnw = lite[lite['mode'] == 'no_weight'].set_index('part')
    lrows = [['Body part', 'Phone, weight', 'PC hybrid, weight', 'Phone, no weight', 'PC hybrid, no weight']]
    for p in PARTS:
        lrows.append([p, f'{lw.loc[p, "MAE"]:.2f}', f'{mw.loc[COMB, p]:.2f}', f'{lnw.loc[p, "MAE"]:.2f}',
                      f'{mnw.loc[COMB, p]:.2f}'])
    lrows.append(['Overall accuracy', f'{lw["accuracy"].mean():.2f}%', f'{aw.loc[COMB, ov]:.2f}%',
                  f'{lnw["accuracy"].mean():.2f}%', f'{anw.loc[COMB, ov]:.2f}%'])
    st += [H1('8. Feasibility of a phone version'),
           P('The large 3D models (CameraHMR, SHAPY) need a graphics card and cannot run on a phone. MediaPipe, the '
             'silhouette CNN, the two priors and the combiner are small and can. A phone-only combiner using just these '
             'models was trained and tested with the same nested cross-validation:'),
           table(lrows, [W * 0.2, W * 0.2, W * 0.2, W * 0.2, W * 0.2], hl=(8,)),
           N('Average error in cm; last row overall accuracy.'),
           P(f'With the weight entered, the phone version is almost as accurate as the PC hybrid '
             f'({lw["accuracy"].mean():.1f}% against {aw.loc[COMB, ov]:.1f}%). Without the weight it drops much more '
             f'({lnw["accuracy"].mean():.1f}%), because CameraHMR is what keeps the PC version accurate then. A phone app '
             'should therefore require the weight. It would guide the customer live (distance, phone tilt, pose, turning '
             'to the side, light, automatic capture) using MediaPipe on the phone. This is planned as future work.')]

    # ── 9 large-scale training (BODY-fit / NOMO-3D) ──────────────────────────
    st += [H1('9. Large-scale training experiment'),
           P('To test whether much more training data helps, a new silhouette network was trained on BODY-fit (Tampere '
             'University, CC BY 4.0): 4,132 real people with 3D body scans and about 50 scanner measurements each, 170 times '
             'more people than the 24 tape-measured photos. Every fitted 3D body was rendered as a front and a side '
             'silhouette, prepared exactly like the app\'s photo masks, and given height and weight (from body volume). Three '
             'copies (ResNet-18, same design as the BodyM network) were trained on the GPU on 3,013 people; 229 were used '
             'for checking during training and 558 kept for testing. 292 people who also appear in NOMO-3D (from the same '
             'lab) were removed, so NOMO-3D is a clean outside test.'),
           H2('9.1 Results on 3D scans'),
           table([['Test set', 'Chest', 'Waist', 'Hip', 'Neck', 'Thigh', 'Shoulder', 'Arm'],
                  ['BODY-fit held-out, 558 people, weight', '2.44', '2.40', '1.74', '2.13', '1.88', '2.43', '2.02'],
                  ['BODY-fit held-out, 558 people, no weight', '3.97', '3.99', '3.18', '2.27', '2.40', '2.64', '2.05'],
                  ['NOMO-3D, 356 new people, no weight', '3.74', '3.90', '3.49', '1.32', '2.61', '1.68', '1.53'],
                  ['NOMO-3D, height and gender only (no photos)', '6.24', '7.91', '4.92', '2.49', '3.68', '2.05', '1.78']],
                 [W * 0.37] + [W * 0.09] * 7, hl=(3,)),
           N('Average error in cm. NOMO-3D is tested without weight because its raw scans have holes, so their volume is '
             'not reliable.'),
           P('On scanned bodies the network works well: on NOMO-3D, people it never saw, it roughly halves the error of a '
             'guess from height and gender alone (chest 3.7 against 6.2 cm, waist 3.9 against 7.9 cm).'),
           H2('9.2 Results on the 24 real people'),
           table([['Body part', 'New network alone', 'Hybrid now', 'Hybrid + new network'],
                  ['Chest', '2.89', '2.51', '2.51'], ['Waist', '4.58', '3.36', '3.36'], ['Hip', '3.70', '3.15', '3.15'],
                  ['Neck', '2.55', '1.37', '1.35'], ['Thigh', '1.89', '1.54', '1.60'], ['Shoulder', '3.00', '3.53', '3.53'],
                  ['Arm (sleeve)', '2.28', '2.72', '2.74'],
                  ['Overall accuracy', '', '95.56%', '95.55%']],
                 [W * 0.25, W * 0.25, W * 0.25, W * 0.25], hl=(8,)),
           N('Weight entered, nested cross-validation, average error in cm. Without weight: 95.22% now, 95.16% with the new '
             'network.'),
           P('<b>Result.</b> Adding the new network to the hybrid did not improve accuracy on real photos (95.56% against '
             '95.55%), so it was not added to the app. The reason is the gap between the two kinds of data: BODY-fit '
             'silhouettes come from scans of people in underwear, while customers wear normal clothes and their outline '
             'comes from MediaPipe. Also, only 10 of the 24 people have photos with the arms away from the body, so for the '
             'other 14 the network can only use height and weight. This confirms that the limit of the project is real photos '
             'of clothed people with tape measurements, not the amount of training data.')]

    # ── 10 what did not work + limits ────────────────────────────────────────
    st += [H1('10. What did not work, and the limitations'),
           P('Fifteen other ideas were tested and left out because they did not help on new people, for example: sharper '
             'outlines with SAM 2, fitting the 3D body to the outline, an NHANES prior (about 28,000 people), training on '
             'all datasets together, synthetic augmentation, averaging all models, and HMR 2.0 (CameraHMR is newer and '
             'built on the same design). Removing each model one at a time showed that every model helps at least one '
             'measurement, so none was removed.'),
           P('<b>Limitations.</b><br/>'
             '&bull; Only 24 free people with photos and tape exist, so differences smaller than about 0.5 cm can be chance.<br/>'
             '&bull; Shoulder has only 6 tape values and arm 18, so these results are a rough guide.<br/>'
             '&bull; Loose clothes hide the body shape; tight clothes give better results.<br/>'
             '&bull; It is not yet ready to replace the tape for fitted clothes; chest, waist and hip still show "Check with '
             'tape". Two tailors already differ by 1 to 2 cm, so 98-100% accuracy is not realistic with the current data.<br/>'
             '&bull; Datasets and body models are licensed for non-commercial research only.')]

    # ── 11 conclusion ────────────────────────────────────────────────────────
    st += [H1('11. Conclusion and future work'),
           P(f'TailorCam measures the body from two photos with an average error of {mw.loc[COMB, "Chest"]:.2f} cm on the '
             f'chest, {mw.loc[COMB, "Belly / waist"]:.2f} cm on the waist and {mw.loc[COMB, "Hip"]:.2f} cm on the hip, and '
             'picks the right shirt size for most people. Testing every model separately showed that no single model is '
             'best everywhere, and that the hybrid is the most reliable approach, especially when the weight is not known. '
             'Run on the same people, it is more accurate than the published methods HMR, SHAPY and CameraHMR in 4 of 5 '
             'tests; only on the 10 HBW scans with weight is SHAPY better. Training on 4,132 scanned people worked well '
             'on scans but did not improve real photos, which shows that more real tape-measured customers are what is '
             'needed next.'),
           P('<b>Future work.</b> (1) Collect more tape-measured customers; the app already saves every scan with the '
             'tailor\'s tape values so it can be retrained. (2) Add the BodyM prior to the arm rule and train a shoulder '
             'rule once enough tape values exist. (3) Build the guided phone app. (4) Use the requested BODY-rgb and '
             'Fashion-body datasets when they arrive.')]

    # ── appendix A: Q&A ──────────────────────────────────────────────────────
    st += [PageBreak(), H1('Appendix A. Questions and answers'),
           Paragraph('Q1. Did we train all the models ourselves?', S['q']),
           P('No. Three are pre-trained 3D body models from other research groups (HMR, SHAPY, CameraHMR), adapted here: '
             'weight fitting, measuring code, MediaPipe person detection. The other four (silhouette CNN x3, BodyM prior, '
             'ANSUR prior, combiner) were trained by me. See section 2.1.'),
           Paragraph('Q2. How do the seven parts give one final value?', S['q']),
           P('A learned rule per measurement: keep one model, one model plus offset, or a blend. See section 2.2.'),
           Paragraph('Q3. Is there a better way to combine them?', S['q']),
           P('Six ways were compared; the current one has the lowest error (2.44 cm average with weight, 3.02 cm without, '
             'against 4.32 cm for a simple average without weight).'),
           Paragraph('Q4. How good is each model on its own?', S['q']),
           P('See chapter 6: every model, every body part, time and overall accuracy, with charts and the per-person Excel '
             'data sheets.'),
           Paragraph('Q5. Is HMR the newest model?', S['q']),
           P('No. HMR is from 2018; its update is HMR 2.0 (ICCV 2023). CameraHMR (2025), already used, belongs to that newer '
             'generation, so HMR 2.0 was not added. HMR is kept because it still helps the thigh.'),
           Paragraph('Q6. How does it compare with other work?', S['q']),
           P('See chapter 7. I ran HMR, SHAPY and CameraHMR on the same people as TailorCam. TailorCam is better in 4 of 5 '
             'tests; SHAPY is better on the 10 HBW scans with weight. Commercial apps could not be tested.'),
           Paragraph('Q7. Is the accuracy 95-100%? Can it replace the tape?', S['q']),
           P('Accuracy is 95-97% for chest, waist, hip, neck, thigh and arm. It cannot fully replace the tape for fitted '
             'clothes yet; the app marks those values "Check with tape".'),
           Paragraph('Q8. Did we use every dataset? Would more data help?', S['q']),
           P('Every free dataset with real photos and tape measurements was used (chapter 3), and a network was also trained '
             'on 4,132 scanned people (BODY-fit) and tested on 356 more (NOMO-3D). That did not improve real photos '
             '(chapter 9), so more real tape-measured customers would help most; two university datasets have been requested.'),
           Paragraph('Q9. Does the customer need tight clothes?', S['q']),
           P('No. All results are with normal clothes. Tight clothes would help; remove jackets, keep arms slightly away from '
             'the body, and enter the real height and weight.'),
           Paragraph('Q10. Can it work on a phone?', S['q']),
           P('Yes, with the light models only, if the weight is entered (chapter 8).')]

    # ── appendix B: files ────────────────────────────────────────────────────
    st += [H1('Appendix B. Files and reproducibility'),
           table([['File', 'What it is'],
                  ['notebooks/TailorCam_paper.ipynb', 'all experiments, tables and figures for the paper'],
                  ['reports/TailorCam_Model_Data.xlsx', 'per-person data sheets for every model'],
                  ['scripts/model_accuracy.py', 'accuracy of every model and body part (chapter 6)'],
                  ['scripts/model_timing.py', 'time of every model (section 6.2)'],
                  ['scripts/individual_models.py', 'every model on HBW and the model agency set'],
                  ['scripts/ensemble_methods.py', 'the six ways of combining (section 2.2)'],
                  ['scripts/run_models_on_datasets.py', 'runs all models on all datasets'],
                  ['model/calibration.py', 'the combiner'],
                  ['scripts/make_faculty_report.py', 'builds this report']],
                 [W * 0.42, W * 0.58])]

    doc = Doc(OUT, title='TailorCam Faculty Review Report', author='Md Ashikuzzaman')
    doc.multiBuild(st)
    print('saved', OUT)


if __name__ == '__main__':
    build()
