"""
Full project report for the capstone defense: reports/TailorCam_Defense_Report.pdf

Every number is read from the result files written by the evaluation scripts
(all leak-free: calibration.nested_cv_app). Run these first:
  python scripts/model_accuracy.py
  python scripts/model_timing.py
  python scripts/individual_models.py
  python notebooks/TailorCam_paper.py          (PYTHONIOENCODING=utf-8 on Windows)
  python scripts/ensemble_methods.py

  python scripts/make_defense_report.py
"""

import os
import sys

import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, NextPageTemplate, PageBreak, Paragraph, Spacer, Table)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import ParagraphStyle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
sys.path[:0] = [ROOT, os.path.join(ROOT, 'scripts')]
import make_model_accuracy_pdf as M
from make_model_accuracy_pdf import COMB, MODELS, PARTS, fmt, grid, table, figure
from make_faculty_report import Doc, H1, H2, box, S, P, N, lite_results, MUTED

OUT = os.path.join(ROOT, 'reports', 'TailorCam_Defense_Report.pdf')
OUT_APP = os.path.join(ROOT, 'reports', 'TailorCam_Defense_Appendices.pdf')
APPENDIX = '--appendix' in sys.argv
TAB = M.TAB
FIGP = os.path.join(ROOT, 'reports', 'paper', 'figures')
UI = os.path.join(ROOT, 'reports', 'ui')
OV = 'Overall (all parts it measures)'
CWH = ['Chest', 'Belly / waist', 'Hip']


def bodyfit_combo():
    """Leak-free test of the hybrid with and without the BODY-fit network (real24)."""
    path = os.path.join(TAB, 'bodyfit_combo.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    import model.calibration as C
    from model_accuracy import cv_predict
    from repeated_cv import first
    d0 = pd.read_csv(data_path('predictions', 'real24.csv'))
    rows = []
    for mode in ('with_weight', 'no_weight'):
        d = d0[d0['mode'] == mode]
        base = C.TARGETS if mode == 'with_weight' else first('camerahmr', C.TARGETS)
        for t in ('chest', 'waist', 'hip', 'neck', 'thigh', 'sleeve_length'):
            old = [c for c in base[t] if c in d and d[c].notna().any()]
            for name, cols in (('alone', [f'bodyfit:{t}']), ('hybrid', old), ('hybrid+bodyfit', old + [f'bodyfit:{t}'])):
                ok = d[cols + [f'true:{t}']].notna().all(axis=1)
                X, y = d.loc[ok, cols].values.astype(float), d.loc[ok, f'true:{t}'].values.astype(float)
                e = np.abs(cv_predict(X, y) - y).mean(0)
                rows.append({'mode': mode, 'part': t, 'variant': name, 'MAE': e.mean(), 'acc': 100 - (e / y * 100).mean()})
    r = pd.DataFrame(rows)
    r.round(3).to_csv(path, index=False)
    return r


def build():
    acc, mae, n, t, ext = M.load()
    photo, own, sd, total = M.model_time(t)
    aw, anw, mw, mnw = acc['with_weight'], acc['no_weight'], mae['with_weight'], mae['no_weight']
    rawp = {m: pd.read_csv(os.path.join(TAB, f'model_raw_mae_{m}.csv'), index_col=0) for m in ('with_weight', 'no_weight')}
    main = pd.read_csv(os.path.join(TAB, 'main_results.csv'))
    ksum = pd.read_csv(os.path.join(TAB, 'kfold_summary.csv'))
    size = pd.read_csv(os.path.join(TAB, 'size_classification.csv'))
    ens = pd.read_csv(os.path.join(TAB, 'ensemble_methods.csv'))
    lite = lite_results()
    bfc = bodyfit_combo()
    xds = pd.read_csv(os.path.join(ROOT, 'experiments', 'cross_dataset', 'results.csv'))
    abl = pd.read_csv(os.path.join(TAB, 'ablation.csv'))
    tsum = pd.read_csv(os.path.join(TAB, 'training_summary.csv'))
    BIASP = ['Chest', 'Belly / waist', 'Hip', 'Neck', 'Thigh', 'Arm (sleeve)', 'Shoulder']

    def over_under(errs):
        e = errs.dropna()
        ov, un = e[e > 0.05], e[e < -0.05]
        return [str(len(e)), f'{e.mean():+.2f}', f'{len(ov)} ({len(ov) / len(e) * 100:.0f}%)', f'+{ov.mean():.2f}' if len(ov) else '-',
                f'{len(un)} ({len(un) / len(e) * 100:.0f}%)', f'{un.mean():.2f}' if len(un) else '-', f'{e.max():+.1f}', f'{e.min():+.1f}']
    BHEAD = ['Body part', 'People', 'Bias', 'Too big', 'Avg. too big', 'Too small', 'Avg. too small', 'Largest +', 'Largest -']
    bias_tabs = {}
    signed = {}
    for tg in ('W', 'noW'):
        c_ = pd.read_excel(os.path.join(ROOT, 'reports', 'TailorCam_Model_Data.xlsx'), sheet_name=f'COMBINED {tg}')
        rows_ = [BHEAD]
        for p_ in BIASP:
            e_ = c_[p_ + ' predicted'] - c_[p_ + ' tape']
            signed[(tg, p_)] = e_.dropna().values
            rows_.append([p_] + over_under(e_))
        bias_tabs[tg] = rows_
    bx = pd.read_excel(os.path.join(ROOT, 'reports', 'TailorCam_Model_Data.xlsx'), sheet_name='COMBINED W')
    bx = bx[bx['Chest tape'].notna()]
    bx['ce'] = bx['Chest predicted'] - bx['Chest tape']
    b_over, b_under = bx.loc[bx.ce.idxmax()], bx.loc[bx.ce.idxmin()]
    import matplotlib.pyplot as _plt
    fig_, ax_ = _plt.subplots(figsize=(7.4, 2.8), dpi=200)
    for i_, p_ in enumerate(BIASP):
        v_ = signed[('W', p_)]
        ax_.scatter(np.full(len(v_), i_) + np.random.default_rng(i_).uniform(-0.18, 0.18, len(v_)), v_, s=14,
                    color=['#2a78d6' if x > 0 else '#eb6834' for x in v_], edgecolor='white', linewidth=0.5, zorder=3)
        ax_.plot([i_ - 0.3, i_ + 0.3], [v_.mean()] * 2, color='#0b0b0b', lw=1.5, zorder=4)
    ax_.axhline(0, color='#898781', lw=0.8)
    ax_.set_xticks(range(len(BIASP))); ax_.set_xticklabels([p_.replace('Belly / waist', 'Waist') for p_ in BIASP], fontsize=7.5)
    ax_.set_ylabel('TailorCam minus tape (cm)', fontsize=7.5); ax_.tick_params(axis='y', labelsize=7)
    for sp_ in ('top', 'right'):
        ax_.spines[sp_].set_visible(False)
    ax_.set_title('Each dot is one person (weight entered). Blue = too big, orange = too small, black line = bias', fontsize=8, color='#52514e')
    fig_.tight_layout(); fig_.savefig(os.path.join(M.FIG, 'bias_dots.png')); _plt.close(fig_)
    # odjag (own photos) test
    import make_odjag_report as OJ
    odr = OJ.load()                      # every model + app + app with shop offset, one consistent run
    od_n = odr.person_id.nunique()
    od_raw, od_off = OJ.stats(odr)
    od_o1, od_o2 = OJ.overall(od_raw), OJ.overall(od_off)
    od_used = [k for k, v in __import__('json').load(open(os.path.join(ROOT, 'model', 'calibration', 'shop_offset.json')))
               .get('no_weight', {}).items() if v.get('used')]
    cmb = pd.read_csv(os.path.join(ROOT, 'experiments', 'combined', 'summary.csv'))
    cmb_ov = cmb.groupby('model').accuracy.mean()
    cmb_n = pd.read_csv(os.path.join(ROOT, 'experiments', 'combined', 'people.csv')).person.nunique()
    ODP = {'chest': 'Chest', 'waist': 'Waist', 'hip': 'Hip', 'neck': 'Neck', 'thigh': 'Thigh', 'sleeve_length': 'Sleeve',
           'shoulder_width': 'Shoulder'}
    od_rows = [['Body part', 'Error, app as it is', 'Accuracy', 'Bias', 'Error with shop offset', 'Accuracy with offset']]
    for k_, lab_ in ODP.items():
        g_ = odr[odr.measurement == k_]
        y_, p_ = g_.tape.values, g_.TailorCam.values
        e1 = np.abs(p_ - y_)
        e2 = np.abs(g_['TailorCam + shop offset'].values - y_)          # real pipeline, learned without this person
        od_rows.append([lab_, f'{e1.mean():.2f} cm', f'{100 - (e1 / y_ * 100).mean():.1f}%', f'{(p_ - y_).mean():+.2f} cm',
                        f'{e2.mean():.2f} cm', f'{100 - (e2 / y_ * 100).mean():.1f}%'])
    e2e = pd.read_csv(os.path.join(ROOT, 'experiments', 'odjag', 'shop_offset_e2e.csv'))
    e2e_raw = np.mean([100 - ((g.app - g.tape).abs() / g.tape * 100).mean() for _, g in e2e.groupby('measurement')])
    e2e_off = np.mean([100 - ((g['app + shop offset'] - g.tape).abs() / g.tape * 100).mean() for _, g in e2e.groupby('measurement')])
    od_bias = [BHEAD]
    import json as _json
    import json
    from types import SimpleNamespace
    _raw = _json.load(open(data_path('odjag', 'raw', 'body_measurements.json'), encoding='utf-8'))['measurements']
    _w = odr[odr.measurement == 'waist'].rename(columns={'TailorCam': 'app'}).reset_index(drop=True)
    _gain = pd.Series([abs(_w.app[i] - _w.tape[i]) - abs(_w.app[i] + np.mean(np.delete((_w.tape - _w.app).values, i)) - _w.tape[i])
                       for i in range(len(_w))])
    _k = int((_gain - _gain[_gain > 0].median()).abs().idxmin())      # a typical person the offset helps
    _pid = _w.person_id[_k]
    od_waist_worse = _w.person_id[_gain < 0].tolist()
    _j = next(v for k_, v in _raw.items() if k_.lower() == _pid)
    _ft, _inch = _j['height'].replace('"', '').split("'")
    od_ex = SimpleNamespace(pid=_pid, height_txt=_j['height'].replace('"', ' in'), height_in=int(_ft) * 12 + float(_inch or 0),
                            height_cm=(int(_ft) * 12 + float(_inch or 0)) * 2.54, chest_in=float(_j['chest']),
                            chest_cm=float(_j['chest']) * 2.54,
                            chest_app=float(odr[(odr.person_id == _pid) & (odr.measurement == 'chest')].TailorCam.iloc[0]),
                            w_app=float(_w.app[_k]), w_tape=float(_w.tape[_k]),
                            w_off=float(np.mean(np.delete((_w.tape - _w.app).values, _k))))
    for k_, lab_ in ODP.items():
        g_ = odr[odr.measurement == k_]
        od_bias.append([lab_] + over_under(g_.TailorCam - g_.tape))
    import json
    import model.calibration as CAL
    calib = json.load(open(CAL.CALIB_PATH))
    NICE = {'hybrid': 'silhouette CNN', 'prior': 'BodyM prior', 'hmr': 'HMR', 'shapy': 'SHAPY', 'ansur': 'ANSUR prior',
            'camerahmr': 'CameraHMR'}
    PNAME = {'chest': 'Chest', 'waist': 'Waist', 'hip': 'Hip', 'thigh': 'Thigh', 'neck': 'Neck', 'sleeve_length': 'Sleeve',
             'shoulder_width': 'Shoulder', 'back_length': 'Back length', 'shirt_length': 'Shirt length'}

    def rule_text(p):
        comps = [NICE[c.split(':')[0]] for c in p['components']]
        k, b, w = p['kind'], p['bias'], p['w']
        sb = (' + %.2f cm' % b) if b >= 0 else (' - %.2f cm' % -b)
        if k == 'identity':
            return 'keep the default model', f'{comps[0]} as it is'
        if k == 'bias':
            return 'default model + offset', f'{comps[0]}{sb}'
        if k in ('pick', 'pick+bias'):
            m = comps[int(p['alpha'])]
            return ('one model + offset', f'{m}{sb}') if k == 'pick+bias' else ('one other model', m)
        w0 = 1 - sum(w)
        terms = [f'{w0:.2f} {comps[0]}'] + [('%+.2f %s' % (x, c)).replace('+', '+ ').replace('-', '- ') for x, c in zip(w, comps[1:]) if abs(x) >= 0.005]
        return f'blend of {len(comps)} models', ' '.join(terms) + sb
    rules = {}
    for mode in ('with_weight', 'no_weight'):
        rr = [['Body part', 'Rule', 'Formula used by the app']]
        for t_, p in calib.get(mode, {}).items():
            a_, b_ = rule_text(p)
            rr.append([PNAME.get(t_, t_), a_, b_])
        rules[mode] = rr
    cvt = pd.read_csv(os.path.join(TAB, 'camera_vs_tape.csv'))
    tape_rows = [['Body part', 'Tolerance', 'Error, weight', '80% of people within, weight', 'Label now']]
    for _, r in cvt[cvt.weight == 'known'].iterrows():
        if pd.isna(r['avg error (cm)']):
            continue
        tape_rows.append([r.measurement, f"{r['tolerance (cm)']:.1f} cm", f"{r['avg error (cm)']:.2f} cm", f"{r['80% within (cm)']:.2f} cm",
                          'Camera OK' if r.verdict == 'CAMERA' else 'Check with tape'])
    n_cam = int((cvt.verdict == 'CAMERA').sum())
    ans = pd.read_csv(os.path.join(TAB, 'ansur_prior.csv')).set_index('measurement')
    bmt = pd.read_csv(os.path.join(TAB, 'bodym_test.csv'))
    cnnm = pd.read_csv(os.path.join(TAB, 'individual_models_cnn_members.csv'))
    sad = pd.read_csv(os.path.join(TAB, 'summary_all_datasets.csv'))
    kfs = pd.read_csv(os.path.join(TAB, 'kfold_size_classification.csv'))
    kpf = pd.read_csv(os.path.join(TAB, 'kfold_per_fold.csv'))
    ecor = pd.read_csv(os.path.join(TAB, 'error_correlations.csv'))
    from scipy.stats import wilcoxon
    XL = os.path.join(ROOT, 'reports', 'TailorCam_Model_Data.xlsx')
    sheets = {tg: pd.read_excel(XL, sheet_name=f'COMBINED {tg}') for tg in ('W', 'noW')}
    camsh = {tg: pd.read_excel(XL, sheet_name=f'CameraHMR {tg}') for tg in ('W', 'noW')}
    PART3 = ['Chest', 'Belly / waist', 'Hip']
    grows = [['Group', 'People', 'Chest', 'Waist', 'Hip']]
    for tg, wl in (('W', 'weight'), ('noW', 'no weight')):
        c = sheets[tg]
        for gname, gdf in c.groupby('gender'):
            grows.append([f'{gname}, {wl}', str(len(gdf))] + [f"{gdf[p_ + ' abs error'].mean():.2f}" for p_ in PART3])
    cw = sheets['W']
    bmi = cw.weight_kg / (cw.height_cm / 100) ** 2
    for lab_, msk in ((f'BMI below {bmi.median():.1f}, weight', bmi < bmi.median()), (f'BMI {bmi.median():.1f} or above, weight', bmi >= bmi.median())):
        grows.append([lab_, str(int(msk.sum()))] + [f"{cw.loc[msk, p_ + ' abs error'].mean():.2f}" for p_ in PART3])
    wrows = [['Weight', 'Body part', 'People', 'Hybrid error', 'CameraHMR error', 'p-value', 'Significant?']]
    for tg, wl in (('W', 'entered'), ('noW', 'not entered')):
        for p_ in PART3:
            a1, a2 = sheets[tg][p_ + ' abs error'], camsh[tg][p_ + ' abs error']
            m_ = a1.notna() & a2.notna()
            pv = wilcoxon(a1[m_], a2[m_]).pvalue
            wrows.append([wl, p_, str(int(m_.sum())), f'{a1[m_].mean():.2f}', f'{a2[m_].mean():.2f}', f'{pv:.3f}',
                          'yes' if pv < 0.05 else 'no'])
    cis = [['Measurement', 'Weight', 'MAE (cm)', '95% CI of MAE', 'RMSE (cm)', 'Bias (cm)', 'R2']]
    for _, r in main[main.dataset.str.startswith('real24')].iterrows():
        cis.append([r.measurement, 'entered' if r.weight == 'weight known' else 'not entered', f"{r['MAE (cm)']:.2f}",
                    str(r['95% CI']).replace('\u2013', ' to '), f"{r['RMSE (cm)']:.2f}", f"{r['bias (cm)']:.2f}", f"{r['R²']:.2f}"])
    gpu = t['device'].iloc[0].replace('NVIDIA GeForce ', '')
    W = A4[0] - 36 * mm
    os.makedirs(M.FIG, exist_ok=True)
    f = {k: os.path.join(M.FIG, f'{k}.png') for k in ('heat_w', 'heat_nw', 'parts_w', 'overall', 'time', 'trade', 'vs')}
    M.chart_heatmap(aw, f['heat_w'], 'Accuracy (%) per body part, weight entered. Bold = best in the column')
    M.chart_heatmap(anw, f['heat_nw'], 'Accuracy (%) per body part, no weight. Bold = best in the column')
    M.chart_parts(mw, f['parts_w']); M.chart_overall(acc, f['overall']); M.chart_time(total, own, photo, f['time'])
    M.chart_trade(anw, total, f['trade']); M.chart_vs(mae, f['vs'])
    r24 = main[main.dataset.str.startswith('real24')]
    g = lambda w, m, c: float(r24[(r24.weight == w) & (r24.measurement == m)][c].iloc[0])
    szr = size[size.dataset.str.startswith('real24')]
    sz = lambda w, frm, c: float(szr[(szr.weight == w) & (szr['size from'] == frm)][c].iloc[0])
    ensa = ens[ens.measurement == 'AVERAGE'] if 'measurement' in ens else None

    # same-people comparison with published methods (my own runs)
    exm = ext.set_index(['dataset', 'model'])
    PUB = [('HMR', 'CVPR 2018', 'HMR (2018)'), ('SHAPY', 'CVPR 2022', 'SHAPY (2022)'), ('CameraHMR', '3DV 2025', 'CameraHMR (2025)')]
    sets = []
    for mode, lab in (('with_weight', '24 real people, weight entered'), ('no_weight', '24 real people, no weight')):
        sets.append((lab, {nm: [rawp[mode].loc[k, c] for c in CWH] for nm, _, k in PUB}, [mae[mode].loc[COMB, c] for c in CWH]))
    for ds, lab in (('hbw_val (with weight)', 'HBW 3D scans (10), weight entered'), ('hbw_val (no weight)', 'HBW 3D scans (10), no weight'),
                    ('agency (no weight)', 'Model agency photos (134), no weight')):
        e = exm.loc[ds]
        sets.append((lab, {nm: list(e.loc[k, ['chest', 'waist', 'hip']].astype(float)) for nm, _, k in PUB},
                     list(e.loc['Combiner = TailorCam (ours)', ['chest', 'waist', 'hip']].astype(float))))
    xp = xds.pivot_table(index=['dataset', 'mode', 'model'], columns='measurement', values='MAE after offset').round(2)
    XTAB = [['Tested on', 'Weight', 'Model', 'Chest', 'Waist', 'Hip', 'Thigh', 'Arm', 'Shoulder']]
    for (dsn, md, mo), r in xp.iterrows():
        XTAB.append([dsn, md, mo] + [f"{r[c]:.2f}" for c in ('chest', 'waist', 'hip', 'thigh', 'arm / sleeve', 'shoulder')])
    wins = sum(np.mean(o) < min(np.mean(v) for v in a.values()) for _, a, o in sets)

    st = []
    # ── front matter (EWU CSE400 layout) ─────────────────────────────────────
    sig = lambda who: Paragraph('<br/><br/>______________________________<br/>' + who, S['p'])
    st += [Spacer(1, 26 * mm), Paragraph('TailorCam', S['cover_t']), Spacer(1, 4),
           Paragraph('Measuring the human body from two photos with a hybrid of deep learning models', S['cover_s']),
           Spacer(1, 12 * mm),
           Paragraph('A capstone project report submitted in partial fulfillment of the requirements for the degree of '
                     'Bachelor of Science in Computer Science and Engineering', S['cover_m']), Spacer(1, 4 * mm),
           Paragraph('Course: CSE400, Capstone Project', S['cover_m']), Spacer(1, 10 * mm),
           Paragraph('Submitted by<br/><b>Md Ashikuzzaman</b><br/>ID 2022-3-60-101<br/>Major: Intelligent Systems and Data '
                     'Science', S['cover_m']), Spacer(1, 7 * mm),
           Paragraph('Supervised by<br/><b>Dr. Mohammad Mahdi Hasan</b><br/>Department of Computer Science and Engineering',
                     S['cover_m']), Spacer(1, 10 * mm),
           Paragraph('Department of Computer Science and Engineering<br/>East West University, Dhaka, Bangladesh<br/>November 2026',
                     S['cover_m']), NextPageTemplate('body'), PageBreak()]
    st += [Paragraph('Declaration', S['h1'].clone('fm1', keepWithNext=0)),
           P("I declare that the work in this report, TailorCam: measuring the human body from two photos with a hybrid of "
             "deep learning models, is my own, done under the supervision of Dr. Mohammad Mahdi Hasan, Department of Computer "
             "Science and Engineering, East West University. Models, datasets and papers by other people are named where they "
             "are used and listed in the references. No part of this work has been submitted for any other degree."),
           sig('Md Ashikuzzaman<br/>ID 2022-3-60-101'), Spacer(1, 14 * mm),
           Paragraph('Letter of approval', S['h1'].clone('fm2', keepWithNext=0)),
           P("This capstone project report, TailorCam: measuring the human body from two photos with a hybrid of deep learning "
             "models, submitted by Md Ashikuzzaman (ID 2022-3-60-101) to the Department of Computer Science and Engineering, "
             "East West University, is accepted as satisfactory in partial fulfillment of the requirements for the degree of "
             "Bachelor of Science in Computer Science and Engineering."),
           sig('Dr. Mohammad Mahdi Hasan<br/>Supervisor, Department of Computer Science and Engineering<br/>East West University'),
           sig('Chairperson<br/>Department of Computer Science and Engineering<br/>East West University'), PageBreak()]
    st += [Paragraph('Acknowledgement', S['h1'].clone('fm3', keepWithNext=0)),
           P("I am grateful to my supervisor, Dr. Mohammad Mahdi Hasan, for his guidance and for asking the hard questions that "
             "shaped this project, especially about testing every model on its own and checking the results honestly. I also "
             "thank the Department of Computer Science and Engineering at East West University."),
           P("This project would not have been possible without the people who shared their work openly: the authors of HMR, "
             "SHAPY, CameraHMR, SMPL and MediaPipe; Amazon for the BodyM dataset; the US Army for ANSUR II; the Max Planck "
             "Institute for HBW, the model agency data and BEDLAM; Tampere University for BODY-fit and NOMO-3D; and Unidata for "
             "the free sample photos."), Spacer(1, 10 * mm),
           Paragraph('Abstract', S['h1'].clone('fm4', keepWithNext=0)), P("TailorCam is a desktop app for tailor shops. From a front photo, "
               "a side photo and the customer's height (and weight, if given) it estimates chest, waist, hip, neck, thigh, "
               "shoulder and sleeve length. It runs six measurement models, three of them published 3D body models, and a small "
               "learned combiner decides how to use them for each body part. I tested it on 24 real people measured with a "
               "tape, using nested cross-validation in which the test people take no part in any decision. With the weight "
               f"entered the average error is {g('weight known', 'Chest', 'MAE (cm)'):.2f} cm on the chest "
               f"({g('weight known', 'Chest', 'accuracy (%)'):.1f}% accuracy), {g('weight known', 'Waist', 'MAE (cm)'):.2f} cm "
               f"on the waist and {g('weight known', 'Hip', 'MAE (cm)'):.2f} cm on the hip, and the chest gives the right shirt "
               f"size for {sz('weight known', 'chest', 'accuracy (%)'):.0f}% of people. I also tested every model on its own, "
               f"ran three published methods on the same people (TailorCam was better in {wins} of 5 tests), trained a second "
               "network on 4,132 scanned bodies, and checked the whole evaluation for data leakage."),
           P('<b>Keywords:</b> body measurement, 3D human body, SMPL, deep learning, ensemble, tailoring, computer vision.'),
           PageBreak()]
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle('t0', fontName='UI-B', fontSize=10, leading=14.5),
                       ParagraphStyle('t1', fontName='UI', fontSize=8.8, leading=12, leftIndent=14, textColor=MUTED)]
    st += [Paragraph('Contents', S['h1'].clone('toc_h', spaceBefore=0, keepWithNext=0)), toc, PageBreak()]
    st += [Paragraph('List of abbreviations', S['h1'].clone('fm5', keepWithNext=0)),
           table([['Abbreviation', 'Meaning'], ['CNN', 'convolutional neural network'], ['CV', 'cross-validation'],
                  ['GPU', 'graphics processing unit'], ['HMR', 'Human Mesh Recovery (a 3D body model from a photo)'],
                  ['MAE', 'mean absolute error'], ['MAPE', 'mean absolute percentage error'],
                  ['SMPL / SMPL-X', 'Skinned Multi-Person Linear model, a statistical 3D body model'],
                  ['ANSUR II', 'Anthropometric Survey of US Army Personnel, 2012'], ['BodyM', "Amazon's body measurement dataset"],
                  ['HBW', 'Human Bodies in the Wild (SHAPY test set)'], ['F1', 'harmonic mean of precision and recall'],
                  ['LoA', 'limits of agreement (Bland-Altman)']], [W * 0.25, W * 0.75]), PageBreak()]

    # ── 1 introduction ───────────────────────────────────────────────────────
    st += [H1('1. Introduction'),
           H2('1.1 Motivation'),
           P("A tailor measures every customer by hand. It takes a few minutes, it needs the customer to be in the shop, and "
             "two tailors rarely get exactly the same numbers. Many research groups have released models that turn a photo into "
             "a 3D body, so the idea of measuring a customer with a camera is not new. What is missing is a clear answer to a "
             "simple question: how close do these models really get to a tape, on real people in normal clothes?"),
           H2('1.2 Problem statement'),
           P("Given a front photo, a side photo and the height of a person (and the weight, if they know it), estimate the "
             "chest, waist, hip, neck, thigh, shoulder and sleeve measurements a tailor would take, as close to a tape as "
             "possible, and show honestly how accurate each estimate is."),
           H2('1.3 Objectives'),
           P("&bull; Estimate seven tailoring measurements from a front and a side photo plus height, with weight optional.<br/>"
             "&bull; Combine several models so the result is more reliable than any one of them.<br/>"
             "&bull; Test every model and the combination on real tape-measured people, with no data leakage.<br/>"
             "&bull; Compare the result with published methods on the same people.<br/>"
             "&bull; Deliver a complete shop app: customers, scans, measurements, sizes, orders and PDF reports."),
           H2('1.4 Scope'),
           P("The project covers adults standing in front of a webcam or in uploaded photos, a Windows desktop app for one shop, "
             "and the seven measurements above. It does not try to replace the tape for fitted clothes; values that are not yet "
             "reliable enough are marked 'Check with tape'. A phone version was studied (section 10.2) but not built."),
           H2('1.5 Contributions'),
           P("&bull; A hybrid of six models (HMR, SHAPY, CameraHMR, a silhouette CNN, two height-and-weight models) with a "
             "learned combiner that picks a rule for each body part.<br/>"
             "&bull; Weight fitting: every 3D body is resized so its volume matches the customer's real weight.<br/>"
             "&bull; A per-model evaluation on seven body parts, with time, accuracy and charts.<br/>"
             "&bull; A leakage review that found one problem in my own evaluation; I fixed it, and every number in this report "
             "uses the fixed method.<br/>"
             "&bull; Extra experiments: a phone-only version, training on 4,132 scanned bodies, and turning real photos into "
             "Amazon-style silhouettes."),
           H2('1.6 Organisation of the report'),
           P("Chapter 2 reviews related work. Chapter 3 gives the requirements and the data, chapter 4 the design and "
             "chapter 5 the implementation. Chapter 6 explains how I tested, how accuracy is calculated, and the leakage review. "
             "Chapters 7 to 10 give the results. Chapter 11 looks at efficiency, cost, impact, ethics and project management, "
             "and chapter 12 concludes.")]

    # ── 2 background ─────────────────────────────────────────────────────────
    st += [H1('2. Literature review'),
           H2('2.1 Body models'),
           P("Most methods describe a body with SMPL, a statistical 3D body model with 6,890 points. A small set of shape "
             "numbers (betas) controls the body's size and proportions, and a set of joint angles controls the pose. Once a "
             "method has found the betas for a person, any measurement can be taken on the 3D body. SHAPY uses SMPL-X, a "
             "newer version of the same idea that also models hands and face."),
           H2('2.2 Methods that build a 3D body from a photo'),
           table([['Method', 'Year', 'What it does', 'Used in TailorCam'],
                  ['HMR (Kanazawa et al.)', 'CVPR 2018', 'one network predicts SMPL shape and pose from a photo', 'yes'],
                  ['SPIN, ExPose, TUCH', '2019 to 2021', 'improved pose; body shape still close to average', 'no'],
                  ['STRAPS, Sengupta et al.', '2020 to 2021', 'shape from silhouettes and joints', 'no'],
                  ['SHAPY (Choutas et al.)', 'CVPR 2022', 'shape learned from model-agency measurements', 'yes'],
                  ['HMR 2.0 (4DHumans)', 'ICCV 2023', 'transformer version of HMR', 'considered, not added'],
                  ['CameraHMR (Patel and Black)', '3DV 2025', 'transformer model trained with better camera handling', 'yes']],
                 [W * 0.27, W * 0.14, W * 0.43, W * 0.16]),
           H2('2.3 Methods that measure from silhouettes'),
           P("Amazon released BodyM in 2022: front and side silhouettes of 2,505 people with 14 measurements taken on 3D "
             "scans. Amazon's MeasureNet (2023) uses the same kind of input and reports about 1.7 to 1.8 cm error on its own "
             "private test people, which is the best published result I found. Its training data and code are not public. "
             "My silhouette CNN follows this idea and is trained on the public BodyM data."),
           H2('2.4 Commercial apps'),
           P("Apps such as 3DLOOK, Bodygram and MTailor publish accuracy claims, but their models are closed and paid, so I "
             "could not run them on my test people. I therefore compare only with methods I could run myself (chapter 9)."),
           H2('2.5 Research gap'),
           P("Published methods are usually tested on scans or on their own private people, and rarely on clothed people against a real tape. Each paper reports one model, so it is hard to know which model is best for which body part, and whether combining them helps. This project fills that gap for a tailoring setting: it tests six models and their combination on the same tape-measured people, body part by body part, with a leak-free protocol.")]

    # ── 3 requirement analysis ───────────────────────────────────────────────
    st += [H1('3. Requirement analysis and data'),
           H2('3.1 Stakeholders'),
           P("The main user is the tailor or shop assistant, who runs the scan and reads the results. The customer stands in "
             "front of the camera and wants clothes that fit. The shop owner needs the customer records, orders and reports."),
           H2('3.2 Functional requirements'),
           table([['ID', 'Requirement'],
                  ['FR1', 'Take a front and a side photo with a webcam, or accept two uploaded photos'],
                  ['FR2', 'Accept height (required) and weight (optional)'],
                  ['FR3', 'Estimate chest, waist, hip, neck, thigh, shoulder and sleeve'],
                  ['FR4', 'Convert measurements into clothing sizes'],
                  ['FR5', "Mark every value as 'Camera OK' or 'Check with tape'"],
                  ['FR6', 'Save customers, scans, orders and inventory'],
                  ['FR7', 'Produce a PDF report for a customer'],
                  ['FR8', "Accept the tailor's real tape values and keep them for retraining"]], [W * 0.1, W * 0.9]),
           H2('3.3 Non-functional requirements'),
           table([['ID', 'Requirement', 'How it is met'],
                  ['NFR1', 'Accuracy close to a tape', f"chest error {g('weight known', 'Chest', 'MAE (cm)'):.2f} cm with weight (chapter 7)"],
                  ['NFR2', 'Fast enough for a shop', f'about {total[COMB]:.0f} s of computing per customer (section 8.2)'],
                  ['NFR3', 'Works without internet', 'every model runs on the local PC'],
                  ['NFR4', 'Privacy', 'photos and records stay on the shop computer'],
                  ['NFR5', 'Easy for non-technical staff', 'on-screen guide, voice instructions, mouse and keyboard']],
                 [W * 0.1, W * 0.3, W * 0.6]),
           H2('3.4 Hardware and software requirements'),
           P(f"A Windows PC with a webcam and an NVIDIA graphics card (development and testing used an {gpu}). The software "
             "is Python 3.11 with PyTorch, OpenCV, MediaPipe, NumPy, pandas, scikit-learn and reportlab, plus SQLite for the "
             "shop records. All of it is free."),
           H2('3.5 Datasets: where each is used, why, and how it did')]

    ab = lambda m_, w_, c_: float(abl[(abl.removed == m_) & (abl.weight == w_)][c_].iloc[0])
    bm = lambda sp_, meth_, c_: float(bmt[(bmt['test set'] == sp_) & (bmt.method.str.contains(meth_))][c_].iloc[0])
    xv = lambda ds_, mo_, md_, c_: float(xds[(xds.dataset.str.startswith(ds_)) & (xds.model == mo_) & (xds['mode'] == md_) &
                                            (xds.measurement == c_)]['MAE after offset'].iloc[0])
    ensv = lambda md_, col_: float(ens[(ens['mode'] == md_) & (ens.measurement == 'AVERAGE')][col_].iloc[0])
    bacc = lambda mode, v: bfc[(bfc['mode'] == mode) & (bfc.variant == v)]['acc'].mean()
    TC = 'TailorCam (combined)'
    ds_tab = [['Dataset', 'People', 'Train or test', 'Why we used it', 'Test result', 'Good?'],
              ['real24 (Unidata samples)', '24', 'both: trains the combiner, tested with nested CV',
               'only free set of real photos in normal clothes with tape values, like a tailor shop',
               f"TailorCam {aw.loc[COMB, OV]:.1f}% with weight, {anw.loc[COMB, OV]:.1f}% without", 'yes, best of all models'],
              ['Own photos (odjag)', str(od_n), 'test only', 'people measured by a local tailor, new to the app',
               f"{e2e_raw:.1f}% as it is, {e2e_off:.1f}% with shop offset", 'yes after the shop offset'],
              ['Combined (real24 + own)', str(cmb_n), 'test only (nested CV, no weight)', 'one larger real test set for the defense',
               f"TailorCam {cmb_ov[TC]:.1f}%, CameraHMR {cmb_ov['CameraHMR']:.1f}%", 'yes, second, very close to the best'],
              ['BodyM (Amazon)', '2,018 train; 1,684 + 1,160 test', 'trains silhouette CNN and BodyM prior; test A and B',
               'largest public set of body outlines with measurements',
               f"waist {bm('testA', 'prior', 'waist'):.2f} cm to {bm('testA', 'hybrid', 'waist'):.2f} cm with the CNN", 'yes'],
              ['ANSUR II (US Army)', '6,068', 'trains ANSUR prior (80%), tests on 20%',
               'largest tape survey; fallback from height and weight',
               f"chest {ans.loc['chest', 'known']:.2f} cm, neck {ans.loc['neck', 'known']:.2f} cm on unseen soldiers", 'yes, best for neck'],
              ['BODY-fit (Tampere)', '3,013 train; 558 test', 'trains a second outline network',
               'to see if far more training people help',
               f"chest 2.44 cm on scans; on real photos {bacc('with_weight', 'hybrid'):.1f}% without it, {bacc('with_weight', 'hybrid+bodyfit'):.1f}% with it",
               'no gain, not added'],
              ['NOMO-3D (Tampere)', '356', 'test only', 'independent real 3D scans',
               f"CNN chest {xv('NOMO', 'Silhouette CNN (BodyM)', 'no weight', 'chest'):.2f} cm against {xv('NOMO', 'BodyM prior', 'no weight', 'chest'):.2f} cm from height",
               'yes, the CNN carries over'],
              ['HBW (SHAPY project)', '10', 'test only', 'everyday photos with 3D scan truth', 'see chapter 9',
               'mixed: SHAPY better with weight'],
              ['Model agency (SHAPY project)', '134', 'test only', 'largest photo test, fashion poses',
               '7.29 cm average (CameraHMR 7.82 cm)', 'yes, best of the methods run'],
              ['BEDLAM (synthetic)', '42', 'test only', 'computer-made people with exact bodies', 'sensible on clean truth',
               'not a fair test for CameraHMR'],
              ['NHANES (US survey)', 'about 28,000', 'tried as an extra prior', 'bigger alternative to ANSUR',
               'waist 3.64 to 4.21 cm', 'no, dropped']]
    st += [P("Each dataset has one job. The table shows, for every dataset, whether it trains or tests something, why it was "
             "needed, the test result, and whether that result is good. No person used to train a part is used to test it."),
           table(ds_tab, [W * 0.15, W * 0.1, W * 0.17, W * 0.22, W * 0.24, W * 0.12]),
           N("Errors are average absolute errors in cm; accuracy is 100 minus the average error in percent. CameraHMR was "
             "trained on BEDLAM by its authors. 292 people who are in both BODY-fit and NOMO-3D were removed from BODY-fit "
             "training. Licences: Unidata CC BY-NC-ND 4.0, BodyM CC BY-NC 4.0, ANSUR II public domain, BODY-fit CC BY 4.0, "
             "SHAPY and BEDLAM data under the MPI research licence."),
           P(f"<b>Test dataset for the defense.</b> The main real test set is the 24 real24 people plus our own {od_n} people "
             f"(34 in total, {cmb_n} with usable tape values), tested with nested cross-validation: no person is measured by "
             "rules that learned from them. Our own people were never used to build the app, so for them the result of the "
             "app as it is is a completely new test."),
           H2('3.6 Which model is trained and tested on which dataset'),
           P("Not every model can run on every dataset. HMR, SHAPY and CameraHMR need real colour photos. BodyM has only "
             "silhouettes, BODY-fit and NOMO-3D have only 3D bodies, and ANSUR II is a table of numbers. The combiner needs the "
             "answers of all six models, so it can only run where colour photos exist. T = trained on it, test = tested on it, "
             "x = impossible with this kind of data."),
           table([['Dataset (what it contains)', 'HMR', 'SHAPY', 'Camera HMR', 'Silh. CNN', 'BodyM prior', 'ANSUR prior', 'BODY-fit net', 'Combiner'],
                  ['real24 (colour photos, tape)', 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'T + test (CV)'],
                  ['HBW (colour photos, 3D scan)', 'test', 'test', 'test', 'test', 'test', 'test', '-', 'test'],
                  ['Model agency (photos, sizes)', 'test', 'test', 'test', 'test', 'test', 'test', '-', 'test'],
                  ['BEDLAM (rendered images)', 'test', 'test', 'test*', 'test', 'test', 'test', '-', 'test'],
                  ['BodyM (silhouettes only)', 'x', 'x', 'x', 'T + test', 'T + test', '-', 'test', 'x'],
                  ['ANSUR II (numbers only)', 'x', 'x', 'x', 'x', 'x', 'T + test', 'x', 'x'],
                  ['BODY-fit (3D bodies)', 'x', 'x', 'x', 'test', 'test', 'test', 'T + test', 'x'],
                  ['NOMO-3D (3D scans)', 'x', 'x', 'x', 'test', 'test', 'test', 'test', 'x'],
                  ['Own photos, odjag (front photo, tape)', 'test', 'test', 'test', 'test', 'test', 'test', '-', 'test']],
                 [W * 0.24] + [W * 0.085] * 7 + [W * 0.165]),
           N("* Not a fair test, see the note above. '-' = possible but not run, because the result would add little. "
             "Section 8.6 shows what each training gave, with and without it.")]


    # ── 5 method ─────────────────────────────────────────────────────────────
    st += [H1('4. Design methodology'),
           H2('4.1 System architecture'),
           P("The design is a pipeline. MediaPipe prepares each photo, six models each produce their own set of measurements, and a combiner turns them into one final value per body part. The heavy models run in a separate worker process on the graphics card so the interface never freezes."),
           figure(os.path.join(FIGP, 'pipeline.png'), W, 'Figure 1. The TailorCam pipeline: photos and body details go into six '
                                                         'models, and the combiner gives the final measurements.'),
           H2('4.2 The photo step'),
           P("MediaPipe finds 33 body points and a person mask in each photo. The app checks that the whole body is visible, "
             "guesses whether the photo is a front or side view, and measures how far the arms are from the body (wrist to hip "
             "gap divided by shoulder width). The silhouette CNN only uses the photos when this value is at least 0.45, "
             "because arms pressed against the body hide the waist and hip outline."),
           H2('4.3 3D body models and weight fitting'),
           P("HMR, SHAPY and CameraHMR each predict a 3D body from every photo; the shapes are combined with a median. The "
             "body is scaled to the customer's height. If the weight is known, the shape is changed as little as possible "
             "until the body's volume times 1,010 kg per cubic metre (an average body density) equals the real weight. This "
             "is a small optimisation over the shape numbers. HMR on its own pulls every body toward average, and the weight "
             "corrects much of that."),
           H2('4.4 The virtual tape'),
           P("Measurements are taken at fixed places on the 3D body, the same for every customer. A circumference is the "
             "perimeter of the convex hull of a horizontal slice through the body, which is what a tape does: it bridges the "
             "hollows instead of following every dip in the skin."),
           table([['Body part', 'Where it is measured (HMR and CameraHMR)'],
                  ['Chest', 'largest girth at nipple level, under the arms'],
                  ['Waist', 'smallest girth in a band around the navel'],
                  ['Hip', 'largest girth over the seat'],
                  ['Neck, thigh', 'fixed rings of points on the body model'],
                  ['Shoulder', 'across the back from one shoulder tip to the other, arms hanging down'],
                  ['Sleeve', 'shoulder tip to the centre of the wrist, arm hanging straight']],
                 [W * 0.2, W * 0.8]),
           N('SHAPY uses its own definitions: chest at the nipple, waist at the belly button, hip at crotch height.'),
           H2('4.5 The silhouette CNN'),
           P("Each silhouette is cropped, scaled so the body fills 92% of a 256 by 192 image, and centred, both in training and "
             "in the app, so the network sees the same kind of picture. The network is a ResNet-18 that reads the front and "
             "the side silhouette, plus height, weight and gender as numbers. It does not predict the measurement directly: a "
             "ridge-regression formula from height and weight gives a first guess, and the network predicts a correction and "
             "its own uncertainty (trained with a Laplace likelihood). During training the photos are randomly dropped so the "
             "network also works without them, the weight is randomly hidden, and the silhouettes are rotated, shifted, blurred, "
             "merged and given holes to look like real camera masks. Three copies with different random seeds are averaged. "
             "Training takes 40 passes over BodyM's 2,018 training people on the GPU."),
           H2('4.6 The height-and-weight models'),
           P("The BodyM prior is the ridge formula inside the CNN, used alone. The ANSUR prior is a ridge regression trained "
             "on 6,068 US Army soldiers measured with a tape, using height, weight, gender and age. Neither needs a photo. The "
             "ANSUR prior was trained on 80% of the soldiers and tested on the other 20%:"),
           table([['On 20% held-out soldiers', 'Chest', 'Waist', 'Hip', 'Neck', 'Thigh', 'Sleeve', 'Shoulder']] +
                 [[lab_] + [f"{ans.loc[k_, col_]:.2f}" for k_ in ('chest', 'waist', 'hip', 'neck', 'thigh', 'sleeve_length', 'shoulder_width')]
                  for lab_, col_ in (('Weight known (cm)', 'known'), ('Weight unknown (cm)', 'unknown'))],
                 [W * 0.23] + [W * 0.11] * 7),
           N('Without weight the chest, waist and hip errors more than double: height alone says little about girth.'),
           H2('4.7 The combiner'),
           P("For each body part the combiner chooses between a few simple rules: keep the default model as it is, add a "
             "learned offset, use another single model (with or without an offset), or blend several models with ridge "
             "regression. The choice is made by leave-one-out validation on the training people. A learned rule is kept only "
             "if it beats 'keep the default model' in cross-validation. There are two sets of rules: one for when the weight "
             "is known, and one for when it is not, which is built around CameraHMR. The rules below are read directly from "
             "the app's rule file (model/calibration/real_calibration.json), fitted on all 24 people."),
           Paragraph('<b>Weight entered</b>', S['p']),
           table(rules['with_weight'], [W * 0.13, W * 0.24, W * 0.63]),
           Paragraph('<b>No weight</b>', S['p']),
           table(rules['no_weight'], [W * 0.13, W * 0.24, W * 0.63]),
           N("Shoulder has only 6 tape values, too few for a rule, so the app shows HMR's value. In a blend the weights add up "
             "to 1; a negative weight means one model is used to correct another model's typical mistake. The sleeve rule with "
             "weight uses the BodyM prior; adding it as a choice lowered the sleeve error from 2.72 to 2.61 cm."),
           H2('4.8 Sizes and tape labels'),
           P("Chest and waist are turned into shirt and trouser sizes using size tables for several countries. Each value gets a "
             "label. The rule is strict: a body part is 'Camera OK' only if, in the leak-free test, 80% of the people were within "
             "a tolerance a tailor would accept for that body part. Otherwise it is 'Check with tape'."),
           table(tape_rows, [W * 0.16, W * 0.16, W * 0.18, W * 0.3, W * 0.2]),
           N('These are the numbers the app stores with its rules: one leak-free cross-validation split, so they differ a little '
             'from the 5-split averages in chapter 7.'),
           P(f"With the current 24 test people {'no body part' if n_cam == 0 else str(n_cam) + ' body parts'} passes this rule, "
             "so the app shows 'Check with tape' everywhere. Neck and thigh are the closest. This is deliberate: a shop should "
             "only trust the camera alone once it is proven on its own customers."),
           H2('4.9 Design choices and alternatives'),
           P("Several other designs were built and tested before settling on this one: five other ways of combining the models, a phone-only version, a network trained on 4,132 scanned bodies, and six ways of turning photos into Amazon-style silhouettes. Chapter 10 gives their results. The current design was kept because it had the lowest error on unseen people.")]

    # ── 4 system ─────────────────────────────────────────────────────────────
    ui = lambda a, b, cap: [Table([[Image(os.path.join(UI, a), width=W * 0.49, height=W * 0.49 * 720 / 1280),
                                    Image(os.path.join(UI, b), width=W * 0.49, height=W * 0.49 * 720 / 1280)]],
                                  colWidths=[W * 0.5, W * 0.5], style=[('LEFTPADDING', (0, 0), (-1, -1), 0)]),
                            Paragraph(cap, S['cap'])]
    st += [H1('5. Implementation'),
           H2('5.1 Tools and program structure'),
           P("TailorCam is a Windows desktop app written in Python with OpenCV for the interface and MediaPipe for finding the "
             "body. The heavy models run in a separate worker process with PyTorch on the graphics card, so the interface "
             f"stays responsive. On an {gpu} one customer takes about {total[COMB]:.0f} seconds."),
           H2('5.2 How a scan works'),
           P("The tailor opens or creates a customer and types the height (and weight if known). The scan screen shows an "
             "outline to stand in and a pose score, and a voice gives instructions. After a countdown the app takes the front "
             "photo, asks the customer to turn, and takes the side photo. Photos can also be uploaded instead of using the "
             "webcam; the app then checks which photo is the front and which is the side."),
           *ui('5_scan_guide.png', '6_turn.png', 'Figure 2. Left: the scan guide with the pose outline. Right: the turn instruction.'),
           H2('5.3 Results and shop features'),
           P("The results screen shows the 3D body, every measurement with a label ('Camera OK' or 'Check with tape'), "
             "clothing sizes by country, a size guide, and the customer's earlier scans. The tailor can type real tape values; "
             "the app saves them so the system can be retrained later on the shop's own customers. The app also keeps "
             "customers, orders, inventory and loyalty points in a local SQLite database and makes PDF reports."),
           *ui('8_results.png', '8b_size_guide.png', 'Figure 3. Left: results with the 3D body and tape labels. Right: size guide.'),
           *ui('3_customers.png', '13_order.png', 'Figure 4. Left: customer list. Right: order screen.'),
           Paragraph('<b>Shop offset: what it is and why we added it</b>', S['p']),
           P("<b>The problem.</b> Every tailor places the tape a little differently. One measures the waist at the navel, "
             "another lower; one measures the sleeve from the shoulder tip, another from the neck. The combiner learned its rules "
             "from the tape style of the real24 people. When a local tailor measured our own 10 people, chest, hip and neck were "
             "close, but waist, thigh, sleeve and shoulder were off in the <b>same direction for nearly every person</b>. That is "
             "not a camera error; it is a fixed difference in where this tailor puts the tape. Because of it, the app as it is "
             f"scored {e2e_raw:.1f}% on these people, below HMR alone (section 10.8)."),
           P("<b>The fix.</b> The shop offset adds one fixed correction per measurement, learned from the shop's own customers "
             "whose real tape values the tailor typed into the app: the average of (tape minus app) for that measurement. "
             "scripts/retrain_from_shop.py learns it, saves it in model/calibration/shop_offset.json, and the app applies it "
             "after the combiner (model/calibration.py, apply_shop_offset). Two safety rules: it needs at least 5 tape-measured "
             "customers, and a correction is kept only if it lowers the error when each customer is left out in turn, so a "
             "correction that does not help is never used."),
           P(f"<b>The result.</b> Learned from 9 people and tested on the 10th, repeated for all 10, inside the real app "
             f"pipeline: overall accuracy went from {e2e_raw:.1f}% to {e2e_off:.1f}%, the best of all models, with CameraHMR given the same "
             f"correction almost equal ({od_o2['CameraHMR']:.1f}%). It is cheap "
             "(one number per measurement, no retraining of any network) and it lets one app serve many shops, each with its "
             "own tape style. A new shop needs about 5 to 10 tape-measured customers; until then every value shows 'Check with "
             "tape'. A worked example is in section 6.2, and the full test is in section 10.8."),
           H2('5.4 Training setup'),
           P("The silhouette CNN copies were trained for 40 passes over the BodyM training people, in batches of 64, with the AdamW optimiser, a learning rate of 0.0003 and a one-cycle schedule, on the GPU. The BODY-fit network used the same settings. The two priors and the combiner are closed-form ridge regressions that fit in seconds on the CPU. HMR, SHAPY and CameraHMR were not retrained."),
           P(f"Each CNN copy is a ResNet-18 (about 11 million weights) started from ImageNet weights, with a one-channel input "
             f"for silhouettes. The loss is the Laplace negative log-likelihood, which trains both the value and its "
             f"uncertainty. 10% of the BodyM training people are held back for validation, and the copy with the lowest "
             f"validation error on chest, waist and hip is kept. One pass over the data takes about 15 seconds, so a copy trains "
             f"in about 10 minutes. The best validation errors were "
             f"{', '.join(f'{v:.2f}' for v in tsum['best val MAE (cm)'])} cm, reached at epochs "
             f"{', '.join(str(int(v)) for v in tsum['best epoch'])}."),
           figure(os.path.join(FIGP, 'training_curves.png'), W, 'Figure. Training loss and validation error of the three CNN '
                  'copies. The validation error stops improving after about 20 to 30 epochs, and the best checkpoint is kept, '
                  'so later epochs cannot over-fit the saved model.'),
           H2('5.5 Database design'),
           P("Shop data is stored in a local SQLite file (tailorcam.db) with these tables:"),
           table([['Table', 'What it stores'],
                  ['customers', 'name, phone, email, gender, age, weight, height'],
                  ['measurements', 'one row per scan: date, shoulder, chest, waist, hip, back and shirt length, sleeve, neck, height'],
                  ['orders', 'garment type, style notes, fabric, colour, price, linked to a customer and a scan'],
                  ['inventory', 'fabrics and materials with quantity, unit, minimum level, colour and price'],
                  ['staff_members', 'name, role and whether active'],
                  ['feedback', "the tailor's real tape value next to the camera value, with the error worked out automatically"]],
                 [W * 0.2, W * 0.8]),
           N('The feedback table is what lets the system be retrained on the shop\'s own customers later.'),
           H2('5.6 What I wrote and what I reused'),
           table([['Reused from others', 'Written by me'],
                  ['HMR trained weights (Kanazawa et al.)', 'HMR port from TensorFlow to PyTorch and weight conversion'],
                  ['SHAPY code and weights (Max Planck Institute)', 'running SHAPY on Windows, weight fitting for all 3D models'],
                  ['CameraHMR code and weights (Max Planck Institute)', 'MediaPipe person box and mirror averaging for CameraHMR'],
                  ['SMPL and SMPL-X body models', 'virtual tape measuring code on the 3D body'],
                  ['MediaPipe pose and segmentation (Google)', 'silhouette preparation, CNN design, augmentation and training'],
                  ['ResNet-18 design and ImageNet weights (torchvision)', 'BodyM and ANSUR priors, the combiner, leak-free evaluation'],
                  ['Public datasets (chapter 3)', 'dataset builders, all test scripts, experiments and reports'],
                  ['', 'the whole desktop app: scan flow, upload, results, database, orders, PDF reports']],
                 [W * 0.45, W * 0.55]),
           H2('5.7 How the system developed'),
           P("TailorCam went through three versions. Each one was kept in the code, so the history can be checked."),
           table([['Version', 'How it measured', 'What happened to it'],
                  ['1. Landmark pipeline', 'MediaPipe 3D body points from several views, smoothed with a Kalman filter; widths '
                   'and depths turned into girths with the Ramanujan ellipse formula; refined by a small neural network '
                   '(BodyMeasureNet, an MLP trained on synthetic bodies built from ANSUR II equations) and a Bayesian ridge '
                   'correction; finally a constraint solver nudged the values toward realistic body proportions',
                   'replaced; still in the app as a fallback when the 3D model cannot run'],
                  ['2. HMR 3D body', 'HMR predicts a 3D body from each photo, scaled to the height and fitted to the weight, '
                   'measured with a virtual tape', 'kept as one of the six models'],
                  ['3. Hybrid (current)', 'six models plus a learned combiner, evaluated with leak-free nested '
                   'cross-validation', 'the version described in this report']],
                 [W * 0.18, W * 0.52, W * 0.3]),
           N("Version 1 was not tested on the real24 set with the protocol of this report, so I make no accuracy claim for it. "
             "The feedback workflow from version 1 (the tailor types real tape values next to the camera values) survives in "
             "version 3 as the feedback table and scripts/retrain_from_shop.py.")]

    # ── 6 evaluation ─────────────────────────────────────────────────────────
    st += [H1('6. Testing method and accuracy calculation'),
           H2('6.1 How the test is set up'),
           P("<b>Nested cross-validation.</b> The 24 real people are split into 6 groups of 4. Each group is tested once while "
             "the combiner learns only from the other 20 people. Inside those 20, a second cross-validation chooses the rule "
             "and decides whether a learned rule is kept at all. So the 4 test people never influence anything about the "
             "prediction made for them. Because 24 people is a small number, the whole procedure is repeated with 5 different "
             "random splits and the errors are averaged."),
           P("<b>Single models</b> get exactly the same treatment: each may learn an offset on the training people, and keeps it "
             "only if that helps inside the training people."),
           P("<b>Outside test sets</b> (HBW, model agency, BEDLAM, BodyM test, BODY-fit test, NOMO-3D) are never used to train "
             "or tune anything; the finished models are simply run on them."),
           P("<b>Metrics.</b> Mean absolute error (MAE) in cm; accuracy = 100 minus the mean absolute percentage error; the "
             "share of people within 2 cm; clothing-size accuracy with precision, recall and F1; Bland-Altman limits of "
             "agreement; time per customer.")]

    # ── 6.2 accuracy calculation with worked examples ────────────────────────
    ex = pd.read_excel(os.path.join(ROOT, 'reports', 'TailorCam_Model_Data.xlsx'), sheet_name='COMBINED W')
    exr = ex[ex['Chest tape'].notna()].iloc[[0, 1, 7]]
    bins = [84, 92, 100, 108, 116]
    SZ = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
    lab = lambda v: SZ[int(np.digitize(v, bins))]
    erows = [['Person', 'Tape (cm)', 'TailorCam (cm)', 'Error (cm)', 'Error (%)', 'Accuracy', 'Tape size', 'Our size']]
    for r in exr.itertuples():
        e = abs(r[ex.columns.get_loc('Chest predicted') + 1] - r[ex.columns.get_loc('Chest tape') + 1])
        tp, pr = r[ex.columns.get_loc('Chest tape') + 1], r[ex.columns.get_loc('Chest predicted') + 1]
        erows.append([r.person_id, f'{tp:.1f}', f'{pr:.2f}', f'{e:.2f}', f'{e / tp * 100:.2f}%', f'{100 - e / tp * 100:.2f}%',
                      lab(tp), lab(pr)])
    t0 = exr.iloc[0]
    e0 = abs(t0['Chest predicted'] - t0['Chest tape'])
    loa = pd.read_csv(os.path.join(TAB, 'limits_of_agreement.csv'))
    lc = loa[(loa.test.str.contains('weight known')) & (loa.measurement == 'Chest')].iloc[0]
    st += [H2('6.2 How accuracy is calculated'),
           P("All scores start from one number per person: the difference between the TailorCam value and the tape. Here is "
             f"the chest of the first test person, {t0.person_id}, with the weight entered:"),
           P(f"&bull; Tape: {t0['Chest tape']:.1f} cm. TailorCam: {t0['Chest predicted']:.2f} cm.<br/>"
             f"&bull; Error = |{t0['Chest predicted']:.2f} - {t0['Chest tape']:.1f}| = <b>{e0:.2f} cm</b>.<br/>"
             f"&bull; Percentage error = {e0:.2f} / {t0['Chest tape']:.1f} x 100 = <b>{e0 / t0['Chest tape'] * 100:.2f}%</b>.<br/>"
             f"&bull; Accuracy for this person = 100 - {e0 / t0['Chest tape'] * 100:.2f} = <b>{100 - e0 / t0['Chest tape'] * 100:.2f}%</b>."),
           table(erows, [W * 0.12, W * 0.12, W * 0.14, W * 0.12, W * 0.12, W * 0.13, W * 0.12, W * 0.13]),
           N('Three real test people, chest, weight entered. Each value is the average over the 5 cross-validation splits.'),
           P("<b>From people to one score.</b> The mean absolute error (MAE) is the average of the per-person errors, and the "
             "accuracy is 100 minus the average percentage error (also written as 100 - MAPE). For the chest over all "
             f"{int(g('weight known', 'Chest', 'n'))} test people this gives MAE = {g('weight known', 'Chest', 'MAE (cm)'):.2f} cm "
             f"and accuracy = {g('weight known', 'Chest', 'accuracy (%)'):.2f}%. 'Within 2 cm' is the share of people whose error "
             f"is 2 cm or less ({g('weight known', 'Chest', '≤2cm (%)'):.0f}% for the chest). The whole calculation is repeated for "
             "each of the 5 random splits and averaged."),
           P("<b>Overall accuracy of a model</b> is the plain average of its body-part accuracies. For example, the hybrid with "
             f"weight: ({', '.join(f'{aw.loc[COMB, p_]:.2f}' for p_ in PARTS)}) / 7 = <b>{aw.loc[COMB, OV]:.2f}%</b>."),
           P("<b>Size accuracy.</b> The chest is turned into a shirt size with fixed bands: under 84 cm XS, 84 to 92 S, 92 to "
             "100 M, 100 to 108 L, 108 to 116 XL, 116 and above XXL. A person counts as correct when the size from TailorCam's "
             "value equals the size from the tape value, as in the last two columns above. 'Within 1 size' allows one band "
             "of difference."),
           P("<b>Precision, recall and F1</b> are worked out for each size. Precision is: of the people TailorCam called size M, "
             "how many really are M. Recall is: of the people who really are M, how many TailorCam called M. F1 = 2 x precision x "
             "recall / (precision + recall). The report gives the macro average, which treats every size equally."),
           P(f"<b>Bland-Altman agreement.</b> The bias is the average signed difference (TailorCam minus tape), and the limits of "
             f"agreement are bias plus or minus 1.96 standard deviations. For the chest with weight the bias is "
             f"{lc['bias (cm)']:.2f} cm and 95% of people fall between {lc['lower LoA']:.2f} and {lc['upper LoA']:.2f} cm."),
           P("<b>Time</b> is the wall-clock time for one customer with two photos, after a warm-up run, averaged over 20 runs."),
           P(f"<b>Worked example on our own data (odjag).</b> The tailor wrote values in inches, so each one is first "
             f"converted: cm = inches x 2.54. Height {od_ex.height_txt} = {od_ex.height_in:g} inches x 2.54 = "
             f"<b>{od_ex.height_cm:.1f} cm</b>. Chest {od_ex.chest_in:g} inches x 2.54 = <b>{od_ex.chest_cm:.2f} cm</b>. TailorCam "
             f"gave {od_ex.chest_app:.2f} cm, so the error is |{od_ex.chest_app:.2f} - {od_ex.chest_cm:.2f}| = "
             f"<b>{abs(od_ex.chest_app - od_ex.chest_cm):.2f} cm</b> and the accuracy is 100 - "
             f"{abs(od_ex.chest_app - od_ex.chest_cm) / od_ex.chest_cm * 100:.2f} = "
             f"<b>{100 - abs(od_ex.chest_app - od_ex.chest_cm) / od_ex.chest_cm * 100:.2f}%</b>."),
           P(f"<b>Shop offset, worked example.</b> For {od_ex.pid}'s waist, the offset is learned from the other {od_n - 1} people only: "
             f"the average of (tape - TailorCam) over them is {od_ex.w_off:+.2f} cm. TailorCam gave {od_ex.w_app:.2f} cm, so the "
             f"corrected value is {od_ex.w_app:.2f} {od_ex.w_off:+.2f} = <b>{od_ex.w_app + od_ex.w_off:.2f} cm</b> against a tape "
             f"value of {od_ex.w_tape:.2f} cm: the error falls from {abs(od_ex.w_app - od_ex.w_tape):.2f} to "
             f"{abs(od_ex.w_app + od_ex.w_off - od_ex.w_tape):.2f} cm. An offset is only kept for a body part if, done this way "
             "for every person, it lowers the average error. It is one fixed number, so it can still hurt a person who does "
             f"not follow the shop's usual pattern: for the waist that happened to {len(od_waist_worse)} of {od_n} people "
             f"({', '.join(od_waist_worse)}).")]

    # ── 7 leakage ────────────────────────────────────────────────────────────
    before = {'with': [2.51, 3.36, 3.15, 1.54, 1.37, 2.72], 'without': [3.41, 3.51, 4.49, 2.66, 1.74, 2.33]}
    keys6 = [('Chest', 'Chest'), ('Waist', 'Waist'), ('Hip', 'Hip'), ('Thigh', 'Thigh'), ('Neck', 'Neck'), ('Sleeve', 'Sleeve')]
    lrows = [['Hybrid', 'Chest', 'Waist', 'Hip', 'Thigh', 'Neck', 'Sleeve']]
    for lab, w_ in (('with', 'weight known'), ('without', 'weight unknown')):
        lrows.append([f'Before the fix, {lab} weight'] + [f'{v:.2f}' for v in before[lab]])
        lrows.append([f'<b>After the fix, {lab} weight</b>'] + [f"<b>{g(w_, m, 'MAE (cm)'):.2f}</b>" for _, m in keys6])
    st += [H2('6.3 Data leakage review'),
           P("Before the defense I went through the whole evaluation looking for leakage, meaning any way the test people "
             "could have influenced their own predictions."),
           table([['Check', 'Result'],
                  ['Same person in training and test (real24)', 'clean: nested cross-validation'],
                  ['Duplicate people in real24', 'clean: no repeated body records'],
                  ['Silhouette CNN', 'clean: checkpoint chosen on 10% of the training people; BodyM test sets untouched'],
                  ['BODY-fit and NOMO-3D overlap', 'clean: 292 shared people removed from training'],
                  ['SHAPY on the model agency set', 'clean: SHAPY was trained on the agency train split, tested on the test split'],
                  ['CameraHMR on BEDLAM', 'not fair: CameraHMR was trained on BEDLAM (stated wherever BEDLAM appears)'],
                  ['The "keep the rule only if it helps" step', '<b>leak found and fixed</b> (below)'],
                  ['Design choices tuned on the same 24 people', '<b>cannot be fully removed</b> (below)']],
                 [W * 0.4, W * 0.6]),
           H2('6.4 The leak and the fix'),
           P("The app keeps a learned rule only if it beat 'keep the default model' in cross-validation. My evaluation made "
             "that keep-or-not decision by looking at the errors on the outer test people, so the test people had a small say "
             "in their own predictions. I added a second, inner cross-validation (nested_cv_app in model/calibration.py) that "
             "makes the decision using only the training people, and re-ran every test in this report with it."),
           table(lrows, [W * 0.34] + [W * 0.11] * 6, hl=(2, 4)),
           N('Average error in cm, 24 real people. Every other number in this report uses the fixed method.'),
           P("Most values barely move. Chest with weight rises by about 0.1 cm and hip without weight by about 1 cm, so those "
             "two were the most optimistic before. The app itself did not change; only the estimate of its accuracy did."),
           H2('6.5 What cannot be fully removed'),
           P("Which models to use and which options to switch on were decided while looking at results on these same 24 "
             "people. That is common in small projects, and the only real cure is a fresh test set. The outside test sets are "
             "that check: on the model agency photos the hybrid has the lowest error of the methods I ran, but on the 10 HBW "
             "people with weight entered, SHAPY alone does better. I read that as a sign that some of my choices fit the 24 "
             "people a little too well.")]

    # ── 8 results ────────────────────────────────────────────────────────────
    rrows = [['Measurement', 'MAE, weight', 'Accuracy, weight', 'Within 2 cm', 'MAE, no weight', 'Accuracy, no weight']]
    for _, m in keys6:
        rrows.append([m, f"{g('weight known', m, 'MAE (cm)'):.2f} cm", f"{g('weight known', m, 'accuracy (%)'):.1f}%",
                      f"{g('weight known', m, '≤2cm (%)'):.0f}%", f"{g('weight unknown', m, 'MAE (cm)'):.2f} cm",
                      f"{g('weight unknown', m, 'accuracy (%)'):.1f}%"])
    kf = ksum[ksum.weight == 'known']
    srows = [['Size from', 'Weight', 'Right size', 'Within 1 size', 'Precision', 'Recall', 'F1']]
    for _, r in szr.iterrows():
        srows.append([r['size from'].title(), 'entered' if r.weight == 'weight known' else 'not entered', f"{r['accuracy (%)']:.0f}%",
                      f"{r['within ±1 size (%)']:.0f}%", f"{r['precision (macro)']:.2f}", f"{r['recall (macro)']:.2f}", f"{r['F1 (macro)']:.2f}"])
    st += [H1('7. Results of the final system'),
           P('All numbers are on people the system never learned from, with the leak-free method.'),
           table(rrows, [W * 0.17] + [W * 0.166] * 5),
           N(f"Shoulder: {mw.loc[COMB, 'Shoulder']:.2f} cm with weight (HMR's value, only 6 people)."),
           H2('7.1 Fold by fold (standard 5-fold, weight entered)'),
           table([['Measurement', 'Accuracy mean', 'Spread (sd)', 'MAE mean']] +
                 [[r.measurement, f"{r['accuracy (%) mean']:.1f}%", f"{r['accuracy (%) std']:.1f}%", f"{r['MAE (cm) mean']:.2f} cm"]
                  for _, r in kf.iterrows()], [W * 0.25] * 4),
           H2('7.2 Clothing sizes'),
           table(srows, [W * 0.13, W * 0.15, W * 0.14, W * 0.15, W * 0.14, W * 0.14, W * 0.15]),
           N("With the standard 5-fold split instead of the repeated 6-fold one, the chest gives the right size for "
             f"{kfs.iloc[0]['accuracy (%)']:.0f}% of people with weight (F1 {kfs.iloc[0]['F1 (macro)']:.2f}) and the waist for "
             f"{kfs.iloc[1]['accuracy (%)']:.0f}% (F1 {kfs.iloc[1]['F1 (macro)']:.2f})."),
           P(f"With the weight entered, the chest gives the right shirt size for {sz('weight known', 'chest', 'accuracy (%)'):.0f}% "
             "of people and is never more than one size off. Without the weight this drops to "
             f"{sz('weight unknown', 'chest', 'accuracy (%)'):.0f}%, so entering the weight matters."),
           figure(os.path.join(FIGP, 'bland_altman.png'), W, 'Figure 5. Bland-Altman plots: each dot is one person; the dashed '
                                                            'lines show where 95% of the differences from the tape fall.'),
           figure(os.path.join(FIGP, 'confusion_matrices.png'), W, 'Figure. Size confusion matrices (rows = size from the '
                  'tape, columns = size from TailorCam). Numbers on the diagonal are correct sizes.'),
           H2('7.3 Outside test sets'),
           table([['Test set', 'Chest', 'Waist', 'Hip'],
                  ['HBW 3D scans, weight entered (10)', '4.20 cm', '5.47 cm', '5.09 cm'],
                  ['HBW 3D scans, no weight (10)', '4.77', '9.36', '3.34'],
                  ['Model agency, no weight (134)', '6.84', '6.48', '8.55'],
                  ['BodyM test B, silhouette CNN, weight (1,160 pairs)', '2.47', '2.75', '2.04'],
                  ['BodyM test A, silhouette CNN, weight (1,684 pairs)', '2.29', '1.82', '1.75']] +
                 [[f'BEDLAM* (42), weight {"entered" if r.weight == "weight known" else "not entered"}', f'{r.Chest:.2f}', f'{r.Waist:.2f}', f'{r.Hip:.2f}']
                  for r in sad[sad.dataset.str.startswith('BEDLAM')].itertuples()],
                 [W * 0.46, W * 0.18, W * 0.18, W * 0.18]),
           N("Model agency photos are fashion poses and the truth is an agency size list, not a tape, so errors are larger. "
             "On Amazon's own clean silhouettes the CNN reaches about 97.3% (test B) to 97.7% (test A) accuracy over 13 body parts. "
             "* CameraHMR was trained on BEDLAM, so BEDLAM is optimistic."),
           P("<b>Does the CNN add anything over height and weight?</b> On Amazon's test people, yes: the outlines lower every girth "
             "error compared with the height-and-weight formula alone."),
           table([['BodyM test', 'Method', 'Chest', 'Waist', 'Hip', 'Thigh', 'Arm']] +
                 [[r['test set'], 'height + weight only' if 'prior' in r.method else 'silhouette CNN x3'] +
                  [f"{r[c]:.2f}" for c in ('chest', 'waist', 'hip', 'thigh', 'arm-length')] for _, r in bmt.iterrows()],
                 [W * 0.14, W * 0.26] + [W * 0.12] * 5),
           P("<b>Why three copies?</b> Each copy alone, and their average, on BodyM test B:"),
           table([['Copy', 'Chest', 'Waist', 'Hip']] + [[r.model, f'{r.chest:.2f}', f'{r.waist:.2f}', f'{r.hip:.2f}'] for r in cnnm.itertuples()],
                 [W * 0.4, W * 0.2, W * 0.2, W * 0.2]),
           N('Averaging is about as good as the best single copy and removes the risk of picking an unlucky one.'),
           H2('7.4 Confidence intervals and agreement'),
           P("With only 24 people, a single average can mislead, so each result also has a 95% confidence interval (bootstrap, "
             "2,000 resamples of the people), the RMSE (which punishes large errors more), the bias (average signed error) "
             "and R2 (how much of the person-to-person variation is explained)."),
           table(cis, [W * 0.14, W * 0.13, W * 0.12, W * 0.19, W * 0.14, W * 0.14, W * 0.14]),
           P("The intervals are wide, for example the chest error with weight could be anywhere from about 1.9 to 3.4 cm, "
             "which is the honest effect of a small test set. Biases are close to zero, so TailorCam does not measure "
             "everyone too big or too small. Sleeve has a low R2: its errors are small, but so is the variation between people."),
           H2('7.5 Results by gender and body size'),
           table(grows, [W * 0.4, W * 0.12, W * 0.16, W * 0.16, W * 0.16]),
           N('Average error in cm, 24 real people (10 female, 14 male), leak-free cross-validation.'),
           figure(os.path.join(FIGP, 'error_analysis.png'), W, 'Figure. Error by gender, by BMI and by height (weight entered).'),
           P("With the weight entered, errors are larger for women on the chest and hip, and larger for people with a higher "
             "BMI. Without the weight the pattern changes, and men have the larger chest error. The groups are small (10 and "
             "14 people), so these differences are a warning sign to check with more data, not a firm finding. The Spearman "
             "correlation between the error and BMI is " + ', '.join(f"{r.measurement.lower()} {r[1]:+.2f}" for r in ecor.itertuples(index=False)) +
             ", and with height " + ', '.join(f"{r.measurement.lower()} {r[2]:+.2f}" for r in ecor.itertuples(index=False)) +
             "; only the hip error clearly shrinks for taller people."),
           H2('7.6 Is the difference significant?'),
           P("I compared the hybrid with the best single model, CameraHMR, person by person, with the Wilcoxon signed-rank "
             "test, which does not assume the errors follow a normal curve."),
           table(wrows, [W * 0.14, W * 0.16, W * 0.1, W * 0.15, W * 0.17, W * 0.12, W * 0.16]),
           P("Only the hip with the weight entered is clearly significant, in favour of the hybrid. The other differences are "
             "within what 16 to 20 people can tell apart. So the honest claim is that the hybrid is at least as good as the "
             "best single model on every body part and clearly better on the hip with weight, not that it beats it everywhere."),
           H2('7.7 Too big or too small?'),
           P("Accuracy and MAE ignore the direction of the error. Here I keep the sign, to see whether TailorCam usually "
             "measures bigger or smaller than the tape. The calculation, for each person and body part:"),
           P("&bull; Signed error = TailorCam - tape. Positive = too big, negative = too small.<br/>"
             f"&bull; Example, too big: {b_over.person_id}, chest tape {b_over['Chest tape']:.1f} cm, TailorCam "
             f"{b_over['Chest predicted']:.2f} cm, signed error = {b_over['Chest predicted']:.2f} - {b_over['Chest tape']:.1f} = "
             f"<b>{b_over.ce:+.2f} cm</b>.<br/>"
             f"&bull; Example, too small: {b_under.person_id}, chest tape {b_under['Chest tape']:.1f} cm, TailorCam "
             f"{b_under['Chest predicted']:.2f} cm, signed error = <b>{b_under.ce:+.2f} cm</b>.<br/>"
             "&bull; Bias = average of the signed errors. A bias near 0 means the too-big and too-small errors cancel out.<br/>"
             "&bull; 'Too big' counts the people with a signed error above +0.05 cm, and 'Avg. too big' is their average; the "
             "same for 'too small' below -0.05 cm."),
           Paragraph('<b>Weight entered (24 real people, leak-free)</b>', S['p']),
           table(bias_tabs['W'], [W * 0.14, W * 0.08, W * 0.08, W * 0.12, W * 0.12, W * 0.12, W * 0.13, W * 0.1, W * 0.11]),
           Spacer(1, 4), Paragraph('<b>No weight</b>', S['p']),
           table(bias_tabs['noW'], [W * 0.14, W * 0.08, W * 0.08, W * 0.12, W * 0.12, W * 0.12, W * 0.13, W * 0.1, W * 0.11]),
           N('All values in cm.'),
           figure(os.path.join(M.FIG, 'bias_dots.png'), W, 'Figure. Signed error of every person for every body part.'),
           P("<b>Answer.</b> TailorCam is not always too big or always too small. For every body part except the shoulder, "
             "roughly half the people come out too big and half too small, by similar amounts, so the bias is close to zero "
             "(within about 0.3 cm with weight). That is expected: the combiner learns an offset that removes any constant "
             "difference from the tape. What is left is a random error from person to person, mostly from clothes and body "
             "shape the photo cannot show. Two exceptions: the shoulder comes out too big for 5 of 6 people (about +2.5 cm), "
             "because the app shows HMR's raw value with no learned offset; and without weight the chest leans too big "
             "(+1.25 cm) and a few hips are much too small (largest -26.5 cm for one person), which is why the weight matters.")]

    # ── 9 every model ────────────────────────────────────────────────────────
    trows = [['Model', 'Own time (s)', 'sd (s)', 'With photo step (s)']]
    for m_ in MODELS:
        trows.append([m_, f'{own[m_]:.3f}', f'{sd[m_]:.3f}', f'{total[m_]:.2f}'])
    orows = [['Model', 'Parts', 'Overall, weight', 'Overall, no weight', 'Chest+waist+hip, weight', 'Chest+waist+hip, no weight']]
    for m_ in MODELS:
        orows.append([m_, f"{int(aw.loc[m_, 'Parts measured'])}", fmt(aw.loc[m_, OV], 2, '%'), fmt(anw.loc[m_, OV], 2, '%'),
                      fmt(aw.loc[m_, 'Overall (chest, belly, hip)'], 2, '%'), fmt(anw.loc[m_, 'Overall (chest, belly, hip)'], 2, '%')])
    best_nw = anw.loc[MODELS[:-1], OV].idxmax()
    st += [H1('8. Every model tested separately'),
           P("My supervisor asked for each model to be tested on its own, body part by body part, with its time. The "
             "per-person data sheets are in reports/TailorCam_Model_Data.xlsx."),
           H2('8.1 Error of each body part'),
           Paragraph('<b>Weight entered: average error (cm)</b>', S['p']),
           table([['Model'] + PARTS] + grid(mw, PARTS, 'min', 2), [W * 0.23] + [W * 0.11] * 7, hl=(7,)),
           Spacer(1, 4), Paragraph('<b>No weight: average error (cm)</b>', S['p']),
           table([['Model'] + PARTS] + grid(mnw, PARTS, 'min', 2), [W * 0.23] + [W * 0.11] * 7, hl=(7,)),
           N('Bold = best for that body part. "-" = the model does not measure it.'),
           figure(f['heat_w'], W, 'Figure 6. Accuracy heat map, weight entered. Darker = more accurate.'),
           figure(f['heat_nw'], W, 'Figure 7. Accuracy heat map, no weight.'),
           figure(f['parts_w'], W, 'Figure 8. Average error per body part for each model, weight entered.'),
           H2('8.2 Time'),
           P(f"Measured on the {gpu} for one customer with two photos, after a warm-up run, averaged over "
             f"{int(t['runs'].iloc[0])} runs. The shared photo step takes {photo:.2f} s. Loading all models at start-up takes "
             f"{t.loc['Model loading (once, at app start)', 'mean_s']:.0f} s once."),
           table(trows, [W * 0.4, W * 0.2, W * 0.15, W * 0.25], hl=(7,)),
           figure(f['time'], W, 'Figure 9. Seconds per customer. Grey = shared photo step.'),
           H2('8.3 Overall accuracy'),
           table(orows, [W * 0.26, W * 0.08, W * 0.15, W * 0.17, W * 0.17, W * 0.17], hl=(7,)),
           figure(f['overall'], W, 'Figure 10. Overall accuracy per model.'),
           figure(f['vs'], W, 'Figure 11. Error of the best single model (grey) and the hybrid (blue) for each body part.'),
           H2('8.4 Which model is best'),
           P(f"With the weight entered the hybrid has the best accuracy on chest, waist and hip together "
             f"({aw.loc[COMB, 'Overall (chest, belly, hip)']:.2f}%), is by far the best on the thigh, and has the best overall accuracy of the models "
             f"that measure all seven parts ({aw.loc[COMB, OV]:.2f}%). Single models are better on shoulder and sleeve. Without the weight the "
             f"hybrid is again the best overall ({anw.loc[COMB, OV]:.2f}% against {anw.loc[best_nw, OV]:.2f}% for {best_nw}), "
             "though CameraHMR alone has slightly lower chest and hip errors. CameraHMR is the best single model and also the slowest "
             f"({total['CameraHMR (2025)']:.1f} s). The two height-and-weight models are almost instant but cannot see body "
             "shape. HMR, the oldest model, is the weakest. No single model is best on every body part, which is the reason "
             "for combining them."),
           H2('8.5 What each trained part adds to the final result'),
           P("To see what each part is worth, I removed one model at a time from the combiner, re-ran the full leak-free test, "
             "and measured how much the error changes. A positive number means the error goes up without that model, so the "
             "model helps. A negative number means the result is slightly better without it."),
           table([['Model removed', 'Weight', 'Chest', 'Waist', 'Hip', 'Neck', 'Thigh', 'Sleeve']] +
                 [[{'ansur': 'ANSUR prior', 'camerahmr': 'CameraHMR', 'hmr': 'HMR', 'hybrid': 'Silhouette CNN', 'prior': 'BodyM prior',
                    'shapy': 'SHAPY'}[r.removed], r.weight] +
                  [('<b>+%.2f</b>' % v) if v > 0.05 else '%+.2f' % v for v in (r.Chest, r.Waist, r.Hip, r.Neck, r.Thigh, r.Sleeve)]
                  for r in abl.itertuples()],
                 [W * 0.2, W * 0.12] + [W * 0.113] * 6),
           N('Change in average error (cm) on the 24 real people when the model is removed. Bold = clear help (more than 0.05 cm).'),
           P("Every model helps at least one body part. With the weight entered, the BodyM prior and the silhouette CNN carry "
             "the waist, and HMR, the CNN and ANSUR carry the thigh. CameraHMR adds almost nothing when the weight is known, but "
             "without the weight it is by far the most important model: removing it raises the waist error by about 2.6 cm and "
             "the hip error by about 1.5 cm. So the parts trained on BodyM and ANSUR do most of the work with weight, and the "
             "published 3D models do most of the work without it."),
           H2('8.6 What each training gave: without it and with it'),
           P("For every part I trained, the table compares what TailorCam would have without that training with what it has "
             "with it, and what removing it does to the final combined result (section 8.5). The 3D models were not trained "
             "here: training one needs millions of images, so their published weights are used and fitted to height and weight."),
           table([['Trained part (data)', 'Without this training', 'With this training', 'Effect on the combined result'],
                  ['Silhouette CNN (BodyM)', f"height and weight only: waist {bm('testA', 'prior', 'waist'):.2f} cm on Amazon test A",
                   f"reads the outline: waist {bm('testA', 'hybrid', 'waist'):.2f} cm", f"removing it: waist +{ab('hybrid', 'known', 'Waist'):.2f} cm, thigh +{ab('hybrid', 'known', 'Thigh'):.2f} cm"],
                  ['BodyM prior (BodyM)', 'only the ANSUR prior as a photo-free guess', f"chest {bm('testA', 'prior', 'chest'):.2f} cm from height and weight",
                   f"removing it: waist +{ab('prior', 'known', 'Waist'):.2f} cm"],
                  ['ANSUR prior (ANSUR II)', 'no tape-style fallback', f"chest {ans.loc['chest', 'known']:.2f} cm, neck {ans.loc['neck', 'known']:.2f} cm on unseen soldiers",
                   f"removing it: neck +{ab('ansur', 'known', 'Neck'):.2f} cm, thigh +{ab('ansur', 'known', 'Thigh'):.2f} cm"],
                  ['Combiner (real24)', f"plain average: {ensv('with_weight', 'mean'):.2f} cm (weight), {ensv('no_weight', 'mean'):.2f} cm (no weight)",
                   f"learned rules: {ensv('with_weight', 'current'):.2f} cm and {ensv('no_weight', 'current'):.2f} cm", 'it is the combined result'],
                  ['BODY-fit network (BODY-fit)', 'not part of the app', 'good on scans (chest 2.44 cm)',
                   f"no gain: {bacc('with_weight', 'hybrid'):.2f}% without, {bacc('with_weight', 'hybrid+bodyfit'):.2f}% with, so not added"],
                  ['Shop offset (own photos)', f"{e2e_raw:.1f}% on the own photos", f"{e2e_off:.1f}%", 'adapts the result to one tailor']],
                 [W * 0.2, W * 0.27, W * 0.26, W * 0.27]),
           N('Errors are average absolute errors in cm. Combiner numbers are the average over all body parts (section 10.1).'),
           P("In short: the networks trained on BodyM give TailorCam the ability to read body shape from the outline, the ANSUR "
             "prior gives a reliable answer from height and weight, the combiner turns all of them into one better answer than "
             "any simple average, and the shop offset makes it match the tailor who uses it. The BODY-fit training is the one "
             "that did not help, and it was left out.")]

    # ── 10 comparison ────────────────────────────────────────────────────────
    st += [H1('9. Comparison with published methods'),
           P("To compare fairly I ran three published methods myself, with their authors' released code and trained weights, "
             "on exactly the same people as TailorCam, with the same height, weight and virtual tape. Small changes were needed "
             "to run them here: HMR was converted to PyTorch with its original weights, and CameraHMR uses MediaPipe to find "
             "the person. Errors are average absolute errors in cm on chest, waist and hip.")]
    summ = [['Test set', 'Best other method', 'TailorCam', 'TailorCam better?']]
    for i, (lab, a, o) in enumerate(sets):
        rows = [['Method', 'Published', 'Chest', 'Waist', 'Hip', 'Average']]
        for nm, venue, _ in PUB:
            v = a[nm]
            rows.append([nm, venue] + [f'{x:.2f}' for x in v] + [f'{np.mean(v):.2f}'])
        rows.append(['<b>TailorCam</b>', '2026'] + [f'<b>{x:.2f}</b>' for x in o] + [f'<b>{np.mean(o):.2f}</b>'])
        st += [KeepTogether([H2(f'9.{i + 1} {lab}'), table(rows, [W * 0.24, W * 0.16, W * 0.15, W * 0.15, W * 0.15, W * 0.15], hl=(4,))])]
        bn = min(a, key=lambda k: np.mean(a[k]))
        summ.append([lab, f'{bn} {np.mean(a[bn]):.2f} cm', f'{np.mean(o):.2f} cm', 'Yes' if np.mean(o) < np.mean(a[bn]) else 'No'])
    st += [KeepTogether([H2('9.6 Summary'), table(summ, [W * 0.4, W * 0.24, W * 0.16, W * 0.2]),
           P(f"TailorCam is better than all three published methods in {wins} of 5 tests. It loses on the 10 HBW scans with "
             "the weight entered, where SHAPY is better, and on the 24 real people without weight, where CameraHMR alone is "
             "slightly better (the difference is about 0.1 cm). It wins clearly on the 24 real people with weight and on the "
             "134 model-agency photos.")])]

    # ── 11 experiments ───────────────────────────────────────────────────────
    lw = lite[lite['mode'] == 'with_weight'].set_index('part')
    lnw = lite[lite['mode'] == 'no_weight'].set_index('part')
    phone = [['Body part', 'Phone, weight', 'PC, weight', 'Phone, no weight', 'PC, no weight']]
    for p_ in PARTS:
        phone.append([p_, f"{lw.loc[p_, 'MAE']:.2f}", f'{mw.loc[COMB, p_]:.2f}', f"{lnw.loc[p_, 'MAE']:.2f}", f'{mnw.loc[COMB, p_]:.2f}'])
    phone.append(['Overall accuracy', f"{lw['accuracy'].mean():.2f}%", f'{aw.loc[COMB, OV]:.2f}%',
                  f"{lnw['accuracy'].mean():.2f}%", f'{anw.loc[COMB, OV]:.2f}%'])
    bf = bfc.set_index(['mode', 'part', 'variant'])
    bacc = lambda mode, v: bfc[(bfc['mode'] == mode) & (bfc.variant == v)]['acc'].mean()
    erow = lambda mode: ens[(ens['mode'] == mode) & (ens['measurement'] == 'AVERAGE')] if 'mode' in ens else None
    st += [H1('10. Further experiments'),
           H2('10.1 Six ways of combining the models'),
           P("The combiner is not a simple average. I compared it with five other ways of combining, all learned on training "
             "people only and tested on unseen people over 5 splits: the plain mean, the median, weighting each model by its "
             "accuracy, a small neural network stacker, and picking the single best model automatically. The current rule had "
             "the lowest average error both with weight (2.45 cm against 2.56 cm or more) and without (3.35 cm against 3.49 cm "
             "or more). Simple averaging was clearly worst without weight (4.32 cm), because weak models pull the answer away."),
           H2('10.2 A phone-only version'),
           P("CameraHMR and SHAPY need a graphics card, but MediaPipe, the silhouette CNN, the two priors and the combiner are "
             "small enough for a phone. I trained a combiner with only those models and tested it the same way:"),
           table(phone, [W * 0.2] * 5, hl=(8,)),
           P(f"With the weight the phone version is close to the PC ({lw['accuracy'].mean():.1f}% against {aw.loc[COMB, OV]:.1f}%). "
             f"Without the weight it falls to {lnw['accuracy'].mean():.1f}%, because CameraHMR is what keeps the PC version "
             "accurate then. A phone app should therefore ask for the weight."),
           H2('10.3 Training on 4,132 scanned bodies'),
           P("To see whether more training data helps, I trained a second silhouette network on BODY-fit: 4,132 real people "
             "with 3D scans and scanner measurements. Each fitted body was drawn as a front and side silhouette in the same "
             "format as the app's masks. On scanned bodies it works well: on 558 held-out BODY-fit people the chest, waist and "
             "hip errors are 2.44, 2.40 and 1.74 cm with weight, and on 356 new NOMO-3D people without weight 3.74, 3.90 and "
             "3.49 cm, about half the error of a guess from height and gender. On the 24 real people, however, adding it to "
             f"the hybrid changed the overall accuracy from {bacc('with_weight', 'hybrid'):.2f}% to "
             f"{bacc('with_weight', 'hybrid+bodyfit'):.2f}% with weight and from {bacc('no_weight', 'hybrid'):.2f}% to "
             f"{bacc('no_weight', 'hybrid+bodyfit'):.2f}% without, so I did not add it to the app. Scan silhouettes come from "
             "people in underwear, while customers wear normal clothes, and only about half of the real people stood with their "
             "arms away from the body."),
           H2('10.4 Turning real photos into Amazon-style silhouettes'),
           P("I tested six ways of making an Amazon-style silhouette from a real photo: the MediaPipe mask (always, or only with "
             "arms away), a SAM 2 mask, and rebuilding the body in 3D with CameraHMR or HMR, posing it like Amazon's people "
             "with the arms away, and drawing its outline. The rebuilt CameraHMR outline was the most robust: on HBW it gave "
             "the lowest CNN error of all six (chest 2.20, waist 3.19, hip 3.96 cm with weight), while plain masks broke badly "
             "on everyday photos. Without weight, any real silhouette helped the CNN a lot on the 24 people (from 92.1% to about "
             "94.4% accuracy). Inside the full hybrid, though, no variant beat the current app by more than about 0.3%, which is "
             "within chance for 24 people, so the app was not changed. The code is in experiments/photo_to_amazon."),
           H2('10.5 Every idea tried, and what happened'),
           P("Every idea below was built and tested on people it had not seen. An idea was kept only if it lowered the error. "
             "Rows marked * were measured with the original test, before the leakage fix of section 6.4; they were not re-run "
             "because they were already rejected."),
           table([['Idea', 'Result', 'Decision'],
                  ['CameraHMR mirror averaging (also run on the flipped photo)*', 'no-weight hip 4.93 to 4.49 cm, chest 3.52 to 3.41 cm', 'kept'],
                  ['Weight fitting of every 3D body', 'large gain for HMR, which alone pulls bodies toward average', 'kept'],
                  ['Arms-away rule for the silhouette CNN (0.45)', 'avoids outlines where arms hide the waist (section 10.4)', 'kept'],
                  ['Six ways of combining the models', 'current rule best: 2.45 / 3.35 cm (section 10.1)', 'kept'],
                  ['SAM 2 sharper silhouettes*', 'no gain, waist slightly worse', 'off'],
                  ['Fit one 3D body to the outlines of all photos*', 'fits the clothes: no-weight chest 3.6 to 6.3 cm', 'off'],
                  ['Model disagreement as a per-person confidence*', 'rank correlation with the real error 0.3 or less', 'not used'],
                  ['Tape one value, correct the others*', 'helped fashion models only', 'not used'],
                  ['Train the no-weight combiner on model-agency people*', 'worse on ordinary people', 'not used'],
                  ['NHANES prior, about 28,000 people*', 'waist 3.64 to 4.21 cm', 'off'],
                  ['One combiner on all 781 people of all datasets*', 'hip 3.15 to 3.58 cm', 'off'],
                  ['CameraHMR person-box size*', 'no consistent difference', 'default kept'],
                  ['Combiner scale factor*', 'chest 2.51 to 2.72 cm', 'off'],
                  ['Body density 950 to 1,070 kg per cubic metre*', 'no difference after the combiner', '1,010 kept'],
                  ['Learn the tape height on the 3D body*', 'real24 waist 3.55 to 2.38 cm but HBW 3.90 to 5.31 cm (over-fit)', 'off'],
                  ['Combiner data augmentation (noise, mixup)*', 'gains traded for losses (no-weight hip 4.49 to 5.19 cm)', 'off'],
                  ['Smaller rule sets / "average all models" rule*', 'chest 2.51 to 2.42 cm but thigh 1.54 to 2.35 cm', 'off'],
                  ['Accuracy-weighted rule', 'sleeve slightly better, waist and no-weight neck worse', 'off'],
                  ['Phone-only version (section 10.2)', '95.0% with weight, 92.9% without', 'future work'],
                  ['Network trained on 4,132 scanned bodies (section 10.3)', 'good on scans, no gain on real photos', 'not added'],
                  ['Amazon-style silhouettes from photos, 6 ways (section 10.4)', 'best on HBW, no gain in the hybrid', 'not added'],
                  ['Pose check that refuses photos with arms down', 'proposed; cannot be tested without new photos', 'not added'],
                  ['Removing each model in turn (section 8.5)', 'every model helps at least one body part', 'all kept']],
                 [W * 0.42, W * 0.4, W * 0.18]),
           H2('10.6 Trained networks on datasets they never saw'),
           P("If a network has really learned body shape, it should also work on bodies from a different dataset. I ran each "
             "trained network on every other dataset its input allows. Because each dataset measures in its own way, one offset "
             "per measurement was learned on the training split of the target dataset (never on its test people)."),
           table(XTAB, [W * 0.24, W * 0.12, W * 0.22] + [W * 0.07] * 6),
           N('Average error in cm after the offset. Silhouette CNN = the network trained on Amazon BodyM.'),
           P("The Amazon-trained CNN measures BODY-fit and NOMO-3D bodies about twice as well as height and weight alone (on "
             "NOMO-3D without weight: chest 3.76 against 6.36 cm, waist 3.79 against 7.98 cm), and the BODY-fit network measures "
             "Amazon's test people well too. So the training is not tied to one dataset: the networks learned body shape that "
             "carries over. What does not carry over well is the jump from these clean outlines to real photos of clothed people, "
             "which is why the final combiner is trained on real photos."),
           H2('10.7 Every model considered'),
           table([['Model', 'Kind', 'Status', 'Reason'],
                  ['HMR (2018)', '3D body from photo', 'used', 'helps the thigh (section 8.5)'],
                  ['SHAPY (2022)', '3D body from photo', 'used', 'best single model for chest with weight'],
                  ['CameraHMR (2025)', '3D body from photo', 'used', 'best single model, most important without weight'],
                  ['Silhouette CNN x3 (mine)', 'outline network', 'used', 'carries waist and thigh with weight'],
                  ['BodyM prior (mine)', 'height and weight', 'used', 'carries waist with weight'],
                  ['ANSUR prior (mine)', 'height and weight', 'used', 'best for neck; helps thigh'],
                  ['Combiner (mine)', 'final step', 'used', 'best of six combining methods'],
                  ['BODY-fit network (mine)', 'outline network', 'trained, not added', 'no gain on real photos (10.3)'],
                  ['SAM 2 (Meta, 2024)', 'cut-out', 'tested, not added', 'no gain (10.4, 10.5)'],
                  ['HMR 2.0 (ICCV 2023)', '3D body from photo', 'considered, not added', 'no official Windows-ready copy could be installed; CameraHMR is newer and of the same design'],
                  ['TokenHMR, Multi-HMR, PromptHMR, SMPLest-X, GenHMR', '3D body from photo', 'considered', 'mainly improve pose, not body shape; registration or Linux setups needed'],
                  ['body2fit (2026, GitHub)', 'measurement from photos', 'could not run', "running another developer's model was blocked on this PC"],
                  ['MeasureNet (Amazon, 2023)', 'measurement from outlines', 'not available', 'code and training data are private'],
                  ['3DLOOK, Bodygram, MTailor', 'commercial apps', 'not available', 'closed and paid'],
                  ['Version 1 models (BodyMeasureNet MLP, Bayesian ridge, constraint solver, Kalman filter)', 'landmark pipeline', 'replaced', 'kept only as a fallback (section 5.7)']],
                 [W * 0.3, W * 0.17, W * 0.17, W * 0.36]),
           H2('10.8 Test on our own photos (odjag set)'),
           P(f"As a final real-world check, a tailor measured {od_n} people with a tape (values written by hand in inches and "
             "converted to cm), and each gave one front photo; the name written inside each photo links it to the person. "
             "There was no weight and no side photo, so the app ran in its no-weight mode with its rules unchanged: these "
             "people are completely new to it. The full report is reports/TailorCam_Odjag_Test.pdf."),
           table(od_rows, [W * 0.14, W * 0.18, W * 0.13, W * 0.13, W * 0.21, W * 0.21]),
           N(f"The shop offset is learned from {od_n - 1} people and tested on the remaining one, repeated for all {od_n}."),
           Spacer(1, 4), Paragraph('<b>Too big or too small (app as it is)</b>', S['p']),
           table(od_bias, [W * 0.14, W * 0.08, W * 0.08, W * 0.12, W * 0.12, W * 0.12, W * 0.13, W * 0.1, W * 0.11]),
           P("Chest, hip and neck were close straight away. Waist, thigh, sleeve and shoulder were off in the <b>same "
             "direction for nearly every person</b>. Unlike the 24-person test, where errors went both ways, this is a "
             "constant difference in where this tailor places the tape."),
           Paragraph('<b>Why HMR alone looks better here than the app as it is</b>', S['p']),
           table([['', 'App as it is', 'With shop offset']] +
                 [[m, f"{od_o1[m]:.1f}%", f"{od_o2[m]:.1f}%"] for m in ('TailorCam', 'HMR', 'CameraHMR')],
                 [W * 0.4, W * 0.3, W * 0.3]),
           P(f"Without a shop offset, HMR alone ({od_o1['HMR']:.1f}%) beats the app ({od_o1['TailorCam']:.1f}%). It wins on only "
             f"two body parts: thigh (HMR {od_raw[('HMR', 'thigh')][0]:.2f} cm against TailorCam {od_raw[('TailorCam', 'thigh')][0]:.2f} cm) "
             f"and sleeve ({od_raw[('HMR', 'sleeve_length')][0]:.2f} against {od_raw[('TailorCam', 'sleeve_length')][0]:.2f} cm); on chest, "
             f"hip and neck TailorCam is better (hip {od_raw[('TailorCam', 'hip')][0]:.2f} against {od_raw[('HMR', 'hip')][0]:.2f} cm). "
             "The reason is the combiner's learned corrections, made for the tape style of the first 24 people. For the "
             "sleeve, the no-weight rule adds 5.48 cm to HMR, because on those 24 people HMR measured the arm too short; this "
             f"tailor measures the arm shorter, so HMR is already {od_raw[('HMR', 'sleeve_length')][2]:+.1f} cm and the app ends at "
             f"{od_raw[('TailorCam', 'sleeve_length')][2]:+.1f} cm. For the thigh, the rule leans partly on the ANSUR prior, which is "
             f"{od_raw[('ANSUR prior', 'thigh')][2]:+.1f} cm here. HMR has no learned corrections, so it happens to sit closer to this "
             "tailor. It is not reading the body better: HMR is the weakest model on the 24-person test "
             f"({aw.loc['HMR (2018)', OV]:.1f}% with weight against {aw.loc[COMB, OV]:.1f}% for TailorCam) and on the combined dataset "
             f"({cmb_ov['HMR']:.1f}% against {cmb_ov['TailorCam (combined)']:.1f}%), and once the tape-style shift is removed with a "
             f"shop offset, TailorCam is ahead again ({od_o2['TailorCam']:.1f}% against {od_o2['HMR']:.1f}%)."),
           P("<b>Impact and the fix.</b> In a new shop the app should not be trusted alone at first: every value already shows "
             "'Check with tape'. The shop offset in the app (model/calibration.py, apply_shop_offset) adds one correction per "
             "measurement, learned by scripts/retrain_from_shop.py from the shop's own tape-measured customers, and only where it "
             f"lowered the error with each customer left out in turn. Tested inside the real pipeline on the {od_n} people, overall "
             f"accuracy went from {e2e_raw:.1f}% to {e2e_off:.1f}%. With them as the first shop data, the app now corrects "
             + ', '.join({'chest': 'chest', 'waist': 'waist', 'hip': 'hip', 'neck': 'neck', 'thigh': 'thigh', 'sleeve_length': 'sleeve', 'shoulder_width': 'shoulder'}[k] for k in od_used) + " for this tailor's no-weight scans; the other body parts stay "
             "unchanged because a correction did not help them. A new shop needs about 5 to 10 tape-measured customers to reach this."),
           H2('10.9 Test on the combined dataset'),
           P(f"Finally, the 24 real24 people and the {od_n} own-photo people were put together as one dataset and tested the "
             f"same way as the main test (nested cross-validation, 6 groups x 5 repeated splits), without the weight because part "
             f"of it has none. {cmb_n} people have usable tape values. The full report is reports/TailorCam_Combined_Test.pdf."),
           table([['Model', 'Overall accuracy']] + [[m, f'{v:.1f}%'] for m, v in cmb_ov.sort_values(ascending=False).items()],
                 [W * 0.5, W * 0.5]),
           P(f"TailorCam ({cmb_ov['TailorCam (combined)']:.1f}%) is second, just behind CameraHMR ({cmb_ov['CameraHMR']:.1f}%), and "
             "clearly ahead of the other five models. It is the best of all models on hip and neck. The shoulder pulls it down, "
             "because the app has no learned shoulder rule yet and shows HMR's value: shoulder error "
             f"{cmb[(cmb.model == 'TailorCam (combined)') & (cmb.measurement == 'shoulder_width')].MAE.iloc[0]:.2f} cm against "
             f"{cmb[(cmb.model == 'CameraHMR') & (cmb.measurement == 'shoulder_width')].MAE.iloc[0]:.2f} cm for CameraHMR. Without the "
             "shoulder the two are almost equal ("
             f"{cmb[(cmb.model == 'TailorCam (combined)') & (cmb.measurement != 'shoulder_width')].accuracy.mean():.1f}% against "
             f"{cmb[(cmb.model == 'CameraHMR') & (cmb.measurement != 'shoulder_width')].accuracy.mean():.1f}%). Also, without the weight "
             "the combiner leans mostly on CameraHMR, so it can add only a little on top of it. The two measurers in this dataset tape a "
             "little differently, which one set of combined rules cannot fit at once; that is the job of the shop offset.")]

    # ── 11 impact analysis ───────────────────────────────────────────────────
    st += [H1('11. Impact analysis, cost and project management'),
           H2('11.1 Efficiency'),
           P(f"One customer takes about {total[COMB]:.0f} seconds of computing on the {gpu}, plus the time to stand for two "
             f"photos. Loading the models takes {t.loc['Model loading (once, at app start)', 'mean_s']:.0f} s once when the app "
             "starts. CameraHMR uses most of the time; the phone-only version (section 10.2) would be much faster but needs the "
             "weight to stay accurate."),
           H2('11.2 Cost'),
           P("The software, models and datasets are free for research use, and the app needs no internet or subscription. The "
             "cost to a shop is a Windows PC with a webcam and an NVIDIA graphics card. A version without CameraHMR and SHAPY "
             "would run without a graphics card, at some loss of accuracy when the weight is unknown."),
           H2('11.3 Societal impact'),
           P("A small shop can measure a customer faster and keep a consistent record of each customer's measurements. "
             "Customers who prefer not to be touched can be measured without a tape for a first estimate. The main risk is a "
             "wrong size; the 'Check with tape' label exists so that staff do not trust a value that is not yet reliable."),
           H2('11.4 Environmental impact'),
           P("The app runs on one local computer, and a scan uses a few seconds of GPU time, so its energy use is small. Better "
             "first estimates could reduce clothes that are cut to the wrong size and have to be redone, but I have not measured "
             "that effect."),
           H2('11.5 Ethics, privacy and licences'),
           P("Body photos are personal data. The app keeps everything on the shop's own computer; nothing is uploaded. The "
             "datasets and body models (SMPL, HMR, SHAPY, CameraHMR, BodyM, Unidata samples) are licensed for non-commercial "
             "research, so a commercial version would need licences or models trained on the shop's own data. Dataset photos "
             "are used only for testing and are not redistributed. The report states openly where TailorCam is not better than "
             "other methods."),
           H2('11.6 Economic considerations'),
           P("Because the models are free for research but not for business, selling TailorCam would mean buying licences (for "
             "example for SMPL) or retraining on the shop's own customers. The app already saves tape values for that purpose."),
           H2('11.7 Project management'),
           table([['Phase', 'Work done'],
                  ['1. Proposal and review', 'problem statement, literature review, choice of body models'],
                  ['2. Data', 'searching and downloading every public dataset with photos or measurements'],
                  ['3. Models', 'porting HMR, running SHAPY and CameraHMR, training the CNN and priors'],
                  ['4. Combiner and testing', 'combiner, nested cross-validation, per-model tests, leakage review'],
                  ['5. Application', 'scan flow, photo upload, results, customers, orders and reports'],
                  ['6. Experiments', 'phone version, BODY-fit training, Amazon-style silhouettes'],
                  ['7. Report and defense', 'reports, figures and this document']], [W * 0.3, W * 0.7]),
           P("The code was kept in a git repository, and every result table is produced by a script, so any number can be "
             "rebuilt from its script.")]

    # ── 12 limits / 13 ethics / 14 conclusion ────────────────────────────────
    st += [H1('12. Conclusion'),
           H2('12.1 Limitations and threats to validity'),
           P("&bull; Only 24 free people with real photos and tape values exist, so differences smaller than about 0.5 cm can "
             "be chance, and the 95% intervals are wide.<br/>"
             "&bull; Shoulder has only 6 tape values and sleeve 18, so those results are a rough guide.<br/>"
             "&bull; Design choices were made while looking at the same 24 people (section 6.5).<br/>"
             "&bull; The tape values come from a data vendor; I could not check how they were taken. Two tailors already "
             "differ by 1 to 2 cm, so 98 to 100% accuracy is not realistic with this data.<br/>"
             "&bull; Loose clothes hide the body shape, and many customers stand with their arms down.<br/>"
             "&bull; I did not test how much a wrong height or a guessed weight changes the result, nor lighting and camera "
             "distance in a controlled way; HBW's everyday photos are the only check on this.<br/>"
             "&bull; The measuring pipeline is tested by the evaluation scripts, but the user interface has no automated tests; "
             "it was checked by hand.<br/>"
             "&bull; The test people are a small vendor sample, not the shop's own customers, so results may differ in the shop."),
           H2('12.2 Summary and future work'),
           P(f"TailorCam measures the body from two photos with an average error of {g('weight known', 'Chest', 'MAE (cm)'):.2f} cm "
             f"on the chest, {g('weight known', 'Waist', 'MAE (cm)'):.2f} cm on the waist and "
             f"{g('weight known', 'Hip', 'MAE (cm)'):.2f} cm on the hip when the weight is entered, and picks the right shirt "
             f"size for {sz('weight known', 'chest', 'accuracy (%)'):.0f}% of people. These numbers come from a leak-free test "
             "on people the system never saw. Testing every model on its own showed that none is best everywhere, which is why "
             f"the hybrid works. Run on the same people, it beat HMR, SHAPY and CameraHMR in {wins} of 5 tests. Training on "
             "4,132 scanned bodies worked on scans but not on real photos, which tells me the limit is real photos of clothed "
             "people with tape values, not the amount of training data."),
           P("<b>Future work.</b> Collect tape values from the shop's own customers (the app already saves them) and retrain; "
             "add a shoulder rule once there are 15 or more shoulder values; build the guided phone app, which should require "
             "the weight; and use the BODY-rgb and Fashion-body datasets if their authors share them.")]

    # ── references ───────────────────────────────────────────────────────────
    refs = [
        'Loper, M., Mahmood, N., Romero, J., Pons-Moll, G. and Black, M. J. SMPL: A Skinned Multi-Person Linear Model. ACM Transactions on Graphics (SIGGRAPH Asia), 2015.',
        'Kanazawa, A., Black, M. J., Jacobs, D. W. and Malik, J. End-to-end Recovery of Human Shape and Pose. CVPR, 2018.',
        'Kolotouros, N., Pavlakos, G., Black, M. J. and Daniilidis, K. Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop (SPIN). ICCV, 2019.',
        'Choutas, V., Pavlakos, G., Bolkart, T., Tzionas, D. and Black, M. J. Monocular Expressive Body Regression through Body-Driven Attention (ExPose). ECCV, 2020.',
        'Sengupta, A., Budvytis, I. and Cipolla, R. Synthetic Training for Accurate 3D Human Pose and Shape Estimation in the Wild (STRAPS). BMVC, 2020.',
        'Muller, L., Osman, A. A. A., Tang, S., Huang, C.-H. P. and Black, M. J. On Self-Contact and Human Pose (TUCH). CVPR, 2021.',
        'Choutas, V., Muller, L., Huang, C.-H. P., Tang, S., Tzionas, D. and Black, M. J. Accurate 3D Body Shape Regression using Metric and Semantic Attributes (SHAPY). CVPR, 2022.',
        'Goel, S., Pavlakos, G., Rajasegaran, J., Kanazawa, A. and Malik, J. Humans in 4D: Reconstructing and Tracking Humans with Transformers (HMR 2.0). ICCV, 2023.',
        'Patel, P. and Black, M. J. CameraHMR: Aligning People with Perspective. 3DV, 2025.',
        'Black, M. J., Patel, P., Tesch, J. and Yang, J. BEDLAM: A Synthetic Dataset of Bodies Exhibiting Detailed Lifelike Animated Motion. CVPR, 2023.',
        'Ruiz, N. et al. Human Body Measurement Estimation with Adversarial Augmentation (BodyM). 3DV, 2022. arXiv:2210.05667.',
        'Choudhary, S. et al. (Amazon). MeasureNet: body measurement estimation from front and side images. npj Digital Medicine 6:168, 2023.',
        'Gordon, C. C. et al. 2012 Anthropometric Survey of U.S. Army Personnel (ANSUR II). U.S. Army Natick, 2014.',
        'Yan, S., Wirta, J. and Kamarainen, J.-K. Anthropometric clothing measurements from 3D body scans. Machine Vision and Applications 31, 7, 2020 (NOMO-3D, BODY-fit).',
        'Lugaresi, C. et al. MediaPipe: A Framework for Building Perception Pipelines. arXiv:1906.08172, 2019.',
        'He, K., Zhang, X., Ren, S. and Sun, J. Deep Residual Learning for Image Recognition (ResNet). CVPR, 2016.',
        'Ravi, N. et al. SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714, 2024.',
        'Bland, J. M. and Altman, D. G. Statistical methods for assessing agreement between two methods of clinical measurement. The Lancet, 1986.',
    ]
    st += [H1('References')] + [Paragraph(f'[{i}] {r}', S['small']) for i, r in enumerate(refs, 1)]

    if APPENDIX:     # appendices go in their own PDF (python scripts/make_defense_report.py --appendix)
        st = []
        # ── appendix A: Q&A ──────────────────────────────────────────────────────
        qa = [
            ('Problem and scope', [
                ('What problem does the project solve?', 'Measuring a customer for clothes from two photos instead of a tape (sections 1.1 and 1.2).'),
                ('What is new compared with existing work?', 'Six models combined per body part, every model tested separately on the same tape-measured people, and a leak-free test (sections 1.5 and 2.5).'),
                ('Why a desktop app and not a phone app?', 'Three of the models need a graphics card. A phone-only version was tested and is planned (section 10.2).'),
            ]),
            ('Data', [
                ('Which datasets did you use and where are they from?', 'Section 3.5 lists every dataset in one table with its use and result; section 3.6 shows which model used which dataset.'),
                ('Why only 24 test people?', 'They are all the free data I found with real photos and tape values. Confidence intervals show the effect (section 7.4).'),
                ('Is any person in both training and test?', 'No. Nested cross-validation for real24, and 292 shared people removed between BODY-fit and NOMO-3D (sections 6.3 to 6.5).'),
                ('Why not train the combiner on all datasets?', 'Each dataset measures in a different way; pooling them made real-people results worse (section 10.5).'),
                ('How were the tape values taken?', 'By the data vendor; I could not check their method (section 12.1).'),
            ]),
            ('Method', [
                ('Did you train all the models yourself?', 'No. Section 5.6 lists exactly what is reused and what I wrote; section 3.5 lists what each dataset trains and section 8.6 what it gave.'),
                ('Why combine models instead of picking the best one?', 'No single model is best on every body part (chapter 8), and the best one changes when weight is missing.'),
                ('Why not just average them?', 'Averaging gave 4.32 cm without weight against 3.35 cm for the current rule (section 10.1).'),
                ('Where exactly is each measurement taken?', 'At fixed places on the 3D body (section 4.4); the CNN and priors follow their datasets, and the combiner learns the offsets.'),
                ('How does the weight help?', 'It sets the body volume (section 4.3). Without it chest error rises from ' + f"{g('weight known', 'Chest', 'MAE (cm)'):.2f} to {g('weight unknown', 'Chest', 'MAE (cm)'):.2f} cm."),
                ('Why ResNet-18 and not a bigger network?', 'The input is a small black-and-white silhouette and there are only 2,018 training people, so a small, well-known network was a sensible start. I did not test bigger networks; that is future work. Training details are in section 5.4.'),
                ('What is the loss function and how did you stop over-fitting?', 'Laplace negative log-likelihood, augmentation, and keeping the best validation checkpoint (section 5.4 and its training curves).'),
                ('Did you try other models?', 'Yes; section 10.7 lists every model considered and why it was used or not.'),
                ('What was the first version of the system?', 'A landmark-based pipeline, now only a fallback (section 5.7).'),
                ('Which ideas failed?', 'Section 10.5 lists every idea tried with its result and decision.'),
                ('Is HMR not outdated?', 'Yes, it is from 2018 and is the weakest model, but removing it makes the thigh worse (section 8.5).'),
            ]),
            ('Evaluation', [
                ('How exactly is accuracy calculated?', 'Section 6.2 works through a real person step by step.'),
                ('Why MAE and not only accuracy in percent?', 'A tailor thinks in centimetres. Both are given, plus RMSE, bias and R2 (section 7.4).'),
                ('How do you know there is no data leakage?', 'Sections 6.3 to 6.5: one leak found and fixed, every number re-run.'),
                ('Does TailorCam measure too big or too small?', 'Neither on average; about half too big and half too small, bias near 0 (section 7.7). For a new tailor the error can be one-sided, fixed by a shop offset (section 10.8).'),
                ('Did you test on your own people?', f'Yes, {od_n} people with a tailor tape and one photo each (section 10.8), and all {cmb_n} testable people together (section 10.9).'),
                ('Is the improvement statistically significant?', 'Only partly; section 7.6 gives Wilcoxon p-values.'),
                ('Does it work equally for men and women?', 'Not exactly; section 7.5 shows the differences and why they are uncertain.'),
                ('How did you test each model on its own?', 'Chapter 8, with the same leak-free method, plus time per model (section 8.2).'),
                ('What does each trained part add?', 'Section 8.5 removes one model at a time and measures the change.'),
                ('Did the trained networks work on other datasets?', 'Yes; section 10.6 tests every network on every dataset its input allows.'),
            ]),
            ('Results and comparison', [
                ('What accuracy did you reach?', f"About 95 to 97% per body part; chest {g('weight known', 'Chest', 'MAE (cm)'):.2f} cm with weight (chapter 7)."),
                ('Is it better than published methods?', f'Better than HMR, SHAPY and CameraHMR in {wins} of 5 tests on the same people (chapter 9).'),
                ('Why not compare with MeasureNet or commercial apps?', 'Their code is not public, so they cannot be run on the same people (section 2.4).'),
                ('Can it replace the tape?', 'Not yet. No body part passes the strict 80%-within-tolerance rule, so every value shows "Check with tape" (section 4.8).'),
                ('Why do all values say "Check with tape"?', 'The rule needs 80% of tested people within a tailor tolerance (section 4.8); none passes yet, and the shop should prove it on its own customers first.'),
                ('Which exact rule does the app use for each body part?', 'Section 4.7 lists both rule sets, read directly from the app.'),
                ('Is every fold shown?', 'Yes, Appendix C gives every fold of the 5-fold test.'),
                ('How fast is it?', f"About {total[COMB]:.0f} s per customer on an {gpu} (section 8.2)."),
            ]),
            ('Application and impact', [
                ('How is the data stored?', 'In a local SQLite database (section 5.5).'),
                ('What about privacy and licences?', 'Everything stays on the shop computer; the models are for non-commercial research (section 11.5).'),
                ('What does it cost a shop?', 'A PC with a webcam and an NVIDIA graphics card; the software is free (section 11.2).'),
                ('How did you test the app itself?', 'By hand, screen by screen; there are no automated interface tests (section 12.1).'),
                ('What would you do with more time?', 'Collect tape values from real customers and retrain; build the guided phone app (section 12.2).'),
            ]),
        ]
        st += [PageBreak(), H1('Appendix A. Questions I expect at the defense')]
        st += [P('Each answer points to the part of the report that covers it.')]
        i = 0
        for topic, items in qa:
            st += [Paragraph(f'<b>{topic}</b>', S['p'])]
            for q, a in items:
                i += 1
                st += [Paragraph(f'Q{i}. {q}', S['q']), P(a)]

        st += [H1('Appendix B. Short user manual'),
               P("1. Start the app with <b>python main.py</b>. The models load once (about 20 seconds).<br/>"
                 "2. On the home screen, choose a customer or create a new one and type the name.<br/>"
                 "3. Type the height in cm, and the weight in kg if the customer knows it.<br/>"
                 "4. Start the scan. The customer stands inside the outline, arms slightly away from the body, and follows the "
                 "voice: front photo, turn, side photo. Or press Upload and choose a front and a side photo.<br/>"
                 "5. Read the results: each value shows 'Camera OK' or 'Check with tape', with the suggested sizes.<br/>"
                 "6. If you measure with a tape, type the real values; the app saves them for retraining.<br/>"
                 "7. Save the order or print the PDF report from the results or reports screen.")]

        kp = kpf[kpf.weight == 'known']
        st += [H1('Appendix C. Every fold of the 5-fold test'),
               P('Standard 5-fold cross-validation with the weight entered: each fold is tested on people the combiner never saw, '
                 'and the rule is chosen on the training people of that fold.'),
               table([['Measurement', 'Fold', 'Train', 'Test', 'Rule chosen', 'MAE (cm)', 'Accuracy']] +
                     [[r.measurement, r.fold, str(r['train people']), str(r['test people']), str(r['rule chosen on validation']),
                       f"{r['MAE (cm)']:.2f}", f"{r['accuracy (%)']:.1f}%"] for _, r in kp.iterrows()],
                     [W * 0.15, W * 0.11, W * 0.09, W * 0.09, W * 0.26, W * 0.14, W * 0.16])]
        st += [H1('Appendix E. Files and how to reproduce'),
               table([['Command or file', 'What it does'],
                      ['python main.py', 'runs the app'],
                      ['scripts/run_models_on_datasets.py', 'runs all models on all datasets'],
                      ['scripts/model_accuracy.py', 'per model and body part accuracy, Excel data sheets'],
                      ['scripts/model_timing.py', 'time per model'],
                      ['scripts/individual_models.py', 'models on HBW and the model agency set'],
                      ['notebooks/TailorCam_paper.py', 'main results, k-fold, sizes, Bland-Altman'],
                      ['scripts/ensemble_methods.py', 'the six ways of combining'],
                      ['model/calibration.py', 'the combiner and the leak-free test (nested_cv_app)'],
                      ['model/bodyfit/train_bodyfit.py', 'the BODY-fit network'],
                      ['experiments/photo_to_amazon/', 'the Amazon-silhouette experiment'],
                      ['scripts/make_defense_report.py', 'builds this report']],
                     [W * 0.42, W * 0.58])]

    doc = Doc(OUT_APP if APPENDIX else OUT, title='TailorCam Defense Report', author='Md Ashikuzzaman')
    doc.multiBuild(st)
    print('saved', OUT)


if __name__ == '__main__':
    build()
