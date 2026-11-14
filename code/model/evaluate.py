"""
TailorCam — Model Evaluation Pipeline
======================================
Trains and evaluates the full measurement accuracy pipeline on
synthetic ANSUR II data. Generates:
  - MAE / RMSE / R² table (printed + saved to CSV)
  - 5 publication-quality plots saved to  model/plots/
  - metrics.json with all numbers

Run with:
    python model/evaluate.py
    python -m model.evaluate

For a Data Science capstone board this script demonstrates:
  • Synthetic dataset generation (ANSUR II anthropometry)
  • 4-stage pipeline: Baseline → Geometric → Bayesian → Constraint
  • Train / Test split  (20 000 train, 5 000 test)
  • Per-measurement MAE, RMSE, R²
  • Error-distribution box-plots
  • Predicted-vs-actual scatter
  • Learning-curve (sample-size vs accuracy)
"""

import os
import sys
import math
import json
import time
import csv
import numpy as np

# Non-interactive Matplotlib backend (works without a display)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# ── Path setup ─────────────────────────────────────────────────────────────────
_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
sys.path.insert(0, _ROOT)

from model.bayesian_body import (
    generate_synthetic, BayesianBodyMeasure,
    KEYS, FEAT_NAMES, _ansur_measurements, _simulate_geometric_features,
)
from model.constraint_solver import solve as _constraint_solve

# ── Config ────────────────────────────────────────────────────────────────────
N_TRAIN   = 20_000
N_TEST    = 5_000
PLOT_DIR  = os.path.join(_DIR, 'plots')   # created when evaluate() saves plots

# ── Display labels ────────────────────────────────────────────────────────────
DISPLAY = {
    'shoulder_width': 'Shoulder\nWidth',
    'chest':          'Chest',
    'waist':          'Waist',
    'hip':            'Hip',
    'back_length':    'Back\nLength',
    'shirt_length':   'Shirt\nLength',
    'sleeve_length':  'Sleeve\nLength',
    'neck':           'Neck',
}
DISPLAY_FLAT = {k: v.replace('\n', ' ') for k, v in DISPLAY.items()}

# ANSUR II height fractions — used for Stage 0 baseline
_H_FRAC = {
    'shoulder_width': 0.239,
    'chest':          0.528,
    'waist':          0.455,
    'hip':            0.540,
    'back_length':    0.296,
    'shirt_length':   0.330,
    'sleeve_length':  0.355,
    'neck':           0.198,
}

# ── Palette (warm, matches TailorCam UI) ─────────────────────────────────────
C0 = '#9898A8'   # grey  – Baseline
C1 = '#4A8DB5'   # teal  – Geometric
C2 = '#D79120'   # gold  – Bayesian
C3 = '#1CB428'   # green – Constraint
COLORS = [C0, C1, C2, C3]
STAGE_SHORT = ['Baseline', 'Geometric', '+ Bayesian', '+ Constraint']

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        120,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ellipse(a: float, b: float) -> float:
    """Ramanujan approximation of ellipse circumference."""
    if a <= 0 or b <= 0:
        return 0.0
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + math.sqrt(4.0 - 3.0 * h)))


def _geo_from_X(row: np.ndarray) -> dict:
    """
    Convert raw geometric feature row → measurement dict.
    Mirrors what body_measurer.compute() produces before any ML correction.
    """
    sh_w, ch_w, wa_w, hi_w, bk, ch_d, wa_d, hi_d, sl, h = row
    return {
        'shoulder_width': float(sh_w),
        'chest':          _ellipse(ch_w / 2, ch_d / 2),
        'waist':          _ellipse(wa_w / 2, wa_d / 2),
        'hip':            _ellipse(hi_w / 2, hi_d / 2),
        'back_length':    float(bk),
        'shirt_length':   float(bk) * 1.12,
        'sleeve_length':  float(sl),
        'neck':           float(h) * 0.202,
    }


def _baseline_from_height(h: float) -> dict:
    return {k: h * v for k, v in _H_FRAC.items()}


def _pred_matrix(bm: BayesianBodyMeasure, X: np.ndarray, Y_geo: np.ndarray) -> np.ndarray:
    """Bayesian blend (60% geo + 40% Bayesian) for every test row."""
    Y_bay = np.zeros_like(Y_geo)
    for i, row in enumerate(X):
        feats = dict(zip(FEAT_NAMES, row.astype(float)))
        pred  = bm.predict(feats)
        for j, k in enumerate(KEYS):
            geo_v = Y_geo[i, j]
            bay_v = pred.get(k, {}).get('value', geo_v) if pred else geo_v
            Y_bay[i, j] = geo_v * 0.60 + bay_v * 0.40
    return Y_bay


def _constraint_matrix(X: np.ndarray, Y_bay: np.ndarray) -> np.ndarray:
    """Apply anthropometric constraint solver to every test row."""
    Y_con = np.zeros_like(Y_bay)
    for i, row in enumerate(X):
        h     = float(row[9])
        m_in  = {k: float(Y_bay[i, j]) for j, k in enumerate(KEYS)}
        m_out = _constraint_solve(m_in, h)
        Y_con[i] = [m_out.get(k, Y_bay[i, j]) for j, k in enumerate(KEYS)]
    return Y_con


def _metrics(Y_true, Y_pred):
    mae  = mean_absolute_error(Y_true, Y_pred)
    rmse = math.sqrt(mean_squared_error(Y_true, Y_pred))
    r2   = r2_score(Y_true, Y_pred)
    per  = {k: mean_absolute_error(Y_true[:, j], Y_pred[:, j])
            for j, k in enumerate(KEYS)}
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'per_key': per}


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate():
    print('\n' + '=' * 62)
    print('  TailorCam  —  Model Evaluation Pipeline')
    print('  (Data Science Capstone — Train / Test Analysis)')
    print('=' * 62)

    # ── Step 1: Generate data ─────────────────────────────────────
    print(f'\n[1/6] Generating {N_TRAIN:,} train + {N_TEST:,} test samples …')
    t0 = time.time()
    X_tr, Y_tr = generate_synthetic(N_TRAIN, seed=42)
    X_te, Y_te = generate_synthetic(N_TEST,  seed=999)   # never seen during training
    print(f'      done in {time.time() - t0:.1f} s')

    # ── Step 2: Train Bayesian ensemble ──────────────────────────
    print(f'\n[2/6] Training Bayesian Ridge ensemble  ({len(KEYS)} models) …')
    t0 = time.time()
    bm = BayesianBodyMeasure()
    if not bm.trained:
        bm.ensure_trained(N_TRAIN)
    else:
        print('      (loaded cached model)')
    print(f'      done in {time.time() - t0:.1f} s')

    # ── Step 3: Build prediction matrices for all 4 stages ───────
    print('\n[3/6] Running all pipeline stages on test set …')

    # Stage 0: height-only baseline
    Y_base = np.array([[_baseline_from_height(X_te[i, 9])[k]
                         for k in KEYS] for i in range(N_TEST)], dtype=np.float32)

    # Stage 1: geometric measurement (body_measurer equivalent)
    Y_geo  = np.array([[_geo_from_X(X_te[i])[k]
                         for k in KEYS] for i in range(N_TEST)], dtype=np.float32)

    # Stage 2: + Bayesian Ridge blend
    print('      Bayesian blend …')
    Y_bay  = _pred_matrix(bm, X_te, Y_geo)

    # Stage 3: + Constraint solver
    print('      Constraint solve …')
    Y_con  = _constraint_matrix(X_te, Y_bay)

    stages = [
        ('Baseline\n(Height Only)', Y_base),
        ('Stage 1\n(Geometric)',    Y_geo),
        ('Stage 2\n(+ Bayesian)',   Y_bay),
        ('Stage 3\n(+ Constraint)', Y_con),
    ]

    # ── Step 4: Compute metrics ───────────────────────────────────
    print('\n[4/6] Computing metrics …')
    results = {}
    for label, Y_pred in stages:
        m = _metrics(Y_te, Y_pred)
        results[label] = m
        flat = label.replace('\n', ' ')
        print(f'  {flat:<28}  MAE={m["mae"]:.3f} cm  '
              f'RMSE={m["rmse"]:.3f} cm  R²={m["r2"]:.4f}')

    # ── Step 5: Learning curve ────────────────────────────────────
    print('\n[5/6] Building learning curve …')
    lc_sizes   = [500, 1000, 2000, 4000, 8000, 16000, 20000]
    lc_mae_geo, lc_mae_bay = [], []
    for n in lc_sizes:
        Xn, Yn = generate_synthetic(n, seed=42)
        # Geometric (no model needed — just apply formula)
        Yg = np.array([[_geo_from_X(Xn[i])[k] for k in KEYS]
                        for i in range(n)], dtype=np.float32)
        lc_mae_geo.append(mean_absolute_error(Yn, Yg))
        # Bayesian on same data (train + eval in-sample for learning curve)
        # Use a temporary instance so the main cached model is NOT overwritten
        bm_lc = BayesianBodyMeasure.__new__(BayesianBodyMeasure)
        bm_lc.models = {}; bm_lc.trained = False
        bm_lc._fit_no_save(Xn, Yn)   # train without touching the pkl cache
        Yb = _pred_matrix(bm_lc, Xn, Yg)
        lc_mae_bay.append(mean_absolute_error(Yn, Yb))
        print(f'      n={n:>6}  geo={lc_mae_geo[-1]:.3f}  bay={lc_mae_bay[-1]:.3f}')

    # ── Step 6: Generate plots ────────────────────────────────────
    os.makedirs(PLOT_DIR, exist_ok=True)
    print('\n[6/6] Generating plots …')

    # ── Plot 1: Overall MAE bar chart ─────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    maes  = [results[s[0]]['mae'] for s in stages]
    bars  = ax.bar(STAGE_SHORT, maes, color=COLORS, width=0.5, zorder=3)
    ax.set_ylabel('Mean Absolute Error (cm)', fontsize=12)
    ax.set_title('TailorCam — Pipeline Accuracy Improvement\n'
                 f'(n={N_TEST:,} synthetic test subjects)', fontsize=13, fontweight='bold')
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.set_ylim(0, max(maes) * 1.30)
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f'{mae:.3f} cm', ha='center', va='bottom', fontsize=11, fontweight='bold')
    # Percentage improvement annotation
    base_mae = maes[0]
    for i in range(1, 4):
        pct = (base_mae - maes[i]) / base_mae * 100
        ax.text(i, maes[i] * 0.5, f'−{pct:.0f}%', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')
    plt.tight_layout()
    p1 = os.path.join(PLOT_DIR, 'plot1_mae_comparison.png')
    fig.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {p1}')

    # ── Plot 2: Per-measurement MAE (grouped bar) ─────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    x  = np.arange(len(KEYS))
    w  = 0.19
    for i, (slabel, _) in enumerate(stages):
        vals = [results[slabel]['per_key'][k] for k in KEYS]
        ax.bar(x + (i - 1.5) * w, vals, width=w, color=COLORS[i],
               label=STAGE_SHORT[i], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[k] for k in KEYS], fontsize=9)
    ax.set_ylabel('MAE (cm)', fontsize=12)
    ax.set_title('Per-Measurement Error — All Pipeline Stages',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    plt.tight_layout()
    p2 = os.path.join(PLOT_DIR, 'plot2_per_measurement_mae.png')
    fig.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {p2}')

    # ── Plot 3: Predicted vs Actual (Stage 3, 8 subplots) ─────────
    fig = plt.figure(figsize=(16, 8))
    gs  = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)
    for idx, k in enumerate(KEYS):
        ax  = fig.add_subplot(gs[idx // 4, idx % 4])
        j   = KEYS.index(k)
        n_s = min(800, N_TEST)
        ax.scatter(Y_te[:n_s, j], Y_con[:n_s, j],
                   alpha=0.30, s=6, color=C3, zorder=3)
        lo = min(Y_te[:n_s, j].min(), Y_con[:n_s, j].min()) - 1
        hi = max(Y_te[:n_s, j].max(), Y_con[:n_s, j].max()) + 1
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.2)
        ax.set_title(DISPLAY_FLAT[k], fontsize=9, fontweight='bold')
        ax.set_xlabel('Actual (cm)', fontsize=8)
        ax.set_ylabel('Predicted (cm)', fontsize=8)
        mae_k = results[stages[-1][0]]['per_key'][k]
        r2_k  = r2_score(Y_te[:, j], Y_con[:, j])
        ax.text(0.05, 0.91, f'MAE={mae_k:.2f} cm\nR²={r2_k:.3f}',
                transform=ax.transAxes, fontsize=7.5, color='#1C7A1C',
                va='top')
    fig.suptitle('TailorCam Stage 3 — Predicted vs Actual Body Measurements\n'
                 f'(n={n_s} test subjects)', fontsize=12, fontweight='bold')
    p3 = os.path.join(PLOT_DIR, 'plot3_predicted_vs_actual.png')
    fig.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {p3}')

    # ── Plot 4: Error distribution box-plots (chest/waist/hip) ────
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)
    for ax, k in zip(axes, ['chest', 'waist', 'hip']):
        j    = KEYS.index(k)
        errs = [Y_pred[:, j] - Y_te[:, j] for _, Y_pred in stages]
        bp   = ax.boxplot(errs, patch_artist=True, notch=False, widths=0.5,
                           medianprops=dict(color='black', lw=2),
                           flierprops=dict(marker='.', alpha=0.3, ms=3))
        for patch, col in zip(bp['boxes'], COLORS):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        ax.axhline(0, color='red', lw=1.2, ls='--', alpha=0.7)
        ax.set_xticklabels(STAGE_SHORT, fontsize=8, rotation=15)
        ax.set_ylabel('Error (predicted − actual)  cm', fontsize=9)
        ax.set_title(f'{k.capitalize()} Error Distribution',
                     fontsize=11, fontweight='bold')
        ax.yaxis.grid(True, alpha=0.3)
    plt.suptitle('Prediction Error Distribution by Stage', fontsize=12, fontweight='bold')
    plt.tight_layout()
    p4 = os.path.join(PLOT_DIR, 'plot4_error_distribution.png')
    fig.savefig(p4, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {p4}')

    # ── Plot 5: Learning curve ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lc_sizes, lc_mae_geo, 'o--', color=C1, lw=2, ms=6, label='Geometric only')
    ax.plot(lc_sizes, lc_mae_bay, 's-',  color=C2, lw=2, ms=6, label='+ Bayesian Ridge')
    ax.fill_between(lc_sizes, lc_mae_geo, lc_mae_bay, alpha=0.12, color=C2)
    ax.set_xlabel('Training Set Size (samples)', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (cm)', fontsize=12)
    ax.set_title('Learning Curve — Bayesian Model vs Geometric Baseline',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.35)
    ax.xaxis.grid(True, alpha=0.20)
    ax.set_xscale('log')
    plt.tight_layout()
    p5 = os.path.join(PLOT_DIR, 'plot5_learning_curve.png')
    fig.savefig(p5, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {p5}')

    # ── Save metrics JSON ──────────────────────────────────────────
    metrics_out = {}
    for sn, sdata in results.items():
        key = sn.replace('\n', ' ')
        metrics_out[key] = {
            'mae':              round(sdata['mae'],  3),
            'rmse':             round(sdata['rmse'], 3),
            'r2':               round(sdata['r2'],   4),
            'per_measurement':  {k: round(v, 3) for k, v in sdata['per_key'].items()},
        }
    metrics_path = os.path.join(PLOT_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f'  {metrics_path}')

    # ── Save metrics CSV ───────────────────────────────────────────
    csv_path = os.path.join(PLOT_DIR, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Stage', 'MAE (cm)', 'RMSE (cm)', 'R²'] +
                   [DISPLAY_FLAT[k] + ' MAE' for k in KEYS])
        for sn, sdata in results.items():
            w.writerow(
                [sn.replace('\n', ' '), round(sdata['mae'], 3),
                 round(sdata['rmse'], 3), round(sdata['r2'], 4)] +
                [round(sdata['per_key'][k], 3) for k in KEYS]
            )
    print(f'  {csv_path}')

    # ── Final summary ──────────────────────────────────────────────
    print('\n' + '=' * 62)
    print('  RESULTS SUMMARY')
    print('=' * 62)
    print(f'  {"Stage":<28}  {"MAE":>8}  {"RMSE":>8}  {"R²":>8}')
    print(f'  {"-"*28}  {"-"*8}  {"-"*8}  {"-"*8}')
    for sn, sdata in results.items():
        label = sn.replace('\n', ' ')
        print(f'  {label:<28}  {sdata["mae"]:>8.3f}  '
              f'{sdata["rmse"]:>8.3f}  {sdata["r2"]:>8.4f}')
    print('=' * 62)

    base_mae  = results[stages[0][0]]['mae']
    final_mae = results[stages[-1][0]]['mae']
    pct_imp   = (base_mae - final_mae) / base_mae * 100
    print(f'\n  Overall improvement:  '
          f'{base_mae:.3f} cm  →  {final_mae:.3f} cm  ({pct_imp:.0f}% MAE reduction)')
    print(f'\n  5 plots + metrics saved to:  {PLOT_DIR}')
    print('  ✓ Evaluation complete\n')
    return results


if __name__ == '__main__':
    evaluate()
