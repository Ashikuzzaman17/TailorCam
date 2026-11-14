# %% [markdown]
# # TailorCam: Body Measurements from Photos for Tailor Shops
# ## Complete experiments for the paper (data → training → testing → analysis)
#
# **Author:** Md Ashikuzzaman (ID 2022-3-60-101), Dept. of Computer Science and Engineering,
# East West University, Dhaka. **Supervisor:** Dr. Mohammad Mahdi Hasan.
#
# This notebook produces every number, table and figure needed for the paper. Every
# table is also saved as **CSV and LaTeX** and every figure as **PNG (300 dpi) and PDF** in
# `reports/paper/`, ready to paste into Overleaf / Word.
#
# | Section | Paper section it supports |
# |---|---|
# | 1 Datasets | *Datasets* |
# | 2 Method | *Proposed method* |
# | 3 Training | *Implementation / training details* |
# | 4 Evaluation protocol | *Experimental setup* |
# | 5 Main results (all datasets) | *Results* |
# | 6 Comparison with published models | *Comparison with state of the art* |
# | 7 Ablation study | *Ablation* |
# | 8 Bland–Altman agreement | *Agreement analysis* |
# | 9 Error analysis | *Discussion* |
# | 10 Size classification (accuracy, F1) | *Application results* |
# | 11 Camera vs tape | *Practical use* |
# | 12 Negative results | *Discussion / what did not work* |
# | 13 Runtime | *Efficiency* |
# | 14 Auto-written summary | *Abstract / conclusion* |
# | 15 Limitations | *Limitations and threats to validity* |
#
# **Honesty rules used everywhere:** a person used to train something is never used to test
# it; settings are chosen without the test people (nested cross-validation); every
# real-photo result is averaged over **5 different random splits** because the real
# tape-measured set is small; uncertainty is shown as **95 % bootstrap confidence intervals**.

# %%
import os, sys, re, glob, json, time, warnings
from IPython.display import display, Markdown
warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.getcwd(), '..')) if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts')); os.chdir(ROOT)
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from scipy import stats
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)
plt.rcParams.update({'figure.dpi': 90, 'axes.grid': True, 'grid.alpha': .3, 'font.size': 10})

import model.calibration as C
from model.calibration import fit_all, save, apply_calibration, tape_check, TAPE_TOLERANCE_CM
from evaluate_project import regression_metrics, size_metrics, SIZES, SIZE_BINS

OUT = os.path.join(ROOT, 'reports', 'paper')
os.makedirs(os.path.join(OUT, 'figures'), exist_ok=True); os.makedirs(os.path.join(OUT, 'tables'), exist_ok=True)
RNG_SEEDS = range(5)                         # repeated cross-validation splits
KEYS = ('chest', 'waist', 'hip', 'thigh', 'neck', 'sleeve_length')
GIRTH = ('chest', 'waist', 'hip')
NAMES = {'chest': 'Chest', 'waist': 'Waist', 'hip': 'Hip', 'thigh': 'Thigh', 'neck': 'Neck',
         'sleeve_length': 'Sleeve', 'shoulder_width': 'Shoulder'}


def savefig(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, 'figures', f'{name}.{ext}'), dpi=300, bbox_inches='tight')


def savetab(df, name, caption=''):
    df.to_csv(os.path.join(OUT, 'tables', f'{name}.csv'))
    try:
        tex = df.to_latex(float_format='%.2f', caption=caption, label=f'tab:{name}', escape=True)
        open(os.path.join(OUT, 'tables', f'{name}.tex'), 'w', encoding='utf-8').write(tex)
    except Exception as exc:                        # LaTeX export needs jinja2; CSV is always written
        print(f'(LaTeX for {name} skipped: {exc})')
    return df


def boot_ci(errors, n=2000, seed=0):
    """95 % bootstrap confidence interval of the mean (resampling people)."""
    e = np.asarray(errors, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(e, (n, len(e)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def rows(df):
    return [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
            for r in df.to_dict('records')]


print('Project folder:', ROOT)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU)')
print('PyTorch', torch.__version__, '| numpy', np.__version__, '| pandas', pd.__version__)
print('Outputs →', OUT)

# %% [markdown]
# ---
# ## 1. Datasets
# Eight datasets are used. Only **real colour photos with tape measurements** can train the
# final combiner; the others train individual models or serve as external tests.

# %%
from model.bodym.train_bodym import load_split
bodym = {s: load_split(s) for s in ('train', 'testA', 'testB')}
real = pd.read_csv('../dataset/real_photos/people.csv')
agency = pd.read_csv('../dataset/shapy/model_agency_images/index.csv')
pred = {n: pd.read_csv(f'../dataset/predictions/{n}.csv') for n in ('real24', 'hbw_val', 'agency', 'bedlam')
        if os.path.exists(f'../dataset/predictions/{n}.csv')}
n_ansur = sum(len(pd.read_csv(f'../dataset/ansur2/ANSUR_II_{g}_Public.csv', encoding='latin-1')) for g in ('MALE', 'FEMALE'))

ds_table = pd.DataFrame([
    dict(dataset='BodyM (Amazon, 2022)', people=len(set(bodym['train']['subj'])) + len(set(bodym['testA']['subj'])) + len(set(bodym['testB']['subj'])),
         input='front + side silhouettes', truth='3D-scan measurements', weight='yes',
         role='train hybrid CNN (train) / test (testA, testB)', licence='CC BY-NC 4.0'),
    dict(dataset='Real photos (Unidata samples)', people=len(real), input='front + side colour photos',
         truth='tape measure', weight='yes', role='train + test combiner (nested CV)', licence='CC BY-NC-ND 4.0'),
    dict(dataset='HBW val (SHAPY, CVPR 2022)', people=pred['hbw_val'].person_id.nunique(), input='lab + in-the-wild photos',
         truth='3D body scan', weight='yes (from scan)', role='external test only', licence='MPI research licence'),
    dict(dataset='Model Agency (SHAPY)', people=len(agency), input='fashion photos',
         truth='agency-listed', weight='no', role='external test (official test split)', licence='MPI research licence'),
    dict(dataset='BEDLAM (CVPR 2023)', people=pred['bedlam'].person_id.nunique() if 'bedlam' in pred else 0,
         input='synthetic clothed renders', truth='exact SMPL body', weight='yes (from body)',
         role='external test (*CameraHMR trained on it)', licence='MPI research licence'),
    dict(dataset='ANSUR II (US Army, 2012)', people=n_ansur, input='none (tabular)', truth='tape / calipers',
         weight='yes', role='train height/weight prior', licence='public domain'),
    dict(dataset='NHANES (CDC, 1999–2023)', people='~28,000', input='none (tabular)', truth='tape',
         weight='yes', role='tested as prior (rejected)', licence='public domain'),
    dict(dataset='BEDLAM / AGORA / etc.', people='>100k', input='synthetic', truth='SMPL', weight='–',
         role='pre-training of CameraHMR (by its authors)', licence='research'),
])
savetab(ds_table.set_index('dataset'), 'datasets', 'Datasets used in this work')

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
groups = {'BodyM train': (bodym['train']['h'], bodym['train']['w']),
          'real24': (real.height_cm.values, real.weight_kg.values)}
for n in ('hbw_val', 'bedlam'):
    d = pred[n][pred[n]['mode'] == 'with_weight']
    groups[n] = (d.height_cm.values, d.weight_kg.values)
for name, (h, w) in groups.items():
    ax[0].hist(h, bins=25, alpha=.45, density=True, label=name)
    ax[1].hist(w / (h / 100) ** 2, bins=25, alpha=.45, density=True, label=name)
ax[0].set_xlabel('height (cm)'); ax[1].set_xlabel('BMI (kg/m²)'); ax[0].legend(); ax[0].set_title('Height'); ax[1].set_title('BMI')
box = [real.chest.dropna(), agency.chest_cm.dropna(), pred['hbw_val']['true:chest'].dropna()]
if 'bedlam' in pred:
    box.append(pred['bedlam']['true:chest'].dropna())
ax[2].boxplot(box); ax[2].set_xticks(range(1, len(box) + 1)); ax[2].set_xticklabels(['real24', 'agency', 'HBW', 'BEDLAM'][:len(box)]); ax[2].set_ylabel('chest (cm)'); ax[2].set_title('True chest by dataset')
plt.tight_layout(); savefig(fig, 'datasets_distribution'); plt.show()

# %% [markdown]
# ---
# ## 2. Proposed method
# A **hybrid** of six measurement sources and a learned **combiner**:
#
# | Source | Type | Input | Trained on |
# |---|---|---|---|
# | Hybrid CNN ensemble (ours) | ResNet-18 ×3, prior + image correction, Laplace uncertainty | front + side silhouettes, height, weight, gender | BodyM train |
# | BodyM prior (ours) | ridge regression | height, weight, gender | BodyM train |
# | ANSUR prior (ours) | ridge on polynomial features | height, weight, gender, age | ANSUR II (6,068) |
# | HMR (Kanazawa 2018) + weight fit (ours) | 3D SMPL body; shape adjusted so volume × density = weight | photo | public weights |
# | SHAPY (Choutas 2022) + weight fit | 3D SMPL-X body | photo | public weights |
# | CameraHMR (Patel & Black 2025) + mirror TTA + weight fit (ours) | 3D SMPL body + camera FoV | photo | public weights |
# | **Combiner (ours)** | per measurement: identity / offset / pick / ridge blend, chosen by nested CV | the 6 answers above | real tape-measured people |
#
# Measurements are taken on the 3D bodies with a virtual tape: convex-hull girths of
# horizontal slices (chest at nipple level, waist = narrowest near the navel, hip = fullest
# over the seat), sleeve and shoulder on an arms-down copy of the body.

# %%
fig, ax = plt.subplots(figsize=(15, 5)); ax.axis('off'); ax.set_xlim(0, 15); ax.set_ylim(0, 6)
def boxat(x, y, w, h, txt, col):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=col, ec='k', lw=1.2)); ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=9)
def arrow(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', lw=1.3))
boxat(0.2, 3.6, 2.2, 1.2, 'Front + side\nphotos', '#dde')
boxat(0.2, 1.0, 2.2, 1.2, 'Height, weight,\ngender, age', '#dde')
boxat(3.2, 3.6, 2.2, 1.2, 'MediaPipe\nkeypoints + mask', '#eed')
srcs = [('Hybrid CNN ×3', 5.2), ('HMR + weight fit', 4.3), ('SHAPY + weight fit', 3.4), ('CameraHMR + TTA\n+ weight fit', 2.5),
        ('BodyM prior', 1.6), ('ANSUR prior', 0.7)]
for t, y in srcs:
    boxat(6.4, y - 0.35, 2.6, 0.7, t, '#ded')
    arrow(9.0, y, 10.2, 3.0)
for t, y in srcs[:4]:
    arrow(5.4, 4.2, 6.4, y)
for t, y in srcs[4:]:
    arrow(2.4, 1.6, 6.4, y)
arrow(2.4, 4.2, 3.2, 4.2); arrow(2.4, 1.6, 6.4, 5.2)
boxat(10.2, 2.4, 2.2, 1.2, 'Combiner\n(nested CV,\nreal tape data)', '#fdd')
arrow(12.4, 3.0, 13.0, 3.0)
boxat(13.0, 2.2, 1.9, 1.6, 'chest, waist,\nhip, thigh,\nneck, sleeve,\nshoulder', '#fff')
ax.set_title('TailorCam pipeline'); savefig(fig, 'pipeline'); plt.show()

# %% [markdown]
# ---
# ## 3. Training
# ### 3a. Hybrid CNN ensemble on BodyM
# Split by **person**: 90 % train / 10 % validation of the BodyM train set; the best epoch is
# chosen on validation; testA and testB (487 people) are never seen. Augmentation during
# training: random scale / shift / rotation of the silhouettes plus *real-world mask*
# augmentation (blur, holes, arm merging) so the model copes with MediaPipe masks.
# Optimiser Adam, Laplace negative-log-likelihood loss, 3 seeds.

# %%
def read_log(path):
    out = []
    for line in open(path):
        m = re.search(r'epoch\s+(\d+)/(\d+)\s+loss ([\d.]+)\s+val chest/waist/hip MAE ([\d.]+)', line)
        if m: out.append((int(m[1]), float(m[3]), float(m[4])))
    return pd.DataFrame(out, columns=['epoch', 'train loss', 'val MAE (cm)'])
logs = {f'seed {i}': read_log(f'model/bodym/logs/train_s{i}.log') for i in range(3) if os.path.exists(f'model/bodym/logs/train_s{i}.log')}
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
for n, d in logs.items():
    ax[0].plot(d['epoch'], d['train loss'], label=n); ax[1].plot(d['epoch'], d['val MAE (cm)'], label=n)
ax[0].set_title('Training loss'); ax[1].set_title('Validation MAE, chest/waist/hip (cm)')
for a in ax: a.set_xlabel('epoch'); a.legend()
plt.tight_layout(); savefig(fig, 'training_curves'); plt.show()
savetab(pd.DataFrame({n: {'epochs': int(d.epoch.max()), 'best val MAE (cm)': d['val MAE (cm)'].min(),
                          'best epoch': int(d.loc[d['val MAE (cm)'].idxmin(), 'epoch'])} for n, d in logs.items()}).T,
        'training_summary', 'Hybrid CNN training (BodyM)')

# %%
from model.bodym.bodym_net import MEASUREMENTS
from model.bodym.hybrid import HybridNet, member_paths, predict_split, DEV
nets = []
for p in member_paths():
    net = HybridNet().to(DEV); net.load_state_dict(torch.load(p, map_location=DEV, weights_only=True)); nets.append(net)
bodym_res = []
for s in ('testA', 'testB'):
    D = bodym[s]
    runs = [predict_split(n, D) for n in nets]
    P, Pp = np.mean([r['image'] for r in runs], 0), np.mean([r['prior'] for r in runs], 0)
    for m in ('chest', 'waist', 'hip', 'thigh', 'shoulder-breadth', 'arm-length'):
        j = MEASUREMENTS.index(m)
        for meth, X in (('height+weight prior', Pp), ('hybrid ensemble', P)):
            e = np.abs(X[:, j] - D['y'][:, j]); lo, hi = boot_ci(e)
            bodym_res.append({'test set': s, 'method': meth, 'measurement': m, 'MAE (cm)': e.mean(),
                              '95% CI': f'{lo:.2f}–{hi:.2f}', 'accuracy (%)': 100 - (e / D['y'][:, j] * 100).mean()})
bodym_tab = pd.DataFrame(bodym_res)
savetab(bodym_tab.pivot_table(index=['test set', 'method'], columns='measurement', values='MAE (cm)'),
        'bodym_test', 'BodyM test sets (487 unseen people): MAE in cm')

# %% [markdown]
# ### 3b. Height/weight priors
# ANSUR II: 80 % train / 20 % test split **by person** (seed 0). Shown: the prior's own test error.

# %%
from model.ansur_prior import load_data as ansur_data, AnsurPrior, TARGETS as AT
ad = ansur_data(); test = np.random.default_rng(0).random(len(ad)) < 0.2
ap = AnsurPrior(); ares = []
for with_w in (True, False):
    for key, col in AT.items():
        sub = ad[test]
        p_ = np.array([ap.predict(r.height_cm, r.weight_kg if with_w else None, r.gender, r.Age)[key] for r in sub.itertuples()])
        e = np.abs(p_ - sub[col].values / 10)
        ares.append({'weight': 'known' if with_w else 'unknown', 'measurement': key, 'test people': len(e), 'MAE (cm)': e.mean()})
savetab(pd.DataFrame(ares).pivot_table(index='measurement', columns='weight', values='MAE (cm)'),
        'ansur_prior', 'ANSUR II prior, held-out 20 % of soldiers')

# %% [markdown]
# ### 3c. The combiner (final training on all real tape-measured people)
# For each measurement the combiner compares: **identity** (first model as is), **offset**,
# **pick another model** (± offset) and **ridge blends** (α = 1, 10, 100, 1000), choosing by
# leave-one-out validation; it is only adopted if it beats identity on held-out folds. Two
# versions: weight known (default model = hybrid CNN) and weight unknown (default = CameraHMR).
# The final combiners below are trained on all real24 people and saved for the app.

# %%
T_W = C.TARGETS
T_NW = {k: (['camerahmr:' + k] + [c for c in v if c != 'camerahmr:' + k] if 'camerahmr:' + k in v else v) for k, v in C.TARGETS.items()}
r24 = pred['real24']
calib_w, _ = fit_all(rows(r24[r24['mode'] == 'with_weight']), 6, T_W)
calib_nw, _ = fit_all(rows(r24[r24['mode'] == 'no_weight']), 6, T_NW)
save(calib_w, 'with_weight'); save(calib_nw, 'no_weight')
def describe(c):
    if c['kind'] == 'identity': return f"keep {c['components'][0]}"
    if c['kind'].startswith('pick'): return f"{c['kind']}: {c['components'][c['w'].index(1.0) + 1]}"
    if c['kind'] == 'blend': return f"blend α={c['alpha']:g} of {len(c['components'])} models"
    return c['kind']
savetab(pd.DataFrame([{'weight': m, 'measurement': t, 'people': c['n'], 'final rule': describe(c)}
                      for m, cal in (('known', calib_w), ('unknown', calib_nw)) for t, c in cal.items()]).set_index(['weight', 'measurement']),
        'combiner_rules', 'Rule chosen by the combiner for each measurement')

# %% [markdown]
# ---
# ## 4. Evaluation protocol
# * **real24 (tape)** – repeated nested cross-validation: 6 outer folds = test people;
#   leave-one-out inside the training part chooses the combiner rule; repeated with **5 random
#   splits**; each person's error is averaged over the splits.
# * **External tests** (never used for training the combiner): HBW (3D scans, 10 people),
#   Model Agency official test split (134 people, no weight), BEDLAM (42 synthetic people;
#   *CameraHMR saw BEDLAM during its own training, so BEDLAM is optimistic*), BodyM testA/testB.
# * **Metrics:** MAE, RMSE, bias, R², MAPE, **accuracy = 100 − MAPE**, % within 1/2/3 cm,
#   95 % bootstrap CI of the MAE, and clothing-size classification (accuracy, precision,
#   recall, F1).

# %%
def cv_predictions(df, mode, targets, keys=KEYS, seeds=RNG_SEEDS):
    """Repeated nested CV → per target: y, per-split predictions (S x n), person info."""
    x = df[df['mode'] == mode]; out = {}
    for t in keys:
        if t not in targets or f'true:{t}' not in x: continue
        comps = [c for c in targets[t] if c in x and x[c].notna().any()]
        u = x[x[comps + [f'true:{t}']].notna().all(axis=1)]
        if len(u) < 8: continue
        X, y = u[comps].values.astype(float), u[f'true:{t}'].values.astype(float)
        P = []
        for s in seeds:
            P.append(C.nested_cv_app(X, y, 6, seed=s))   # keep-or-not decided on training people only
        out[t] = dict(y=y, P=np.array(P), person=u.person_id.values, gender=u.gender.values,
                      h=u.height_cm.values, w=u.weight_kg.values)
    return out


def metrics_from_cv(r):
    y, P = r['y'], r['P']
    per_split = [regression_metrics(y, p) for p in P]
    m = {k: float(np.mean([d[k] for d in per_split])) for k in per_split[0] if k != 'n'}
    person_err = np.abs(P - y).mean(0); lo, hi = boot_ci(person_err)
    return {'n': len(y), 'MAE (cm)': m['MAE (cm)'], 'MAE sd over splits': float(np.std([d['MAE (cm)'] for d in per_split])),
            '95% CI': f'{lo:.2f}–{hi:.2f}', 'RMSE (cm)': m['RMSE (cm)'], 'bias (cm)': m['bias (cm)'], 'R²': m['R²'],
            'accuracy (%)': 100 - m['MAPE (%)'], '≤1cm (%)': m['≤1cm (%)'], '≤2cm (%)': m['≤2cm (%)'], '≤3cm (%)': m['≤3cm (%)']}


def external(df, mode, keys=GIRTH):
    x = df[df['mode'] == mode]; out = {}
    for t in keys:
        if f'true:{t}' not in x: continue
        ys, ps = [], []
        for r in x.to_dict('records'):
            comps = {k: v for k, v in r.items() if ':' in k and not k.startswith('true:') and pd.notna(v)}
            v = apply_calibration(comps, weight_known=(mode == 'with_weight')).get(t)
            if v is not None and pd.notna(r.get(f'true:{t}')):
                ys.append(r[f'true:{t}']); ps.append(v)
        if len(ys) >= 3:
            out[t] = dict(y=np.array(ys, float), P=np.array([ps], float))
    return out

cv_w = cv_predictions(r24, 'with_weight', T_W)
cv_nw = cv_predictions(r24, 'no_weight', T_NW)
print('done: real24 repeated nested CV')

# %% [markdown]
# ---
# ## 5. Main results: every dataset, every measurement

# %%
tests = {('real24 (tape, CV)', 'weight known'): cv_w, ('real24 (tape, CV)', 'weight unknown'): cv_nw,
         ('HBW (3D scan)', 'weight known'): external(pred['hbw_val'], 'with_weight'),
         ('HBW (3D scan)', 'weight unknown'): external(pred['hbw_val'], 'no_weight'),
         ('Model Agency test', 'weight unknown'): external(pred['agency'][pred['agency'].split == 'test'], 'no_weight')}
if 'bedlam' in pred:
    tests[('BEDLAM* (synthetic)', 'weight known')] = external(pred['bedlam'], 'with_weight', KEYS)
    tests[('BEDLAM* (synthetic)', 'weight unknown')] = external(pred['bedlam'], 'no_weight', KEYS)
main = pd.DataFrame([{'dataset': d, 'weight': w, 'measurement': NAMES[t], **metrics_from_cv(r)}
                     for (d, w), res in tests.items() for t, r in res.items()]).set_index(['dataset', 'weight', 'measurement'])
savetab(main.round(2), 'main_results', 'Main results on all datasets (people never used for training)')
display(main.round(2))

# %%
# BodyM rows join the same summary (silhouette test sets, hybrid CNN alone)
summary = main['MAE (cm)'].unstack('measurement')[[NAMES[t] for t in GIRTH]]
for s in ('testA', 'testB'):
    b = bodym_tab[(bodym_tab['test set'] == s) & (bodym_tab.method == 'hybrid ensemble')].set_index('measurement')['MAE (cm)']
    summary.loc[(f'BodyM {s} (silhouettes)', 'weight known'), :] = [b['chest'], b['waist'], b['hip']]
acc = main['accuracy (%)'].unstack('measurement')[[NAMES[t] for t in GIRTH]]
summary = summary.join(acc, rsuffix=' acc. (%)')
savetab(summary.round(2), 'summary_all_datasets', 'MAE (cm) and accuracy (%) of chest/waist/hip on every test set')
display(summary.round(2))

fig, ax = plt.subplots(figsize=(12, 5))
summary[[NAMES[t] for t in GIRTH]].plot.barh(ax=ax); ax.set_xlabel('MAE (cm), lower is better'); ax.invert_yaxis()
ax.set_title('Chest / waist / hip error on every test set'); plt.tight_layout(); savefig(fig, 'summary_all_datasets'); plt.show()

# %%
# all real-photo test people together (real24 CV + HBW + agency test), each person counted once per measurement
pool = []
for (d, w), res in tests.items():
    if d.startswith('BEDLAM'): continue
    for t, r in res.items():
        if t in GIRTH:
            for e, y in zip(np.abs(r['P'] - r['y']).mean(0), r['y']):
                pool.append({'weight': w, 'measurement': NAMES[t], 'abs err': e, 'pct': e / y * 100})
pool = pd.DataFrame(pool)
pooled = pool.groupby(['weight', 'measurement']).agg(people=('abs err', 'size'), MAE=('abs err', 'mean'),
                                                     accuracy=('pct', lambda v: 100 - v.mean()))
savetab(pooled.round(2), 'pooled_real_photos', 'All real-photo test people pooled (real24 CV + HBW + agency test)')
display(pooled.round(2))

# %% [markdown]
# ### 5b. 5-fold cross-validation report (per fold)
# The classic k-fold table. The tape-measured people are split into **5 folds**; each fold is
# the **test** set once while the combiner is trained on the other 4 folds, with **validation**
# (leave-one-out) inside those 4 folds choosing its rule. So every fold has its own
# train / validation / test people, and no person is ever in two of them.
#
# * **Accuracy (%)** = 100 − mean percentage error of the measurement.
# * **Within ±2 cm (%)** = share of test people measured within 2 cm of the tape.
# * **Size accuracy (%)** = correct clothing size (XS–XXL) from that measurement.
# * **Precision / recall / F1** are computed on all test folds together (one fold holds only
#   3–4 people; an F1 from 3 people is meaningless).

# %%
K = 5
kf_rows, kf_cls, kf_folds = [], [], {}
for mode, T in (('with_weight', T_W), ('no_weight', T_NW)):
    x = r24[r24['mode'] == mode]
    wlabel = 'known' if mode == 'with_weight' else 'unknown'
    for t in KEYS:
        if t not in T or f'true:{t}' not in x: continue
        comps = [c for c in T[t] if c in x and x[c].notna().any()]
        u = x[x[comps + [f'true:{t}']].notna().all(axis=1)]
        if len(u) < 10: continue
        X, y = u[comps].values.astype(float), u[f'true:{t}'].values.astype(float)
        raw, cal, fold, chosen = C.nested_cv(X, y, K, seed=0)
        kf_folds[(wlabel, t)] = fold
        for f in range(K):
            te = fold == f
            e = np.abs(cal[te] - y[te])
            r = {'weight': wlabel, 'measurement': NAMES[t], 'fold': f'Fold {f + 1}', 'train people': int((~te).sum()),
                 'test people': int(te.sum()), 'rule chosen on validation': chosen[f],
                 'MAE (cm)': e.mean(), 'accuracy (%)': 100 - (e / y[te] * 100).mean(), 'within ±2 cm (%)': (e <= 2).mean() * 100}
            if t in SIZE_BINS:
                r['size accuracy (%)'] = (np.digitize(y[te], SIZE_BINS[t]) == np.digitize(cal[te], SIZE_BINS[t])).mean() * 100
            kf_rows.append(r)
        if t in SIZE_BINS:
            m, _ = size_metrics(y, cal, t)
            kf_cls.append({'weight': wlabel, 'size from': NAMES[t], 'people': len(y), **m})
kf = pd.DataFrame(kf_rows)
num = ['MAE (cm)', 'accuracy (%)', 'within ±2 cm (%)', 'size accuracy (%)']
per_fold = kf.set_index(['weight', 'measurement', 'fold'])
savetab(per_fold.round(2), 'kfold_per_fold', '5-fold cross-validation: every fold (test = unseen people)')
kf_summary = kf.groupby(['weight', 'measurement'], sort=False)[num].agg(['mean', 'std'])
kf_summary.columns = [f'{a} {b}' for a, b in kf_summary.columns]
savetab(kf_summary.round(2), 'kfold_summary', '5-fold cross-validation: mean and SD over the 5 folds')
print('Per-fold results (weight known, chest):')
display(per_fold.loc[('known', 'Chest')].round(2))
print('Mean ± SD over the 5 folds:')
display(pd.DataFrame({c: kf_summary[f'{c} mean'].round(2).astype(str) + ' ± ' + kf_summary[f'{c} std'].round(2).astype(str)
                      for c in num}))
kf_cls = pd.DataFrame(kf_cls).set_index(['weight', 'size from'])
savetab(kf_cls.round(3), 'kfold_size_classification', '5-fold CV: clothing-size classification (all test folds)')
print('Clothing-size classification over all 5 test folds:'); display(kf_cls.round(3))

# %%
fig, ax = plt.subplots(1, 2, figsize=(16, 4.2))
for a, w in zip(ax, ('known', 'unknown')):
    s = kf[(kf.weight == w) & kf.measurement.isin([NAMES[t] for t in GIRTH + ('thigh', 'neck')])]
    s.pivot_table(index='fold', columns='measurement', values='accuracy (%)').plot.bar(ax=a, rot=0)
    a.set_ylim(85, 100); a.set_ylabel('accuracy (%)'); a.set_title(f'5-fold CV accuracy per fold, weight {w}')
plt.tight_layout(); savefig(fig, 'kfold_accuracy'); plt.show()

# %% [markdown]
# ---
# ## 6. Comparison with published models
# Each published model alone, **out of the box** and **with a learned offset** (the offset is
# learned inside the same nested CV, so the comparison is fair), versus the TailorCam
# combiner. Paired Wilcoxon signed-rank test on per-person errors (real24, 5-split average).

# %%
BASE = {'HMR (2018)': 'hmr', 'SHAPY (2022)': 'shapy', 'CameraHMR (2025)': 'camerahmr',
        'Hybrid CNN (ours)': 'hybrid', 'ANSUR prior (ours)': 'ansur', 'BodyM prior (ours)': 'prior'}
comp = []
for mode, ours in (('with_weight', cv_w), ('no_weight', cv_nw)):
    x = r24[r24['mode'] == mode]
    for t in GIRTH + ('thigh', 'neck'):
        if t not in ours: continue
        e_ours = np.abs(ours[t]['P'] - ours[t]['y']).mean(0)
        row = {'weight': 'known' if mode == 'with_weight' else 'unknown', 'measurement': NAMES[t],
               'TailorCam (ours)': e_ours.mean()}
        for name, m in BASE.items():
            col = f'{m}:{t}'
            if col not in x or not x[col].notna().any(): continue
            r_ = cv_predictions(x.assign(**{'mode': mode}), mode, {t: [col]}, keys=(t,))
            if t not in r_: continue
            idx = [list(r_[t]['person']).index(p) for p in ours[t]['person'] if p in r_[t]['person']]
            e_off = np.abs(r_[t]['P'] - r_[t]['y']).mean(0)
            raw = np.abs(x.set_index('person_id').loc[ours[t]['person'], col].values - ours[t]['y'])
            row[f'{name} raw'] = raw.mean(); row[f'{name} +offset'] = e_off.mean()
            if len(idx) == len(e_ours):
                try:
                    row[f'p ({name})'] = stats.wilcoxon(e_ours, e_off[idx]).pvalue
                except ValueError:
                    pass
        comp.append(row)
comp = pd.DataFrame(comp).set_index(['weight', 'measurement'])
savetab(comp.round(3), 'comparison_published', 'TailorCam vs published models on real24 (MAE cm; p = Wilcoxon vs ours)')
display(comp[[c for c in comp.columns if not c.startswith('p (')]].round(2))
display(comp[[c for c in comp.columns if c.startswith('p (')]].round(3))

# %% [markdown]
# **How to read this table (important for the paper).** Without the weight, TailorCam is
# clearly better than every published model (e.g. waist 3.5 cm vs 7.4–7.9 cm for HMR/SHAPY,
# p < 0.01). With the weight known, some *single* models with an offset are as good or, in
# hindsight, better on one measurement (chest: SHAPY + offset 2.25 vs 2.51 cm). The combiner
# could choose exactly that rule, but with ~20 people the choice itself is noisy. We tested
# smaller rule sets to reduce that noise (`scripts/candidate_experiment.py`): removing
# the blends gives chest 2.42 cm but thigh 1.54 → 2.35 cm and no-weight hip 4.49 → 5.21 cm,
# so no rule set is better overall. Picking SHAPY for chest *after* seeing this table would be
# test-set selection, so we do not do it; more tape-measured people will settle it.

# %%
fig, ax = plt.subplots(1, 2, figsize=(16, 4.5))
for a, w in zip(ax, ('known', 'unknown')):
    sub = comp.loc[w][['TailorCam (ours)'] + [c for c in comp.columns if c.endswith('+offset')]]
    sub = sub.loc[[NAMES[t] for t in GIRTH if NAMES[t] in sub.index]]
    sub.T.plot.bar(ax=a, rot=30); a.set_ylabel('MAE (cm)'); a.set_title(f'real24, weight {w}: ours vs single models (+offset)')
plt.tight_layout(); savefig(fig, 'comparison_published'); plt.show()

# %% [markdown]
# ### 6b. Comparison with results published by other groups (indicative)
# Numbers below come from Choudhary et al., *npj Digital Medicine* 6:168 (2023), Table 2: men,
# **Human Solutions dataset**, 215 people, MAE in cm, two views (front + side) where
# available. They were measured on **different people** with a different tape protocol, so
# this table places TailorCam among published methods; it is not a head-to-head test.
# MeasureNet (Amazon Halo) was trained on a large proprietary dataset.

# %%
lit = pd.DataFrame([
    ('SPIN (Kolotouros 2019)', 16.25, 12.83, 10.47, 'public 3D-pose model'),
    ('STRAPS (Sengupta 2020)', 5.03, 6.64, 7.36, 'public'),
    ('Sengupta et al. 2021, sex-specific', 5.08, 4.28, 3.91, 'public'),
    ('Smith et al.', 2.59, 2.56, 2.40, 'published'),
    ('MeasureNet, 2 views (Amazon 2023)', 1.78, 1.75, 1.60, 'company, proprietary data'),
], columns=['method', 'chest (cm)', 'waist (cm)', 'hip (cm)', 'note']).set_index('method')
ours = main.loc[('real24 (tape, CV)', 'weight known')]['MAE (cm)']
lit.loc['TailorCam (ours), real24, weight known'] = [ours['Chest'], ours['Waist'], ours['Hip'], 'this work, 5-split nested CV']
savetab(lit.round(2), 'literature_comparison', 'Published results (Choudhary et al. 2023, Table 2) and TailorCam; different test people')
display(lit.round(2))

# %% [markdown]
# ---
# ## 7. Ablation study
# Remove one source at a time from the combiner (repeated nested CV, real24). A positive
# Δ means the measurement gets **worse** without that source, i.e. the source is useful.

# %%
from repeated_cv import score
abl = []
for mode, T in (('with_weight', T_W), ('no_weight', T_NW)):
    full = score(r24, mode, T)
    for drop in ('hybrid', 'prior', 'ansur', 'hmr', 'shapy', 'camerahmr'):
        T2 = {k: [c for c in v if not c.startswith(drop + ':')] for k, v in T.items()}
        T2 = {k: v for k, v in T2.items() if v}
        res = score(r24, mode, T2)
        for t in full:
            if t in res:
                abl.append({'weight': 'known' if mode == 'with_weight' else 'unknown', 'removed': drop,
                            'measurement': NAMES[t], 'Δ MAE (cm)': res[t][0] - full[t][0]})
abl = pd.DataFrame(abl).pivot_table(index=['weight', 'removed'], columns='measurement', values='Δ MAE (cm)')
savetab(abl.round(2), 'ablation', 'Ablation: change in MAE when one source is removed (positive = source helps)')
display(abl.round(2))
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
for a, w in zip(ax, ('known', 'unknown')):
    im = a.imshow(abl.loc[w].values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    a.set_yticks(range(len(abl.loc[w]))); a.set_yticklabels(abl.loc[w].index)
    a.set_xticks(range(abl.shape[1])); a.set_xticklabels(abl.columns, rotation=30); a.grid(False)
    for i in range(abl.loc[w].shape[0]):
        for j in range(abl.shape[1]):
            v = abl.loc[w].values[i, j]
            if np.isfinite(v): a.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=8)
    a.set_title(f'weight {w}: Δ MAE when removed (red = useful)')
plt.colorbar(im, ax=ax); savefig(fig, 'ablation'); plt.show()

# %% [markdown]
# ---
# ## 8. Agreement with the tape: Bland–Altman plots
# x = mean of tape and TailorCam, y = TailorCam − tape. Solid line = mean bias; dashed lines
# = 95 % limits of agreement (bias ± 1.96 SD). This is the standard agreement figure in
# anthropometry papers.

# %%
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
loa = []
for i, (label, res) in enumerate((('real24, weight known (CV)', cv_w), ('real24, weight unknown (CV)', cv_nw))):
    for j, t in enumerate(GIRTH):
        if t not in res: continue
        y, p = res[t]['y'], res[t]['P'].mean(0)
        mean, diff = (y + p) / 2, p - y
        b, sd = diff.mean(), diff.std(ddof=1)
        a = ax[i, j]; a.scatter(mean, diff, s=25)
        a.axhline(b, color='k'); a.axhline(b + 1.96 * sd, color='r', ls='--'); a.axhline(b - 1.96 * sd, color='r', ls='--')
        a.set_title(f'{NAMES[t]} — {label}', fontsize=9); a.set_xlabel('mean of tape and TailorCam (cm)'); a.set_ylabel('TailorCam − tape (cm)')
        loa.append({'test': label, 'measurement': NAMES[t], 'bias (cm)': b, 'lower LoA': b - 1.96 * sd, 'upper LoA': b + 1.96 * sd})
plt.tight_layout(); savefig(fig, 'bland_altman'); plt.show()
savetab(pd.DataFrame(loa).set_index(['test', 'measurement']).round(2), 'limits_of_agreement', 'Bland–Altman bias and 95% limits of agreement')

# %% [markdown]
# ---
# ## 9. Error analysis: who is measured worst?

# %%
ea = []
for t in GIRTH:
    r = cv_w[t]
    for p, g, h, w, y, e in zip(r['person'], r['gender'], r['h'], r['w'], r['y'], np.abs(r['P'] - r['y']).mean(0)):
        ea.append({'person': p, 'gender': g, 'height': h, 'BMI': w / (h / 100) ** 2, 'measurement': NAMES[t], 'true': y, 'abs err': e})
ea = pd.DataFrame(ea)
fig, ax = plt.subplots(1, 3, figsize=(17, 4))
ea.boxplot(column='abs err', by=['measurement', 'gender'], ax=ax[0], rot=45); ax[0].set_title('error by gender'); ax[0].set_xlabel('')
for t in GIRTH:
    s = ea[ea.measurement == NAMES[t]]; ax[1].scatter(s.BMI, s['abs err'], label=NAMES[t], s=20)
    ax[2].scatter(s.height, s['abs err'], label=NAMES[t], s=20)
ax[1].set_xlabel('BMI'); ax[2].set_xlabel('height (cm)')
for a in ax[1:]: a.set_ylabel('abs error (cm)'); a.legend()
plt.suptitle('real24, weight known (5-split CV)'); plt.tight_layout(); savefig(fig, 'error_analysis'); plt.show()
corr = pd.DataFrame([{'measurement': NAMES[t], **{f'Spearman ρ vs {v}': stats.spearmanr(ea[ea.measurement == NAMES[t]][v],
                     ea[ea.measurement == NAMES[t]]['abs err']).correlation for v in ('BMI', 'height')}} for t in GIRTH]).set_index('measurement')
savetab(corr.round(2), 'error_correlations', 'Correlation of error with BMI and height')
display(corr.round(2))
display(ea.sort_values('abs err', ascending=False).head(8).round(1))

# %% [markdown]
# ---
# ## 10. Clothing-size classification (accuracy, precision, recall, F1)
# Chest and waist are converted to sizes XS–XXL (chest bins 84/92/100/108/116 cm; waist
# 70/78/86/94/102 cm) for both the tape and TailorCam values.

# %%
sz, cms = [], {}
for (d, w), res in tests.items():
    for t in ('chest', 'waist'):
        if t not in res: continue
        per = [size_metrics(res[t]['y'], p, t) for p in res[t]['P']]
        m = {k: np.mean([x[0][k] for x in per]) for k in per[0][0]}
        sz.append({'dataset': d, 'weight': w, 'size from': t, 'people': len(res[t]['y']), **m})
        cms[(d, w, t)] = per[0][1]
sz = pd.DataFrame(sz).set_index(['dataset', 'weight', 'size from'])
savetab(sz.round(3), 'size_classification', 'Clothing-size classification on every test set')
display(sz.round(3))
show = [k for k in cms if k[0].startswith('real24')] + [k for k in cms if k[0].startswith('Model')]
fig, ax = plt.subplots(1, len(show), figsize=(4.3 * len(show), 4))
for a, k in zip(np.atleast_1d(ax), show):
    cm = cms[k]; a.imshow(cm, cmap='Blues'); a.grid(False)
    a.set_xticks(range(6)); a.set_xticklabels(SIZES); a.set_yticks(range(6)); a.set_yticklabels(SIZES)
    for i in range(6):
        for j in range(6):
            if cm[i, j]: a.text(j, i, cm[i, j], ha='center', va='center', fontsize=8)
    a.set_title(f'{k[0][:10]} {k[1]}\n{k[2]} size', fontsize=9); a.set_xlabel('predicted'); a.set_ylabel('true')
plt.tight_layout(); savefig(fig, 'confusion_matrices'); plt.show()

# %% [markdown]
# ---
# ## 11. Can the camera replace the tape? (per measurement)
# A measurement is marked **camera** only if 80 % of tested people were within the tailoring
# tolerance. The app shows this as a CAM / TAPE tag on every result.

# %%
tc = pd.DataFrame([{'weight': 'known' if wk else 'unknown', 'measurement': NAMES.get(k, k), 'tolerance (cm)': TAPE_TOLERANCE_CM[k],
                    'avg error (cm)': v['expected_error'], '80% within (cm)': v['p80'], 'verdict': v['status'].upper()}
                   for wk in (True, False) for k, v in tape_check(wk).items()]).set_index(['weight', 'measurement'])
savetab(tc, 'camera_vs_tape', 'Camera vs tape verdict per measurement'); display(tc)

# %% [markdown]
# ---
# ## 12. What did not work (negative results)
# Every idea below was implemented and tested with the same honest protocol; none was kept
# unless it helped on people it had not seen. Scripts are in `scripts/` and `model/`.
#
# | Idea | Test | Result | Decision |
# |---|---|---|---|
# | SAM 2 sharper silhouettes (`model/segmenter.py`) | real24 | no gain (waist slightly worse) | off |
# | Fit one body to all photos' outlines (`model/silhouette_fit.py`) | real24, HBW, BEDLAM | "inside" rule never active; "fill" fits clothes: chest (no weight) 3.6 → 6.3 cm | off |
# | Model disagreement as per-person confidence | 1,184 measurements | rank correlation ≤ 0.3 | not used |
# | Tape one value, correct the others | agency → others | helps fashion models only | not used |
# | Train no-weight combiner on agency models | agency → real24/HBW | worse on ordinary people | not used |
# | NHANES prior, ~28k people (`model/nhanes_prior.py`) | combiner CV | waist 3.64 → 4.21 cm | off |
# | One combiner on all 781 people (`model/pooled_combiner.py`) | 5-split nested CV | not better (hip 3.15 → 3.58) | off |
# | CameraHMR person-box size | real24, HBW, BEDLAM | no consistent difference | default kept |
# | Combiner scale factor | 5-split CV | chest 2.51 → 2.72 | off |
# | Body density 950–1070 kg/m³ | 5-split CV | no difference after combiner | 1010 kept |
# | Learn tape height on the 3D body | real24 → HBW | real24 waist 3.55 → 2.38 but HBW 3.90 → 5.31 (overfit) | off |
# | Combiner augmentation: noise / mixup (`scripts/augment_experiment.py`) | 5-split CV | trades gains for losses (e.g. no-weight hip 4.49 → 5.19) | off |
# | Smaller combiner rule sets / "average all models" rule (`scripts/candidate_experiment.py`) | 5-split CV | chest 2.51 → 2.42 but thigh 1.54 → 2.35, no-weight hip 4.49 → 5.21 | off |
# | **CameraHMR mirror test-time augmentation** | 5-split CV | no-weight hip 4.93 → 4.49, chest 3.52 → 3.41 | **kept** |
#
# The augmentation result is reproduced live below.

# %%
from augment_experiment import make_augmented_fit, _orig_fit
aug_rows = []
for mode, T in (('with_weight', T_W), ('no_weight', T_NW)):
    base = score(r24, mode, T)
    for name, kind, k in (('noise k=0.5', 'noise', 0.5), ('mixup α=0.4', 'mixup', 0.4)):
        C.fit_candidate = make_augmented_fit(kind, k)
        res = score(r24, mode, T); C.fit_candidate = _orig_fit
        for t in base:
            aug_rows.append({'weight': mode, 'augmentation': name, 'measurement': NAMES[t],
                             'without (cm)': base[t][0], 'with (cm)': res[t][0]})
aug = pd.DataFrame(aug_rows).pivot_table(index=['weight', 'augmentation'], columns='measurement', values=['without (cm)', 'with (cm)'])
savetab(aug.round(2), 'augmentation', 'Combiner data augmentation (5-split nested CV)'); display(aug.round(2))

# %% [markdown]
# ---
# ## 13. Runtime
# Measured through the same service the app uses (models already loaded), on this computer.

# %%
RUN_TIMING = True
if RUN_TIMING:
    from model.hmr.service import HMRService
    from output import load_photo
    svc = HMRService(); svc.start(); t0 = time.time()
    while not svc.ready and time.time() - t0 < 300: time.sleep(1)
    load_s = time.time() - t0; times = []
    for pid in real.person_id[:5]:
        r_ = real.set_index('person_id').loc[pid]
        fr = [('FRONT', load_photo(f'../dataset/real_photos/{pid}/front.jpg'), None), ('LEFT', load_photo(f'../dataset/real_photos/{pid}/side.jpg'), None)]
        t1 = time.time(); svc.measure(r_.height_cm, fr, weight_kg=r_.weight_kg, gender=r_.gender, timeout=120); times.append(time.time() - t1)
    rt = pd.DataFrame({'value': {'GPU': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                                 'model loading (s, once at start)': round(load_s, 1),
                                 'one customer, 2 photos (s), mean of 5': round(float(np.mean(times)), 2)}})
    try: svc.stop()
    except Exception: pass
    savetab(rt, 'runtime', 'Runtime'); display(rt)

# %% [markdown]
# ---
# ## 14. Auto-written summary (numbers filled in from the results above)

# %%
def g(d, w, t, col='MAE (cm)'):
    try: return main.loc[(d, w, NAMES[t]), col]
    except KeyError: return np.nan
rw, rn = 'real24 (tape, CV)', 'real24 (tape, CV)'
szc = sz.loc[('real24 (tape, CV)', 'weight known', 'chest')]
text = f"""**Summary.** We present TailorCam, a hybrid system that estimates tailoring measurements from a
front and a side colour photo plus the customer's height (and optionally weight). Six sources —
three published 3D body-shape models (HMR, SHAPY, CameraHMR) adapted to the customer's weight, a
silhouette CNN ensemble trained on BodyM, and two anthropometric priors (BodyM, ANSUR II) — are
merged by a combiner trained with nested cross-validation on real tape-measured people. With the
weight known, the mean absolute error on unseen people (5-split repeated nested CV) is
{g(rw,'weight known','chest'):.2f} cm for chest, {g(rw,'weight known','waist'):.2f} cm for waist and
{g(rw,'weight known','hip'):.2f} cm for hip (accuracy {g(rw,'weight known','chest','accuracy (%)'):.1f} %,
{g(rw,'weight known','waist','accuracy (%)'):.1f} % and {g(rw,'weight known','hip','accuracy (%)'):.1f} %);
neck {g(rw,'weight known','neck'):.2f} cm and thigh {g(rw,'weight known','thigh'):.2f} cm. Without the weight:
chest {g(rn,'weight unknown','chest'):.2f}, waist {g(rn,'weight unknown','waist'):.2f}, hip {g(rn,'weight unknown','hip'):.2f} cm.
On the independent HBW 3D-scan test set (weight known) the errors are {g('HBW (3D scan)','weight known','chest'):.2f} /
{g('HBW (3D scan)','weight known','waist'):.2f} / {g('HBW (3D scan)','weight known','hip'):.2f} cm. The predicted chest size
is correct for {szc['accuracy (%)']:.0f} % of people and within one size for {szc['within ±1 size (%)']:.0f} %
(macro F1 = {szc['F1 (macro)']:.2f}). Fourteen further ideas were tested and are reported as negative results."""
display(Markdown(text)); open(os.path.join(OUT, 'summary.md'), 'w', encoding='utf-8').write(text)
print('\nFiles written:'); [print('  ', os.path.relpath(p, ROOT)) for p in sorted(glob.glob(os.path.join(OUT, '*', '*.png')) + glob.glob(os.path.join(OUT, '*', '*.tex')))]

# %% [markdown]
# ---
# ## 15. Limitations and threats to validity
# * **Small tape-measured set.** Only 24 free real people exist with photos + tape (16–20 per
#   measurement). Differences below ~0.5 cm between methods are within split-to-split noise
#   (see "MAE sd over splits"). Data requests have been sent to the authors of NOMO BODY-rgb
#   (194 people) and Fashion-body (30 people).
# * **Different measuring definitions.** Agency values are self-reported and rounded; HBW
#   measures at SHAPY's landmarks; NHANES measures the waist at the iliac crest. Cross-dataset
#   errors therefore include definition differences.
# * **Clothing.** Loose clothes hide the body; every attempt to extract more from the outline
#   failed (Section 12). Tight clothing during the scan is the most promising practical lever.
# * **BEDLAM is optimistic** for CameraHMR, which was trained on BEDLAM.
# * **Large bodies.** The Bland–Altman plot shows one person with a hip of ~125 cm
#   under-estimated by 12 cm (19 cm without weight). Very large bodies are rare in every
#   training set; scans of large customers should be tape-checked.
# * **Population.** No dataset contains South-Asian customers specifically; the shop's own
#   tape-measured scans (`scripts/retrain_from_shop.py`) are needed to adapt to them.
# * **Licences.** All datasets are for non-commercial research (see Section 1).
