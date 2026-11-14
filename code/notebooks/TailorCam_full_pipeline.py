# %% [markdown]
# # TailorCam: Complete Training & Evaluation Notebook
#
# This notebook shows every step of how TailorCam's measurement system is built
# and tested: the data, the models, training, validation and testing, with
# results shown as tables and charts.
#
# ### The final system: a hybrid of several models
# ```
#  camera photo ──► MediaPipe ──► silhouettes ──► (A) Hybrid ensemble: 3 CNNs trained on BodyM ──┐
#               ├─► (B) HMR 3D body model (2018), adjusted to the real body weight ──────────────────┤
#               └─► (E) SHAPY 3D body model (CVPR 2022), adjusted to the real body weight ───────────┤
#  height + weight + gender + age ─► (C) Priors: BodyM (2,018 people) + ANSUR II (6,068, tape) ───────┤
#                                                                                                       ▼
#               (D) Combiner: learned on REAL colour photos of tape-measured people
#                                                                                                       ▼
#                                       chest · waist · hip · thigh · neck · shoulder · sleeve
# ```
#
# | Part | What happens | Data |
# |---|---|---|
# | 1 | Look at the datasets | BodyM (2,505), real photos (24), SHAPY agency photos (715), HBW scans (10) |
# | 2 | Train model A (train / validation split) | BodyM train |
# | 3 | Test model A on unseen people | BodyM testA + testB (487 people) |
# | 4 | Full pipeline step by step; every model on every real-photo dataset | 749 real people |
# | 5 | Train the combiner (D) with nested cross-validation | 24 tape-measured people (+ agency check) |
# | 6 | Final results: MAE, R², accuracy, F1 and more | real24 (CV), HBW, agency test |

# %%
import os, sys, re, glob, json, time, warnings
from IPython.display import display
warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.getcwd(), '..')) if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd, cv2, torch
import matplotlib.pyplot as plt
try:
    get_ipython().run_line_magic('matplotlib', 'inline')    # show charts inside the notebook
except NameError:
    pass
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
plt.rcParams.update({'figure.dpi': 90, 'axes.grid': True, 'grid.alpha': .3})

from model.bodym.bodym_net import MEASUREMENTS
from model.bodym.train_bodym import load_split
from model.bodym.hybrid import HybridNet, member_paths, predict_split, DEV
from evaluate_project import regression_metrics, size_metrics, SIZES, SIZE_BINS
print('Project folder:', ROOT)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU)')
print('Trained models found:', [os.path.basename(p) for p in member_paths()])

# %% [markdown]
# ---
# ## Part 1: The datasets
# ### 1a. BodyM (Amazon), used for training
# 2,505 real people with front and side **silhouettes** (the real photos were
# removed for privacy), height, weight, gender and 14 measurements from 3D
# body scans. The official split keeps every person in only one set.

# %%
bodym = {s: load_split(s) for s in ('train', 'testA', 'testB')}
def n_people(d, g):
    return len(set(d['subj'][d['g'] == g]))

pd.DataFrame([{'split': s, 'people': len(set(d['subj'])), 'photo pairs': len(d['y']),
               'men': n_people(d, 'male'), 'women': n_people(d, 'female'),
               'height (cm) mean': round(float(d['h'].mean()), 1),
               'weight (kg) mean': round(float(d['w'].mean()), 1)} for s, d in bodym.items()])

# %%
tr = bodym['train']
fig, ax = plt.subplots(1, 10, figsize=(18, 3.6))
for i, j in enumerate(np.random.default_rng(3).choice(len(tr['y']), 5, replace=False)):
    ax[2 * i].imshow(tr['front'][j], cmap='gray'); ax[2 * i + 1].imshow(tr['side'][j], cmap='gray')
    ax[2 * i].set_title(f"{tr['g'][j]}, {tr['h'][j]:.0f} cm", fontsize=9)
    ax[2 * i + 1].set_title(f"{tr['w'][j]:.0f} kg, chest {tr['y'][j][MEASUREMENTS.index('chest')]:.0f}", fontsize=9)
for a in ax: a.axis('off')
plt.suptitle('BodyM training examples (front + side silhouettes after pre-processing)'); plt.show()

fig, ax = plt.subplots(1, 4, figsize=(16, 3))
for a, m in zip(ax, ('chest', 'waist', 'hip', 'thigh')):
    a.hist(tr['y'][:, MEASUREMENTS.index(m)], bins=40, color='tab:blue'); a.set_title(f'{m} (cm) — train set')
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 1b. Real colour photos with tape measurements, used for the final combiner and test
# These are all the **free** sources found on the internet (October 2026), merged and
# de-duplicated by `scripts/build_real_photo_dataset.py`:
# - ud-biometrics (Hugging Face): 6 people
# - UniqueData (Hugging Face): 21 people, 3 of whom duplicate the first set
#
# That gives **24 unique people**. The provider marked some values "to be
# reviewed"; those are **not used** for training or scoring.

# %%
people = pd.read_csv('../dataset/real_photos/people.csv')
show = ['person_id', 'gender', 'age', 'height_cm', 'weight_kg', 'chest', 'waist', 'hip', 'thigh',
        'neck', 'shoulder_width', 'arm_length', 'flagged']
people[show]

# %%
n = len(people); cols = 8
fig, ax = plt.subplots(int(np.ceil(n / cols * 2)), cols, figsize=(18, 2.6 * np.ceil(n / cols * 2)))
ax = ax.ravel()
for i, r in people.iterrows():
    for k, v in enumerate(('front', 'side')):
        img = cv2.cvtColor(cv2.imread(f"../dataset/real_photos/{r.person_id}/{v}.jpg"), cv2.COLOR_BGR2RGB)
        a = ax[2 * i + k]; a.imshow(img); a.axis('off')
        a.set_title(f"{r.person_id} {v}\n{r.height_cm:.0f}cm {r.weight_kg:.0f}kg", fontsize=8)
for a in ax[2 * n:]: a.axis('off')
plt.suptitle('All 24 real people (front + side photos)', y=1.0); plt.tight_layout(); plt.show()

# %%
targets = ['chest', 'waist', 'hip', 'thigh', 'neck', 'shoulder_width', 'arm_length']
usable = pd.DataFrame({t: [(pd.notna(r[t]) and t not in str(r.flagged).split(';')) for _, r in people.iterrows()]
                       for t in targets}, index=people.person_id)
fig, ax = plt.subplots(figsize=(9, 6))
ax.imshow(usable.values.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_yticks(range(len(targets))); ax.set_yticklabels(targets)
ax.set_xticks(range(n)); ax.set_xticklabels(usable.index, rotation=90, fontsize=8)
ax.set_title('Which tape values are usable (green) vs missing / flagged (red)'); ax.grid(False); plt.show()
print('usable values per measurement:', usable.sum().to_dict())

# %% [markdown]
# ### 1c. SHAPY Model Agency data: real photos of 715 people
# From the SHAPY project (Max Planck Institute, CVPR 2022). Fashion-model agencies
# list each model's height, chest, waist and hip. About 4,700 models were
# released as web links, but only the **715 whose photos are still online** could be
# downloaded (`scripts/download_model_agency.py`). The SHAPY authors' **official
# train / validation / test split** is used as-is.
#
# Limits: the measurements are listed by the agencies (not taped by researchers),
# there is **no body weight**, and the photos are fashion poses.

# %%
agency = pd.read_csv('../dataset/shapy/model_agency_images/index.csv')
display(pd.DataFrame({'people': agency.groupby('split').size(),
                      'photos': agency.groupby('split')['n_images'].sum()}).T[['train', 'val', 'test']])
fig, ax = plt.subplots(1, 5, figsize=(18, 3))
for a, (c, t) in zip(ax, [('height_cm', 'height'), ('chest_cm', 'chest'), ('waist_cm', 'waist'),
                          ('hip_cm', 'hip'), ('agency', 'agency')]):
    if c == 'agency':
        agency[c].value_counts().plot.bar(ax=a, rot=30)
    else:
        a.hist(agency[c], bins=30)
    a.set_title(f'{t} (cm)' if c != 'agency' else 'people per agency')
plt.tight_layout(); plt.show()
fig, ax = plt.subplots(1, 8, figsize=(18, 3.6))
for a, (_, r) in zip(ax, agency.sample(8, random_state=1).iterrows()):
    p = sorted(glob.glob(f'../dataset/shapy/model_agency_images/{r.agency}/{r.model_id}/*.jpg'))[0]
    a.imshow(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)); a.axis('off')
    a.set_title(f'{r.split}: {r.height_cm:.0f}cm\nC{r.chest_cm:.0f} W{r.waist_cm:.0f} H{r.hip_cm:.0f}', fontsize=8)
plt.suptitle('Model agency examples'); plt.show()

# %% [markdown]
# ### 1d. HBW (Human Bodies in the Wild): 10 people with 3D body scans
# Real photos taken in a lab and in everyday life, plus each person's **3D body scan**.
# The true chest, waist, hip, height **and weight** are calculated from the scan with
# SHAPY's official landmark definitions (`model/shapy_measure.py`). HBW is used **only
# for testing**, never for training.

# %%
from model.shapy_measure import load_obj, measure_smplx
hbw_gt = []
for obj in sorted(glob.glob('../dataset/shapy/HBW_low_resolution/smplx/val/*.obj')):
    V_, F_ = load_obj(obj)
    hbw_gt.append({'subject': os.path.basename(obj)[:-4], **measure_smplx(V_, F_)})
display(pd.DataFrame(hbw_gt))
sub = sorted(glob.glob('../dataset/shapy/HBW_low_resolution/images/val_small_resolution/012_*'))[0]
imgs_ = sorted(glob.glob(sub + '/*/*.png'))
fig, ax = plt.subplots(1, 7, figsize=(18, 3.6))
for a, p in zip(ax[:6], [imgs_[i] for i in np.linspace(0, len(imgs_) - 1, 6).astype(int)]):
    a.imshow(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)); a.axis('off')
    a.set_title(p.split(os.sep)[-2][:14], fontsize=8)
ax[6].imshow(cv2.cvtColor(cv2.imread('../dataset/shapy/HBW_low_resolution/smplx/val/012.png'), cv2.COLOR_BGR2RGB))
ax[6].axis('off'); ax[6].set_title('3D scan (truth)')
plt.suptitle('HBW subject 012: photos + true 3D body'); plt.show()

# %% [markdown]
# ---
# ## Part 2: Training model A (hybrid ensemble) on BodyM
# **Split:** the BodyM train set is divided by **person** (never by photo, so the same
# person can't appear in both): 90% for training, 10% for validation. The
# validation score picks the best epoch. The test sets are not used at all here.
#
# **Model:** a ResNet-18 that reads both silhouettes and corrects a height+weight prior,
# and also predicts its own uncertainty. Three copies are trained with different
# random seeds and averaged (an *ensemble*).
#
# Set `TRAIN_AGAIN = True` to retrain live (about 10 minutes per model on an
# RTX 4070). Otherwise the saved models and their recorded training logs are used.

# %%
TRAIN_AGAIN = False
subj = np.unique(tr['subj'])
val_subj = set(np.random.default_rng(0).choice(subj, len(subj) // 10, replace=False))
split_sizes = {'train (people)': len(subj) - len(val_subj), 'validation (people)': len(val_subj),
               'testA (people)': len(set(bodym['testA']['subj'])), 'testB (people)': len(set(bodym['testB']['subj']))}
plt.figure(figsize=(8, 2.6))
plt.barh(list(split_sizes), list(split_sizes.values()), color=['tab:blue', 'tab:orange', 'tab:green', 'tab:green'])
for i, v in enumerate(split_sizes.values()): plt.text(v + 10, i, str(v), va='center')
plt.title('BodyM split: every person is in exactly one set'); plt.gca().invert_yaxis(); plt.show()

if TRAIN_AGAIN:
    import subprocess
    for seed in (0, 1, 2):
        subprocess.run([sys.executable, '-m', 'model.bodym.hybrid', '--epochs', '40',
                        '--seed', str(seed), '--no-eval'], check=True)

# %%
def read_log(path):
    rows = []
    for line in open(path):
        m = re.search(r'epoch\s+(\d+)/(\d+)\s+loss ([\d.]+)\s+val chest/waist/hip MAE ([\d.]+)', line)
        if m: rows.append((int(m[1]), float(m[3]), float(m[4])))
    return pd.DataFrame(rows, columns=['epoch', 'train loss', 'validation MAE (cm)'])

logs = {f'model {i}': read_log(f'model/bodym/logs/train_s{i}.log') for i in range(3)
        if os.path.exists(f'model/bodym/logs/train_s{i}.log')}
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
for name, df in logs.items():
    ax[0].plot(df['epoch'], df['train loss'], label=name)
    ax[1].plot(df['epoch'], df['validation MAE (cm)'], label=name)
    b = df['validation MAE (cm)'].idxmin()
    ax[1].scatter(df.loc[b, 'epoch'], df.loc[b, 'validation MAE (cm)'], s=60, zorder=5)
ax[0].set_title('Training loss (goes down = learning)'); ax[0].set_xlabel('epoch'); ax[0].legend()
ax[1].set_title('Validation error: chest/waist/hip MAE (dot = saved best)'); ax[1].set_xlabel('epoch'); ax[1].legend()
plt.show()
pd.DataFrame({n: {'epochs run': int(d['epoch'].max()), 'best validation MAE (cm)': d['validation MAE (cm)'].min(),
                  'best epoch': int(d.loc[d['validation MAE (cm)'].idxmin(), 'epoch'])} for n, d in logs.items()}).T

# %% [markdown]
# ---
# ## Part 3: Testing model A on BodyM test sets (487 people never seen in training)
# Three versions are compared:
# - **height+weight only**: the prior, with no camera at all (the baseline)
# - **hybrid ensemble**: the prior corrected by the silhouettes (what TailorCam uses)
# - **hybrid, no weight**: for when the customer doesn't give a weight

# %%
nets = []
for p in member_paths():
    net = HybridNet().to(DEV); net.load_state_dict(torch.load(p, map_location=DEV, weights_only=True)); nets.append(net)

bodym_pred = {}
for s in ('testA', 'testB'):
    D = bodym[s]
    runs = [predict_split(n, D) for n in nets]
    bodym_pred[s] = {'height+weight only': np.mean([r['prior'] for r in runs], 0),
                     'hybrid ensemble': np.mean([r['image'] for r in runs], 0),
                     'hybrid, no weight': np.mean([predict_split(n, D, use_weight=False)['image'] for n in nets], 0)}
    for i, r in enumerate(runs):
        bodym_pred[s][f'single model {i}'] = r['image']

rows = []
for s in ('testA', 'testB'):
    for meth, P in bodym_pred[s].items():
        for m in ('chest', 'waist', 'hip', 'thigh', 'shoulder-breadth', 'arm-length'):
            j = MEASUREMENTS.index(m)
            rows.append({'test set': s, 'method': meth, 'measurement': m,
                         'MAE (cm)': np.abs(P[:, j] - bodym[s]['y'][:, j]).mean()})
cmp = pd.DataFrame(rows).pivot_table(index=['test set', 'method'], columns='measurement', values='MAE (cm)').round(2)
cmp

# %%
fig, ax = plt.subplots(1, 2, figsize=(16, 4.5))
for a, s in zip(ax, ('testA', 'testB')):
    sub = cmp.loc[s][['chest', 'waist', 'hip', 'thigh']]
    sub = sub.loc[['height+weight only', 'single model 0', 'hybrid ensemble']]
    sub.T.plot.bar(ax=a, rot=0, color=['tab:gray', 'tab:orange', 'tab:green'])
    a.set_title(f'BodyM {s}: average error in cm (lower is better)'); a.set_ylabel('MAE (cm)')
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Full metric table for the hybrid ensemble (testB, 400 people)
# - **MAE**: average error in cm
# - **RMSE**: like MAE, but punishes large errors more
# - **R²**: share of the person-to-person variation the model explains (1 = perfect)
# - **≤1/2/3 cm**: % of people predicted within that tolerance

# %%
D, P = bodym['testB'], bodym_pred['testB']['hybrid ensemble']
pd.DataFrame([{'measurement': m, **regression_metrics(D['y'][:, j], P[:, j])}
              for j, m in enumerate(MEASUREMENTS)]).set_index('measurement').round(2)

# %%
fig, ax = plt.subplots(1, 4, figsize=(17, 4))
for a, m in zip(ax, ('chest', 'waist', 'hip', 'thigh')):
    j = MEASUREMENTS.index(m)
    a.scatter(D['y'][:, j], P[:, j], s=6, alpha=.4)
    lo, hi = D['y'][:, j].min(), D['y'][:, j].max(); a.plot([lo, hi], [lo, hi], 'r--')
    a.set_title(f'testB {m}: predicted vs true'); a.set_xlabel('true (cm)'); a.set_ylabel('predicted (cm)')
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Size classification on BodyM: accuracy, precision, recall, F1
# Predicted and true chest/waist are converted to shirt sizes (XS…XXL) and scored as
# a classification task.

# %%
rows = []
fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
k = 0
for s in ('testA', 'testB'):
    for m in ('chest', 'waist'):
        j = MEASUREMENTS.index(m)
        sm, cm = size_metrics(bodym[s]['y'][:, j], bodym_pred[s]['hybrid ensemble'][:, j], m)
        rows.append({'test set': s, 'size from': m, **sm})
        ax[k].imshow(cm, cmap='Blues'); ax[k].grid(False)
        ax[k].set_xticks(range(6)); ax[k].set_xticklabels(SIZES); ax[k].set_yticks(range(6)); ax[k].set_yticklabels(SIZES)
        for a_ in range(6):
            for b_ in range(6):
                if cm[a_, b_]: ax[k].text(b_, a_, cm[a_, b_], ha='center', va='center', fontsize=8)
        ax[k].set_title(f'{s} {m} size  acc {sm["accuracy (%)"]:.0f}%'); ax[k].set_xlabel('predicted'); ax[k].set_ylabel('true')
        k += 1
plt.tight_layout(); plt.show()
pd.DataFrame(rows).round(3)

# %% [markdown]
# ---
# ## Part 4: The full pipeline on a real colour photo, step by step
# This is exactly what the TailorCam app does after a scan. Three 3D body models are
# shown side by side: **HMR** (2018), **SHAPY** (CVPR 2022, built for body shape) and
# **CameraHMR** (3DV 2025, trained on the BEDLAM synthetic clothed-people dataset; it
# also estimates the camera's field of view, which removes a big source of shape error).

# %%
from model.hmr.hmr_body import HMRBody
from model.hmr.hmr_worker import run_request, _load_hybrid, _load_shapy, _load_camerahmr, arms_away
from model.hmr.body_points import load_rings
from model.keypoints import mediapipe_to_body25
from model.shapy_measure import SMPLXShape
from output import render_smpl_mesh
hb, rings, hy, sh, chmr = HMRBody(), load_rings(), _load_hybrid(), _load_shapy(), _load_camerahmr()
print('SHAPY loaded:', sh is not None, '| CameraHMR loaded:', chmr is not None)
if not hasattr(sh, 'shape'):
    sh.shape = SMPLXShape()

ex = people.iloc[7]
front = cv2.imread(f'../dataset/real_photos/{ex.person_id}/front.jpg')
side = cv2.imread(f'../dataset/real_photos/{ex.person_id}/side.jpg')
kf, mf = hb.analyze(front); ks, ms = hb.analyze(side)
pf = hb.predict(front, kf)
V_hmr = hb.rest_mesh(hb.fit_weight(pf['betas'], ex.height_cm, ex.weight_kg))
mesh_hmr = cv2.cvtColor(render_smpl_mesh(V_hmr, hb.faces, 300, 450), cv2.COLOR_BGR2RGB)
sp = sh.predict([('front', front, mediapipe_to_body25(kf))])['front']
V_sh = sh.shape.shaped(sh.shape.fit_weight(sp['betas'], ex.height_cm, ex.weight_kg)).cpu().numpy()
mesh_sh = cv2.cvtColor(render_smpl_mesh(V_sh, sh.shape.faces, 300, 450), cv2.COLOR_BGR2RGB)
cp = chmr.predict(front, kf)
V_ch = chmr.rest_mesh(chmr.fit_weight(cp['betas'], ex.height_cm, ex.weight_kg))
mesh_ch = cv2.cvtColor(render_smpl_mesh(V_ch, chmr.faces, 300, 450), cv2.COLOR_BGR2RGB)

vis = cv2.cvtColor(front, cv2.COLOR_BGR2RGB).copy()
for x, y, v in kf:
    if v > .5:
        cv2.circle(vis, (int(x), int(y)), 7, (255, 0, 0), -1)
fig, ax = plt.subplots(1, 8, figsize=(25, 5))
panels = [(cv2.cvtColor(front, cv2.COLOR_BGR2RGB), '1. front photo'), (vis, '2. keypoints (MediaPipe)'),
          (mf, '3. front silhouette'), (ms, '4. side silhouette'), (pf['crop'], '5. HMR input'),
          (mesh_hmr, '6a. HMR 3D body'), (mesh_sh, '6b. SHAPY 3D body'), (mesh_ch, '6c. CameraHMR 3D body')]
for a, (im, t) in zip(ax, panels):
    a.imshow(im, cmap='gray'); a.set_title(t); a.axis('off')
plt.suptitle(f'{ex.person_id}: {ex.height_cm:.0f} cm, {ex.weight_kg:.0f} kg, '
             f'arms-away score {arms_away(kf):.2f}'); plt.show()

# %% [markdown]
# ### Running every model on every real-photo dataset
# `scripts/run_models_on_datasets.py` runs the complete pipeline for every person in
# every dataset and stores **each model's own answer** (`hybrid:`, `prior:`, `hmr:`,
# `shapy:`, `camerahmr:`, `ansur:`) next to the truth. It takes a few hours for all
# datasets, so the saved results are loaded here. Set `RUN_MODELS = True` to recompute them.
#
# **BEDLAM** (CVPR 2023) is a fourth test set: photo-realistic *synthetic* people in
# simulated clothes, whose exact 3D body is known, so the true measurements have no tape
# error (`scripts/build_bedlam_dataset.py`). Caveat: CameraHMR was trained on BEDLAM, so
# **its** BEDLAM score is not a fair test — HMR, SHAPY, the hybrid and the priors never saw it.

# %%
RUN_MODELS = False
if RUN_MODELS:
    import subprocess
    for f in glob.glob('../dataset/predictions/*.csv'):
        os.remove(f)
    subprocess.run([sys.executable, 'scripts/run_models_on_datasets.py'], check=True)
pred = {n: pd.read_csv(f'../dataset/predictions/{n}.csv') for n in ('real24', 'agency', 'hbw_val', 'bedlam')
        if os.path.exists(f'../dataset/predictions/{n}.csv')}
pd.DataFrame([{'dataset': n, 'people': d.person_id.nunique(), 'runs': len(d),
               'modes': ', '.join(sorted(d['mode'].unique())),
               'with SHAPY result': int(d['shapy:chest'].notna().sum()),
               'with CameraHMR result': int(d['camerahmr:chest'].notna().sum())}
              for n, d in pred.items()])

# %% [markdown]
# ### Which single model is best on real photos? (before combining)
# Average error (cm) of each model on its own. **prior** = height (+weight) only, no camera.
# For the agency data only the **test** people are shown.

# %%
def model_table(df, keys=('chest', 'waist', 'hip', 'neck', 'sleeve_length', 'thigh')):
    rows = []
    for mode in sorted(df['mode'].unique()):
        d = df[df['mode'] == mode]
        for src in ('prior', 'hybrid', 'hmr', 'shapy', 'camerahmr', 'ansur'):
            r = {'mode': mode, 'model': src}
            for k in keys:
                if f'true:{k}' not in d or f'{src}:{k}' not in d:
                    continue
                ok = d[f'true:{k}'].notna() & d[f'{src}:{k}'].notna()
                if ok.any():
                    r[k] = round(float((d.loc[ok, f'{src}:{k}'] - d.loc[ok, f'true:{k}']).abs().mean()), 2)
            rows.append(r)
    return pd.DataFrame(rows)

for n, d in pred.items():
    print(f'\n=== {n} ===  (MAE in cm, lower is better)')
    display(model_table(d if n != 'agency' else d[d.split == 'test']))

# %%
fig, ax = plt.subplots(1, len(pred), figsize=(6 * len(pred), 4.2))
for a, (n, d) in zip(ax, pred.items()):
    t = model_table(d if n != 'agency' else d[d.split == 'test'])
    mode = 'no_weight' if n == 'agency' else 'with_weight'
    t[t['mode'] == mode].set_index('model')[['chest', 'waist', 'hip']].T.plot.bar(ax=a, rot=0)
    a.set_ylabel('MAE (cm)'); a.set_title(f'{n} ({mode.replace("_", " ")}): single models')
plt.tight_layout(); plt.show()

# %% [markdown]
# ---
# ## Part 5: Training the final combiner
# The combiner learns, per measurement, which models to trust and how to fix each
# one's bias. It can keep one model, pick another single model (with or without a
# bias fix), or blend several. Six models feed it: **hybrid**, **prior** (BodyM height+weight),
# **HMR**, **SHAPY**, **CameraHMR** and **ANSUR** (6,068 tape-measured people).
#
# | Mode | Trained on | How it's tested | External test |
# |---|---|---|---|
# | **weight known** | 24 real-photo people (tape) | nested 6-fold cross-validation | HBW (10) |
# | **weight unknown** | the same 24 people, weight hidden | nested 6-fold cross-validation | agency test (134), HBW (10) |
#
# ### 5a. Why the weight-unknown combiner is NOT trained on the fashion models
# A first version learned the weight-unknown combiner from the 705 agency models (official
# train/val/test split). It was very good on other fashion models, but **worse than SHAPY
# alone on ordinary people**: models are slimmer than the general population, so the
# combiner learned a "make everyone slimmer" bias. The table proves it:

# %%
from model.calibration import fit_split, fit_all, save, apply_calibration, predict_candidate, TARGETS, CALIB_PATH

def to_rows(df):
    return [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
            for r in df.to_dict('records')]

GIRTH = {k: TARGETS[k] for k in ('chest', 'waist', 'hip')}
ag = pred['agency']
cal_ag, rep_ag = fit_split(*(to_rows(ag[ag.split == s]) for s in ('train', 'val', 'test')), GIRTH)
cmp_rows = []
for name, d in (('agency test', ag[ag.split == 'test']),
                ('real24 (ordinary people)', pred['real24'][pred['real24']['mode'] == 'no_weight']),
                ('HBW (ordinary people)', pred['hbw_val'][pred['hbw_val']['mode'] == 'no_weight'])):
    r = {'test set': name}
    for t in ('chest', 'waist', 'hip'):
        c = cal_ag[t]
        ok = d[f'true:{t}'].notna()
        X = d.loc[ok, c['components']].values.astype(float)
        r[f'{t}: agency-trained'] = round(float(np.abs(predict_candidate(c, X) - d.loc[ok, f'true:{t}']).mean()), 2)
        r[f'{t}: SHAPY alone'] = round(float((d.loc[ok, f'shapy:{t}'] - d.loc[ok, f'true:{t}']).abs().mean()), 2)
    cmp_rows.append(r)
pd.DataFrame(cmp_rows)

# %% [markdown]
# So the weight-unknown combiner is trained on **ordinary tape-measured people** instead,
# starting from **CameraHMR**, the best single photo model when the weight is unknown
# (see the single-model tables above). The agency data is kept as a large **external test**.
#
# ### 5b. Weight known: nested cross-validation on the 24 tape-measured people

# %%
r24 = pred['real24']
calib_w, rep_w = fit_all(to_rows(r24[r24['mode'] == 'with_weight']), folds=6)
rep = rep_w['chest']
fold_mat = np.array([[0 if rep['fold'][i] == f else 1 for i in range(rep['n'])] for f in range(6)])
fig, ax = plt.subplots(figsize=(14, 3))
ax.imshow(fold_mat, cmap='coolwarm_r', aspect='auto'); ax.grid(False)
ax.set_yticks(range(6)); ax.set_yticklabels([f'round {f + 1}' for f in range(6)])
ax.set_xticks(range(rep['n'])); ax.set_xticklabels(rep['people'], rotation=90, fontsize=8)
ax.set_title('Cross-validation rounds: red = TEST people, blue = TRAIN (+ validation inside)'); plt.show()

def combiner_table(rep):
    out = []
    for t, r in rep.items():
        f = r['final']
        model = (f['components'][f['w'].index(1.0) + 1] if f['kind'].startswith('pick')
                 else f['components'][0])
        out.append({'measurement': t, 'people': r['n'], 'adopted': f['kind'], 'main model': model,
                    'choice per round': ', '.join(r['chosen_per_fold']),
                    'TEST MAE before (cm)': f['test_mae_before'], 'TEST MAE after (cm)': f['test_mae_after'],
                    'app uses (cm)': f['test_mae_after'] if f['kind'] != 'identity' else f['test_mae_before']})
    return pd.DataFrame(out)

combiner_table(rep_w)

# %% [markdown]
# ### 5c. Weight unknown: nested cross-validation, starting from CameraHMR

# %%
NW_TARGETS = {k: (['camerahmr:' + k] + [c for c in v if c != 'camerahmr:' + k] if 'camerahmr:' + k in v else v)
              for k, v in TARGETS.items()}
calib_nw, rep_nw = fit_all(to_rows(r24[r24['mode'] == 'no_weight']), folds=6, targets=NW_TARGETS)
combiner_table(rep_nw)

# %%
save(calib_w, 'with_weight')
save(calib_nw, 'no_weight')
print('Saved both combiners for the app ->', CALIB_PATH)

# %% [markdown]
# ---
# ## Part 6: Final results of the complete TailorCam system
# Every number comes from people **not used to train the combiner**:
# - real24 via **cross-validation** (both modes)
# - **HBW** (10 people), the **agency test** set (134 people) and **BEDLAM**: never used to train the combiner
#   (\* BEDLAM was used to train CameraHMR itself, so its BEDLAM rows are optimistic)

# %%
def final_predictions(df, mode):
    out = df[df['mode'] == mode].copy()
    for t in ('chest', 'waist', 'hip'):
        vals = []
        for r in out.to_dict('records'):
            comps = {k: v for k, v in r.items()
                     if ':' in k and not k.startswith('true:') and isinstance(v, (int, float)) and pd.notna(v)}
            vals.append(apply_calibration(comps, weight_known=(mode == 'with_weight')).get(t, r[f'hybrid:{t}']))
        out[f'final:{t}'] = vals
    return out

def test_pred(r):
    """Test-fold predictions of what the app really uses: the combiner if it was
    adopted, otherwise the first model unchanged."""
    return r['cal'] if r['final']['kind'] != 'identity' else r['raw']

def cv_frame(rep, keys=('chest', 'waist', 'hip')):
    people_ = sorted({p for t in keys if t in rep for p in rep[t]['people']})
    f = pd.DataFrame({'person_id': people_})
    for t in keys:
        d_ = dict(zip(rep[t]['people'], zip(rep[t]['y'], test_pred(rep[t])))) if t in rep else {}
        f[f'true:{t}'] = [d_.get(p, (np.nan, np.nan))[0] for p in people_]
        f[f'final:{t}'] = [d_.get(p, (np.nan, np.nan))[1] for p in people_]
    return f

tests = {
    'real24 (with weight, CV)': cv_frame(rep_w),
    'HBW (with weight)': final_predictions(pred['hbw_val'], 'with_weight'),
    'real24 (no weight, CV)': cv_frame(rep_nw),
    'HBW (no weight)': final_predictions(pred['hbw_val'], 'no_weight'),
    'Agency test (no weight)': final_predictions(ag[ag.split == 'test'], 'no_weight'),
}
if 'bedlam' in pred:
    tests['BEDLAM* (with weight)'] = final_predictions(pred['bedlam'], 'with_weight')
    tests['BEDLAM* (no weight)'] = final_predictions(pred['bedlam'], 'no_weight')

rows = []
for name, d in tests.items():
    for t in ('chest', 'waist', 'hip'):
        ok = d[f'true:{t}'].notna() & d[f'final:{t}'].notna()
        if ok.sum() >= 3:
            rows.append({'test': name, 'measurement': t,
                         **regression_metrics(d.loc[ok, f'true:{t}'].values, d.loc[ok, f'final:{t}'].values)})
final = pd.DataFrame(rows).round(2)
final

# %% [markdown]
# ### All tailoring measurements (real24, cross-validated)

# %%
allm = []
for mode, rep_ in (('with weight', rep_w), ('no weight', rep_nw)):
    for t, r in rep_.items():
        e = np.abs(test_pred(r) - r['y'])
        allm.append({'mode': mode, 'measurement': t, 'people': r['n'], 'MAE (cm)': round(float(e.mean()), 2),
                     '<=2cm (%)': round(float((e <= 2).mean() * 100), 1),
                     '<=3cm (%)': round(float((e <= 3).mean() * 100), 1)})
pd.DataFrame(allm).pivot_table(index='measurement', columns='mode', values=['MAE (cm)', '<=3cm (%)'])

# %%
piv = final.pivot_table(index='test', columns='measurement', values='MAE (cm)')[['chest', 'waist', 'hip']]
ax = piv.plot.barh(figsize=(11, 4.5)); ax.set_xlabel('MAE (cm), lower is better')
ax.set_title('Final TailorCam system: average error on real photos (people never trained on)'); plt.show()

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
for a, t in zip(axes, ('chest', 'waist', 'hip')):
    for name, d in tests.items():
        ok = d[f'true:{t}'].notna() & d[f'final:{t}'].notna()
        a.scatter(d.loc[ok, f'true:{t}'], d.loc[ok, f'final:{t}'], s=14, alpha=.6, label=name)
    a.plot([55, 140], [55, 140], 'r--'); a.set_xlim(55, 140); a.set_ylim(55, 140)
    a.set_xlabel('true (cm)'); a.set_ylabel('TailorCam (cm)'); a.set_title(t)
axes[0].legend(fontsize=7); plt.tight_layout(); plt.show()

# %% [markdown]
# ### Size classification: accuracy, precision, recall, F1

# %%
size_rows = []
fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
k = 0
for name in ('real24 (with weight, CV)', 'Agency test (no weight)'):
    d = tests[name]
    for t in ('chest', 'waist'):
        ok = d[f'true:{t}'].notna() & d[f'final:{t}'].notna()
        sm, cm = size_metrics(d.loc[ok, f'true:{t}'].values, d.loc[ok, f'final:{t}'].values, t)
        size_rows.append({'test': name, 'size from': t, 'people': int(ok.sum()), **sm})
        ax[k].imshow(cm, cmap='Greens'); ax[k].grid(False)
        ax[k].set_xticks(range(6)); ax[k].set_xticklabels(SIZES)
        ax[k].set_yticks(range(6)); ax[k].set_yticklabels(SIZES)
        for i in range(6):
            for j in range(6):
                if cm[i, j]:
                    ax[k].text(j, i, cm[i, j], ha='center', va='center', fontsize=8)
        ax[k].set_title(f'{name[:12]} {t}: acc {sm["accuracy (%)"]:.0f}%', fontsize=9)
        ax[k].set_xlabel('predicted'); ax[k].set_ylabel('true'); k += 1
plt.tight_layout(); plt.show()
pd.DataFrame(size_rows).round(3)

# %% [markdown]
# ### Can the camera replace the tape? Per measurement
# A measurement is marked **camera** only if 80 % of the tested (unseen) people were within
# the tailoring tolerance; otherwise the app tags it **TAPE** on the results screen. The
# table updates itself when `scripts/retrain_from_shop.py` adds your own tape-measured customers.

# %%
from model.calibration import tape_check, TAPE_TOLERANCE_CM
tc = []
for wk in (True, False):
    for k, v in tape_check(wk).items():
        tc.append({'weight': 'known' if wk else 'unknown', 'measurement': k,
                   'tolerance (cm)': TAPE_TOLERANCE_CM[k], 'avg error (cm)': v['expected_error'],
                   '80% of people within (cm)': v['p80'], 'tested on': v['tested_on'],
                   'verdict': v['status'].upper()})
pd.DataFrame(tc).set_index(['weight', 'measurement'])

# %% [markdown]
# ### Ideas tested and rejected (kept out of the app because they did not help)
# | Idea | Test | Result |
# |---|---|---|
# | SAM 2 sharper silhouettes | real24 | no gain (waist slightly worse) |
# | Fit one body to all photos' outlines (`model/silhouette_fit.py`) | real24, HBW, BEDLAM | "inside the outline" never active; "fill the outline" fits the clothes: chest (no weight) 3.6 → 6.3 cm |
# | Model disagreement as a per-person confidence | 1,184 measurements, 4 datasets | weak (rank correlation ≤ 0.3; none for waist/hip) |
# | Tape one measurement, correct the others | trained on agency train, tested elsewhere | helps fashion models, hurts ordinary people and every with-weight case |
# | Train the no-weight combiner on agency models | agency → real24/HBW | worse than a single model on ordinary people |
# | NHANES prior (`model/nhanes_prior.py`, up to 27,800 tape-measured US adults) | combiner CV, real24 | waist 3.64 → 4.21 cm with weight; nothing better |
# | One combiner trained on ALL datasets (`model/pooled_combiner.py`, 781 people) | nested CV on real24, 5 repeated splits | not better (with weight: chest 2.57 vs 2.53, hip 3.58 vs 3.15) |
# | CameraHMR: tighter / looser person box | real24, HBW, BEDLAM | no consistent difference |
# | Combiner "scale" candidate (offset + scale factor) | 5 repeated nested-CV splits | chest 2.51 → 2.72 cm, worse |
# | Body density for the weight fit (950–1070 kg/m³) | 5 repeated splits | no difference after the combiner |
# | Learn the tape height on the 3D body (39 levels) | real24 nested CV, then HBW | real24 waist 3.55 → 2.38, but HBW waist 3.90 → 5.31: does not generalise |
#
# These are scored over **5 repeated splits** (`scripts/repeated_cv.py`): with only 16–20 people
# per measurement, one split can move a result by ±0.5 cm by luck alone.
#
# **Adopted:** CameraHMR mirror test-time augmentation (each photo + its mirror image, shapes
# averaged): small but consistent gain on real photos (HBW waist without weight 9.6 → 9.2 cm,
# real24 chest without weight 3.5 → 3.4 cm).
#
# **Conclusion:** with loose clothes, the photos carry no more body information than the
# models already use. The remaining levers are tight-fitting clothes during the scan and
# tape-measured customers from the shop.

# %% [markdown]
# ---
# ## Summary: average error (MAE, cm) on people never used for training

# %%
summ = []
for s in ('testA', 'testB'):
    P = bodym_pred[s]['hybrid ensemble']
    summ.append({'test': f'BodyM {s} (silhouettes, with weight)', 'people': len(set(bodym[s]['subj'])),
                 **{m: round(np.abs(P[:, MEASUREMENTS.index(m)] - bodym[s]['y'][:, MEASUREMENTS.index(m)]).mean(), 2)
                    for m in ('chest', 'waist', 'hip')}})
for name, d in tests.items():
    summ.append({'test': name, 'people': int(d['true:chest'].notna().sum()),
                 **{t: round(float((d[f'final:{t}'] - d[f'true:{t}']).abs().mean()), 2)
                    for t in ('chest', 'waist', 'hip')}})
pd.DataFrame(summ)

# %% [markdown]
# **Limitations**
# - The agency measurements are listed by agencies (rounded, sometimes flattering), not taped.
# - Each dataset defines "hip" slightly differently (agency: fullest hip; HBW: at crotch height;
#   real24: tape at the pelvis), so cross-dataset tests include definition differences.
# - Only 10 HBW and 24 real24 people have full truth, so those numbers carry wide uncertainty.
# - BEDLAM people are synthetic (rendered), and CameraHMR was trained on BEDLAM; the BEDLAM
#   pack used here is a "zoom" scene, so only people seen down to the knees are included.
#
# **How to improve further:** scan 20–30 real customers in TailorCam (arms away from the body,
# fitted clothes), tape-measure them, add them to `../dataset/real_photos/`, and re-run this notebook.
