# %% [markdown]
# # TailorCam: Body Measurement Estimation (BodyM)
#
# **Goal:** predict body measurements (chest, waist, hip, …) from a front and a
# side silhouette plus height and weight.
#
# **Dataset:** Amazon BodyM, a public dataset with 2,505 real people. Each person
# has front and side silhouettes, height, weight, gender and 14 measurements
# taken from 3D body scans.
# Reference: Ruiz et al., *Human Body Measurement Estimation with Adversarial
# Augmentation*, 3DV 2022.
#
# **Model: hybrid.** A height+weight regression gives a reliable first estimate.
# A ResNet-18 looks at both silhouettes and corrects it, and also predicts how
# confident it is.
#
# **Evaluation:**
# 1. Regression metrics per measurement: MAE, RMSE, R², MAPE, and % within ±1/2/3 cm.
# 2. Comparison with a height+weight baseline that uses no images.
# 3. Size classification (S…XXL from chest and waist): accuracy, precision,
#    recall, F1 and a confusion matrix.
#
# **On Kaggle:** turn on *Accelerator → GPU* and *Internet → On*, then Run All.
# This takes about 15 minutes on a T4 GPU.

# %%
import os, csv, time, random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

EPOCHS = int(os.environ.get('EPOCHS', 40))      # 40 = full training
SEED = 0
DATA = os.environ.get('BODYM_DIR',
                      '/kaggle/working/bodym' if os.path.exists('/kaggle') else '../dataset/bodym')
OUT = os.environ.get('OUT_DIR', '/kaggle/working' if os.path.exists('/kaggle') else 'notebooks/output')
os.makedirs(OUT, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print('device:', DEV, '| data:', DATA, '| epochs:', EPOCHS)

# %% [markdown]
# ## 1. Download the dataset
# BodyM is in a public S3 bucket, so no AWS account is needed. Size is about 140 MB.

# %%
def download_bodym(dest):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    pages = s3.get_paginator('list_objects_v2').paginate(Bucket='amazon-bodym')
    keys = [o['Key'] for p in pages for o in p.get('Contents', [])]
    todo = [k for k in keys if not os.path.exists(os.path.join(dest, k))]
    print(f'{len(keys)} files in bucket, {len(todo)} to download')
    for i, k in enumerate(todo):
        path = os.path.join(dest, k)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        s3.download_file('amazon-bodym', k, path)
        if i % 2000 == 0:
            print(f'  {i}/{len(todo)}')

if not os.path.exists(os.path.join(DATA, 'testB', 'measurements.csv')):
    download_bodym(DATA)
for split in ('train', 'testA', 'testB'):
    n = len(os.listdir(os.path.join(DATA, split, 'mask')))
    p = sum(1 for _ in open(os.path.join(DATA, split, 'hwg_metadata.csv'))) - 1
    print(f'{split:6s}: {p:5d} people, {n:5d} front+side silhouette pairs')

# %% [markdown]
# ## 2. Pre-processing
# Each silhouette is cropped to the person, scaled so the body fills 92% of the
# height, and centred on a 256×192 image. Absolute size is removed on purpose,
# because the model gets the real height as a number.

# %%
H, W, FILL = 256, 192, 0.92
MEASUREMENTS = ['ankle', 'arm-length', 'bicep', 'calf', 'chest', 'forearm', 'height',
                'hip', 'leg-length', 'shoulder-breadth', 'shoulder-to-crotch',
                'thigh', 'waist', 'wrist']
NM = len(MEASUREMENTS)

def prep_mask(m):
    m = (m > 127).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[big, :4]
    body = (lab[y:y + h, x:x + w] == big).astype(np.uint8) * 255
    s = min(FILL * H / h, 0.98 * W / w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    body = cv2.resize(body, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((H, W), np.uint8)
    out[(H - nh) // 2:(H - nh) // 2 + nh, (W - nw) // 2:(W - nw) // 2 + nw] = body
    return (out > 127).astype(np.uint8)

def load_split(split):
    cache = os.path.join(DATA, f'nb_cache_{split}.npz')
    if os.path.exists(cache):
        return dict(np.load(cache, allow_pickle=True))
    d = os.path.join(DATA, split)
    meas = {r['subject_id']: r for r in csv.DictReader(open(os.path.join(d, 'measurements.csv')))}
    meta = {r['subject_id']: r for r in csv.DictReader(open(os.path.join(d, 'hwg_metadata.csv')))}
    rows = {k: [] for k in ('front', 'side', 'y', 'h', 'w', 'g', 'subj')}
    for r in csv.DictReader(open(os.path.join(d, 'subject_to_photo_map.csv'))):
        sid, pid = r['subject_id'], r['photo_id'].strip()
        a = cv2.imread(os.path.join(d, 'mask', pid + '.png'), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(os.path.join(d, 'mask_left', pid + '.png'), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None or sid not in meas:
            continue
        pa, pb = prep_mask(a), prep_mask(b)
        if pa is None or pb is None:
            continue
        rows['front'].append(pa); rows['side'].append(pb)
        rows['y'].append([float(meas[sid][k]) for k in MEASUREMENTS])
        rows['h'].append(float(meta[sid]['height_cm'])); rows['w'].append(float(meta[sid]['weight_kg']))
        rows['g'].append(meta[sid]['gender']); rows['subj'].append(sid)
    out = {k: np.array(v, np.float32 if k in ('y', 'h', 'w') else None) for k, v in rows.items()}
    np.savez_compressed(cache, **out)
    return out

t0 = time.time()
data = {s: load_split(s) for s in ('train', 'testA', 'testB')}
print(f'loaded in {time.time() - t0:.0f}s')

# %% [markdown]
# ## 3. Explore the data

# %%
tr = data['train']
fig, ax = plt.subplots(1, 8, figsize=(16, 3.5))
for i, j in enumerate(np.random.choice(len(tr['y']), 4, replace=False)):
    ax[2 * i].imshow(tr['front'][j], cmap='gray'); ax[2 * i].set_title(f"front {tr['g'][j]}")
    ax[2 * i + 1].imshow(tr['side'][j], cmap='gray')
    ax[2 * i + 1].set_title(f"side {tr['h'][j]:.0f}cm {tr['w'][j]:.0f}kg")
for a in ax: a.axis('off')
plt.tight_layout(); plt.savefig(f'{OUT}/samples.png'); plt.show()

stats = pd.DataFrame(tr['y'], columns=MEASUREMENTS).describe().T[['mean', 'std', 'min', 'max']].round(1)
print('Training-set measurement statistics (cm):'); print(stats)
print('\nGender counts:', dict(zip(*np.unique(tr['g'], return_counts=True))))

# %% [markdown]
# ## 4. Model
# The prior is a ridge regression on height, weight, BMI and gender, expanded to
# degree-2 polynomial features. The image branch is a ResNet-18 (ImageNet-
# pretrained) shared by both views. It predicts a correction to the prior and a
# confidence (Laplace scale) for each measurement. It is trained with the images
# randomly switched off, so it can fall back to the prior when the photos are bad.

# %%
def scalar_features(h, w, g):
    gg = 1.0 if g == 'male' else (-1.0 if g == 'female' else 0.0)
    if w:
        bmi = w / (h / 100) ** 2
        return np.array([(h - 170) / 10, (w - 75) / 15, (bmi - 25) / 5, 1.0, gg], np.float32)
    return np.array([(h - 170) / 10, 0, 0, 0, gg], np.float32)

NS = 5
PAIRS = list(combinations_with_replacement(range(NS), 2))

def poly(s):
    if torch.is_tensor(s):
        return torch.cat([torch.ones_like(s[:, :1]), s] + [s[:, i:i+1] * s[:, j:j+1] for i, j in PAIRS], 1)
    return np.concatenate([np.ones_like(s[:, :1]), s] + [s[:, i:i+1] * s[:, j:j+1] for i, j in PAIRS], 1)

class HybridNet(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        conv = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        conv.weight.data = r.conv1.weight.data.sum(1, keepdim=True)
        r.conv1, r.fc = conv, nn.Identity()
        self.encoder = r
        self.view_embed = nn.Parameter(torch.zeros(2, 512))
        self.head = nn.Sequential(nn.Linear(1024 + NS + 1, 512), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 2 * NM))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('prior_W', torch.zeros(1 + NS + len(PAIRS), NM))
        self.register_buffer('y_std', torch.ones(NM))
        self.register_buffer('prior_mae', torch.ones(NM))

    def prior(self, sc):
        return poly(sc) @ self.prior_W

    def forward(self, front, side, sc, ok):
        f = (self.encoder(front * 2 - 1) + self.view_embed[0]) * ok[:, None]
        s = (self.encoder(side * 2 - 1) + self.view_embed[1]) * ok[:, None]
        out = self.head(torch.cat([f, s, sc, ok[:, None]], 1))
        mean = self.prior(sc) + out[:, :NM] * self.y_std
        scale = self.prior_mae * torch.exp(out[:, NM:].clamp(-4, 3))
        return mean, scale

# %% [markdown]
# ## 5. Training
# - 10% of training **people** are held out for validation (never by photo, to
#   avoid leakage).
# - Augmentation imitates real camera masks: rotation, scale, shift, mirroring,
#   ragged or fatter/thinner edges, arms merged with the torso, blur, and holes.
# - Weight is hidden for 20% of samples, so the model also works without it.

# %%
subj = np.unique(tr['subj'])
val_subj = set(np.random.default_rng(0).choice(subj, len(subj) // 10, replace=False))
is_val = np.array([s in val_subj for s in tr['subj']])
tr_idx, va_idx = np.where(~is_val)[0], np.where(is_val)[0]
print(f'train: {len(tr_idx)} pairs ({len(subj) - len(val_subj)} people)   '
      f'validation: {len(va_idx)} pairs ({len(val_subj)} people)')

def scalars(D, idx, hide_w=None):
    return torch.from_numpy(np.array([
        scalar_features(float(D['h'][i]), None if (hide_w is not None and hide_w[k]) else float(D['w'][i]),
                        str(D['g'][i])) for k, i in enumerate(idx)])).to(DEV)

def imgs(D, idx):
    t = lambda a: torch.from_numpy(a[idx].astype(np.float32))[:, None].to(DEV)
    return t(D['front']), t(D['side'])

def augment(x):
    n = x.shape[0]
    ang = (torch.rand(n, device=DEV) - .5) * np.radians(6)
    sc = 1 + (torch.rand(n, device=DEV) - .5) * .08
    tx, ty = (torch.rand(n, device=DEV) - .5) * .08, (torch.rand(n, device=DEV) - .5) * .06
    th = torch.stack([torch.stack([torch.cos(ang) / sc, -torch.sin(ang) / sc, tx], 1),
                      torch.stack([torch.sin(ang) / sc, torch.cos(ang) / sc, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), align_corners=False)
    flip = torch.rand(n, device=DEV) < .5
    x[flip] = x[flip].flip(-1)
    r = np.random.randint(0, 3)
    if r:
        x = F.max_pool2d(x, 2 * r + 1, 1, r) if np.random.rand() < .5 else -F.max_pool2d(-x, 2 * r + 1, 1, r)
    x = (x > .5).float()
    sel = torch.rand(n, device=DEV) < .35                       # arms merged with torso
    if sel.any():
        k = int(np.random.choice([9, 15, 21, 27]))
        y = F.max_pool2d(x[sel], (3, k), 1, (1, k // 2)); x[sel] = -F.max_pool2d(-y, (3, k), 1, (1, k // 2))
    sel = torch.rand(n, device=DEV) < .35                       # loose clothes / blur
    if sel.any():
        k = int(np.random.choice([5, 9, 13]))
        x[sel] = (F.avg_pool2d(x[sel], k, 1, k // 2) > np.random.uniform(.25, .5)).float()
    for i in torch.where(torch.rand(n, device=DEV) < .15)[0].tolist():   # holes
        h, w = np.random.randint(10, 40), np.random.randint(10, 40)
        y0, x0 = np.random.randint(0, H - h), np.random.randint(0, W - w)
        x[i, :, y0:y0 + h, x0:x0 + w] = 0
    return x

@torch.no_grad()
def predict(net, D, use_weight=True, use_images=True, bs=128):
    net.eval(); P, B = [], []
    for i in range(0, len(D['y']), bs):
        idx = np.arange(i, min(i + bs, len(D['y'])))
        f, s = imgs(D, idx)
        sc = scalars(D, idx, None if use_weight else np.ones(len(idx), bool))
        ok = torch.full((len(idx),), 1.0 if use_images else 0.0, device=DEV)
        m1, s1 = net(f, s, sc, ok); m2, s2 = net(f, s.flip(-1), sc, ok)
        P.append(((m1 + m2) / 2).cpu().numpy()); B.append(((s1 + s2) / 2).cpu().numpy())
    return np.concatenate(P), np.concatenate(B)

# prior: ridge regression on height/weight features (with and without weight)
from sklearn.linear_model import Ridge
X = np.array([scalar_features(float(tr['h'][i]), w, str(tr['g'][i]))
              for hide in (False, True) for i in tr_idx
              for w in [None if hide else float(tr['w'][i])]])
Y = np.concatenate([tr['y'][tr_idx]] * 2)
ridge = Ridge(alpha=1.0, fit_intercept=False).fit(poly(X), Y)

net = HybridNet().to(DEV)
net.prior_W.copy_(torch.from_numpy(ridge.coef_.T.astype(np.float32)))
net.prior_mae.copy_(torch.from_numpy(np.abs(poly(X) @ ridge.coef_.T - Y).mean(0).astype(np.float32)))
net.y_std.copy_(torch.from_numpy(tr['y'][tr_idx].std(0)))

BS, LR = 64, 3e-4
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * (len(tr_idx) // BS), pct_start=.1)
yt = torch.from_numpy(tr['y']).to(DEV)
va = {k: v[va_idx] for k, v in tr.items()}
KEY3 = [MEASUREMENTS.index(k) for k in ('chest', 'waist', 'hip')]
hist, best, rng = [], 1e9, np.random.default_rng(SEED)
for ep in range(EPOCHS):
    net.train(); losses = []
    perm = rng.permutation(tr_idx)
    for i in range(len(tr_idx) // BS):
        idx = perm[i * BS:(i + 1) * BS]
        f, s = imgs(tr, idx)
        f, s = augment(f), augment(s)
        sc = scalars(tr, idx, rng.random(BS) < .2)
        ok = (torch.rand(BS, device=DEV) > .2).float()
        mean, scale = net(f, s, sc, ok)
        loss = ((yt[idx] - mean).abs() / scale + torch.log(scale)).mean()   # Laplace NLL
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); losses.append(loss.item())
    pv, _ = predict(net, va)
    v = np.abs(pv - va['y']).mean(0)[KEY3].mean()
    hist.append((np.mean(losses), v))
    if v < best:
        best = v; torch.save(net.state_dict(), f'{OUT}/tailorcam_hybrid.pt')
    print(f'epoch {ep + 1:2d}/{EPOCHS}  train loss {np.mean(losses):.3f}  '
          f'val chest/waist/hip MAE {v:.2f} cm  (best {best:.2f})')

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot([h[0] for h in hist]); ax[0].set_title('training loss'); ax[0].set_xlabel('epoch')
ax[1].plot([h[1] for h in hist]); ax[1].set_title('validation MAE chest/waist/hip (cm)'); ax[1].set_xlabel('epoch')
plt.tight_layout(); plt.savefig(f'{OUT}/training_curves.png'); plt.show()

# %% [markdown]
# ## 6. Test-set results: regression metrics
# Both test sets contain **only people never seen in training**.
# - **testA**: 87 people, controlled capture
# - **testB**: 400 people, varied capture

# %%
net.load_state_dict(torch.load(f'{OUT}/tailorcam_hybrid.pt', map_location=DEV))

def regression_table(y, p):
    err = p - y
    rows = []
    for j, m in enumerate(MEASUREMENTS):
        e = err[:, j]; a = np.abs(e)
        rows.append({'measurement': m,
                     'MAE (cm)': a.mean(), 'RMSE (cm)': np.sqrt((e ** 2).mean()),
                     'R²': 1 - (e ** 2).sum() / ((y[:, j] - y[:, j].mean()) ** 2).sum(),
                     'MAPE (%)': (a / y[:, j]).mean() * 100,
                     '≤1cm (%)': (a <= 1).mean() * 100, '≤2cm (%)': (a <= 2).mean() * 100,
                     '≤3cm (%)': (a <= 3).mean() * 100})
    return pd.DataFrame(rows).set_index('measurement').round(2)

results, preds = {}, {}
for split in ('testA', 'testB'):
    D = data[split]
    p, sigma = predict(net, D)
    preds[split] = (p, sigma)
    t = regression_table(D['y'], p)
    results[split] = t
    print(f'\n=== {split}: {len(set(D["subj"]))} people, {len(D["y"])} photo pairs - hybrid model ===')
    print(t.to_string())
    t.to_csv(f'{OUT}/metrics_{split}.csv')

# %% [markdown]
# ### Comparison: how much do the silhouettes help?
# The same model is evaluated with the images switched off (prior = height +
# weight only), and with weight hidden.

# %%
cmp_rows = []
for split in ('testA', 'testB'):
    D = data[split]
    variants = {
        'height+weight only (no images)': predict(net, D, use_images=False)[0],
        'HYBRID (images + height + weight)': preds[split][0],
        'height only (no images, no weight)': predict(net, D, use_weight=False, use_images=False)[0],
        'images + height (no weight)': predict(net, D, use_weight=False)[0],
    }
    for name, p in variants.items():
        mae = np.abs(p - D['y']).mean(0)
        cmp_rows.append({'test set': split, 'method': name,
                         **{m: round(mae[MEASUREMENTS.index(m)], 2)
                            for m in ('chest', 'waist', 'hip', 'thigh', 'shoulder-breadth', 'arm-length')}})
cmp = pd.DataFrame(cmp_rows)
print('MAE (cm), lower is better'); print(cmp.to_string(index=False))
cmp.to_csv(f'{OUT}/method_comparison.csv', index=False)

# %% [markdown]
# ## 7. Size classification: accuracy, precision, recall, F1
# Shops sell sizes, not centimetres. The true and predicted chest and waist are
# converted to standard size classes, then scored as a classification task.
#
# | Size | Chest (cm) | Waist (cm) |
# |---|---|---|
# | XS | < 84 | < 70 |
# | S | 84–92 | 70–78 |
# | M | 92–100 | 78–86 |
# | L | 100–108 | 86–94 |
# | XL | 108–116 | 94–102 |
# | XXL | ≥ 116 | ≥ 102 |

# %%
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
SIZES = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
BINS = {'chest': [84, 92, 100, 108, 116], 'waist': [70, 78, 86, 94, 102]}
to_size = lambda v, b: np.digitize(v, b)

cls_rows = []
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for r, split in enumerate(('testA', 'testB')):
    D, (p, _) = data[split], preds[split]
    for c, m in enumerate(('chest', 'waist')):
        j = MEASUREMENTS.index(m)
        yt_, yp_ = to_size(D['y'][:, j], BINS[m]), to_size(p[:, j], BINS[m])
        acc = accuracy_score(yt_, yp_)
        within1 = (np.abs(yt_ - yp_) <= 1).mean()
        pr, rc, f1, _ = precision_recall_fscore_support(yt_, yp_, average='macro', zero_division=0)
        _, _, f1w, _ = precision_recall_fscore_support(yt_, yp_, average='weighted', zero_division=0)
        cls_rows.append({'test set': split, 'size from': m, 'accuracy (%)': acc * 100,
                         'within ±1 size (%)': within1 * 100, 'precision (macro)': pr,
                         'recall (macro)': rc, 'F1 (macro)': f1, 'F1 (weighted)': f1w})
        cm = confusion_matrix(yt_, yp_, labels=range(6))
        ax = axes[r, c]; ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(6)); ax.set_xticklabels(SIZES); ax.set_yticks(range(6)); ax.set_yticklabels(SIZES)
        for a in range(6):
            for b in range(6):
                if cm[a, b]: ax.text(b, a, cm[a, b], ha='center', va='center', fontsize=8)
        ax.set_xlabel('predicted size'); ax.set_ylabel('true size')
        ax.set_title(f'{split} — {m} size (accuracy {acc * 100:.1f}%)')
plt.tight_layout(); plt.savefig(f'{OUT}/confusion_matrices.png'); plt.show()
cls = pd.DataFrame(cls_rows).round(3)
print(cls.to_string(index=False)); cls.to_csv(f'{OUT}/size_classification.csv', index=False)

D, (p, _) = data['testB'], preds['testB']
j = MEASUREMENTS.index('chest')
print('\nDetailed report, testB chest size:')
print(classification_report(to_size(D['y'][:, j], BINS['chest']), to_size(p[:, j], BINS['chest']),
                            labels=range(6), target_names=SIZES, zero_division=0))

# %% [markdown]
# ## 8. Visual check: predicted vs true, and error distribution

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for c, m in enumerate(('chest', 'waist', 'hip', 'thigh')):
    j = MEASUREMENTS.index(m)
    D, (p, _) = data['testB'], preds['testB']
    ax = axes[0, c]; ax.scatter(D['y'][:, j], p[:, j], s=6, alpha=.5)
    lo, hi = D['y'][:, j].min(), D['y'][:, j].max(); ax.plot([lo, hi], [lo, hi], 'r--')
    ax.set_title(f'testB {m}: predicted vs true'); ax.set_xlabel('true (cm)'); ax.set_ylabel('predicted (cm)')
    ax = axes[1, c]; ax.hist(p[:, j] - D['y'][:, j], bins=40)
    ax.axvline(0, color='r'); ax.set_title(f'{m} error (cm)')
plt.tight_layout(); plt.savefig(f'{OUT}/pred_vs_true.png'); plt.show()

# %% [markdown]
# ### Is the model's confidence honest?
# For a Laplace distribution, about 76% of true values should fall within ±2σ
# of the prediction. A well-calibrated model gets close to that.

# %%
for split in ('testA', 'testB'):
    D, (p, s) = data[split], preds[split]
    inside = (np.abs(p - D['y']) <= 2 * s).mean(0)
    print(split, 'within ±2σ:', {m: f'{inside[MEASUREMENTS.index(m)] * 100:.0f}%'
                                 for m in ('chest', 'waist', 'hip', 'thigh')})

# %% [markdown]
# ### Results by gender

# %%
for split in ('testA', 'testB'):
    D, (p, _) = data[split], preds[split]
    for g in ('male', 'female'):
        sel = D['g'] == g
        mae = np.abs(p[sel] - D['y'][sel]).mean(0)
        print(f'{split} {g:6s} ({sel.sum():4d} pairs): ' + '  '.join(
            f'{m} {mae[MEASUREMENTS.index(m)]:.2f}' for m in ('chest', 'waist', 'hip')))

# %% [markdown]
# ## 9. Summary
# The files saved to the output folder are listed below. Download them for the
# report.

# %%
print('Headline (MAE in cm, people never seen in training):')
for split in ('testA', 'testB'):
    t = results[split]
    print(f"  {split}: chest {t.loc['chest', 'MAE (cm)']:.2f} | waist {t.loc['waist', 'MAE (cm)']:.2f} | "
          f"hip {t.loc['hip', 'MAE (cm)']:.2f} | within ±3 cm (chest) {t.loc['chest', '≤3cm (%)']:.0f}%")
print('\nSize classification:')
print(cls[['test set', 'size from', 'accuracy (%)', 'within ±1 size (%)', 'F1 (macro)']].to_string(index=False))
print('\nSaved:', sorted(os.listdir(OUT)))
