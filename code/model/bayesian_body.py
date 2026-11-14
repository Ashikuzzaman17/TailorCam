"""
BayesianBodyMeasure — scikit-learn BayesianRidge ensemble for body measurements.

Architecture
-----------
Input  (12): geometric intermediate measurements from body_measurer.py
             [shoulder_w, chest_w, waist_w, hip_w, back_len,
              chest_d, waist_d, hip_d, sleeve_raw, height_cm,
              age, has_back]
Output  (8): corrected measurements in cm, each with uncertainty σ

Training
--------
Synthetic data is generated on first run (~4 seconds, 100k samples).
Distribution skewed toward South-Asian (Bangladesh) height and BMI ranges.
Age effect on body composition included per NHANES longitudinal regression.
The trained models are cached to bayesian_body.pkl next to this file.

Usage in main thread (Python 3.14, no torch required):
    bm = BayesianBodyMeasure()
    bm.ensure_trained()           # loads or generates + trains
    pred = bm.predict(geo_feats)  # returns dict[str, dict{'value','sigma'}]
"""

import os
import pickle
import numpy as np

_DIR = os.path.dirname(__file__)
_PKL = os.path.join(_DIR, 'bayesian_body.pkl')

# Output measurement order (must match ANSUR II regressions below)
KEYS = [
    'shoulder_width', 'chest', 'waist', 'hip',
    'back_length',    'shirt_length', 'sleeve_length', 'neck',
]

# Input feature order — 12 features (was 10; added age + has_back)
FEAT_NAMES = [
    'shoulder_w', 'chest_w', 'waist_w', 'hip_w', 'back_len',
    'chest_d', 'waist_d', 'hip_d', 'sleeve_raw', 'height_cm',
    'age', 'has_back',
]

# Model version stamp — increment when FEAT_NAMES changes so stale .pkl is rejected
_MODEL_VERSION = 2   # v1=10 features, v2=12 features with age+has_back

N_SAMPLES_DEFAULT = 100_000


# ── South-Asian anthropometric constants ──────────────────────────────────────
# ANSUR II baselines are US Army (mostly Western proportions).
# Bangladesh / South-Asian research corrections (Deurenberg et al. 2002,
# Ramachandran 2012, NCD-RisC 2016):
#   • Mean male height: 164–167 cm vs 175.6 cm in ANSUR II
#   • Mean female height: 151–154 cm vs 162 cm in ANSUR II
#   • Central obesity: waist circumference ~2.5 cm higher per BMI unit vs Western
#   • Narrower shoulders relative to height (shoulder/height ratio: 0.228 vs 0.231)
#   • Similar hip circumference (pelvis width not significantly different)
#
# We implement these as *additive corrections* applied on top of ANSUR II formulas.

_SA_SH_FACTOR   = 0.986   # shoulder slightly narrower (98.6% of ANSUR II pred)
_SA_WAIST_EXTRA = 0.60    # extra cm of waist per BMI unit above 18.5 (central obesity)
_SA_HIP_FACTOR  = 1.004   # hip slightly wider relative to shoulder


# ── Age effect on body composition (NHANES + SHIP longitudinal data) ─────────
# Regression on adults 18–80 yr from population survey meta-analysis:
#   • Waist: +0.28 cm / year after age 30 (abdominal fat accumulation)
#   • Shoulder: −0.05 cm / year after age 45 (sarcopenia, fat redistribution)
#   • Chest: +0.10 cm / year after age 35 (thoracic cage expansion, slight)
#   • Hip: −0.10 cm / year after age 55 (pelvic muscle atrophy)
#   • Height: −0.04 cm / year after age 40 (spinal compression — negligible here)

def _age_correction(age: float, key: str) -> float:
    """Return correction in cm to add to ANSUR II baseline due to age."""
    a = max(18.0, float(age))
    if key == 'waist':
        return max(0.0, a - 30.0) * 0.28
    if key == 'shoulder_width':
        return max(0.0, a - 45.0) * (-0.05)
    if key == 'chest':
        return max(0.0, a - 35.0) * 0.10
    if key == 'hip':
        return max(0.0, a - 55.0) * (-0.10)
    return 0.0


# ── Synthetic data generation (ANSUR II + South-Asian + age) ─────────────────

def _ansur_measurements(h: float, bmi: float, female: bool,
                        age: float = 30.0,
                        south_asian: bool = True) -> np.ndarray:
    """
    Ground-truth measurements (cm) from anthropometric regression.
    Includes South-Asian correction and age-based body-composition shift.
    """
    if female:
        sh  = 36.8 + (h - 162.0) * 0.214 + (bmi - 26.1) * 0.19
        ch  = 88.2 + (h - 162.0) * 0.350 + (bmi - 26.1) * 1.38
        wa  = 74.6 + (h - 162.0) * 0.160 + (bmi - 26.1) * 2.10
        hip = 95.8 + (h - 162.0) * 0.220 + (bmi - 26.1) * 1.82
        nk  = 32.4 + (h - 162.0) * 0.080 + (bmi - 26.1) * 0.42
    else:
        sh  = 40.6 + (h - 175.6) * 0.218 + (bmi - 25.0) * 0.16
        ch  = 97.0 + (h - 175.6) * 0.390 + (bmi - 25.0) * 1.15
        wa  = 80.4 + (h - 175.6) * 0.175 + (bmi - 25.0) * 2.20
        hip = 95.2 + (h - 175.6) * 0.215 + (bmi - 25.0) * 1.48
        nk  = 37.2 + (h - 175.6) * 0.095 + (bmi - 25.0) * 0.31

    # ── South-Asian corrections ───────────────────────────────────────────────
    if south_asian:
        sh  *= _SA_SH_FACTOR
        wa  += max(0.0, bmi - 18.5) * _SA_WAIST_EXTRA
        hip *= _SA_HIP_FACTOR

    # ── Age corrections ───────────────────────────────────────────────────────
    sh  += _age_correction(age, 'shoulder_width')
    ch  += _age_correction(age, 'chest')
    wa  += _age_correction(age, 'waist')
    hip += _age_correction(age, 'hip')

    bk = h * 0.296
    sr = bk * 1.12
    sl = h * 0.355
    return np.array([sh, ch, wa, hip, bk, sr, sl, nk])


def _simulate_geometric_features(
        ground_truth: np.ndarray, h: float, age: float,
        has_back: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Simulate the noisy geometric intermediates that body_measurer.py produces.

    The geometric method underestimates depths (side-view occlusion) by ~10-15%
    and overestimates shoulder/hip by ~5% due to joint-landmark offset.  We add
    these systematic biases + random per-sample noise so the Bayesian model
    learns to correct them.

    When has_back=1.0, shoulder/hip width error is reduced (BACK view fusion
    gives cleaner widths), so depth correction matters more.
    """
    sh, ch, wa, hip, bk, sr, sl, nk = ground_truth

    # ── Width features ────────────────────────────────────────────────────────
    # When BACK view is available (has_back=1), width noise is ±0.9 cm instead of ±1.5 cm
    width_noise = 0.9 if has_back > 0.5 else 1.5

    shoulder_w = sh + rng.normal(0, width_noise)
    chest_w    = sh * 0.895  + rng.normal(0, width_noise * 1.2)
    waist_w    = hip * 0.755 / 1.435 + rng.normal(0, 2.2)
    hip_w      = hip / 3.14159 + rng.normal(0, width_noise)
    back_len   = bk + rng.normal(0, 0.8)

    # ── Depth features (systematically under-estimated from SIDE view) ────────
    true_cd = ch  / (2 * np.pi) * 2   # approx AP depth from circumference
    true_wd = wa  / (2 * np.pi) * 2
    true_hd = hip / (2 * np.pi) * 2

    chest_d  = true_cd * rng.uniform(0.78, 0.95)
    waist_d  = true_wd * rng.uniform(0.72, 0.93)
    hip_d    = true_hd * rng.uniform(0.80, 0.96)

    # ── Sleeve ────────────────────────────────────────────────────────────────
    sleeve_raw = sl * rng.uniform(0.96, 1.10)

    return np.array([
        shoulder_w, chest_w, waist_w, hip_w, back_len,
        chest_d, waist_d, hip_d, sleeve_raw, h,
        age, has_back,
    ])


def generate_synthetic(n: int = N_SAMPLES_DEFAULT,
                        seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns X (n, 12), Y (n, 8) for training the Bayesian models.

    Height distribution is South-Asian biased:
      • 40% of samples from South-Asian height range (male 150–175, female 140–165)
      • 60% from global range (145–205)  → covers all body types
    Age range: 16–70 years (realistic tailor clientele)
    BMI range: 16–42
    Sex: 50/50
    """
    rng = np.random.default_rng(seed)

    # Build height distribution: bimodal toward South-Asian + global
    n_sa  = int(n * 0.40)
    n_glb = n - n_sa

    female_sa  = rng.integers(0, 2, n_sa).astype(bool)
    female_glb = rng.integers(0, 2, n_glb).astype(bool)

    # South-Asian heights (cm)
    h_sa = np.where(
        female_sa,
        rng.uniform(140, 165, n_sa),   # female SA
        rng.uniform(150, 175, n_sa)    # male SA
    )
    h_glb = np.where(
        female_glb,
        rng.uniform(145, 185, n_glb),
        rng.uniform(155, 200, n_glb)
    )

    heights = np.concatenate([h_sa, h_glb])
    female  = np.concatenate([female_sa, female_glb])
    sa_flag = np.concatenate([np.ones(n_sa, bool), np.zeros(n_glb, bool)])

    bmis    = rng.uniform(16, 42, n)
    ages    = rng.uniform(16, 70, n)
    has_bk  = rng.integers(0, 2, n).astype(float)  # 50% of scans have back view

    # Shuffle so SA and global are intermixed
    perm    = rng.permutation(n)
    heights = heights[perm]; female = female[perm]; sa_flag = sa_flag[perm]
    bmis    = bmis[perm]; ages = ages[perm]; has_bk = has_bk[perm]

    X = np.zeros((n, 12), dtype=np.float32)
    Y = np.zeros((n, 8),  dtype=np.float32)

    for i in range(n):
        gt    = _ansur_measurements(heights[i], bmis[i], female[i],
                                    age=ages[i], south_asian=bool(sa_flag[i]))
        Y[i]  = gt
        X[i]  = _simulate_geometric_features(gt, heights[i], ages[i],
                                             has_bk[i], rng)

    return X, Y


# ── Bayesian Ridge ensemble ───────────────────────────────────────────────────

class BayesianBodyMeasure:
    """
    One BayesianRidge per output measurement, trained on South-Asian-aware
    ANSUR II synthetic data with age and BACK-view features.
    """

    def __init__(self):
        self.models: dict = {}
        self.trained       = False
        self._try_load()

    # ------------------------------------------------------------------
    def _try_load(self):
        if os.path.exists(_PKL):
            try:
                with open(_PKL, 'rb') as f:
                    data = pickle.load(f)
                # Accept both old dict-of-models and new versioned format
                if isinstance(data, dict) and '__version__' in data:
                    if data['__version__'] == _MODEL_VERSION:
                        self.models = data['models']
                        self.trained = len(self.models) == len(KEYS)
                        if self.trained:
                            print(f'[BayesianBody] Loaded v{_MODEL_VERSION} from {_PKL}')
                    else:
                        print(f'[BayesianBody] Stale v{data["__version__"]} → retrain '
                              f'(current v{_MODEL_VERSION})')
                else:
                    # Old 10-feature model — reject silently
                    print(f'[BayesianBody] Old format .pkl → retrain '
                          f'(now v{_MODEL_VERSION} with 12 features)')
            except Exception as e:
                print(f'[BayesianBody] Load failed ({e}) — will retrain')

    # ------------------------------------------------------------------
    def ensure_trained(self, n: int = N_SAMPLES_DEFAULT):
        """Train if not already cached.  Called once at startup."""
        if self.trained:
            return
        print(f'[BayesianBody] Generating {n:,} samples (SA-aware, age, has_back)...')
        X, Y = generate_synthetic(n)
        self._fit(X, Y)

    # ------------------------------------------------------------------
    def _fit(self, X: np.ndarray, Y: np.ndarray):
        from sklearn.linear_model import BayesianRidge
        from sklearn.preprocessing import StandardScaler

        self.models = {}
        print('[BayesianBody] Training 8 BayesianRidge models...')
        for i, key in enumerate(KEYS):
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            m = BayesianRidge(max_iter=300, tol=1e-4, compute_score=False)
            m.fit(Xs, Y[:, i])
            self.models[key] = {'model': m, 'scaler': scaler}

        self.trained = True
        try:
            with open(_PKL, 'wb') as f:
                pickle.dump({'__version__': _MODEL_VERSION, 'models': self.models}, f)
            print(f'[BayesianBody] Saved v{_MODEL_VERSION} → {_PKL}')
        except Exception as e:
            print(f'[BayesianBody] Save failed: {e}')

    # ------------------------------------------------------------------
    def _fit_no_save(self, X: np.ndarray, Y: np.ndarray):
        """Train without saving to disk — used by evaluate.py learning curve."""
        from sklearn.linear_model import BayesianRidge
        from sklearn.preprocessing import StandardScaler
        self.models = {}
        for i, key in enumerate(KEYS):
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            m = BayesianRidge(max_iter=300, tol=1e-4, compute_score=False)
            m.fit(Xs, Y[:, i])
            self.models[key] = {'model': m, 'scaler': scaler}
        self.trained = True

    # ------------------------------------------------------------------
    def predict(self, geo_features: dict) -> dict:
        """
        Parameters
        ----------
        geo_features : dict with keys matching FEAT_NAMES
            (shoulder_w, chest_w, waist_w, hip_w, back_len,
             chest_d, waist_d, hip_d, sleeve_raw, height_cm,
             age, has_back)
            Missing keys default to sensible values:
              age=30, has_back=0

        Returns
        -------
        dict[str → {'value': float, 'sigma': float}]
            Empty dict on failure.
        """
        if not self.trained:
            return {}

        # Defaults for optional features
        feats = dict(geo_features)
        feats.setdefault('age',      30.0)
        feats.setdefault('has_back',  0.0)

        try:
            x = np.array(
                [feats.get(k, 0.0) for k in FEAT_NAMES],
                dtype=np.float32
            ).reshape(1, -1)
        except Exception:
            return {}

        results = {}
        for key in KEYS:
            entry = self.models.get(key)
            if entry is None:
                continue
            Xs = entry['scaler'].transform(x)
            val, std = entry['model'].predict(Xs, return_std=True)
            results[key] = {
                'value': round(float(val[0]), 1),
                'sigma': round(float(std[0]), 2),
            }
        return results


# ── Confidence label helpers ──────────────────────────────────────────────────

def confidence_label(sigma: float) -> tuple[str, str]:
    """Returns (label, level) where level in ('high','medium','low')."""
    if sigma < 1.5:
        return ('High Confidence',  'high')
    if sigma < 3.0:
        return ('Medium Confidence', 'medium')
    return ('Low Confidence — retake side view', 'low')
