"""
Final combiner ("calibration") learned from REAL colour photos with tape
measurements (dataset/real_photos).

Every measurement has several estimates from the different models, e.g. for
the chest:
    hybrid:chest   BodyM silhouette ensemble (height+weight prior + images)
    prior:chest    height + weight + gender only
    hmr:chest      HMR 3D body adjusted to the real weight
The combiner learns, per measurement, how much to trust each one and how to
remove a systematic bias (e.g. a different tape definition of "sleeve"):

    value = default + bias + Σ w_i · (other_i − default)

With only ~24 people the risk is overfitting, so for each measurement three
candidates are compared and the winner is picked by cross-validation:
    identity   keep the default model's value (nothing learned)
    bias       default + a learned constant offset
    blend      bias + ridge-regularised weights on the other models
    pick(+bias)        use one other model as is (or with an offset)
    scale:j            one model with an offset AND a scale factor (shrunk
                       halfway towards 1), for models that are too big for big
                       people and too small for small ones (ENABLE_SCALE)
A candidate is only adopted if it beats "identity" in cross-validation; when
this rule is evaluated (nested_cv_app) the decision is made on training people only.

Evaluation is nested: the OUTER folds are the test sets (never touched
while fitting); inside each training part, leave-one-out cross-validation
is the validation step that chooses the candidate and its strength.
"""

import os
import json
import numpy as np

CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'calibration', 'real_calibration.json')

# target (TailorCam key) → candidate components, default first
TARGETS = {
    'chest':          ['hybrid:chest', 'prior:chest', 'hmr:chest', 'shapy:chest', 'ansur:chest', 'camerahmr:chest'],
    'waist':          ['hybrid:waist', 'prior:waist', 'hmr:waist', 'shapy:waist', 'ansur:waist', 'camerahmr:waist'],
    'hip':            ['hybrid:hip', 'prior:hip', 'hmr:hip', 'shapy:hip', 'ansur:hip', 'camerahmr:hip'],
    'thigh':          ['hybrid:thigh', 'prior:thigh', 'hmr:thigh', 'ansur:thigh', 'camerahmr:thigh'],
    'neck':           ['hmr:neck', 'ansur:neck', 'camerahmr:neck'],
    'shoulder_width': ['hmr:shoulder_width', 'hybrid:shoulder-breadth', 'ansur:shoulder_width', 'camerahmr:shoulder_width'],
    'sleeve_length':  ['hmr:sleeve_length', 'hybrid:arm-length', 'ansur:sleeve_length', 'camerahmr:sleeve_length',
                       'prior:arm-length'],   # BodyM prior added 2026-11-04: sleeve 2.72 -> 2.61 cm (5 splits), 2.68 -> 2.48 (20 splits)
    'back_length':    ['hmr:back_length', 'ansur:back_length', 'camerahmr:back_length'],
}
ALPHAS = [1.0, 10.0, 100.0, 1000.0]
ENABLE_MEAN = False    # 'mean' (average of all models + offset): tested 2026-10-20, no gain, off
ENABLE_BLEND = True    # ridge blends of the models (more parameters)
ENABLE_SCALE = False   # tested 2026-10-20: no gain (chest 2.51 -> 2.72), off
ENABLE_INVERR = False  # 'inverr' (1/error-weighted models): tested 2026-10-29, sleeve -0.09 cm but waist +0.07, no-weight neck +0.10; off

_cache = {'mtime': None, 'calib': {}}


# ── Model ─────────────────────────────────────────────────────────────────────
def _design(X):
    """X: n x k component values (default first) → n x (k-1) differences."""
    return X[:, 1:] - X[:, :1]


def fit_candidate(kind, X, y, alpha=None):
    """Return {'bias': b, 'w': [...]} for one candidate."""
    x0 = X[:, 0]
    if kind == 'identity':
        return {'bias': 0.0, 'w': [0.0] * (X.shape[1] - 1)}
    if kind == 'bias' or X.shape[1] == 1:
        return {'bias': float(np.mean(y - x0)), 'w': [0.0] * (X.shape[1] - 1)}
    if kind == 'scale':                         # y = bias + s · model[alpha], s shrunk towards 1
        x = X[:, alpha]
        xc, yc = x - x.mean(), y - y.mean()
        sxx = float(xc @ xc)
        s_ = (float(xc @ yc) + sxx) / (2 * sxx) if sxx > 0 else 1.0
        w = [0.0] * (X.shape[1] - 1)
        if alpha > 0:
            w[alpha - 1] = s_
        return {'bias': float(y.mean() - s_ * x.mean()), 'w': w, 'a0': s_}
    if kind == 'inverr':                        # weight each offset-corrected model by 1 / its training error
        off = (y[:, None] - X).mean(0)
        err = np.abs(X + off - y[:, None]).mean(0)
        v = 1 / np.maximum(err, 1e-6); v /= v.sum()
        return {'bias': float(v @ off), 'w': [float(a) for a in v[1:]]}
    if kind == 'mean':                          # equal-weight average of every model + offset
        k = X.shape[1]
        return {'bias': float(np.mean(y - X.mean(1))), 'w': [1.0 / k] * (k - 1)}
    if kind in ('pick', 'pick+bias'):          # use only model number `alpha`
        w = [0.0] * (X.shape[1] - 1)
        w[alpha - 1] = 1.0
        b = float(np.mean(y - X[:, alpha])) if kind == 'pick+bias' else 0.0
        return {'bias': b, 'w': w}
    D = _design(X)
    Dm, rm = D.mean(0), (y - x0).mean()
    Dc, rc = D - Dm, (y - x0) - rm
    w = np.linalg.solve(Dc.T @ Dc + alpha * np.eye(D.shape[1]), Dc.T @ rc)
    return {'bias': float(rm - Dm @ w), 'w': [float(v) for v in w]}


def predict_candidate(p, X):
    out = p.get('a0', 1.0) * X[:, 0] + p['bias']
    if X.shape[1] > 1:
        out = out + _design(X) @ np.array(p['w'])
    return out


def _candidates(k):
    c = [('identity', None), ('bias', None)]
    if k > 1:
        c += [(kind, j) for j in range(1, k) for kind in ('pick', 'pick+bias')]
        if ENABLE_BLEND:
            c += [('blend', a) for a in ALPHAS]
        if ENABLE_MEAN:
            c += [('mean', None)]
        if ENABLE_INVERR:
            c += [('inverr', None)]
    if ENABLE_SCALE:
        c += [('scale', j) for j in range(k)]
    return c


def select(X, y):
    """Leave-one-out validation over candidates → (kind, alpha, loo_mae)."""
    n, best = len(y), None
    for kind, alpha in _candidates(X.shape[1]):
        errs = []
        for i in range(n):
            tr = np.arange(n) != i
            p = fit_candidate(kind, X[tr], y[tr], alpha)
            errs.append(abs(predict_candidate(p, X[i:i + 1])[0] - y[i]))
        mae = float(np.mean(errs))
        if best is None or mae < best[2] - 1e-9:
            best = (kind, alpha, mae)
    return best


def nested_cv(X, y, folds=6, seed=0):
    """Outer k-fold test predictions for (uncalibrated, calibrated)."""
    n = len(y)
    order = np.random.default_rng(seed).permutation(n)
    fold_of = np.empty(n, int)
    fold_of[order] = np.arange(n) % folds
    raw, cal, chosen = X[:, 0].copy(), np.empty(n), []
    for f in range(folds):
        te, tr = fold_of == f, fold_of != f
        kind, alpha, _ = select(X[tr], y[tr])
        p = fit_candidate(kind, X[tr], y[tr], alpha)
        cal[te] = predict_candidate(p, X[te])
        chosen.append(f'blend(a={alpha:g})' if kind == 'blend' else
                      f'{kind}:{alpha}' if kind.startswith('pick') or kind == 'scale' else kind)
    return raw, cal, fold_of, chosen


def nested_cv_app(X, y, folds=6, seed=0, inner=5):
    """
    Outer-fold test predictions of the app's full rule, with no test-fold leak.
    fit_all() keeps a learned candidate only if it beat identity in cross-validation;
    here that keep-or-not decision is made inside each outer training part (an inner
    nested_cv on the training people only), so the outer test people decide nothing.
    """
    n = len(y)
    order = np.random.default_rng(seed).permutation(n)
    fold_of = np.empty(n, int)
    fold_of[order] = np.arange(n) % folds
    out = np.empty(n)
    for f in range(folds):
        te, tr = fold_of == f, fold_of != f
        r_in, c_in, _, _ = nested_cv(X[tr], y[tr], min(inner, int(tr.sum())), seed=seed + 1)
        adopt = np.abs(c_in - y[tr]).mean() < np.abs(r_in - y[tr]).mean()
        kind, alpha, _ = select(X[tr], y[tr]) if adopt else ('identity', None, None)
        out[te] = predict_candidate(fit_candidate(kind, X[tr], y[tr], alpha), X[te])
    return out


# ── Fit on a table of people ──────────────────────────────────────────────────
def fit_all(rows, folds=6, targets=None):
    """
    rows: list of dicts with component values ('hybrid:chest', …) and the tape
    truth under 'true:<target>' (missing / flagged values = None).
    Returns (calibration dict, report dict).
    """
    calib, report = {}, {}
    for target, comps in (targets or TARGETS).items():
        comps = [c for c in comps if any(r.get(c) is not None for r in rows)]
        use = [r for r in rows if r.get(f'true:{target}') is not None
               and all(r.get(c) is not None for c in comps)]
        if len(use) < 8 or not comps:
            continue
        X = np.array([[r[c] for c in comps] for r in use], float)
        y = np.array([r[f'true:{target}'] for r in use], float)
        raw, cal, fold_of, chosen = nested_cv(X, y, folds)
        mae_raw, mae_cal = np.abs(raw - y).mean(), np.abs(cal - y).mean()
        kind, alpha, loo = select(X, y)
        adopt = mae_cal < mae_raw            # only keep it if it helped on test folds
        p = fit_candidate(kind if adopt else 'identity', X, y, alpha)
        used = np.abs(nested_cv_app(X, y, folds) - y)    # leak-free test error of the app's rule
        calib[target] = {'components': comps, 'kind': kind if adopt else 'identity',
                         'alpha': alpha, **p, 'n': len(y),
                         'test_mae_before': round(float(mae_raw), 2),
                         'test_mae_after': round(float(mae_cal), 2),
                         'test_mae': round(float(used.mean()), 2),
                         'test_p80': round(float(np.percentile(used, 80)), 2)}
        report[target] = {'n': len(y), 'y': y, 'raw': raw, 'cal': cal, 'fold': fold_of,
                          'chosen_per_fold': chosen, 'final': calib[target],
                          'people': [r.get('person_id') for r in use]}
    return calib, report


def fit_split(train, val, test, targets=None):
    """
    Large-data version (e.g. SHAPY model-agency data, official split):
    candidates are fitted on TRAIN, the winner is chosen on VALIDATION, then
    refitted on train+val and scored once on TEST.
    Returns (calibration dict, report dict).
    """
    calib, report = {}, {}
    for target, comps in (targets or TARGETS).items():
        comps = [c for c in comps if any(r.get(c) is not None for r in train)]
        sets = []
        for rows in (train, val, test):
            use = [r for r in rows if r.get(f'true:{target}') is not None
                   and all(r.get(c) is not None for c in comps)]
            sets.append((np.array([[r[c] for c in comps] for r in use], float).reshape(-1, len(comps)),
                         np.array([r[f'true:{target}'] for r in use], float), use))
        (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, ute) = sets
        if len(ytr) < 8 or len(yva) < 3 or not comps:
            continue
        scores = []
        for kind, alpha in _candidates(len(comps)):
            p = fit_candidate(kind, Xtr, ytr, alpha)
            scores.append((float(np.abs(predict_candidate(p, Xva) - yva).mean()), kind, alpha))
        best_val, kind, alpha = min(scores, key=lambda t: t[0])
        ident_val = [sc for sc in scores if sc[1] == 'identity'][0][0]
        adopt = best_val < ident_val
        p = fit_candidate(kind if adopt else 'identity',
                          np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]), alpha)
        cal = predict_candidate(p, Xte) if len(yte) else np.array([])
        calib[target] = {'components': comps, 'kind': kind if adopt else 'identity',
                         'alpha': alpha, **p, 'n_train': len(ytr), 'n_val': len(yva),
                         'n_test': len(yte),
                         'test_mae_before': round(float(np.abs(Xte[:, 0] - yte).mean()), 2) if len(yte) else None,
                         'test_mae_after': round(float(np.abs(cal - yte).mean()), 2) if len(yte) else None}
        report[target] = {'n': len(yte), 'y': yte, 'raw': Xte[:, 0] if len(yte) else yte,
                          'cal': cal, 'X': Xte, 'components': comps, 'final': calib[target],
                          'val_scores': scores, 'people': [r.get('person_id') for r in ute]}
    return calib, report


def save(calib, mode='with_weight', path=CALIB_PATH):
    """Store the combiner for one mode ('with_weight' / 'no_weight')."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    allc = _load(path)
    allc[mode] = calib
    with open(path, 'w') as f:
        json.dump(allc, f, indent=2)


def _load(path=CALIB_PATH):
    if not os.path.exists(path):
        return {}
    c = json.load(open(path))
    if c and 'with_weight' not in c and 'no_weight' not in c:   # old single-mode file
        c = {'with_weight': c}
    return c


# ── Which measurements still need the tape? ───────────────────────────────────
# How far off a measurement may be before the garment fits worse (cm). Typical
# tailoring tolerances; change them to your shop's practice.
TAPE_TOLERANCE_CM = {'neck': 1.5, 'shoulder_width': 1.5, 'chest': 2.5, 'waist': 2.5, 'hip': 2.5,
                     'thigh': 2.0, 'sleeve_length': 2.0, 'back_length': 2.0, 'shirt_length': 2.5}


def tape_check(weight_known=True, tolerance=None, path=CALIB_PATH):
    """
    Per measurement: the error measured on people never trained on (cross-
    validation), and whether the camera is good enough ('camera') or the tailor
    should tape it ('tape'). A measurement is 'camera' only if 80 % of tested
    people were within the tolerance. Untested measurements are always 'tape'.
    """
    tol = {**TAPE_TOLERANCE_CM, **(tolerance or {})}
    calib = _load(path).get('with_weight' if weight_known else 'no_weight', {})
    out = {}
    for key, t in tol.items():
        c = calib.get(key)
        if not c or 'test_p80' not in c:
            out[key] = {'status': 'tape', 'expected_error': None, 'p80': None, 'tested_on': 0}
            continue
        out[key] = {'status': 'camera' if c['test_p80'] <= t else 'tape',
                    'expected_error': c['test_mae'], 'p80': c['test_p80'], 'tested_on': c['n']}
    return out


# ── App side ──────────────────────────────────────────────────────────────────
def apply_calibration(components, weight_known=True, path=CALIB_PATH):
    """components: {'hybrid:chest': 98.1, ...} → {target: calibrated cm}."""
    if not os.path.exists(path):
        return {}
    mt = os.path.getmtime(path)
    if _cache['mtime'] != mt:
        _cache['calib'], _cache['mtime'] = _load(path), mt
    out = {}
    for target, c in _cache['calib'].get('with_weight' if weight_known else 'no_weight', {}).items():
        if not all(k in components for k in c['components']):
            continue
        if c['kind'] == 'identity':        # keep the first model as is (may not be the hybrid)
            out[target] = round(float(components[c['components'][0]]), 1)
            continue
        X = np.array([[components[k] for k in c['components']]], float)
        out[target] = round(float(predict_candidate(c, X)[0]), 1)
    return out


# ── Shop offset: adapt to one tailor's way of taping ──────────────────────────
# A tailor may place the tape a little differently from the people the combiner
# learned from (tested: waist, thigh, sleeve and shoulder were off in the same
# direction for nearly every customer). One offset per measurement, learned from
# the shop's own tape-measured customers, is added AFTER the combiner. No file =
# no change. A measurement is only corrected if the offset lowered the error in
# leave-one-customer-out validation (experiments/odjag/shop_fix.py: 92.2 → 96.7 %).
SHOP_PATH = os.path.join(os.path.dirname(CALIB_PATH), 'shop_offset.json')
SHOP_MIN = 5            # customers needed per measurement before an offset is used


def app_value(components, target, weight_known):
    """The value the app shows before any shop offset (combiner, or HMR where there is no rule)."""
    v = apply_calibration(components, weight_known).get(target)
    return v if v is not None else components.get(f'hmr:{target}')


def fit_shop_offsets(rows, min_n=SHOP_MIN):
    """rows: shop rows (components + 'true:<key>' + 'mode') → {mode: {key: {...}}}."""
    keys = sorted({k[5:] for r in rows for k in r if k.startswith('true:')})
    out = {}
    for mode in ('with_weight', 'no_weight'):
        res = {}
        for t in keys:
            pairs = []
            for r in rows:
                y = r.get(f'true:{t}')
                if r.get('mode') != mode or y is None:
                    continue
                comps = {k: v for k, v in r.items() if ':' in k and not k.startswith('true:') and v is not None}
                p = app_value(comps, t, mode == 'with_weight')
                if p is not None:
                    pairs.append((float(p), float(y)))
            if len(pairs) < min_n:
                continue
            p, y = np.array(pairs).T
            d = y - p
            loo = np.abs(p + (d.sum() - d) / (len(d) - 1) - y)     # offset learned without this customer
            before = np.abs(p - y)
            res[t] = {'offset': round(float(d.mean()), 2), 'n': len(d), 'mae_before': round(float(before.mean()), 2),
                      'mae_after_loo': round(float(loo.mean()), 2), 'used': bool(loo.mean() < before.mean())}
        out[mode] = res
    return out


def apply_shop_offset(values, weight_known=True, path=SHOP_PATH):
    """values: {key: cm} shown by the app → {key: cm} for the keys a shop offset applies to."""
    if not path or not os.path.exists(path):
        return {}
    shop = json.load(open(path)).get('with_weight' if weight_known else 'no_weight', {})
    return {k: round(float(values[k]) + s['offset'], 1) for k, s in shop.items()
            if s.get('used') and values.get(k) is not None}
