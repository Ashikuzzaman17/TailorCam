"""
Train correction models — one per measurement type.

Usage:
  1. Collect scan data via the main app (model/training_data.csv is auto-created).
  2. Open training_data.csv and fill in the real_* columns with actual tape measurements.
  3. Run:  python model/train_model.py

Output: model/correction_model.pkl
  (loaded automatically by measurements.py on next app launch)
"""

import os
import csv
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

KEYS      = ['shoulder_width', 'chest', 'waist', 'back_length', 'shirt_length', 'sleeve_length', 'neck']
DATA_FILE = os.path.join(os.path.dirname(__file__), 'training_data.csv')
MODEL_OUT = os.path.join(os.path.dirname(__file__), 'correction_model.pkl')

MIN_SAMPLES = 30  # Minimum rows needed before training is meaningful


def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"No training data found at {DATA_FILE}")

    cam, real = {k: [] for k in KEYS}, {k: [] for k in KEYS}
    with open(DATA_FILE, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows where real measurements are not filled in yet
            if not all(row.get(f"real_{k}", '').strip() for k in KEYS):
                continue
            for k in KEYS:
                cam[k].append(float(row[f"cam_{k}"]))
                real[k].append(float(row[f"real_{k}"]))

    return cam, real


def train():
    cam, real = load_data()

    n_samples = len(cam['chest'])
    print(f"Found {n_samples} complete training rows.")

    if n_samples < MIN_SAMPLES:
        print(f"Warning: {MIN_SAMPLES} samples recommended for good accuracy. "
              f"You have {n_samples}. Training anyway...")

    bundle = {}

    for key in KEYS:
        X = np.array(cam[key]).reshape(-1, 1)
        y = np.array(real[key])

        # Use Random Forest if enough data, else simple Ridge regression
        if n_samples >= 50:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = Ridge(alpha=1.0)

        scores = cross_val_score(model, X, y, cv=min(5, n_samples),
                                 scoring='neg_mean_absolute_error')
        mae = -scores.mean()

        model.fit(X, y)
        bundle[key] = model

        print(f"  {key:<16}  MAE = {mae:.2f} cm  (cross-val, {n_samples} samples)")

    joblib.dump(bundle, MODEL_OUT)
    print(f"\nCorrection model saved → {MODEL_OUT}")
    print("Restart the app to apply the correction model automatically.")


if __name__ == "__main__":
    train()
