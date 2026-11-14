"""
Saves raw camera-estimated measurements to CSV after every scan.
Later, the tailor enters real tape measurements beside each row.
That paired data is used by train_model.py to build the correction model.

CSV columns:
  cam_shoulder, cam_chest, cam_waist, cam_back, cam_shirt, cam_sleeve, cam_neck,
  real_shoulder, real_chest, real_waist, real_back, real_shirt, real_sleeve, real_neck,
  timestamp
"""

import csv
import os
import datetime

KEYS = ['shoulder_width', 'chest', 'waist', 'back_length', 'shirt_length', 'sleeve_length', 'neck']
DATA_FILE = os.path.join(os.path.dirname(__file__), 'training_data.csv')


class DataCollector:
    def __init__(self):
        self._ensure_header()

    def save(self, tracker, measurements):
        """Append one scan row (camera values only; real values left blank for tailor to fill)."""
        row = {f"cam_{k}": measurements.get(k, '') for k in KEYS}
        row.update({f"real_{k}": '' for k in KEYS})
        row['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._write_row(row)

    def save_photo_view(self, _view_idx, _kps):
        """Placeholder — called per-view during photo capture; logging added later."""

    def save_measurements(self, measurements):
        """Save final SMPL-derived measurements (no tracker needed)."""
        row = {f"cam_{k}": measurements.get(k, '') for k in KEYS}
        row.update({f"real_{k}": '' for k in KEYS})
        row['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._write_row(row)

    def _ensure_header(self):
        if not os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames())
                writer.writeheader()

    def _write_row(self, row):
        with open(DATA_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames())
            writer.writerow(row)

    @staticmethod
    def _fieldnames():
        cam_cols  = [f"cam_{k}"  for k in KEYS]
        real_cols = [f"real_{k}" for k in KEYS]
        return cam_cols + real_cols + ['timestamp']
