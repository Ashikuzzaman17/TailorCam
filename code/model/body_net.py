"""
BodyMeasureNet — lightweight MLP for body measurement regression.

Architecture
------------
Input  (298)  : 33 landmarks × 3-D × 3 views (normalised to height) + 1 height
Hidden         : 512 → 256 → 128 with BatchNorm, ReLU, Dropout(0.15)
Skip           : residual path from input → penultimate layer
Output  (8)   : measurements as fractions of height

Outputs (fraction × height_cm = cm):
  shoulder_width, chest, waist, hip,
  back_length, shirt_length, sleeve_length, neck

Training: py -3.13 model/train_body_net.py   (~15 min on RTX 4070)
Inference: body_worker.py (Python 3.13 subprocess)
"""

import torch
import torch.nn as nn

N_LM       = 33
N_VIEWS    = 3
N_FEAT     = N_LM * 3 * N_VIEWS + 1   # 298
N_MEASURES = 8

# Measurement names (output order)
MEASURE_KEYS = [
    'shoulder_width', 'chest', 'waist', 'hip',
    'back_length', 'shirt_length', 'sleeve_length', 'neck',
]

# Population-mean fractions of height (ANSUR II) — used to initialise last bias
MEAN_FRACS = torch.tensor([
    0.239,   # shoulder_width
    0.528,   # chest
    0.455,   # waist
    0.540,   # hip
    0.296,   # back_length
    0.330,   # shirt_length
    0.355,   # sleeve_length
    0.198,   # neck
])


class _Block(nn.Module):
    def __init__(self, in_d, out_d, drop=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, out_d),
            nn.BatchNorm1d(out_d),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )
        self.skip = nn.Linear(in_d, out_d, bias=False) if in_d != out_d else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class BodyMeasureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_norm = nn.LayerNorm(N_FEAT)
        self.b1 = _Block(N_FEAT, 512)
        self.b2 = _Block(512,    256)
        self.b3 = _Block(256,    128)
        self.head = nn.Linear(128, N_MEASURES)

        # Initialise head bias so un-trained outputs ≈ population mean fractions
        with torch.no_grad():
            self.head.bias.copy_(MEAN_FRACS)
            self.head.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 298)  →  fractions : (B, 8)"""
        x = self.input_norm(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return self.head(x)    # fractions of height, no final activation


def features_from_views(views_world_lms: dict, height_cm: float) -> torch.Tensor:
    """
    Convert {view_name: world_landmarks} + height_cm → feature tensor (1, 298).

    world_landmarks can be a list of MediaPipe landmark objects or dicts.
    Landmarks are normalised by height so the model is scale-invariant.
    """
    import numpy as np

    VIEW_ORDER = ['FRONT', 'LEFT', 'RIGHT']
    feat = []
    norm = height_cm / 100.0   # convert cm to metres (world unit scale)

    for view in VIEW_ORDER:
        lms = views_world_lms.get(view)
        buf = np.zeros(N_LM * 3, dtype=np.float32)
        if lms is not None:
            for i in range(min(len(lms), N_LM)):
                lm = lms[i]
                if hasattr(lm, 'x'):
                    buf[i*3]   = lm.x / norm
                    buf[i*3+1] = lm.y / norm
                    buf[i*3+2] = lm.z / norm
                elif isinstance(lm, dict):
                    buf[i*3]   = lm.get('x', 0) / norm
                    buf[i*3+1] = lm.get('y', 0) / norm
                    buf[i*3+2] = lm.get('z', 0) / norm
                else:
                    buf[i*3:i*3+3] = np.asarray(lm[:3]) / norm
        feat.append(buf)

    feat.append(np.array([1.0], dtype=np.float32))   # normalised height (always 1)
    x = np.concatenate(feat)
    return torch.from_numpy(x).unsqueeze(0)           # (1, 298)
