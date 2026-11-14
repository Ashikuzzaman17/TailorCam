"""
Silhouette → body measurement network, trained on Amazon BodyM.

Inputs : front + side silhouettes (prep.prep_mask), height (cm), weight (kg,
         optional), gender (male / female / unknown)
Outputs: the 14 BodyM measurements in cm.

A shared ResNet-18 encodes each view; the two feature vectors are joined with
the height/weight/gender numbers and an MLP predicts all measurements.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision

from .prep import prep_mask

MEASUREMENTS = ['ankle', 'arm-length', 'bicep', 'calf', 'chest', 'forearm',
                'height', 'hip', 'leg-length', 'shoulder-breadth',
                'shoulder-to-crotch', 'thigh', 'waist', 'wrist']

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bodym_net.pt')


def scalar_features(height_cm, weight_kg, gender):
    """
    Normalised numeric inputs. weight_kg may be None (unknown) — the model is
    trained to work without it, just less accurately.
    gender: 'male' | 'female' | anything else = unknown.
    """
    g = 1.0 if gender == 'male' else (-1.0 if gender == 'female' else 0.0)
    if weight_kg:
        bmi = weight_kg / (height_cm / 100.0) ** 2
        w, b, has_w = (weight_kg - 75) / 15, (bmi - 25) / 5, 1.0
    else:
        w = b = has_w = 0.0
    return np.array([(height_cm - 170) / 10, w, b, has_w, g], np.float32)


N_SCALARS = 5


class BodyMNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        # 1-channel silhouettes: fold the RGB filters into one
        conv = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        conv.weight.data = r.conv1.weight.data.sum(1, keepdim=True)
        r.conv1 = conv
        r.fc = nn.Identity()
        self.encoder = r
        self.view_embed = nn.Parameter(torch.zeros(2, 512))   # front / side
        self.head = nn.Sequential(
            nn.Linear(512 * 2 + N_SCALARS, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, len(MEASUREMENTS)))
        # target normalisation, filled in by training
        self.register_buffer('y_mean', torch.zeros(len(MEASUREMENTS)))
        self.register_buffer('y_std', torch.ones(len(MEASUREMENTS)))

    def forward(self, front, side, scalars):
        """front/side: N x 1 x H x W in {0,1}; scalars: N x 5. Returns standardised outputs."""
        f = self.encoder(front * 2 - 1) + self.view_embed[0]
        s = self.encoder(side * 2 - 1) + self.view_embed[1]
        return self.head(torch.cat([f, s, scalars], 1))

    def to_cm(self, out):
        return out * self.y_std + self.y_mean


class BodyMPredictor:
    """Load the trained network and predict measurements for one person."""

    def __init__(self, path=WEIGHTS_PATH, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = BodyMNet()
        self.net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        self.net.eval().to(self.device)

    @torch.no_grad()
    def predict(self, front_mask, side_mask, height_cm, weight_kg=None, gender='unknown'):
        f, s = prep_mask(front_mask), prep_mask(side_mask)
        if f is None or s is None:
            return None
        t = lambda m: torch.from_numpy(m)[None, None].to(self.device)
        sc = torch.from_numpy(scalar_features(height_cm, weight_kg, gender))[None].to(self.device)
        # average with the mirrored side view (BodyM side views face one way)
        out = (self.net(t(f), t(s), sc) + self.net(t(f), t(s[:, ::-1].copy()), sc)) / 2
        cm = self.net.to_cm(out)[0].cpu().numpy()
        return {k: round(float(v), 1) for k, v in zip(MEASUREMENTS, cm)}
