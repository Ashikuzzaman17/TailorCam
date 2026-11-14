"""
PyTorch port of HMR (Kanazawa et al., "End-to-end Recovery of Human Shape
and Pose", CVPR 2018).

Encoder : TF-slim ResNet-v2-50 (pre-activation), global-average-pooled → 2048
Regressor: 3 iterative-error-feedback steps of fc1024-fc1024-fc85
Output  : theta = [cam(3: s, tx, ty), pose(72 axis-angle), betas(10)]

Weights are converted from the original TF checkpoint by convert_weights.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_BN_EPS = 1e-5
NUM_CAM, NUM_POSE, NUM_BETAS = 3, 72, 10
NUM_THETA = NUM_CAM + NUM_POSE + NUM_BETAS   # 85


def _pad_same(x, k, stride):
    """TF 'SAME' padding for a stride-1 conv, or slim conv2d_same for stride>1."""
    if stride == 1:
        p = (k - 1) // 2
        return F.pad(x, (p, k - 1 - p, p, k - 1 - p))
    pad_total = k - 1
    beg = pad_total // 2
    return F.pad(x, (beg, pad_total - beg, beg, pad_total - beg))


class Bottleneck(nn.Module):
    def __init__(self, in_ch, depth, base, stride):
        super().__init__()
        self.stride = stride
        self.preact = nn.BatchNorm2d(in_ch, eps=_BN_EPS)
        self.has_proj = in_ch != depth
        if self.has_proj:
            self.shortcut = nn.Conv2d(in_ch, depth, 1, stride=stride, bias=True)
        self.conv1 = nn.Conv2d(in_ch, base, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(base, eps=_BN_EPS)
        self.conv2 = nn.Conv2d(base, base, 3, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(base, eps=_BN_EPS)
        self.conv3 = nn.Conv2d(base, depth, 1, bias=True)

    def forward(self, x):
        pre = F.relu(self.preact(x))
        if self.has_proj:
            sc = self.shortcut(pre)
        else:
            sc = x[:, :, ::self.stride, ::self.stride] if self.stride > 1 else x
        r = F.relu(self.bn1(self.conv1(pre)))
        r = F.relu(self.bn2(self.conv2(_pad_same(r, 3, self.stride))))
        r = self.conv3(r)
        return sc + r


class ResNetV2_50(nn.Module):
    # (base_depth, num_units, stride of the LAST unit) — TF-slim convention
    _BLOCKS = [(64, 3, 2), (128, 4, 2), (256, 6, 2), (512, 3, 1)]

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, bias=True)
        blocks, in_ch = [], 64
        for base, n, stride in self._BLOCKS:
            units = []
            for u in range(n):
                s = stride if u == n - 1 else 1
                units.append(Bottleneck(in_ch, base * 4, base, s))
                in_ch = base * 4
            blocks.append(nn.Sequential(*units))
        self.blocks = nn.ModuleList(blocks)
        self.postnorm = nn.BatchNorm2d(2048, eps=_BN_EPS)

    def forward(self, x):
        x = self.conv1(_pad_same(x, 7, 2))
        # slim max_pool2d 3x3/2 with SAME padding (pad only at the end)
        h, w = x.shape[2:]
        ph = max((-(-h // 2) - 1) * 2 + 3 - h, 0)
        pw = max((-(-w // 2) - 1) * 2 + 3 - w, 0)
        x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2),
                  value=float('-inf'))
        x = F.max_pool2d(x, 3, 2)
        for b in self.blocks:
            x = b(x)
        x = F.relu(self.postnorm(x))
        return x.mean(dim=(2, 3))


class HMR(nn.Module):
    def __init__(self, num_stage=3):
        super().__init__()
        self.num_stage = num_stage
        self.encoder = ResNetV2_50()
        self.fc1 = nn.Linear(2048 + NUM_THETA, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, NUM_THETA)
        self.register_buffer('mean_theta', torch.zeros(1, NUM_THETA))

    def forward(self, img):
        """img: N x 3 x 224 x 224, RGB scaled to [-1, 1]. Returns N x 85 theta."""
        feat = self.encoder(img)
        theta = self.mean_theta.expand(img.shape[0], -1)
        for _ in range(self.num_stage):
            h = F.relu(self.fc1(torch.cat([feat, theta], 1)))
            h = F.relu(self.fc2(h))
            theta = theta + self.fc3(h)
        return theta


def tf_to_torch_state(tf_vars):
    """Map TF checkpoint variable names to the HMR module's state_dict."""
    sd = {}

    def conv(dst, src, bias=True):
        sd[dst + '.weight'] = torch.from_numpy(tf_vars[src + '/weights'].transpose(3, 2, 0, 1).copy())
        if bias:
            sd[dst + '.bias'] = torch.from_numpy(tf_vars[src + '/biases'].copy())

    def bn(dst, src):
        sd[dst + '.weight'] = torch.from_numpy(tf_vars[src + '/gamma'].copy())
        sd[dst + '.bias'] = torch.from_numpy(tf_vars[src + '/beta'].copy())
        sd[dst + '.running_mean'] = torch.from_numpy(tf_vars[src + '/moving_mean'].copy())
        sd[dst + '.running_var'] = torch.from_numpy(tf_vars[src + '/moving_variance'].copy())
        sd[dst + '.num_batches_tracked'] = torch.tensor(0)

    r = 'resnet_v2_50'
    conv('encoder.conv1', f'{r}/conv1')
    for bi, (_, n, _) in enumerate(ResNetV2_50._BLOCKS):
        for u in range(n):
            src = f'{r}/block{bi + 1}/unit_{u + 1}/bottleneck_v2'
            dst = f'encoder.blocks.{bi}.{u}'
            bn(dst + '.preact', src + '/preact')
            if src + '/shortcut/weights' in tf_vars:
                conv(dst + '.shortcut', src + '/shortcut')
            conv(dst + '.conv1', src + '/conv1', bias=False)
            bn(dst + '.bn1', src + '/conv1/BatchNorm')
            conv(dst + '.conv2', src + '/conv2', bias=False)
            bn(dst + '.bn2', src + '/conv2/BatchNorm')
            conv(dst + '.conv3', src + '/conv3')
    bn('encoder.postnorm', f'{r}/postnorm')

    for i in (1, 2, 3):
        sd[f'fc{i}.weight'] = torch.from_numpy(tf_vars[f'3D_module/fc{i}/weights'].T.copy())
        sd[f'fc{i}.bias'] = torch.from_numpy(tf_vars[f'3D_module/fc{i}/biases'].copy())
    sd['mean_theta'] = torch.from_numpy(tf_vars['mean_param'].copy())
    return sd
