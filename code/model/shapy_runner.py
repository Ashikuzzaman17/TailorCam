"""
SHAPY (Choutas et al., CVPR 2022) inside TailorCam.

Loads the trained SHAPY_A regressor once and predicts the SMPL-X body shape
for any list of photos + body keypoints. SHAPY's original code (2021, Linux)
runs unchanged through the compatibility layer in dataset/shapy/shapy_compat.

Requires (from shapy.is.tue.mpg.de, extracted):
  assets/shapy/shapy_code      github.com/muelea/shapy
  assets/shapy/shapy_data      shapy_data.zip  (trained models)
  assets/shapy/smplx/models    SMPL-X v1.1
Licence: SHAPY / SMPL-X are for non-commercial research only.
"""

import os
import sys
import json
import shutil
import tempfile
import contextlib

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
SHAPY_DIR = data_path('shapy')
_REGRESSOR = os.path.join(SHAPY_DIR, 'shapy_code', 'regressor')
_CFG = os.path.join(_REGRESSOR, 'configs', 'b2a_expose_hrnet_demo.yaml')
_CKPT_DIR = os.path.join(SHAPY_DIR, 'shapy_data', 'trained_models', 'shapy', 'SHAPY_A')


def is_available():
    return all(os.path.exists(p) for p in (_CFG, _CKPT_DIR,
               os.path.join(SHAPY_DIR, 'smplx', 'models', 'smplx', 'SMPLX_NEUTRAL.npz')))


@contextlib.contextmanager
def _cwd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class ShapyRunner:
    def __init__(self):
        compat = os.path.join(SHAPY_DIR, 'shapy_compat')
        if compat not in sys.path:
            sys.path.insert(0, compat)
        import run_shapy  # noqa: F401  (applies the compatibility fixes)
        import torch
        from loguru import logger
        from omegaconf import OmegaConf
        from human_shape.config.defaults import conf as default_conf
        from human_shape.models.build import build_model
        from human_shape.utils import Checkpointer
        logger.remove()
        logger.add(sys.stderr, level='WARNING')

        self.torch, self.OmegaConf = torch, OmegaConf
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with _cwd(_REGRESSOR):
            cfg = default_conf.copy()
            cfg.merge_with(OmegaConf.load(_CFG))
            cfg.merge_with(OmegaConf.from_dotlist([
                f'output_folder={_CKPT_DIR}', 'part_key=pose',
                'datasets.batch_size=16', 'datasets.pose_shape_ratio=1.0',
                'datasets.pose.openpose.img_folder=images',
                'datasets.pose.openpose.keyp_folder=openpose',
                'datasets.pose.num_workers.train=0', 'datasets.pose.num_workers.val=0',
                'datasets.pose.num_workers.test=0',
                'network.smplx.compute_measurements=False']))
            cfg.is_training = False
            for part in ('pose', 'shape'):
                splits = cfg.datasets.get(part, {}).get('splits', {})
                if splits:
                    splits['train'], splits['val'], splits['test'] = [], [], []
            cfg.datasets.pose.splits.test = ['openpose']
            self.cfg = cfg
            model = build_model(cfg)['network'].to(self.device)
            ckpt = os.path.join(_CKPT_DIR, cfg.checkpoint_folder)
            Checkpointer(model, save_dir=ckpt, pretrained=cfg.pretrained).load_checkpoint()
            self.model = model.eval()

    def predict_folder(self, folder):
        """
        folder/images/<name>.jpg + folder/openpose/<name>.json →
        {name: {'betas': (10,), 'v_shaped': (10475, 3)}}
        """
        from human_shape.data import build_all_data_loaders
        from human_shape.data.structures.image_list import to_image_list
        torch = self.torch
        self.cfg.datasets.pose.openpose.data_folder = os.path.abspath(folder)
        out = {}
        with _cwd(_REGRESSOR), torch.no_grad():
            loaders = build_all_data_loaders(self.cfg, split='test', shuffle=False,
                                             enable_augment=False, return_full_imgs=True)
            loader = loaders['pose']
            loader = loader[0] if isinstance(loader, list) else loader
            for full_imgs, body_imgs, targets in loader:
                if body_imgs is None:
                    continue
                full = to_image_list(full_imgs)
                full = full.to(device=self.device) if full is not None else None
                targets = [t.to(self.device) for t in targets]
                res = self.model(body_imgs.to(self.device), targets, full_imgs=full,
                                 device=self.device)['stage_02']
                betas = res['betas'].detach().cpu().numpy()
                v_shaped = res['v_shaped'].detach().cpu().numpy()
                for i, t in enumerate(targets):
                    name = os.path.splitext(t.get_field('fname'))[0]
                    out[name] = {'betas': betas[i], 'v_shaped': v_shaped[i]}
        return out

    def measure_person(self, items, height_cm, weight_kg=None):
        """
        items: (name, bgr_image, body25) for one person (several photos).
        Shape = median of the per-photo SHAPY shapes; measured at the true
        height and, when known, adjusted to the true weight.
        Returns (measurements dict, per-photo betas array) or (None, None).
        """
        from model.shapy_measure import SMPLXShape
        if not hasattr(self, 'shape'):
            self.shape = SMPLXShape()
        pred = self.predict(items)
        if not pred:
            return None, None
        betas = np.array([p['betas'] for p in pred.values()])
        return self.shape.measure(np.median(betas, 0), height_cm, weight_kg), betas

    def predict(self, items):
        """items: list of (name, bgr_image, body25 keypoints 25x3) → predict_folder()."""
        from model.keypoints import openpose_json
        tmp = tempfile.mkdtemp(prefix='shapy_')
        try:
            os.makedirs(os.path.join(tmp, 'images'))
            os.makedirs(os.path.join(tmp, 'openpose'))
            for name, img, k25 in items:
                cv2.imwrite(os.path.join(tmp, 'images', f'{name}.jpg'), img)
                with open(os.path.join(tmp, 'openpose', f'{name}.json'), 'w') as f:
                    json.dump(openpose_json(k25), f)
            return self.predict_folder(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
