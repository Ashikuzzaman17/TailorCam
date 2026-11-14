"""
Where TailorCam keeps its files. The project has two folders side by side:

  code/        everything the program needs (this folder)
    assets/      model files the app needs to run: CameraHMR, SHAPY, SMPL-X
    app_data/    the shop's own data: saved scans and extra tape-measured customers
  dataset/     every dataset, used only for training and testing (the app never reads it)

Any of them can live somewhere else, e.g. datasets on another drive:
  set TAILORCAM_DATASETS=D:\\TailorCam_datasets      (also TAILORCAM_ASSETS, TAILORCAM_APP_DATA)
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.environ.get('TAILORCAM_DATASETS', os.path.join(os.path.dirname(ROOT), 'dataset'))
ASSETS = os.environ.get('TAILORCAM_ASSETS', os.path.join(ROOT, 'assets'))
APP_DATA = os.environ.get('TAILORCAM_APP_DATA', os.path.join(ROOT, 'app_data'))

_APP = {'shop_scans', 'shop_extra'}
_SHAPY_MODEL = {'shapy_code', 'shapy_compat', 'shapy_data', 'smplx'}


def dataset(*parts):
    return os.path.join(DATASETS, *parts)


def asset(*parts):
    return os.path.join(ASSETS, *parts)


def app_data(*parts):
    return os.path.join(APP_DATA, *parts)


def data_path(first, *rest):
    """Where an old 'data/<first>/...' path lives now."""
    if first in _APP:
        return app_data(first, *rest)
    if first == 'camerahmr':
        return asset(first, *rest)
    if first == 'shapy' and (not rest or rest[0] in _SHAPY_MODEL):
        return asset(first, *rest)          # SHAPY's code, weights and SMPL-X; its test sets stay in datasets/
    return dataset(first, *rest)
