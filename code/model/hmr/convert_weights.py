"""
One-time conversion of the original HMR TensorFlow checkpoint to PyTorch.

Usage:
  python model/hmr/convert_weights.py <ckpt_prefix> <smpl_faces.npy>

  ckpt_prefix  e.g. .../hmr_models/models/model.ckpt-667589
  smpl_faces   smpl_faces.npy from the HMR repo (src/tf_smpl/)

Writes (next to this file):
  hmr_weights.pt   encoder + regressor + mean theta
  smpl_hmr.npz     SMPL arrays stored inside the same checkpoint
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tf_ckpt_reader import load_checkpoint
from hmr_net import HMR, tf_to_torch_state

_HERE = os.path.dirname(os.path.abspath(__file__))


def main(ckpt_prefix, faces_path):
    tf_vars = load_checkpoint(ckpt_prefix)
    print(f'Read {len(tf_vars)} TF variables')

    model = HMR()
    model.load_state_dict(tf_to_torch_state(tf_vars), strict=True)
    torch.save(model.state_dict(), os.path.join(_HERE, 'hmr_weights.pt'))

    nv = tf_vars['v_template'].shape[0]
    np.savez_compressed(
        os.path.join(_HERE, 'smpl_hmr.npz'),
        v_template=tf_vars['v_template'],
        shapedirs=tf_vars['shapedirs'].reshape(10, nv, 3),
        posedirs=tf_vars['posedirs'].reshape(207, nv, 3),
        J_regressor=tf_vars['J_regressor'],
        weights=tf_vars['lbs_weights'],
        cocoplus_regressor=tf_vars['cocoplus_regressor'],
        faces=np.load(faces_path).astype(np.int32),
    )
    print('Saved hmr_weights.pt and smpl_hmr.npz')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
