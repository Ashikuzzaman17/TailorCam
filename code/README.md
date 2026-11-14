# Live-camera-body-measurement
Live camera body measurement uses computer vision and AI to scan your shape, typically requiring capture, reconstruction, and delivery steps.

## HMR 3D body model (`model/hmr/`)

Each scan snapshots clean frames during the countdown of every view. When the
scan ends, HMR (Kanazawa et al., CVPR 2018, ported to PyTorch) predicts the SMPL
body shape from every frame. The shapes are median-averaged, the body is scaled
to the customer's height, and the measurements are taken tape-style on the 3D
mesh. If HMR is unavailable, the app falls back to the landmark-based pipeline.

- `hmr_weights.pt`, `smpl_hmr.npz`: converted from the original TF checkpoint
  (`model.ckpt-667589`) with `convert_weights.py`. TensorFlow is not needed.
- `customBodyPoints.txt`: SMPL vertex rings that locate each measurement.
- Try it on photos: `python -m model.hmr.hmr_worker --image a.jpg b.jpg --height 175`
- The worker runs in its own process. Set `TAILORCAM_TORCH_PYTHON` to pick the
  Python that has torch installed.

Licence: SMPL and the HMR weights are for non-commercial research use only.
Commercial use needs a licence from Meshcapade / MPI.
