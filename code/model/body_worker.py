"""
Body measurement inference worker — runs in Python 3.13.

Usage (called automatically by main.py):
  py -3.13 model/body_worker.py <path_to_input.json>

Input JSON (written by main.py to a temp file):
  {
    "height_cm": 175.0,
    "views": {
      "FRONT": [[x,y,z], ...],   // 33 landmarks
      "LEFT":  [[x,y,z], ...],
      "RIGHT": [[x,y,z], ...]
    }
  }

Output JSON (printed to stdout):
  { "shoulder_width": 42.1, "chest": 98.3, ... }

Stdout is machine-parseable JSON only — errors go to stderr.
"""

import sys, os, json, traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

WEIGHT_PATH = os.path.join(_HERE, 'body_weights.pt')


def _load_input():
    """Read input JSON from temp file (arg1) or fallback to stdin."""
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        with open(sys.argv[1], encoding='utf-8') as fh:
            raw = fh.read().strip()
        try:
            os.remove(sys.argv[1])
        except Exception:
            pass
    else:
        raw = sys.stdin.read().strip()
    return json.loads(raw)


def _run_net(height_cm, views_raw):
    """Load model + run inference; returns dict of measurement cm."""
    import torch
    from body_net import BodyMeasureNet, MEASURE_KEYS, features_from_views

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BodyMeasureNet().to(device)

    if os.path.isfile(WEIGHT_PATH):
        ckpt = torch.load(WEIGHT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        print(f'[body_worker] loaded weights (epoch {ckpt.get("epoch","?")})',
              file=sys.stderr)
    else:
        print('[body_worker] WARN: body_weights.pt not found — using untrained model.',
              file=sys.stderr)

    model.eval()

    # Convert raw lists [[x,y,z], ...] to objects with .x / .y / .z
    class _LM:
        __slots__ = ('x', 'y', 'z')
        def __init__(self, xyz): self.x, self.y, self.z = xyz

    views = {k: [_LM(pt) for pt in v] for k, v in views_raw.items()}

    with torch.no_grad():
        feat = features_from_views(views, height_cm).to(device)   # (1, 298)
        fracs = model(feat).cpu().squeeze(0).numpy()               # (8,)

    results = {k: round(float(fracs[i]) * height_cm, 1)
               for i, k in enumerate(MEASURE_KEYS)}
    return results


def main():
    try:
        data       = _load_input()
        height_cm  = float(data['height_cm'])
        views_raw  = data.get('views', {})

        results = _run_net(height_cm, views_raw)
        print(json.dumps(results), flush=True)

    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
