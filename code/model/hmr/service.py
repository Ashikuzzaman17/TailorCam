"""
TailorCam-side client for the HMR worker.

The worker runs in its own Python process (it needs torch, which the main
app's interpreter may not have). It is started in the background when the
app launches, so the model is already loaded when the first scan finishes.

Interpreter search order:
  1. TAILORCAM_TORCH_PYTHON environment variable (full path to python.exe)
  2. the app's own interpreter
  3. the Windows launcher:  py -3.13, py -3.12, py -3.11
"""

import os
import sys
import json
import queue
import shutil
import tempfile
import threading
import subprocess

import cv2

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CHECK = 'import torch, cv2, scipy, numpy'


def _candidate_pythons():
    env = os.environ.get('TAILORCAM_TORCH_PYTHON')
    if env:
        yield [env]
    yield [sys.executable]
    if shutil.which('py'):
        for v in ('3.13', '3.12', '3.11'):
            yield ['py', f'-{v}']


def _works(cmd):
    try:
        return subprocess.run(cmd + ['-c', _CHECK], capture_output=True,
                              timeout=60).returncode == 0
    except Exception:
        return False


class HMRService:
    """Start with HMRService().start(); check .ready; call .measure(...)."""

    def __init__(self):
        self.ready = False
        self.last_error = None
        self.failed = None          # error text if the worker cannot run
        self._proc = None
        self._replies = queue.Queue()
        self._next_id = 0
        self._lock = threading.Lock()
        self._dir = os.path.join(tempfile.gettempdir(), 'tailorcam_hmr')

    def start(self):
        threading.Thread(target=self._boot, daemon=True).start()
        return self

    def _boot(self):
        from model.hmr.hmr_body import is_available
        if not is_available():
            self.failed = 'HMR weights missing (run model/hmr/convert_weights.py)'
            print(f'[HMR] {self.failed}')
            return
        cmd = next((c for c in _candidate_pythons() if _works(c)), None)
        if cmd is None:
            self.failed = 'no Python with torch found (set TAILORCAM_TORCH_PYTHON)'
            print(f'[HMR] {self.failed}')
            return
        try:
            self._proc = subprocess.Popen(
                cmd + ['-m', 'model.hmr.hmr_worker', '--serve'],
                cwd=_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=None, text=True, bufsize=1)
        except Exception as exc:
            self.failed = f'worker launch failed: {exc}'
            print(f'[HMR] {self.failed}')
            return
        for line in self._proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get('ready'):
                self.ready = True
                print(f"[HMR] worker ready ({' '.join(cmd)}, {msg.get('device')})")
            else:
                self._replies.put(msg)
        self.ready = False
        if self.failed is None:
            self.failed = 'worker exited'
            print(f'[HMR] {self.failed}')

    def measure(self, height_cm, frames, weight_kg=None, gender='unspecified', age=None,
                timeout=20.0, auto_view=False):
        """
        frames: list of (view_name, bgr_image, keypoints[[x,y,vis]...]).
        weight_kg / gender: optional, but weight roughly halves girth errors.
        Returns the worker reply dict (see hmr_worker.py) or None.
        """
        self.last_error = None
        if not self.ready or not frames or not height_cm:
            return None
        os.makedirs(self._dir, exist_ok=True)
        payload = []
        for i, (view, img, kps) in enumerate(frames):
            path = os.path.join(self._dir, f'{i:02d}_{view}.jpg')
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            payload.append({'path': path, 'view': view, 'keypoints': kps})
        with self._lock:
            self._next_id += 1
            rid = self._next_id
            try:
                self._proc.stdin.write(json.dumps(
                    {'id': rid, 'height_cm': float(height_cm),
                     'weight_kg': float(weight_kg) if weight_kg else None,
                     'gender': gender, 'age': age, 'frames': payload,
                     'auto_view': bool(auto_view)}) + '\n')
                self._proc.stdin.flush()
            except Exception as exc:
                print(f'[HMR] send failed: {exc}')
                return None
            while True:
                try:
                    reply = self._replies.get(timeout=timeout)
                except queue.Empty:
                    print('[HMR] timed out waiting for worker')
                    return None
                if reply.get('id') == rid:
                    break
        if not reply.get('ok'):
            print(f"[HMR] failed: {reply.get('error')}")
            self.last_error = reply           # keeps 'photo_checks' so the app can say what was wrong
            return None
        return reply

    def close(self):
        self.failed = 'closed'
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.write(json.dumps({'cmd': 'quit'}) + '\n')
                self._proc.stdin.flush()
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
