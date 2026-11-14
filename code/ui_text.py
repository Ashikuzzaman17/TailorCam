"""
Sharp, anti-aliased UI text for the OpenCV interface.

OpenCV's built-in Hershey fonts are thin stroke fonts that look blocky and
blurry; every professional app uses a proper typeface. These two functions are
drop-in replacements for cv2.putText / cv2.getTextSize (same arguments, same
return values), drawing Segoe UI (the Windows UI font) through Pillow.

Mapping from the old arguments:
  fontScale 1.0  ≈ 30 px type (Hershey scale 1.0 has ~22 px capitals)
  thickness 1    → Regular, 2 → Semibold, ≥3 or FONT_HERSHEY_DUPLEX → Bold
Rendered words are cached as small alpha masks, so drawing is fast enough for
the 30 fps camera loop. If no TrueType font is found, OpenCV text is used.
"""

import os
from collections import OrderedDict

import cv2
import numpy as np

_FONT_DIRS = [os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts'),
              '/usr/share/fonts/truetype/dejavu', '/Library/Fonts']
_FILES = {'regular': ['segoeui.ttf', 'DejaVuSans.ttf', 'Arial.ttf'],
          'semibold': ['seguisb.ttf', 'segoeui.ttf', 'DejaVuSans.ttf', 'Arial.ttf'],
          'bold': ['segoeuib.ttf', 'DejaVuSans-Bold.ttf', 'Arial Bold.ttf']}
SCALE_TO_PX = 30.0

_paths, _fonts = {}, {}
_cache = OrderedDict()
_CACHE_MAX = 4000

try:
    from PIL import Image, ImageDraw, ImageFont
    for w, names in _FILES.items():
        for d in _FONT_DIRS:
            hit = next((os.path.join(d, n) for n in names if os.path.exists(os.path.join(d, n))), None)
            if hit:
                _paths[w] = hit
                break
    AVAILABLE = 'regular' in _paths
except ImportError:                       # Pillow missing → plain OpenCV text
    AVAILABLE = False


def _weight(font_face, thickness, scale):
    if font_face == cv2.FONT_HERSHEY_DUPLEX or thickness >= 3:
        return 'bold'
    if thickness >= 2 and scale >= 0.42:
        return 'semibold'
    return 'regular'


def _font(weight, px):
    key = (weight, px)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(_paths.get(weight, _paths['regular']), px)
    return _fonts[key]


def _mask(text, weight, px):
    """→ (alpha mask uint8, left, top) relative to the baseline-left point."""
    key = (text, weight, px)
    hit = _cache.get(key)
    if hit is not None:
        _cache.move_to_end(key)
        return hit
    f = _font(weight, px)
    l, t, r, b = f.getbbox(text, anchor='ls')
    w, h = max(1, r - l), max(1, b - t)
    im = Image.new('L', (w, h), 0)
    ImageDraw.Draw(im).text((-l, -t), text, font=f, fill=255, anchor='ls')
    hit = (np.asarray(im, np.float32) / 255.0, l, t)
    _cache[key] = hit
    if len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)
    return hit


def _px(scale):
    return max(11, int(round(SCALE_TO_PX * float(scale))))   # 11 px = smallest readable size


def put_text(img, text, org, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255),
             thickness=1, lineType=None, bottomLeftOrigin=False):
    """Same arguments as cv2.putText; org is the baseline-left point."""
    text = str(text)
    if not AVAILABLE or not text.strip():
        return cv2.putText(img, text, (int(org[0]), int(org[1])), fontFace, fontScale, color,
                           max(1, int(thickness)), cv2.LINE_AA)
    if DEFER:                                 # drawn sharply later by present()
        _record(img, text, org, fontFace, fontScale, color, thickness)
        return img
    a, l, t = _mask(text, _weight(fontFace, thickness, fontScale), _px(fontScale))
    x0, y0 = int(org[0]) + l, int(org[1]) + t
    H, W = img.shape[:2]
    ax0, ay0 = max(0, -x0), max(0, -y0)
    x1, y1 = min(W, x0 + a.shape[1]), min(H, y0 + a.shape[0])
    x0c, y0c = max(0, x0), max(0, y0)
    if x1 <= x0c or y1 <= y0c:
        return img
    m = a[ay0:ay0 + (y1 - y0c), ax0:ax0 + (x1 - x0c)][..., None]
    roi = img[y0c:y1, x0c:x1].astype(np.float32)
    col = np.array(color[:3], np.float32)
    img[y0c:y1, x0c:x1] = (roi * (1 - m) + col * m).astype(np.uint8)
    return img


def text_size(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, thickness=1):
    """Same return value as cv2.getTextSize: ((width, height above baseline), baseline)."""
    text = str(text)
    if not AVAILABLE:
        return cv2.getTextSize(text, fontFace, fontScale, max(1, int(thickness)))
    f = _font(_weight(fontFace, thickness, fontScale), _px(fontScale))
    w = int(round(f.getlength(text))) if text else 0
    asc, desc = f.getmetrics()
    cap = -f.getbbox('H', anchor='ls')[1]
    return (w, cap), desc


# ── Crisp display at the window's real resolution ────────────────────────────
# Screens are laid out at their design size (the camera size, e.g. 1280×720).
# Stretching that picture to a bigger / maximised / fullscreen window blurs the
# text. With DEFER on, put_text only *records* text; present() stretches the
# shapes to the window and then draws every recorded text freshly at the
# window's resolution, so text stays sharp at any window size.
DEFER = False
_pending = {}                                 # id(image) → [(text, weight, px, x, y, colour)]
_view = {'s': 1.0, 'ox': 0, 'oy': 0}


def _record(img, text, org, fontFace, fontScale, color, thickness):
    _pending.setdefault(id(img), []).append(
        (text, _weight(fontFace, thickness, fontScale), _px(fontScale), int(org[0]), int(org[1]), color))


def _blend(img, a, x0, y0, color):
    H, W = img.shape[:2]
    ax0, ay0 = max(0, -x0), max(0, -y0)
    x1, y1 = min(W, x0 + a.shape[1]), min(H, y0 + a.shape[0])
    x0c, y0c = max(0, x0), max(0, y0)
    if x1 <= x0c or y1 <= y0c:
        return
    m = a[ay0:ay0 + (y1 - y0c), ax0:ax0 + (x1 - x0c)][..., None]
    roi = img[y0c:y1, x0c:x1].astype(np.float32)
    img[y0c:y1, x0c:x1] = (roi * (1 - m) + np.array(color[:3], np.float32) * m).astype(np.uint8)


def bake(img):
    """Draw the recorded text of `img` into it now (call before covering it with a modal)."""
    for text, wgt, px, x, y, col in _pending.pop(id(img), []):
        a, l, t = _mask(text, wgt, px)
        _blend(img, a, x + l, y + t, col)


def present(img, win_w, win_h, bg=(23, 17, 13)):
    """Fit `img` into a win_w × win_h picture (letterboxed) and draw its text sharply."""
    fh, fw = img.shape[:2]
    if win_w < 50 or win_h < 50:
        win_w, win_h = fw, fh
    s = min(win_w / fw, win_h / fh)
    nw, nh = max(1, int(round(fw * s))), max(1, int(round(fh * s)))
    ox, oy = (win_w - nw) // 2, (win_h - nh) // 2
    if (nw, nh) == (fw, fh) and (ox, oy) == (0, 0):
        out = img.copy()
    else:
        out = np.full((win_h, win_w, 3), bg, np.uint8)
        out[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    for text, wgt, px, x, y, col in _pending.get(id(img), []):
        a, l, t = _mask(text, wgt, max(11, int(round(px * s))))
        _blend(out, a, int(round(x * s)) + ox + l, int(round(y * s)) + oy + t, col)
    _pending.clear()
    _view.update(s=s, ox=ox, oy=oy)
    return out


def to_design(x, y):
    """Window (mouse) coordinates → screen layout coordinates."""
    return int((x - _view['ox']) / _view['s']), int((y - _view['oy']) / _view['s'])
