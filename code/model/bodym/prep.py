"""
Silhouette pre-processing shared by training (BodyM masks) and the app
(MediaPipe segmentation masks), so both look identical to the network.

A mask is binarised, cropped to the largest person blob, scaled so the body
fills a fixed fraction of the canvas height, and centred. Absolute size is
deliberately thrown away; the network gets the true height as a number.
"""

import numpy as np
import cv2

H, W = 256, 192           # network input (portrait 4:3)
_FILL = 0.92              # body height as a fraction of H


def prep_mask(mask):
    """mask: HxW (uint8 0/255 or float 0..1) → H x W float32 in {0,1}, or None."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    m = (m > (127 if m.dtype == np.uint8 else 0.5)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[big, :4]
    if h < 40:
        return None
    body = (lab[y:y + h, x:x + w] == big).astype(np.uint8) * 255
    s = min(_FILL * H / h, 0.98 * W / w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    body = cv2.resize(body, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((H, W), np.uint8)
    oy, ox = (H - nh) // 2, (W - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = body
    return (out > 127).astype(np.float32)
