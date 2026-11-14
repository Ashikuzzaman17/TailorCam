"""
Convert MediaPipe pose landmarks (33 points, what TailorCam detects) to the
OpenPose BODY_25 layout (what SHAPY expects).
"""

import numpy as np

# BODY_25 index → MediaPipe index (or a pair to average)
_MAP = {
    0: 0,            # nose
    1: (11, 12),     # neck = mid-shoulders
    2: 12, 3: 14, 4: 16,          # right shoulder, elbow, wrist
    5: 11, 6: 13, 7: 15,          # left shoulder, elbow, wrist
    8: (23, 24),     # mid-hip
    9: 24, 10: 26, 11: 28,        # right hip, knee, ankle
    12: 23, 13: 25, 14: 27,       # left hip, knee, ankle
    15: 5, 16: 2, 17: 8, 18: 7,   # right eye, left eye, right ear, left ear
    19: 31, 20: 31, 21: 29,       # left big toe, small toe (≈ foot index), heel
    22: 32, 23: 32, 24: 30,       # right big toe, small toe, heel
}


def mediapipe_to_body25(kps33, min_vis=0.3):
    """kps33: list of (x_px, y_px, visibility) → 25 x 3 array (x, y, confidence)."""
    k = np.asarray(kps33, np.float32)
    out = np.zeros((25, 3), np.float32)
    for o, m in _MAP.items():
        if isinstance(m, tuple):
            a, b = k[m[0]], k[m[1]]
            out[o] = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, min(a[2], b[2])]
        else:
            out[o] = k[m]
    out[out[:, 2] < min_vis] = 0           # OpenPose marks missing joints as 0,0,0
    out[20, 2] *= 0.5                       # small toe is only approximated
    out[23, 2] *= 0.5
    return out


def openpose_json(body25):
    """OpenPose JSON dict for one person (the format SHAPY's loader reads)."""
    return {'version': 1.3, 'people': [{
        'person_id': [-1],
        'pose_keypoints_2d': [float(v) for v in np.asarray(body25).reshape(-1)],
        'face_keypoints_2d': [], 'hand_left_keypoints_2d': [], 'hand_right_keypoints_2d': [],
        'pose_keypoints_3d': [], 'face_keypoints_3d': [],
        'hand_left_keypoints_3d': [], 'hand_right_keypoints_3d': []}]}
