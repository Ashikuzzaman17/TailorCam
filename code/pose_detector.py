import cv2
import os
import time
import urllib.request
import mediapipe as mp


MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

# Skeleton connections for manual drawing (new API has no built-in draw util)
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(17,19),
    (12,14),(14,16),(16,18),(16,20),(18,20),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
]


class PoseLandmark:
    NOSE=0; LEFT_EAR=7; RIGHT_EAR=8
    LEFT_SHOULDER=11; RIGHT_SHOULDER=12
    LEFT_ELBOW=13;    RIGHT_ELBOW=14
    LEFT_WRIST=15;    RIGHT_WRIST=16
    LEFT_HIP=23;      RIGHT_HIP=24
    LEFT_KNEE=25;     RIGHT_KNEE=26
    LEFT_ANKLE=27;    RIGHT_ANKLE=28


class PoseDetector:
    def __init__(self, min_detection=0.7, min_tracking=0.7):
        self.LM = PoseLandmark
        self._ensure_model()

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection,
            min_pose_presence_confidence=min_detection,
            min_tracking_confidence=min_tracking,
            output_segmentation_masks=True,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        # VIDEO mode requires strictly increasing timestamps
        self._ts_ms = 0

    def _ensure_model(self):
        if not os.path.exists(MODEL_PATH):
            print("Downloading pose model (first launch only — ~25 MB)...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Model ready.")

    def process(self, frame):
        self._ts_ms += 33  # ~30 fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect_for_video(mp_img, self._ts_ms)

    def draw_landmarks(self, frame, results):
        if not results.pose_landmarks:
            return frame
        h, w = frame.shape[:2]
        lms = results.pose_landmarks[0]
        for a, b in POSE_CONNECTIONS:
            if a < len(lms) and b < len(lms):
                p1 = (int(lms[a].x * w), int(lms[a].y * h))
                p2 = (int(lms[b].x * w), int(lms[b].y * h))
                cv2.line(frame, p1, p2, (0, 200, 255), 2)
        for lm in lms:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
        return frame

    def get_screen_landmarks(self, results, frame_w, frame_h):
        if not results.pose_landmarks:
            return None
        return [
            {'x': lm.x * frame_w, 'y': lm.y * frame_h,
             'visibility': getattr(lm, 'visibility', 1.0)}
            for lm in results.pose_landmarks[0]
        ]

    def get_world_landmarks(self, results):
        if not results.pose_world_landmarks:
            return None
        return results.pose_world_landmarks[0]

    def get_segmentation_mask(self, results):
        """Float32 H×W array, 0=background 1=person. None if unavailable."""
        if not results.segmentation_masks:
            return None
        return results.segmentation_masks[0].numpy_view()

    def close(self):
        self._landmarker.close()
