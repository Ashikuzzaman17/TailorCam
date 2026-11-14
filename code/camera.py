import cv2
import time


class Camera:
    _MAX_FAILS = 45   # consecutive bad reads before reopen attempt

    def __init__(self, index=0, width=1280, height=720):
        self._preferred = index
        self._w = width
        self._h = height
        self.cap = None
        self._fail_streak = 0
        if not self._open():
            raise RuntimeError(
                'No camera found. Webcam may be unplugged or '
                'still held by a previous process. Wait a moment and retry.')

    def _open(self):
        for idx in list(dict.fromkeys([self._preferred, 0, 1, 2])):
            # Use default backend (CAP_ANY / MSMF on Windows).
            # CAP_DSHOW is broken by-index in OpenCV 4.13.
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                continue

            # Request resolution — camera may deliver a different size, that's fine.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._h)
            cap.set(cv2.CAP_PROP_FPS, 30)

            # MSMF sometimes needs a moment to hand off the first frame.
            # Retry for up to 5 seconds before giving up on this index.
            for attempt in range(50):
                ret, frame = cap.read()
                if ret:
                    self.cap = cap
                    self._preferred = idx
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f'[Camera] index={idx}  {w}x{h}  '
                          f'(ready in {attempt + 1} attempt(s))')
                    return True
                time.sleep(0.1)

            cap.release()
        return False

    def read(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if ret:
            self._fail_streak = 0
            return frame
        self._fail_streak += 1
        return None

    @property
    def needs_reopen(self):
        return self._fail_streak >= self._MAX_FAILS

    def reopen(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        time.sleep(2.0)
        self._fail_streak = 0
        return self._open()

    def release(self):
        if self.cap:
            self.cap.release()

    @property
    def width(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap else self._w

    @property
    def height(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap else self._h
