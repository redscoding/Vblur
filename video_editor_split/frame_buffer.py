"""Small bounded frame cache used by video_processor.py (UTF-8)."""
from collections import OrderedDict
import cv2

class FrameRingBuffer:
    def __init__(self, max_frames=12, max_bytes=96 * 1024 * 1024):
        self.max_frames = max(1, int(max_frames))
        self.max_bytes = max(1, int(max_bytes))
        self._frames = OrderedDict()
        self._bytes = 0

    def put(self, frame_no, frame):
        if frame is None:
            return
        key = int(frame_no)
        copy = frame.copy()
        size = int(copy.nbytes)
        old = self._frames.pop(key, None)
        if old is not None:
            self._bytes -= int(old.nbytes)
        if size > self.max_bytes:
            return
        self._frames[key] = copy
        self._bytes += size
        while self._frames and (len(self._frames) > self.max_frames or self._bytes > self.max_bytes):
            _, removed = self._frames.popitem(last=False)
            self._bytes -= int(removed.nbytes)

    def get_frame(self, frame_no, cap):
        key = int(frame_no)
        frame = self._frames.get(key)
        if frame is not None:
            self._frames.move_to_end(key)
            return frame.copy()
        resume = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cap.set(cv2.CAP_PROP_POS_FRAMES, key)
        ok, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, resume)
        return frame.copy() if ok and frame is not None else None

    def clear(self):
        self._frames.clear()
        self._bytes = 0
