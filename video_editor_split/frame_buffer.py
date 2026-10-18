"""Bounded in-memory ring buffer for recent OpenCV video frames.

The cache is deliberately bounded by both frame count and bytes. Cache misses seek
the source video and restore its decoder position; misses are not inserted, so a
backward scan cannot evict the most recent forward frames.
"""
from collections import OrderedDict
import threading
import cv2


class FrameRingBuffer:
    def __init__(self, max_frames=12, max_bytes=96 * 1024 * 1024):
        self.max_frames = max(1, int(max_frames))
        self.max_bytes = max(1, int(max_bytes))
        self._frames = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()
        self.cache_hits = 0
        self.cache_misses = 0
        self.disk_reads = 0

    @property
    def bytes_used(self):
        with self._lock:
            return self._bytes

    @property
    def frame_count(self):
        with self._lock:
            return len(self._frames)

    def put(self, frame_no, frame):
        if frame is None:
            return
        idx = int(frame_no)
        stored = frame.copy()
        size = int(stored.nbytes)
        if size > self.max_bytes:
            return
        with self._lock:
            old = self._frames.pop(idx, None)
            if old is not None:
                self._bytes -= int(old.nbytes)
            self._frames[idx] = stored
            self._bytes += size
            # FIFO by insertion/forward decode order; enforce both caps.
            while self._frames and (len(self._frames) > self.max_frames or self._bytes > self.max_bytes):
                _, evicted = self._frames.popitem(last=False)
                self._bytes -= int(evicted.nbytes)

    def get_cached(self, frame_no):
        with self._lock:
            frame = self._frames.get(int(frame_no))
            if frame is None:
                return None
            self.cache_hits += 1
            return frame.copy()

    def get_frame(self, frame_no, cap):
        """Return cached frame or seek/read it, restoring cap's previous position."""
        cached = self.get_cached(frame_no)
        if cached is not None:
            return cached
        with self._lock:
            self.cache_misses += 1
        resume_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
            ok, frame = cap.read()
            with self._lock:
                self.disk_reads += 1
            return frame.copy() if ok and frame is not None else None
        finally:
            cap.set(cv2.CAP_PROP_POS_FRAMES, resume_pos)

    def stats(self):
        with self._lock:
            return {
                "frames": len(self._frames),
                "bytes": self._bytes,
                "max_frames": self.max_frames,
                "max_bytes": self.max_bytes,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "disk_reads": self.disk_reads,
            }

    def clear(self):
        with self._lock:
            self._frames.clear()
            self._bytes = 0
