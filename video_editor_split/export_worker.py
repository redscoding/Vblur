"""Video export worker. Kept separate from OCR/tracking logic."""
import cv2
from PyQt5.QtCore import QThread, pyqtSignal


class VideoProcessorThread(QThread):
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(str)

    def __init__(self, video_path, output_path, mosaic_tasks, end_frame=None):
        super().__init__()
        self.video_path, self.output_path, self.mosaic_tasks = video_path, output_path, mosaic_tasks
        self.end_frame = None if end_frame is None else max(0, int(end_frame))

    @staticmethod
    def _apply_mosaic(frame, roi, strength):
        height, width = frame.shape[:2]
        x, y, w, h = [int(round(float(v))) for v in roi]
        x = max(0, min(x, width - 1)); y = max(0, min(y, height - 1))
        w = max(0, min(w, width - x)); h = max(0, min(h, height - y))
        if w <= 0 or h <= 0:
            return
        area = frame[y:y+h, x:x+w]
        strength = max(1, int(strength))
        sw, sh = max(1, w // strength), max(1, h // strength)
        frame[y:y+h, x:x+w] = cv2.resize(cv2.resize(area, (sw, sh), interpolation=cv2.INTER_LINEAR),
                                           (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _rois_for_frame(task, frame_no):
        if task.get("type") != "tracking":
            roi = task.get("roi")
            return [roi] if roi else []
        value = task.get("tracking_data", {}).get(frame_no)
        if value is None:
            return []
        if isinstance(value, (tuple, list)) and len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
            return [value]
        return list(value) if isinstance(value, (list, tuple)) else []

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished_signal.emit(""); return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        source_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        limit_frame = source_total - 1 if self.end_frame is None else min(source_total - 1, self.end_frame)
        total = max(0, limit_frame + 1)
        out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not out.isOpened():
            cap.release(); self.finished_signal.emit(""); return
        current = 0
        try:
            while current <= limit_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                for task in self.mosaic_tasks:
                    if int(task.get("start", 0)) <= current <= int(task.get("end", total - 1)):
                        for roi in self._rois_for_frame(task, current):
                            self._apply_mosaic(frame, roi, task.get("strength", 15))
                out.write(frame)
                current += 1
                if current % 15 == 0:
                    self.progress_signal.emit(current, total)
            self.progress_signal.emit(current, total)
        except Exception:
            self.finished_signal.emit(""); return
        finally:
            cap.release(); out.release()
        self.finished_signal.emit(self.output_path)
