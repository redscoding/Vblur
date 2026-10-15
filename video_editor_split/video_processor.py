"""Fast manual-initialization tracking for video mosaic.

Option B v5:
- The user manually selects one target in the current frame.
- Every subsequent frame uses full-resolution, full-frame 2D template matching as the primary method.
- KCF/MOSSE is deliberately not used to decide the output box, avoiding local-tracker drift.
- OpenCV matchTemplate evaluates every legal x/y placement for each configured scale.
- EasyOCR is optional and only used as a last-resort local recovery when target_text is supplied.

tracking_data format:
    {absolute_frame_index: [(x, y, width, height), ...]}

Requires: opencv-contrib-python, numpy, PyQt5.
EasyOCR is optional unless OCR recovery is used.
"""
from __future__ import annotations

import time
import math
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class OCRTrackingThread(QThread):
    """Compatibility name retained so the existing GUI can import this class."""

    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict, int, bool)
    monitor_frame_signal = pyqtSignal(object, int, str)
    monitor_data_signal = pyqtSignal(dict)

    # Tune these after measuring your own videos.
    MONITOR_INTERVAL = 1
    PROGRESS_INTERVAL = 10
    MIN_TEMPLATE_SCORE = 0.48
    STALE_GRACE_FRAMES = 1
    OCR_RETRY_COOLDOWN = 12
    # Every scale scans all legal (x, y) positions in the full-resolution frame.
    TEMPLATE_SCALES = (0.85, 1.0, 1.15)

    def __init__(self, video_path, start_frame, total_frames, target_text="",
                 initial_roi=None, tracker_name="KCF"):
        super().__init__()
        self.video_path = str(video_path)
        self.start_frame = max(0, int(start_frame))
        self.total_frames = max(0, int(total_frames))
        self.target_text = str(target_text or "").strip()
        self.initial_roi = tuple(map(int, initial_roi)) if initial_roi else None
        self.tracker_name = str(tracker_name or "KCF").upper()
        self._cancelled = False

    def requestInterruption(self):
        self._cancelled = True
        super().requestInterruption()

    def _report(self, stage, message, frame_no=None, **extra):
        self.monitor_data_signal.emit({
            "stage": stage, "message": message, "frame": frame_no,
            "clock": time.monotonic(), **extra
        })

    @staticmethod
    def _clip(frame, roi):
        h, w = frame.shape[:2]
        x, y, rw, rh = [int(round(float(v))) for v in roi]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        rw = max(1, min(rw, w - x))
        rh = max(1, min(rh, h - y))
        return x, y, rw, rh

    @staticmethod
    def _center(roi):
        x, y, w, h = roi
        return x + w / 2.0, y + h / 2.0

    @staticmethod
    def _create_tracker(name="KCF"):
        """Prefer fast KCF, then MOSSE, then CSRT as a compatibility fallback."""
        candidates = []
        if name == "MOSSE":
            candidates += [
                ("legacy", "TrackerMOSSE_create"),
                ("root", "TrackerMOSSE_create"),
                ("legacy", "TrackerKCF_create"),
                ("root", "TrackerKCF_create"),
            ]
        else:
            candidates += [
                ("legacy", "TrackerKCF_create"),
                ("root", "TrackerKCF_create"),
                ("legacy", "TrackerMOSSE_create"),
                ("root", "TrackerMOSSE_create"),
            ]
        candidates += [
            ("legacy", "TrackerCSRT_create"),
            ("root", "TrackerCSRT_create"),
        ]
        for scope, attr in candidates:
            owner = getattr(cv2, "legacy", None) if scope == "legacy" else cv2
            factory = getattr(owner, attr, None) if owner is not None else None
            if callable(factory):
                try:
                    return factory(), attr.replace("Tracker", "").replace("_create", "")
                except Exception:
                    pass
        raise RuntimeError(
            "找不到 OpenCV 追蹤器。請安裝 opencv-contrib-python："
            "python -m pip install -U opencv-contrib-python"
        )

    @staticmethod
    def _gray_template(frame, roi):
        x, y, w, h = roi
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Equalization makes the reference somewhat less sensitive to mild lighting shifts.
        gray = cv2.equalizeHist(gray)
        return gray

    @staticmethod
    def _patch_score(template, frame, roi):
        """Compare the current ROI to the initial template at a fixed template size."""
        if template is None:
            return 0.0
        x, y, w, h = roi
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0 or w < 3 or h < 3:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (template.shape[1], template.shape[0]),
                          interpolation=cv2.INTER_AREA if w >= template.shape[1] else cv2.INTER_LINEAR)
        gray = cv2.equalizeHist(gray)
        try:
            score = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)[0, 0]
            if not np.isfinite(score):
                return 0.0
            return float(max(-1.0, min(1.0, score)))
        except cv2.error:
            return 0.0

    def _template_search(self, frame, last_roi, template):
        """Exhaustive full-resolution template search over every valid location.

        No frame downscaling and no local-only refinement: for each configured
        template scale, cv2.matchTemplate evaluates all legal placements across
        the entire original-resolution frame. OpenCV performs the sliding
        comparison in optimized native code (rather than a Python pixel loop).
        """
        if template is None:
            return None, 0.0
        fh, fw = frame.shape[:2]
        th, tw = template.shape[:2]
        if th < 3 or tw < 3:
            return None, 0.0

        # Convert the whole frame only once per frame; all pixel locations are
        # still searched at the original resolution.
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)
        best_roi, best_score = None, -1.0

        for scale in self.TEMPLATE_SCALES:
            sw = max(3, int(round(tw * scale)))
            sh = max(3, int(round(th * scale)))
            if sw > fw or sh > fh:
                continue
            if sw == tw and sh == th:
                candidate_template = template
            else:
                candidate_template = cv2.resize(
                    template, (sw, sh),
                    interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                )
            # This response map covers every possible (x, y) placement in the
            # full-resolution frame for this template size.
            response = cv2.matchTemplate(
                frame_gray, candidate_template, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(response)
            if np.isfinite(max_val) and max_val > best_score:
                best_score = float(max_val)
                best_roi = self._clip(frame, (max_loc[0], max_loc[1], sw, sh))

        if best_roi is None:
            return None, 0.0
        return best_roi, max(0.0, best_score)

    @staticmethod
    def _normalize(text):
        return "".join(ch for ch in str(text).casefold() if ch.isalnum())

    def _ocr_recover_local(self, frame, roi, reader):
        """Optional OCR recovery in a bounded local region, never an unconditional full-frame scan."""
        if not self.target_text:
            return None, reader
        try:
            import easyocr
            if reader is None:
                reader = easyocr.Reader(["ch_tra", "en"], gpu=False)
            fh, fw = frame.shape[:2]
            x, y, w, h = roi
            pad_x, pad_y = max(w, 30), max(h, 20)
            x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
            x2, y2 = min(fw, x + w + pad_x), min(fh, y + h + pad_y)
            crop = frame[y1:y2, x1:x2]
            target = self._normalize(self.target_text)
            if crop.size == 0:
                return None, reader
            results = reader.readtext(crop, detail=1, paragraph=False, decoder="greedy",
                                      canvas_size=640, min_size=8)
            best = None
            best_conf = 0.0
            for item in results or []:
                if len(item) < 3:
                    continue
                text, conf = str(item[1]), float(item[2])
                if target not in self._normalize(text) or conf < 0.20:
                    continue
                pts = np.asarray(item[0], dtype=np.float32).reshape(-1, 2)
                bx1, by1 = pts.min(axis=0)
                bx2, by2 = pts.max(axis=0)
                bw, bh = bx2 - bx1, by2 - by1
                pad_w, pad_h = max(2, int(bw * 0.15)), max(2, int(bh * 0.25))
                candidate = self._clip(frame, (x1 + bx1 - pad_w, y1 + by1 - pad_h,
                                                bw + 2 * pad_w, bh + 2 * pad_h))
                # Do not let OCR jump to a far-away duplicate occurrence.
                old_c, new_c = self._center(roi), self._center(candidate)
                diag = max(20.0, math.hypot(w, h))
                if math.hypot(old_c[0] - new_c[0], old_c[1] - new_c[1]) > diag * 3.0:
                    continue
                if conf > best_conf:
                    best, best_conf = candidate, conf
            return best, reader
        except Exception as exc:
            self._report("ocr_recovery_error", f"OCR 補救失敗：{type(exc).__name__}: {exc}")
            return None, reader

    def _draw_monitor(self, frame, frame_no, roi, state, score, tracker_label):
        shown = frame.copy()
        if roi is not None:
            x, y, w, h = roi
            colors = {
                "INITIAL": (0, 220, 0),
                "TRACKING": (255, 210, 0),
                "TEMPLATE_RECOVERED": (0, 220, 220),
                "TEMPLATE_MATCHED": (0, 220, 220),
                "OCR_RECOVERED": (0, 220, 0),
                "UNCERTAIN": (0, 0, 255),
                "LOST": (0, 0, 255),
            }
            color = colors.get(state, (255, 255, 255))
            cv2.rectangle(shown, (x, y), (x+w, y+h), color, 2)
            cv2.putText(shown, f"{state} score={score:.2f}", (x, max(18, y-7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        cv2.putText(shown, f"F{frame_no} | {tracker_label}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        self.monitor_frame_signal.emit(shown, int(frame_no), state)

    def run(self):
        cap = None
        data = {}
        last_frame = self.start_frame
        success = False
        cancelled = False
        reader = None
        try:
            if self.initial_roi is None or len(self.initial_roi) != 4 or self.initial_roi[2] <= 0 or self.initial_roi[3] <= 0:
                raise RuntimeError("缺少有效的手動初始框。請在預覽影格中手動框選目標。")

            self.progress_signal.emit(self.start_frame, "正在載入影片並初始化全畫面比對...")
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError("無法開啟影片檔案。")
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            ok, first_frame = cap.read()
            if not ok:
                raise RuntimeError(f"無法讀取起始影格 F{self.start_frame}。")

            roi = self._clip(first_frame, self.initial_roi)
            x, y, w, h = roi
            if w < 5 or h < 5:
                raise RuntimeError("初始框太小，請重新框選較完整的目標。")
            template = self._gray_template(first_frame, roi)
            # Intentionally do not use KCF/MOSSE to determine the final box.
            # Those trackers are local and can drift when the target moves.
            tracker_label = "GLOBAL_TEMPLATE_MATCH"
            self._report("initialized", f"手動初始框完成；追蹤器={tracker_label}",
                         self.start_frame, bbox=roi, tracker=tracker_label)
            data[self.start_frame] = [roi]
            current_roi = roi
            last_valid_roi = roi
            lost_count = 0
            last_ocr_attempt = -10**9
            last_frame = self.start_frame
            self._draw_monitor(first_frame, self.start_frame, roi, "INITIAL", 1.0, tracker_label)

            frame_no = self.start_frame + 1
            while True:
                if self.isInterruptionRequested() or self._cancelled:
                    cancelled = True
                    self._report("cancelled", "使用者取消追蹤", last_frame)
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                last_frame = frame_no
                frame_started = time.monotonic()
                # Primary method: exhaustive 2D template search on EVERY frame.
                # No KCF box is accepted as the answer; this searches all valid x/y
                # placements in the original-resolution frame at each configured scale.
                found, match_score = self._template_search(frame, last_valid_roi, template)
                candidate = None
                score = match_score
                state = "LOST"

                if found is not None and match_score >= self.MIN_TEMPLATE_SCORE:
                    candidate = self._clip(frame, found)
                    state = "TEMPLATE_MATCHED"
                    lost_count = 0

                if candidate is None:
                    lost_count += 1
                    # OCR is a rare fallback, gated by cooldown and only if a target string exists.
                    if (self.target_text and frame_no - last_ocr_attempt >= self.OCR_RETRY_COOLDOWN):
                        last_ocr_attempt = frame_no
                        self._report("ocr_recovery", f"追蹤信心不足，局部 OCR 補救中 (F{frame_no})",
                                     frame_no, score=match_score)
                        ocr_roi, reader = self._ocr_recover_local(frame, last_valid_roi, reader)
                        if ocr_roi is not None:
                            candidate = ocr_roi
                            score = 0.75
                            state = "OCR_RECOVERED"
                            tracker_label = "GLOBAL_TEMPLATE_MATCH+OCR"
                            lost_count = 0
                    if candidate is None and lost_count <= self.STALE_GRACE_FRAMES:
                        # Brief grace period only; old coordinates are not silently written
                        # for an extended run of frames.
                        candidate = last_valid_roi
                        state = "UNCERTAIN"
                        score = 0.0

                if candidate is not None and state != "UNCERTAIN":
                    current_roi = candidate
                    last_valid_roi = candidate
                    data[frame_no] = [candidate]
                elif candidate is not None and state == "UNCERTAIN":
                    # Conservative short grace period to reduce a one-frame gap.
                    data[frame_no] = [candidate]
                else:
                    # Do not reuse stale boxes indefinitely: that can mosaic the wrong location.
                    data[frame_no] = []

                if frame_no % self.MONITOR_INTERVAL == 0:
                    self._draw_monitor(frame, frame_no, candidate, state, score, tracker_label)
                if frame_no % self.PROGRESS_INTERVAL == 0:
                    elapsed = time.monotonic() - frame_started
                    self.progress_signal.emit(frame_no, f"追蹤 F{frame_no}/{self.total_frames} | {state} | {elapsed:.3f}s/frame")
                    self._report("tracking", f"F{frame_no}/{self.total_frames} | {state}",
                                 frame_no, tracker=tracker_label, score=score,
                                 lost_count=lost_count, frame_elapsed_seconds=elapsed)
                frame_no += 1

            success = (not cancelled) and bool(data) and any(bool(v) for v in data.values())
            self._report("finished", "追蹤完成" if success else ("追蹤已取消" if cancelled else "未取得追蹤資料"),
                         last_frame, tracked_frames=sum(bool(v) for v in data.values()),
                         total_frames=len(data), tracker=tracker_label, success=success)
        except Exception as exc:
            self._report("error", f"執行錯誤：{type(exc).__name__}: {exc}", last_frame)
            success = (not cancelled) and bool(data) and any(bool(v) for v in data.values())
        finally:
            if cap is not None:
                cap.release()
            self.finished_signal.emit(data, last_frame, success)


class VideoProcessorThread(QThread):
    """Frame-accurate mosaic exporter; compatible with the existing GUI task format."""
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(str)

    def __init__(self, video_path, output_path, mosaic_tasks):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.mosaic_tasks = mosaic_tasks

    @staticmethod
    def _apply_mosaic(frame, roi, strength):
        height, width = frame.shape[:2]
        x, y, w, h = [int(round(float(v))) for v in roi]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(0, min(w, width - x))
        h = max(0, min(h, height - y))
        if w <= 0 or h <= 0:
            return
        area = frame[y:y+h, x:x+w]
        strength = max(1, int(strength))
        sw, sh = max(1, w // strength), max(1, h // strength)
        frame[y:y+h, x:x+w] = cv2.resize(
            cv2.resize(area, (sw, sh), interpolation=cv2.INTER_LINEAR),
            (w, h), interpolation=cv2.INTER_NEAREST
        )

    @staticmethod
    def _rois_for_frame(task, frame_no):
        if task.get("type") != "tracking":
            roi = task.get("roi")
            return [roi] if roi else []
        value = task.get("tracking_data", {}).get(frame_no)
        if value is None:
            return []
        if isinstance(value, (tuple, list)) and len(value) == 4 and all(
                isinstance(v, (int, float)) for v in value):
            return [value]
        return list(value) if isinstance(value, (list, tuple)) else []

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished_signal.emit("")
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(
            self.output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not out.isOpened():
            cap.release()
            self.finished_signal.emit("")
            return
        current = 0
        ok_output = True
        try:
            while True:
                if self.isInterruptionRequested():
                    ok_output = False
                    break
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
            ok_output = False
        finally:
            cap.release()
            out.release()
        self.finished_signal.emit(self.output_path if ok_output and current > 0 else "")
