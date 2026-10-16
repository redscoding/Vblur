"""Hybrid tracking + redetection for a PyQt5 video mosaic application.

Features
--------
* Manual ROI initialization.
* CSRT primary tracker (KCF fallback when CSRT is unavailable).
* Kalman constant-velocity prediction.
* Candidate validation using displacement, size and appearance similarity.
* Recovery in stages: local template search -> coarse full-frame search -> OCR
  (when a target string is supplied).
* Explicit TRACKING / SUSPECT / REDETECT / LOST states and per-frame output.
* Compatible with the existing GUI signals and VideoProcessorThread API.

Important: this is a practical classical-CV baseline, not a guarantee of perfect
tracking. Review LOST / UNCERTAIN intervals before using an exported video for
privacy-sensitive purposes.
"""
from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

Box = Tuple[int, int, int, int]


class _OpticalFlowTracker:
    """Small OpenCV-only fallback when opencv-contrib trackers are unavailable."""
    def __init__(self):
        self.prev_gray = None
        self.points = None
        self.box = None

    def init(self, frame, bbox):
        x, y, w, h = [int(round(float(v))) for v in bbox]
        self.box = (x, y, w, h)
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(self.prev_gray.shape, dtype=np.uint8)
        mask[max(0, y):max(0, y) + h, max(0, x):max(0, x) + w] = 255
        self.points = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=80, qualityLevel=0.01,
                                              minDistance=4, mask=mask, blockSize=7)
        return self.points is not None and len(self.points) >= 3

    def update(self, frame):
        if self.prev_gray is None or self.box is None or self.points is None or len(self.points) < 3:
            return False, self.box or (0, 0, 1, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.points, None, winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        if next_points is None or status is None:
            self.prev_gray = gray
            return False, self.box
        good_old = self.points[status.reshape(-1) == 1].reshape(-1, 2)
        good_new = next_points[status.reshape(-1) == 1].reshape(-1, 2)
        if len(good_new) < 3:
            self.prev_gray = gray
            self.points = None
            return False, self.box
        delta = np.median(good_new - good_old, axis=0)
        x, y, w, h = self.box
        fh, fw = gray.shape[:2]
        nx = int(round(x + float(delta[0])))
        ny = int(round(y + float(delta[1])))
        nx = max(0, min(nx, fw - 1))
        ny = max(0, min(ny, fh - 1))
        w, h = min(w, fw - nx), min(h, fh - ny)
        self.box = (nx, ny, max(1, w), max(1, h))
        # Keep points that remain inside the updated box; replenish if too few remain.
        inside = ((good_new[:, 0] >= nx) & (good_new[:, 0] <= nx + w) &
                  (good_new[:, 1] >= ny) & (good_new[:, 1] <= ny + h))
        kept = good_new[inside].astype(np.float32).reshape(-1, 1, 2)
        if len(kept) < 8:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask[ny:ny+h, nx:nx+w] = 255
            replenished = cv2.goodFeaturesToTrack(gray, maxCorners=80, qualityLevel=0.01,
                                                   minDistance=4, mask=mask, blockSize=7)
            if replenished is not None:
                kept = replenished
        self.points = kept if kept is not None and len(kept) >= 3 else None
        self.prev_gray = gray
        return self.points is not None, self.box


class OCRTrackingThread(QThread):
    """Compatibility name retained for the current main.py GUI."""

    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict, int, bool)
    monitor_frame_signal = pyqtSignal(object, int, str)
    monitor_data_signal = pyqtSignal(dict)

    MONITOR_INTERVAL = 2
    PROGRESS_INTERVAL = 10
    OCR_INTERVAL = 12
    OCR_RETRY_COOLDOWN = 8
    MIN_APPEARANCE_SCORE = 0.42
    LOCAL_SEARCH_RADIUS = 2.0
    GLOBAL_SEARCH_MAX_WIDTH = 720
    MAX_LOST_FRAMES_BEFORE_GLOBAL = 2
    UNCERTAIN_GRACE_FRAMES = 1

    def __init__(self, video_path, start_frame, total_frames, target_text="",
                 initial_roi=None, tracker_name="CSRT"):
        super().__init__()
        self.video_path = str(video_path)
        self.start_frame = max(0, int(start_frame))
        self.total_frames = max(0, int(total_frames))
        self.target_text = str(target_text or "").strip()
        self.initial_roi = tuple(map(int, initial_roi)) if initial_roi else None
        self.tracker_name = str(tracker_name or "CSRT").upper()
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
    def _clip(frame, roi) -> Box:
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
    def _diag(roi):
        return max(8.0, math.hypot(roi[2], roi[3]))

    @staticmethod
    def _create_tracker(name="CSRT"):
        """Create CSRT first; fall back to KCF/MOSSE based on installed OpenCV."""
        order = [name.upper(), "CSRT", "KCF", "MOSSE"]
        seen = set()
        for tracker in order:
            if tracker in seen:
                continue
            seen.add(tracker)
            attr = {"CSRT": "TrackerCSRT_create", "KCF": "TrackerKCF_create",
                    "MOSSE": "TrackerMOSSE_create"}.get(tracker)
            if not attr:
                continue
            for owner in (getattr(cv2, "legacy", None), cv2):
                factory = getattr(owner, attr, None) if owner is not None else None
                if callable(factory):
                    try:
                        return factory(), tracker
                    except Exception:
                        pass
        # OpenCV's standard wheel may not contain CSRT/KCF/MOSSE. Use optical flow
        # as a fallback so the app remains runnable; the same validation and
        # redetection pipeline still guards against drift.
        return _OpticalFlowTracker(), "OPTICAL_FLOW"

    @staticmethod
    def _new_kalman(box):
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        cx, cy = OCRTrackingThread._center(box)
        kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
        return kf

    @staticmethod
    def _gray_patch(frame, roi, size=None):
        x, y, w, h = roi
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        if size:
            gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA if w > size[0] else cv2.INTER_LINEAR)
        return gray

    def _appearance_score(self, frame, roi, template):
        if template is None:
            return 0.0
        patch = self._gray_patch(frame, roi, (template.shape[1], template.shape[0]))
        if patch is None or patch.shape != template.shape:
            return 0.0
        try:
            score = float(cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)[0, 0])
            return max(-1.0, min(1.0, score)) if np.isfinite(score) else 0.0
        except cv2.error:
            return 0.0

    def _local_template_search(self, frame, predicted_box, template):
        """Search a bounded region around predicted location, at a few scales."""
        if template is None:
            return None, 0.0
        fh, fw = frame.shape[:2]
        px, py, pw, ph = self._clip(frame, predicted_box)
        radius_x = max(pw * self.LOCAL_SEARCH_RADIUS, 40)
        radius_y = max(ph * self.LOCAL_SEARCH_RADIUS, 30)
        x1 = max(0, int(px + pw / 2 - radius_x))
        y1 = max(0, int(py + ph / 2 - radius_y))
        x2 = min(fw, int(px + pw / 2 + radius_x))
        y2 = min(fh, int(py + ph / 2 + radius_y))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, 0.0
        gray = cv2.equalizeHist(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        th, tw = template.shape[:2]
        best_box, best_score = None, -1.0
        for scale in (0.85, 1.0, 1.15):
            sw, sh = max(8, int(tw * scale)), max(8, int(th * scale))
            if sw > gray.shape[1] or sh > gray.shape[0]:
                continue
            t = template if (sw == tw and sh == th) else cv2.resize(
                template, (sw, sh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            )
            try:
                response = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(response)
            except cv2.error:
                continue
            if np.isfinite(score) and score > best_score:
                best_score = float(score)
                best_box = self._clip(frame, (x1 + loc[0], y1 + loc[1], sw, sh))
        return best_box, max(0.0, best_score)

    def _global_template_search(self, frame, template):
        """Coarse downscaled full-frame scan, then refine around the best candidate."""
        if template is None:
            return None, 0.0
        fh, fw = frame.shape[:2]
        scale_down = min(1.0, self.GLOBAL_SEARCH_MAX_WIDTH / float(fw))
        small = cv2.resize(frame, (max(1, int(fw * scale_down)), max(1, int(fh * scale_down))),
                           interpolation=cv2.INTER_AREA) if scale_down < 1 else frame
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        th, tw = template.shape[:2]
        small_w, small_h = max(5, int(tw * scale_down)), max(5, int(th * scale_down))
        if small_w >= gray.shape[1] or small_h >= gray.shape[0]:
            return None, 0.0
        base = cv2.resize(template, (small_w, small_h), interpolation=cv2.INTER_AREA) if scale_down < 1 else template
        best_box, best_score = None, -1.0
        for scale in (0.85, 1.0, 1.15):
            sw, sh = max(5, int(small_w * scale)), max(5, int(small_h * scale))
            if sw >= gray.shape[1] or sh >= gray.shape[0]:
                continue
            t = base if (sw == small_w and sh == small_h) else cv2.resize(
                base, (sw, sh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            )
            try:
                response = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(response)
            except cv2.error:
                continue
            if np.isfinite(score) and score > best_score:
                best_score = float(score)
                if scale_down < 1:
                    best_box = self._clip(frame, (loc[0] / scale_down, loc[1] / scale_down,
                                                   sw / scale_down, sh / scale_down))
                else:
                    best_box = self._clip(frame, (loc[0], loc[1], sw, sh))
        if best_box is None:
            return None, 0.0
        # Refine at original resolution in a small area around coarse candidate.
        bx, by, bw, bh = best_box
        rx1, ry1 = max(0, bx - bw), max(0, by - bh)
        rx2, ry2 = min(fw, bx + 2 * bw), min(fh, by + 2 * bh)
        crop = frame[ry1:ry2, rx1:rx2]
        target_w, target_h = bw, bh
        if crop.size and target_w < crop.shape[1] and target_h < crop.shape[0]:
            crop_gray = cv2.equalizeHist(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
            # Keep refinement bounded; template scale is allowed to vary slightly.
            for scale in (0.9, 1.0, 1.1):
                sw, sh = max(5, int(tw * scale)), max(5, int(th * scale))
                if sw >= crop_gray.shape[1] or sh >= crop_gray.shape[0]:
                    continue
                t = template if (sw == tw and sh == th) else cv2.resize(
                    template, (sw, sh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
                )
                try:
                    response = cv2.matchTemplate(crop_gray, t, cv2.TM_CCOEFF_NORMED)
                    _, score, _, loc = cv2.minMaxLoc(response)
                except cv2.error:
                    continue
                if np.isfinite(score) and score > best_score:
                    best_score = float(score)
                    best_box = self._clip(frame, (rx1 + loc[0], ry1 + loc[1], sw, sh))
        return best_box, max(0.0, best_score)

    @staticmethod
    def _normalize(text):
        return "".join(ch for ch in str(text).casefold() if ch.isalnum())

    def _ocr_search(self, frame, reader, region=None):
        """Find target_text in local region or full frame. Returns (box, conf, reader)."""
        if not self.target_text:
            return None, 0.0, reader
        try:
            if reader is None:
                import easyocr
                reader = easyocr.Reader(["ch_tra", "en"], gpu=False)
            fh, fw = frame.shape[:2]
            if region is None:
                x1, y1, x2, y2 = 0, 0, fw, fh
                crop = frame
            else:
                x, y, w, h = self._clip(frame, region)
                pad_x, pad_y = max(w, 40), max(h, 30)
                x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                x2, y2 = min(fw, x + w + pad_x), min(fh, y + h + pad_y)
                crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None, 0.0, reader
            results = reader.readtext(crop, detail=1, paragraph=False, decoder="greedy",
                                      canvas_size=960 if region is None else 640, min_size=8)
            target = self._normalize(self.target_text)
            best_box, best_conf = None, 0.0
            for item in results or []:
                if len(item) < 3:
                    continue
                text, conf = str(item[1]), float(item[2])
                normalized = self._normalize(text)
                if not target or (target not in normalized and normalized not in target):
                    continue
                if conf < 0.25:
                    continue
                pts = np.asarray(item[0], dtype=np.float32).reshape(-1, 2)
                bx1, by1 = pts.min(axis=0)
                bx2, by2 = pts.max(axis=0)
                bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
                # Expand OCR text box to cover a useful mosaic region.
                pad_w, pad_h = max(4, int(bw * 0.22)), max(3, int(bh * 0.30))
                out_w = bw + 2 * pad_w
                out_h = bh + 2 * pad_h
                # OCR often returns only the glyph rectangle, while the manually
                # selected ROI may include a full plate/object. Keep a useful mask
                # area by expanding around the OCR center to at least 80% of the
                # initial ROI dimensions.
                initial = getattr(self, "_initial_roi", None)
                if initial is not None:
                    out_w = max(out_w, int(initial[2] * 0.80))
                    out_h = max(out_h, int(initial[3] * 0.80))
                center_x, center_y = x1 + (bx1 + bx2) / 2.0, y1 + (by1 + by2) / 2.0
                candidate = self._clip(frame, (center_x - out_w / 2, center_y - out_h / 2,
                                                out_w, out_h))
                if conf > best_conf:
                    best_box, best_conf = candidate, conf
            return best_box, best_conf, reader
        except Exception as exc:
            self._report("ocr_error", f"OCR 重檢測失敗：{type(exc).__name__}: {exc}")
            return None, 0.0, reader

    def _valid_candidate(self, frame, candidate, predicted, last_box, template,
                         appearance_threshold=None):
        if candidate is None:
            return False, 0.0, "no_candidate"
        candidate = self._clip(frame, candidate)
        x, y, w, h = candidate
        if w < 5 or h < 5:
            return False, 0.0, "box_too_small"
        if last_box is not None:
            old_w, old_h = last_box[2], last_box[3]
            area_ratio = (w * h) / max(1.0, old_w * old_h)
            if area_ratio < 0.30 or area_ratio > 3.2:
                return False, 0.0, "size_jump"
        # Permit faster movement than a strictly local tracker, but reject wild jumps.
        if predicted is not None:
            c1, c2 = self._center(candidate), self._center(predicted)
            max_dist = max(100.0, 4.5 * self._diag(last_box or candidate))
            if math.hypot(c1[0] - c2[0], c1[1] - c2[1]) > max_dist:
                return False, 0.0, "position_jump"
        appearance = self._appearance_score(frame, candidate, template)
        threshold = self.MIN_APPEARANCE_SCORE if appearance_threshold is None else appearance_threshold
        if appearance < threshold:
            return False, appearance, "appearance_low"
        return True, appearance, "ok"

    def _draw_monitor(self, frame, frame_no, roi, state, score, tracker_label, note=""):
        shown = frame.copy()
        colors = {
            "INITIAL": (0, 220, 0), "TRACKING": (0, 220, 0),
            "SUSPECT": (0, 190, 255), "REDETECT": (255, 180, 0),
            "LOST": (0, 0, 255), "OCR_RECOVERED": (255, 255, 0),
            "TEMPLATE_RECOVERED": (255, 255, 0), "UNCERTAIN": (0, 0, 255),
        }
        color = colors.get(state, (255, 255, 255))
        if roi is not None:
            x, y, w, h = self._clip(frame, roi)
            cv2.rectangle(shown, (x, y), (x + w, y + h), color, 2)
            label = f"{state} score={score:.2f}"
            cv2.putText(shown, label, (x, max(18, y - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2, cv2.LINE_AA)
        cv2.putText(shown, f"F{frame_no} | {tracker_label} | {note[:60]}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        self.monitor_frame_signal.emit(shown, int(frame_no), state)

    def run(self):
        cap = None
        data = {}
        last_frame = self.start_frame
        success = False
        cancelled = False
        reader = None
        tracked_count = 0
        lost_count = 0
        try:
            if self.initial_roi is None or len(self.initial_roi) != 4 or self.initial_roi[2] <= 0 or self.initial_roi[3] <= 0:
                raise RuntimeError("缺少有效的手動初始框。請先在預覽影格中手動框選目標。")
            self.progress_signal.emit(self.start_frame, "正在載入影片並初始化混合追蹤...")
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError("無法開啟影片檔案。")
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            ok, first_frame = cap.read()
            if not ok:
                raise RuntimeError(f"無法讀取起始影格 F{self.start_frame}。")
            current_box = self._clip(first_frame, self.initial_roi)
            self._initial_roi = current_box
            if current_box[2] < 5 or current_box[3] < 5:
                raise RuntimeError("初始框太小，請重新框選較完整的目標。")
            template = self._gray_patch(first_frame, current_box)
            tracker, tracker_label = self._create_tracker(self.tracker_name)
            tracker.init(first_frame, tuple(map(float, current_box)))
            kalman = self._new_kalman(current_box)
            last_valid_box = current_box
            last_ocr_frame = -10**9
            self._report("initialized", f"初始化完成；追蹤器={tracker_label}；模式={'OCR' if self.target_text else '一般物體'}",
                         self.start_frame, bbox=current_box, tracker=tracker_label)
            data[self.start_frame] = [current_box]
            tracked_count += 1
            self._draw_monitor(first_frame, self.start_frame, current_box, "INITIAL", 1.0, tracker_label)

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
                prediction = kalman.predict()
                pcx, pcy = float(prediction[0, 0]), float(prediction[1, 0])
                px, py, pw, ph = last_valid_box
                predicted_box = self._clip(frame, (pcx - pw / 2, pcy - ph / 2, pw, ph))

                candidate = None
                appearance = 0.0
                reason = "tracker_failed"
                tracker_ok = False
                try:
                    ok_track, raw_box = tracker.update(frame)
                    if ok_track:
                        raw_box = self._clip(frame, raw_box)
                        valid, appearance, reason = self._valid_candidate(
                            frame, raw_box, predicted_box, last_valid_box, template
                        )
                        if valid:
                            candidate = raw_box
                            tracker_ok = True
                        else:
                            self._report("suspect", f"追蹤候選未通過檢查：{reason}", frame_no,
                                         candidate=raw_box, appearance=appearance)
                except Exception as exc:
                    reason = f"tracker_exception:{type(exc).__name__}"

                state = "TRACKING" if tracker_ok else "SUSPECT"
                score = appearance
                # OCR periodically corrects text targets; local OCR first to keep it cheaper.
                if self.target_text and (frame_no % self.OCR_INTERVAL == 0 or not tracker_ok):
                    if frame_no - last_ocr_frame >= self.OCR_RETRY_COOLDOWN:
                        last_ocr_frame = frame_no
                        ocr_box, ocr_conf, reader = self._ocr_search(
                            frame, reader, region=last_valid_box if lost_count < self.MAX_LOST_FRAMES_BEFORE_GLOBAL else None
                        )
                        if ocr_box is not None:
                            # OCR identity is a strong signal; allow a lower visual score but
                            # still require plausible size and location unless global search ran.
                            ocr_valid, ocr_app, ocr_reason = self._valid_candidate(
                                frame, ocr_box, predicted_box, last_valid_box, template,
                                appearance_threshold=0.18
                            )
                            if ocr_valid or lost_count >= self.MAX_LOST_FRAMES_BEFORE_GLOBAL:
                                candidate = ocr_box
                                tracker_ok = True
                                state = "OCR_RECOVERED"
                                score = ocr_conf
                                try:
                                    tracker, tracker_label = self._create_tracker(self.tracker_name)
                                    tracker.init(frame, tuple(map(float, candidate)))
                                except Exception as exc:
                                    self._report("tracker_reinit_error", f"OCR 校正後重新初始化追蹤器失敗：{exc}", frame_no)
                                self._report("ocr_recovered", f"OCR 找回目標（信心 {ocr_conf:.2f}）", frame_no,
                                             bbox=ocr_box, ocr_confidence=ocr_conf, appearance=ocr_app)

                # Redetection is only run when normal tracking/OCR did not produce a valid box.
                if candidate is None:
                    state = "REDETECT"
                    self._report("redetect", f"追蹤異常（{reason}），正在重新搜尋", frame_no)
                    recovered, recover_score = self._local_template_search(frame, predicted_box, template)
                    if recovered is not None and recover_score >= 0.50:
                        valid, app, why = self._valid_candidate(
                            frame, recovered, predicted_box, last_valid_box, template,
                            appearance_threshold=0.48
                        )
                        if valid:
                            candidate, score, state = recovered, app, "TEMPLATE_RECOVERED"
                    # If local search fails for a few frames, do a coarse global scan.
                    if candidate is None and lost_count + 1 >= self.MAX_LOST_FRAMES_BEFORE_GLOBAL:
                        recovered, recover_score = self._global_template_search(frame, template)
                        if recovered is not None and recover_score >= 0.58:
                            valid, app, why = self._valid_candidate(
                                frame, recovered, predicted_box, last_valid_box, template,
                                appearance_threshold=0.50
                            )
                            # Global recovery can be far from prediction; require appearance.
                            if valid or app >= 0.60:
                                candidate, score, state = recovered, max(app, recover_score), "TEMPLATE_RECOVERED"
                    if candidate is not None:
                        try:
                            tracker, tracker_label = self._create_tracker(self.tracker_name)
                            tracker.init(frame, tuple(map(float, candidate)))
                        except Exception as exc:
                            self._report("tracker_reinit_error", f"重新初始化追蹤器失敗：{exc}", frame_no)

                if candidate is not None:
                    current_box = self._clip(frame, candidate)
                    last_valid_box = current_box
                    cx, cy = self._center(current_box)
                    measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
                    kalman.correct(measurement)
                    lost_count = 0
                    data[frame_no] = [current_box]
                    tracked_count += 1
                else:
                    lost_count += 1
                    state = "LOST"
                    # Short one-frame grace: include last known box only briefly. It is marked
                    # uncertain in the monitor; later LOST frames have no box in tracking_data.
                    if lost_count <= self.UNCERTAIN_GRACE_FRAMES:
                        data[frame_no] = [last_valid_box]
                        state = "UNCERTAIN"
                    else:
                        data[frame_no] = []

                if frame_no % self.MONITOR_INTERVAL == 0 or state in ("LOST", "UNCERTAIN", "OCR_RECOVERED"):
                    self._draw_monitor(frame, frame_no, candidate if candidate is not None else (last_valid_box if state == "UNCERTAIN" else None),
                                       state, score, tracker_label, reason)
                if frame_no % self.PROGRESS_INTERVAL == 0:
                    elapsed = time.monotonic() - frame_started
                    self.progress_signal.emit(frame_no, f"混合追蹤 F{frame_no}/{self.total_frames} | {state} | {elapsed:.3f}s/frame")
                    self._report("tracking", f"F{frame_no}/{self.total_frames} | {state}", frame_no,
                                 tracker=tracker_label, score=score, lost_count=lost_count,
                                 frame_elapsed_seconds=elapsed, reason=reason)
                frame_no += 1

            success = (not cancelled) and bool(data) and any(bool(v) for v in data.values())
            self._report("finished", "追蹤完成" if success else ("追蹤已取消" if cancelled else "未取得追蹤資料"),
                         last_frame, tracked_frames=tracked_count, total_frames=len(data),
                         tracker=tracker_label, success=success,
                         lost_frames=sum(1 for v in data.values() if not v))
        except Exception as exc:
            self._report("error", f"執行錯誤：{type(exc).__name__}: {exc}", last_frame)
            success = (not cancelled) and bool(data) and any(bool(v) for v in data.values())
        finally:
            if cap is not None:
                cap.release()
            self.finished_signal.emit(data, last_frame, success)


class VideoProcessorThread(QThread):
    """Frame-accurate mosaic exporter compatible with existing GUI task format."""
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
        return list(value) if isinstance(value, (tuple, list)) else []

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished_signal.emit("")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
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
