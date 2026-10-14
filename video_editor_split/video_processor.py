"""Multi-target OCR tracking and frame-accurate mosaic export.

Requires: opencv-contrib-python, easyocr, PyQt5.
tracking_data format: {absolute_frame_index: [(x, y, w, h), ...]}
"""
import math
import time
import cv2
import easyocr
from PyQt5.QtCore import QThread, pyqtSignal


class OCRTrackingThread(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict, int, bool)
    monitor_frame_signal = pyqtSignal(object, int, str)
    monitor_data_signal = pyqtSignal(dict)

    OCR_INTERVAL = 12
    MIN_CONFIDENCE = 0.20
    MAX_PREDICTED_GAP = 4
    DETECT_CANVAS_SIZE = 1920

    def __init__(self, video_path, start_frame, total_frames, target_text):
        super().__init__()
        self.video_path = video_path
        self.start_frame = max(0, int(start_frame))
        self.total_frames = max(0, int(total_frames))
        self.target_text = str(target_text or "")

    def _report(self, stage, message, frame_no=None, **extra):
        self.monitor_data_signal.emit({"stage": stage, "message": message,
            "frame": frame_no, "clock": time.monotonic(), **extra})

    @staticmethod
    def _create_tracker():
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        raise RuntimeError("目前的 OpenCV 沒有 CSRT tracker，請安裝 opencv-contrib-python。")

    @staticmethod
    def _clip(frame, roi):
        h, w = frame.shape[:2]
        x, y, rw, rh = [int(round(float(v))) for v in roi]
        x = max(0, min(x, w - 1)); y = max(0, min(y, h - 1))
        rw = max(1, min(rw, w - x)); rh = max(1, min(rh, h - y))
        return (x, y, rw, rh)

    @staticmethod
    def _center(r):
        return r[0] + r[2] / 2.0, r[1] + r[3] / 2.0

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _normalize(text):
        # Case-insensitive; ignore whitespace and common punctuation but preserve letters/numbers.
        return "".join(ch for ch in str(text).casefold() if ch.isalnum())

    def _ocr_all(self, reader, frame, frame_no):
        """Detect text boxes first, then recognize each detector box independently.

        Calling readtext() on a full image may group nearby text boxes before recognition.
        detect()+recognize() makes the detector's boxes explicit and avoids relying on
        a single returned text line to represent all instances.
        """
        started = time.monotonic()
        target = self._normalize(self.target_text)
        h, w = frame.shape[:2]
        found = []
        raw_texts = []
        try:
            horizontal_batch, free_batch = reader.detect(
                frame, min_size=8, text_threshold=0.6, low_text=0.35,
                link_threshold=0.35, canvas_size=self.DETECT_CANVAS_SIZE,
                mag_ratio=1.0, slope_ths=0.1, ycenter_ths=0.35,
                height_ths=0.35, width_ths=0.1, add_margin=0.05,
                reformat=True)
            horizontal = horizontal_batch[0] if horizontal_batch else []
            free = free_batch[0] if free_batch else []
            # Recognize each detector result separately. This prevents one recognition
            # result from hiding another independent occurrence of the target.
            boxes = []
            for box in horizontal:
                if len(box) >= 4:
                    x1, x2, y1, y2 = [int(round(float(v))) for v in box[:4]]
                    boxes.append((max(0, x1), max(0, y1), min(w, x2), min(h, y2), None))
            for poly in free:
                pts = poly.tolist() if hasattr(poly, "tolist") else poly
                xs = [float(p[0]) for p in pts]; ys = [float(p[1]) for p in pts]
                boxes.append((max(0, int(min(xs))), max(0, int(min(ys))),
                              min(w, int(max(xs))), min(h, int(max(ys))), pts))

            for x1, y1, x2, y2, poly in boxes:
                if x2 <= x1 or y2 <= y1:
                    continue
                try:
                    if poly is None:
                        rec = reader.recognize(frame, horizontal_list=[[x1, x2, y1, y2]],
                                               free_list=[], detail=1, paragraph=False,
                                               decoder="greedy", batch_size=1, workers=0)
                    else:
                        rec = reader.recognize(frame, horizontal_list=[], free_list=[poly],
                                               detail=1, paragraph=False, decoder="greedy",
                                               batch_size=1, workers=0)
                except Exception:
                    # Fallback for EasyOCR versions with different recognize signatures.
                    crop = frame[y1:y2, x1:x2]
                    rec = reader.readtext(crop, detail=1, paragraph=False,
                                          decoder="greedy", width_ths=0.1, ycenter_ths=0.1)
                if not rec:
                    continue
                # One detector box is recognized at a time; use its best text result.
                best = max(rec, key=lambda r: float(r[2]) if len(r) > 2 else 0.0)
                text = str(best[1]) if len(best) > 1 else ""
                conf = float(best[2]) if len(best) > 2 else 0.0
                raw_texts.append({"text": text, "confidence": conf})
                if conf < self.MIN_CONFIDENCE or not target or target not in self._normalize(text):
                    continue
                bw, bh = x2 - x1, y2 - y1
                px, py = max(2, int(bw * 0.15)), max(2, int(bh * 0.25))
                roi = self._clip(frame, (x1-px, y1-py, bw+2*px, bh+2*py))
                found.append((roi, text, conf))
        except Exception as exc:
            self._report("error", f"OCR 偵測失敗：{type(exc).__name__}: {exc}", frame_no)

        # Deduplicate only strongly overlapping boxes; never dedupe by text alone.
        found.sort(key=lambda item: item[2], reverse=True)
        unique = []
        for item in found:
            if not any(self._iou(item[0], old[0]) > 0.75 for old in unique):
                unique.append(item)
        elapsed = time.monotonic() - started
        self._report("ocr_done", f"OCR 完成，耗時 {elapsed:.2f} 秒；符合目標 {len(unique)} 個",
                     frame_no, elapsed_seconds=elapsed, result_count=len(raw_texts),
                     target_count=len(unique), target_text=self.target_text,
                     ocr_results=raw_texts[:20])
        return unique, elapsed

    def _new_track(self, frame, detection, track_id):
        roi, text, confidence = detection
        tracker = self._create_tracker()
        tracker.init(frame, roi)
        return {"id": track_id, "tracker": tracker, "roi": roi, "lost": 0,
                "text": text, "confidence": confidence}

    def _draw_monitor(self, frame, frame_no, rois, mode):
        shown = frame.copy()
        for track_id, roi in rois:
            x, y, w, h = roi
            cv2.rectangle(shown, (x, y), (x+w, y+h), (0, 220, 0), 2)
            cv2.putText(shown, f"ID {track_id}", (x, max(18, y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(shown, f"Frame {frame_no} | {mode} | targets={len(rois)}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        self.monitor_frame_signal.emit(shown, int(frame_no), mode)

    def run(self):
        cap = None
        data = {}
        last_frame = self.start_frame
        success = False
        try:
            self.progress_signal.emit(self.start_frame, "正在載入 OCR 模型...")
            self._report("loading", "正在載入 EasyOCR 模型", self.start_frame)
            reader = easyocr.Reader(["ch_tra", "en"], gpu=False)
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError("無法開啟影片檔案。")
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("無法讀取起始影格。")

            detections, _ = self._ocr_all(reader, frame, self.start_frame)
            tracks = []
            next_id = 1
            for detection in detections:
                tracks.append(self._new_track(frame, detection, next_id)); next_id += 1
            if not tracks:
                self._report("not_found", f"起始影格找不到目標文字：{self.target_text}", self.start_frame)
                return

            def snapshot(frame_no, active_tracks):
                # Record all active boxes for every processed frame, not one ROI per frame.
                rois = []
                for t in active_tracks:
                    if t["lost"] <= self.MAX_PREDICTED_GAP:
                        roi = tuple(map(int, self._clip(frame, t["roi"])))
                        if not any(self._iou(roi, old) > 0.80 for old in rois):
                            rois.append(roi)
                data[frame_no] = rois
                return [(t["id"], tuple(map(int, self._clip(frame, t["roi"]))))
                        for t in active_tracks if t["lost"] <= self.MAX_PREDICTED_GAP]

            data[self.start_frame] = [t["roi"] for t in tracks]
            self._draw_monitor(frame, self.start_frame, [(t["id"], t["roi"]) for t in tracks], "INITIAL OCR")
            self._report("locked", f"起始影格找到 {len(tracks)} 個符合目標的區域", self.start_frame,
                         target_text=self.target_text, target_count=len(tracks), tracked_frames=1)
            self.progress_signal.emit(self.start_frame, f"已找到 {len(tracks)} 個目標，開始追蹤")
            last_frame = self.start_frame
            stop_reason = "已處理至影片結尾"

            for i in range(self.start_frame + 1, self.total_frames):
                if self.isInterruptionRequested():
                    stop_reason = "使用者取消追蹤"; break
                ret, frame = cap.read()
                if not ret:
                    stop_reason = "影片讀取結束或讀取失敗"; break
                last_frame = i
                # Update every tracker on every frame.
                for t in tracks:
                    ok, bbox = t["tracker"].update(frame)
                    if ok and bbox[2] > 1 and bbox[3] > 1:
                        t["roi"] = self._clip(frame, bbox)
                        t["lost"] = 0
                    else:
                        t["lost"] += 1

                should_ocr = ((i - self.start_frame) % self.OCR_INTERVAL == 0 or
                              not any(t["lost"] <= self.MAX_PREDICTED_GAP for t in tracks))
                if should_ocr:
                    detections, elapsed = self._ocr_all(reader, frame, i)
                    # Greedy one-to-one association using IoU and center distance.
                    pairs = []
                    for ti, t in enumerate(tracks):
                        for di, (roi, _text, _conf) in enumerate(detections):
                            c1, c2 = self._center(t["roi"]), self._center(roi)
                            dist = math.hypot(c1[0]-c2[0], c1[1]-c2[1])
                            scale = max(20.0, math.hypot(t["roi"][2], t["roi"][3]))
                            iou = self._iou(t["roi"], roi)
                            # A strict size-scaled gate avoids assigning one row's text to another row.
                            if iou >= 0.05 or dist <= max(35.0, scale * 1.8):
                                pairs.append((iou * 2.0 - dist / max(scale, 1.0), ti, di))
                    pairs.sort(reverse=True)
                    used_tracks, used_dets = set(), set()
                    for _score, ti, di in pairs:
                        if ti in used_tracks or di in used_dets:
                            continue
                        t = tracks[ti]
                        roi, text, conf = detections[di]
                        t["roi"], t["lost"], t["text"], t["confidence"] = roi, 0, text, conf
                        t["tracker"] = self._create_tracker(); t["tracker"].init(frame, roi)
                        used_tracks.add(ti); used_dets.add(di)
                    # Every unmatched detection becomes a new independent target.
                    for di, detection in enumerate(detections):
                        if di not in used_dets:
                            tracks.append(self._new_track(frame, detection, next_id)); next_id += 1
                    # Remove stale trackers only after the grace period.
                    tracks = [t for t in tracks if t["lost"] <= self.MAX_PREDICTED_GAP]
                    self._report("tracking", f"多目標追蹤中：{sum(t['lost'] == 0 for t in tracks)} 個正常、{len(tracks)} 個有效框",
                                 i, target_count=len(tracks), elapsed_seconds=elapsed,
                                 lost_count=sum(t["lost"] > 0 for t in tracks), patience=self.MAX_PREDICTED_GAP)

                rois_with_ids = [(t["id"], tuple(map(int, self._clip(frame, t["roi"]))))
                                 for t in tracks if t["lost"] <= self.MAX_PREDICTED_GAP]
                # Deduplicate identical boxes while keeping distinct, separated occurrences.
                unique_with_ids = []
                for tid, roi in rois_with_ids:
                    if not any(self._iou(roi, old_roi) > 0.85 for _, old_roi in unique_with_ids):
                        unique_with_ids.append((tid, roi))
                data[i] = [roi for _, roi in unique_with_ids]  # explicit entry for each frame
                if i % 3 == 0:
                    self._draw_monitor(frame, i, unique_with_ids, "MULTI TRACKING" if unique_with_ids else "SEARCHING")
                    self.progress_signal.emit(i, f"多目標追蹤：F{i}/{self.total_frames}，有效框 {len(unique_with_ids)}")
            success = bool(data)
            self._report("finished", stop_reason, last_frame, tracked_frames=len(data),
                         target_count=max((len(v) for v in data.values()), default=0), success=success)
        except Exception as exc:
            self._report("error", f"執行錯誤：{type(exc).__name__}: {exc}", last_frame)
            success = bool(data)
        finally:
            if cap is not None:
                cap.release()
            self.finished_signal.emit(data, last_frame, success)


class VideoProcessorThread(QThread):
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(str)

    def __init__(self, video_path, output_path, mosaic_tasks):
        super().__init__()
        self.video_path, self.output_path, self.mosaic_tasks = video_path, output_path, mosaic_tasks

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
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not out.isOpened():
            cap.release(); self.finished_signal.emit(""); return
        current = 0
        try:
            while True:
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
