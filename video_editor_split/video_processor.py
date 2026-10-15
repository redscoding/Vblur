"""Multi-target OCR tracking and frame-accurate mosaic export.

Requires: opencv-contrib-python, easyocr, PyQt5.
tracking_data format: {absolute_frame_index: [(x, y, w, h), ...]}
"""
import math
import time
import cv2
import easyocr
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

try:
    from global_feature_gate import GlobalFeatureGate
except ImportError:  # Keep the original OCR workflow usable if the optional module is missing.
    GlobalFeatureGate = None


class OCRTrackingThread(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict, int, bool)
    monitor_frame_signal = pyqtSignal(object, int, str)
    monitor_data_signal = pyqtSignal(dict)

    # OCR association runs periodically; CSRT updates each frame and uncertain jumps are rejected.
    # Search is deliberately cheaper than steady-state tracking. A coarse search can
    # miss a target that appears for fewer than SEARCH_STEP frames; the full-resolution
    # burst after the first hit compensates for initialization misses around that point.
    SEARCH_STEP = 3
    SEARCH_MAX_DIM = 960
    SEARCH_CANVAS_SIZE = 1280
    INIT_BURST_BEFORE = 2
    INIT_BURST_AFTER = 5
    OCR_INTERVAL = 6
    ADAPTIVE_OCR_INTERVAL = 2
    MIN_CONFIDENCE = 0.20
    MAX_PREDICTED_GAP = 10
    DETECT_CANVAS_SIZE = 1920
    MONITOR_INTERVAL = 2  # update the visual monitor every N frames
    MOTION_JUMP_DIAGONALS = 1.8
    MATCH_MIN_SCORE = 0.27

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

    def _ocr_all(self, reader, frame, frame_no, *, coordinate_frame=None,
                 canvas_size=None, report=True):
        """Detect every occurrence of the requested text in one EasyOCR pass.

        `frame` may be a downscaled search image. If `coordinate_frame` is supplied,
        boxes are mapped back to that original frame's coordinate system.
        """
        started = time.monotonic()
        target = self._normalize(self.target_text)
        found = []
        raw_texts = []
        try:
            results = reader.readtext(
                frame, detail=1, paragraph=False, decoder="greedy",
                min_size=8, text_threshold=0.6, low_text=0.35,
                link_threshold=0.35, canvas_size=int(canvas_size or self.DETECT_CANVAS_SIZE),
                mag_ratio=1.0, slope_ths=0.1, ycenter_ths=0.35,
                height_ths=0.35, width_ths=0.1, add_margin=0.05,
                batch_size=16, workers=0,
            )
            h, w = frame.shape[:2]
            if coordinate_frame is not None:
                oh, ow = coordinate_frame.shape[:2]
                sx, sy = ow / float(w), oh / float(h)
                clip_frame = coordinate_frame
            else:
                sx = sy = 1.0
                clip_frame = frame
            for item in results or []:
                if len(item) < 3:
                    continue
                points, text, confidence = item[0], str(item[1]), float(item[2])
                if report:
                    raw_texts.append({"text": text, "confidence": confidence})
                if confidence < self.MIN_CONFIDENCE or not target or target not in self._normalize(text):
                    continue
                pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
                x1, y1 = int(np.floor(pts[:, 0].min())), int(np.floor(pts[:, 1].min()))
                x2, y2 = int(np.ceil(pts[:, 0].max())), int(np.ceil(pts[:, 1].max()))
                bw, bh = x2 - x1, y2 - y1
                if bw <= 0 or bh <= 0:
                    continue
                px, py = max(2, int(bw * 0.15)), max(2, int(bh * 0.25))
                roi_small = self._clip(frame, (x1-px, y1-py, bw+2*px, bh+2*py))
                x, y, rw, rh = roi_small
                roi = self._clip(clip_frame, (x*sx, y*sy, rw*sx, rh*sy))
                found.append((roi, text, confidence))
        except Exception as exc:
            self._report("error", f"OCR 偵測失敗：{type(exc).__name__}: {exc}", frame_no)

        found.sort(key=lambda item: item[2], reverse=True)
        unique = []
        for item in found:
            if not any(self._iou(item[0], old[0]) > 0.82 for old in unique):
                unique.append(item)
        elapsed = time.monotonic() - started
        if report:
            self._report("ocr_done", f"OCR 完成，耗時 {elapsed:.2f} 秒；符合目標 {len(unique)} 個",
                         frame_no, elapsed_seconds=elapsed, result_count=len(raw_texts),
                         target_count=len(unique), target_text=self.target_text,
                         ocr_results=raw_texts[:20])
        return unique, elapsed

    @staticmethod
    def _appearance_signature(frame, roi):
        """A cheap visual fingerprint. It is supporting evidence, never identity proof."""
        x, y, w, h = [int(v) for v in roi]
        fh, fw = frame.shape[:2]
        x, y = max(0, min(x, fw-1)), max(0, min(y, fh-1))
        w, h = max(1, min(w, fw-x)), max(1, min(h, fh-y))
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (32, 16), interpolation=cv2.INTER_AREA)
        # Normalize brightness to reduce sensitivity to mild lighting changes.
        gray = cv2.equalizeHist(gray)
        return gray

    @staticmethod
    def _appearance_similarity(a, b):
        if a is None or b is None or a.shape != b.shape:
            return 0.0
        diff = float(np.mean(cv2.absdiff(a, b))) / 255.0
        return max(0.0, min(1.0, 1.0 - diff))

    def _new_track(self, frame, detection, track_id, frame_no=0):
        roi, text, confidence = detection
        tracker = self._create_tracker()
        tracker.init(frame, roi)
        center = self._center(roi)
        return {"id": track_id, "tracker": tracker, "roi": roi, "lost": 0,
                "text": text, "confidence": confidence, "state": "OCR_CONFIRMED",
                "appearance": self._appearance_signature(frame, roi),
                "center": center, "speed": 0.0, "last_seen_frame": int(frame_no)}

    def _associate_detections(self, frame, detections, tracks, frame_no):
        """Conservative one-to-one matching; never attach a distant new detection by nearest-neighbour alone."""
        pairs = []
        for tid, track in tracks.items():
            if track["lost"] > self.MAX_PREDICTED_GAP:
                continue
            old_roi = track["roi"]
            old_center = self._center(old_roi)
            diag = max(20.0, math.hypot(old_roi[2], old_roi[3]))
            age = max(1, int(frame_no) - int(track.get("last_seen_frame", frame_no-1)))
            gate = max(45.0, diag * 1.25 + float(track.get("speed", 0.0)) * min(age, 8) * 1.5)
            for di, det in enumerate(detections):
                roi = det[0]
                center = self._center(roi)
                dist = math.hypot(center[0]-old_center[0], center[1]-old_center[1])
                if dist > gate:
                    continue
                iou = self._iou(old_roi, roi)
                sig = self._appearance_signature(frame, roi)
                appearance = self._appearance_similarity(track.get("appearance"), sig)
                # A plausible position match is required; appearance alone cannot reconnect a target.
                if iou < 0.02 and dist > max(24.0, diag * 0.55) and appearance < 0.48:
                    continue
                proximity = max(0.0, 1.0 - dist / max(1.0, gate))
                score = 0.40 * proximity + 0.35 * iou + 0.25 * appearance
                if score >= self.MATCH_MIN_SCORE:
                    pairs.append((score, tid, di))
        pairs.sort(reverse=True)
        used_tracks, used_detections, matches = set(), set(), []
        for score, tid, di in pairs:
            if tid in used_tracks or di in used_detections:
                continue
            used_tracks.add(tid); used_detections.add(di)
            matches.append((tid, di, score))
        return matches, used_detections

    def _draw_monitor(self, frame, frame_no, rois, mode):
        """Green=OCR confirmed, cyan=per-frame tracker, yellow=lost/rejected; only eligible boxes are mosaiced."""
        shown = frame.copy()
        color_by_state = {"OCR_CONFIRMED": (0, 220, 0), "TRACKING": (255, 210, 0),
                          "LOST": (0, 190, 255), "JUMP_REJECTED": (0, 0, 255)}
        for item in rois:
            if len(item) == 2:
                track_id, roi = item; state = "TRACKING"; confidence = None
            else:
                track_id, roi, state, confidence = item
            x, y, w, h = [int(v) for v in roi]
            color = color_by_state.get(state, (255, 255, 255))
            cv2.rectangle(shown, (x, y), (x+w, y+h), color, 2)
            label = f"ID {track_id} {state}"
            if confidence is not None:
                label += f" {float(confidence):.2f}"
            cv2.putText(shown, label, (x, max(18, y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
        cv2.putText(shown, f"Frame {frame_no} | {mode} | maskable={sum(1 for r in rois if len(r) >= 3 and r[2] in ('OCR_CONFIRMED','TRACKING'))}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        # Legend is burned into the monitor preview only, not the exported video.
        cv2.putText(shown, "GREEN OCR | CYAN tracked | YELLOW lost | RED motion jump rejected",
                    (12, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        self.monitor_frame_signal.emit(shown, int(frame_no), mode)

    def run(self):
        cap = None
        data = {}
        last_frame = self.start_frame
        success = False
        try:
            self.progress_signal.emit(self.start_frame, "正在載入 OCR 與外觀/運動關聯...")
            self._report("loading", "正在載入 EasyOCR；啟用保守的逐幀追蹤關聯", self.start_frame)
            reader = easyocr.Reader(["ch_tra", "en"], gpu=False)
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError("無法開啟影片檔案。")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("無法讀取起始影格。")

            # We do not force detections onto ByteTrack IDs. A conservative one-to-one
            # association below uses motion gating, overlap and a lightweight appearance signature.

            # Phase 1: fast coarse search. OCR runs on a resized image every few frames,
            # instead of spending a full-resolution OCR pass on every empty frame.
            hit_frame_no = None
            search_frame_no = self.start_frame
            while search_frame_no < self.total_frames:
                if self.isInterruptionRequested():
                    self._report("cancelled", "使用者取消搜尋目標", search_frame_no)
                    return
                if search_frame_no != self.start_frame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, search_frame_no)
                    ret, frame = cap.read()
                if not ret:
                    break
                original_frame = frame
                fh, fw = frame.shape[:2]
                scale = min(1.0, self.SEARCH_MAX_DIM / float(max(fh, fw)))
                if scale < 1.0:
                    search_img = cv2.resize(frame, (max(1, int(round(fw*scale))),
                                                    max(1, int(round(fh*scale)))),
                                            interpolation=cv2.INTER_AREA)
                else:
                    search_img = frame
                detections, _ = self._ocr_all(
                    reader, search_img, search_frame_no,
                    coordinate_frame=original_frame,
                    canvas_size=self.SEARCH_CANVAS_SIZE, report=False)
                if detections:
                    hit_frame_no = search_frame_no
                    self._report("search_hit", f"快速搜尋在第 {hit_frame_no} 幀找到目標，正在高畫質初始化多目標…",
                                 hit_frame_no, target_count=len(detections))
                    break
                self._report("searching", f"快速搜尋中：F{search_frame_no}/{self.total_frames}，尚未找到「{self.target_text}」",
                             search_frame_no, target_text=self.target_text)
                self.progress_signal.emit(search_frame_no,
                    f"快速搜尋 F{search_frame_no}/{self.total_frames}…")
                search_frame_no += self.SEARCH_STEP
                ret = True

            if hit_frame_no is None:
                self._report("not_found", f"從起始影格到影片結尾都找不到目標文字：{self.target_text}",
                             max(self.start_frame, min(search_frame_no, self.total_frames-1)))
                return

            # Phase 2: full-resolution burst scan around the first hit. This is what
            # prevents initialization from stopping after the first detected instance.
            burst_start = max(self.start_frame, hit_frame_no - self.INIT_BURST_BEFORE)
            burst_end = min(self.total_frames - 1, hit_frame_no + self.INIT_BURST_AFTER)
            burst_cache = {}
            burst_frames = {}
            cap.set(cv2.CAP_PROP_POS_FRAMES, burst_start)
            for burst_no in range(burst_start, burst_end + 1):
                if self.isInterruptionRequested():
                    self._report("cancelled", "使用者取消多目標初始化", burst_no)
                    return
                ret, burst_frame = cap.read()
                if not ret:
                    break
                burst_frames[burst_no] = burst_frame
                burst_cache[burst_no], _ = self._ocr_all(
                    reader, burst_frame, burst_no,
                    canvas_size=self.DETECT_CANVAS_SIZE, report=True)

            detected_burst_frames = [n for n, dets in burst_cache.items() if dets]
            if not detected_burst_frames:
                self._report("searching", "快速搜尋命中後，高畫質複查未確認目標；繼續向後搜尋…", hit_frame_no)
                # Avoid treating a low-resolution false positive as a locked target.
                search_frame_no = burst_end + 1
                ret = True
                while search_frame_no < self.total_frames:
                    if self.isInterruptionRequested():
                        self._report("cancelled", "使用者取消搜尋目標", search_frame_no)
                        return
                    cap.set(cv2.CAP_PROP_POS_FRAMES, search_frame_no)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    fh, fw = frame.shape[:2]
                    scale = min(1.0, self.SEARCH_MAX_DIM / float(max(fh, fw)))
                    search_img = cv2.resize(frame, (max(1, int(round(fw*scale))), max(1, int(round(fh*scale)))), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
                    detections, _ = self._ocr_all(reader, search_img, search_frame_no, coordinate_frame=frame, canvas_size=self.SEARCH_CANVAS_SIZE, report=False)
                    if detections:
                        hit_frame_no = search_frame_no
                        burst_start = max(self.start_frame, hit_frame_no-self.INIT_BURST_BEFORE)
                        burst_end = min(self.total_frames-1, hit_frame_no+self.INIT_BURST_AFTER)
                        burst_cache.clear(); burst_frames.clear()
                        cap.set(cv2.CAP_PROP_POS_FRAMES, burst_start)
                        for burst_no in range(burst_start, burst_end+1):
                            ret, burst_frame = cap.read()
                            if not ret: break
                            burst_frames[burst_no] = burst_frame
                            burst_cache[burst_no], _ = self._ocr_all(reader, burst_frame, burst_no, canvas_size=self.DETECT_CANVAS_SIZE, report=True)
                        detected_burst_frames = [n for n, dets in burst_cache.items() if dets]
                        if detected_burst_frames:
                            break
                    self.progress_signal.emit(search_frame_no, f"快速搜尋 F{search_frame_no}/{self.total_frames}…")
                    search_frame_no += self.SEARCH_STEP
                if not detected_burst_frames:
                    self._report("not_found", f"從起始影格到影片結尾都找不到可確認的目標文字：{self.target_text}",
                                 max(self.start_frame, min(search_frame_no, self.total_frames-1)))
                    return

            # Start live trackers on the latest burst frame with confirmed detections.
            # Earlier burst detections are retained for immediate frame-accurate mosaicing.
            initial_frame_no = max(detected_burst_frames)
            frame = burst_frames[initial_frame_no]
            detections = burst_cache[initial_frame_no]
            tracks = {}
            next_track_id = 1
            feature_gate = GlobalFeatureGate() if GlobalFeatureGate is not None else None
            for detection in detections:
                # Every distinct OCR box starts as its own track; no path is inferred between boxes.
                if any(self._iou(detection[0], t["roi"]) >= 0.80 for t in tracks.values()):
                    continue
                tracks[next_track_id] = self._new_track(frame, detection, next_track_id, initial_frame_no)
                if feature_gate is not None:
                    feature_gate.add_template(frame, detection[0])
                next_track_id += 1

            # Save direct OCR boxes for every frame in the initialization burst. This gives
            # the export an immediate mask even when CSRT's first update would be imperfect.
            for burst_no, burst_dets in burst_cache.items():
                data[burst_no] = [tuple(map(int, d[0])) for d in burst_dets]
            cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_no + 1)
            ret = True

            def record_frame(frame_no, current_frame):
                rois = []
                draw = []
                for tid, t in list(tracks.items()):
                    if t["lost"] > self.MAX_PREDICTED_GAP:
                        continue
                    roi = tuple(map(int, self._clip(current_frame, t["roi"])))
                    state = t.get("state", "TRACKING")
                    # Important: only a valid per-frame tracker update or direct OCR confirmation
                    # contributes a mask. A stale box is drawn as LOST for diagnostics, never masked.
                    draw_state = "JUMP_REJECTED" if state == "JUMP_REJECTED" else (state if t["lost"] == 0 else "LOST")
                    draw.append((tid, roi, draw_state, t.get("confidence")))
                    if t["lost"] != 0 or state in ("LOST", "JUMP_REJECTED"):
                        continue
                    if not any(self._iou(roi, old) > 0.88 for old in rois):
                        rois.append(roi)
                # OCR detections in the initialization burst are direct observations and are masked
                # on their own frames only. This does not create any interpolated path.
                for detection in burst_cache.get(frame_no, []):
                    roi = tuple(map(int, self._clip(current_frame, detection[0])))
                    if not any(self._iou(roi, old) > 0.88 for old in rois):
                        rois.append(roi)
                data[frame_no] = rois
                return draw

            data[initial_frame_no] = [tuple(map(int, d[0])) for d in detections]
            initial = [(tid, t["roi"], "OCR_CONFIRMED", t.get("confidence")) for tid, t in tracks.items()]
            self._draw_monitor(frame, initial_frame_no, initial, "OCR INITIALIZATION")
            self._report("locked", f"高畫質初始化完成：第 {initial_frame_no} 幀有 {len(detections)} 個目標區域，建立 {len(tracks)} 個追蹤 ID",
                         initial_frame_no, target_text=self.target_text, target_count=len(detections), tracked_frames=len(burst_cache))
            self.progress_signal.emit(initial_frame_no,
                f"初始化找到 {len(detections)} 個目標，開始追蹤")
            last_frame = initial_frame_no
            stop_reason = "已處理至影片結尾"

            for i in range(initial_frame_no + 1, self.total_frames):
                if self.isInterruptionRequested():
                    stop_reason = "使用者取消追蹤"
                    break
                ret, frame = cap.read()
                if not ret:
                    stop_reason = "影片讀取結束或讀取失敗"
                    break
                last_frame = i

                # Update every frame. Reject implausibly large CSRT jumps instead of
                # letting the box teleport to another similar-looking object.
                for t in tracks.values():
                    # Once a tracker is lost or a jump was rejected, do not let CSRT
                    # silently declare itself recovered. Only a later OCR association may
                    # re-lock the ID; this avoids masking a different object after drift.
                    if t["lost"] > 0:
                        t["lost"] += 1
                        if t.get("state") != "JUMP_REJECTED":
                            t["state"] = "LOST"
                        continue
                    previous_roi = t["roi"]
                    previous_center = self._center(previous_roi)
                    ok, bbox = t["tracker"].update(frame) if t.get("tracker") is not None else (False, previous_roi)
                    if ok and bbox[2] > 1 and bbox[3] > 1:
                        candidate_roi = self._clip(frame, bbox)
                        candidate_center = self._center(candidate_roi)
                        distance = math.hypot(candidate_center[0]-previous_center[0], candidate_center[1]-previous_center[1])
                        diag = max(20.0, math.hypot(previous_roi[2], previous_roi[3]))
                        max_step = max(35.0, diag * self.MOTION_JUMP_DIAGONALS)
                        if distance <= max_step:
                            t["speed"] = 0.70 * float(t.get("speed", 0.0)) + 0.30 * distance
                            t["roi"] = candidate_roi
                            t["center"] = candidate_center
                            t["lost"] = 0
                            t["state"] = "TRACKING"
                        else:
                            t["lost"] += 1
                            t["state"] = "JUMP_REJECTED"
                    else:
                        t["lost"] += 1
                        t["state"] = "LOST"

                active_tracks = [t for t in tracks.values() if t["lost"] <= self.MAX_PREDICTED_GAP]
                adaptive_interval = self.ADAPTIVE_OCR_INTERVAL if any(t["lost"] > 0 for t in active_tracks) else self.OCR_INTERVAL
                should_ocr = (i in burst_cache or
                              (i - initial_frame_no) % adaptive_interval == 0 or
                              not active_tracks)
                # The gate is deliberately used only during recovery, when no track has
                # a valid current-frame observation. Normal periodic OCR remains unchanged.
                recovery_mode = not any(t["lost"] == 0 for t in active_tracks)
                if (should_ocr and recovery_mode and i not in burst_cache and
                        feature_gate is not None):
                    decision = feature_gate.should_run_ocr(frame)
                    should_ocr = decision.should_run_ocr
                    if not should_ocr and decision.score is not None:
                        # Low-cost diagnostic event; this does not alter tracking or masks.
                        self._report("feature_gate_skip",
                                     f"全域特徵門控略過本次 OCR（相似度 {decision.score:.3f}）",
                                     i, feature_score=decision.score,
                                     gate_reason=decision.reason,
                                     gate_skipped=feature_gate.skip_count)
                elapsed = 0.0
                if should_ocr:
                    if i in burst_cache:
                        detections = burst_cache[i]
                    else:
                        detections, elapsed = self._ocr_all(reader, frame, i)
                        if feature_gate is not None and recovery_mode:
                            feature_gate.mark_ocr_run()
                    matches, used_detection_indices = self._associate_detections(frame, detections, tracks, i)
                    for tid, di, match_score in matches:
                        detection = detections[di]
                        roi, text, conf = detection
                        t = tracks[tid]
                        t.update({"roi": roi, "lost": 0, "text": text, "confidence": conf,
                                  "state": "OCR_CONFIRMED", "last_seen_frame": i,
                                  "center": self._center(roi)})
                        if feature_gate is not None:
                            feature_gate.add_template(frame, roi)
                        # Keep the original appearance template stable; do not learn from uncertain matches.
                        t["tracker"] = self._create_tracker()
                        t["tracker"].init(frame, roi)

                    # Unmatched detections start new IDs instead of being attached to the nearest
                    # existing track by position alone. This prevents a far-away duplicate from
                    # being treated as the same moving object.
                    for di, detection in enumerate(detections):
                        if di in used_detection_indices:
                            continue
                        if any(self._iou(detection[0], t["roi"]) > 0.82 for t in tracks.values() if t["lost"] == 0):
                            continue
                        tracks[next_track_id] = self._new_track(frame, detection, next_track_id, i)
                        if feature_gate is not None:
                            feature_gate.add_template(frame, detection[0])
                        next_track_id += 1

                    # Keep briefly lost tracks available for conservative re-identification,
                    # but never use their stale box as a mosaic ROI.
                    tracks = {tid: t for tid, t in tracks.items() if t["lost"] <= self.MAX_PREDICTED_GAP}
                    self._report("tracking", f"外觀/運動關聯完成：OCR 偵測 {len(detections)} 個，追蹤 ID {len(tracks)} 個",
                                 i, target_count=len(tracks), elapsed_seconds=elapsed,
                                 lost_count=sum(t["lost"] > 0 for t in tracks.values()),
                                 patience=self.MAX_PREDICTED_GAP, detection_count=len(detections))

                rois_with_ids = record_frame(i, frame)
                if i % self.MONITOR_INTERVAL == 0:
                    mode = "OCR CONFIRMED + CSRT" if any(t["lost"] == 0 for t in tracks.values()) else "SEARCHING / LOST"
                    self._draw_monitor(frame, i, rois_with_ids, mode)
                    self.progress_signal.emit(i, f"追蹤 F{i}/{self.total_frames}，可遮罩框 {len(data.get(i, []))}")

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
