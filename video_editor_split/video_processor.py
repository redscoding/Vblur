"""Multi-target tracking + sparse full-frame/local RapidOCR verification + mosaic export.

Requires: opencv-contrib-python, rapidocr-onnxruntime, PyQt5. EasyOCR/PyTorch is not used.
tracking_data format: {absolute_frame_index: [(x, y, w, h), ...]}
"""
import math
import time
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


class IntuitionisticFuzzyDecision:
    """Small IFS evidence combiner. Scores are heuristics and must be calibrated per dataset."""

    @staticmethod
    def evaluate(ocr_confidence, motion_score, appearance_score, iou_score):
        values = [float(np.clip(v, 0.0, 1.0)) for v in
                  (ocr_confidence, motion_score, appearance_score, iou_score)]
        ocr, motion, appearance, iou = values
        support = 0.50 * ocr + 0.22 * motion + 0.18 * appearance + 0.10 * iou
        opposition = 0.38 * (1.0 - motion) + 0.30 * (1.0 - appearance) + 0.32 * (1.0 - ocr)
        # Construct an intuitionistic fuzzy pair with mu + nu <= 1.
        mu = support * (1.0 - opposition)
        nu = opposition * (1.0 - support)
        mu = float(np.clip(mu, 0.0, 1.0))
        nu = float(np.clip(nu, 0.0, 1.0 - mu))
        pi = float(np.clip(1.0 - mu - nu, 0.0, 1.0))
        if mu >= 0.28 and mu > nu:
            decision = "ACCEPT"
        elif pi >= 0.36 or abs(mu - nu) < 0.08:
            decision = "UNCERTAIN"
        else:
            decision = "REJECT"
        return {"mu": mu, "nu": nu, "pi": pi, "decision": decision}


class SortMotionModel:
    """SORT-style constant-velocity Kalman model for one bounding box.

    State: [center_x, center_y, width, height, vx, vy, vw, vh].
    OCR detections correct the model; accepted CSRT boxes provide intermediate updates.
    """

    def __init__(self, roi):
        x, y, w, h = [float(v) for v in roi]
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.025
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.20
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 1.0
        self.kf.statePost = np.array([
            [x + w / 2.0], [y + h / 2.0], [max(1.0, w)], [max(1.0, h)],
            [0.0], [0.0], [0.0], [0.0]
        ], dtype=np.float32)

    @staticmethod
    def _state_to_box(state):
        cx, cy, w, h = [float(v) for v in state[:4].reshape(-1)]
        w, h = max(2.0, w), max(2.0, h)
        return (int(round(cx - w / 2.0)), int(round(cy - h / 2.0)),
                int(round(w)), int(round(h)))

    def predict(self):
        return self._state_to_box(self.kf.predict())

    def correct(self, roi):
        x, y, w, h = [float(v) for v in roi]
        measurement = np.array([[x + w / 2.0], [y + h / 2.0],
                                [max(1.0, w)], [max(1.0, h)]], dtype=np.float32)
        corrected = self.kf.correct(measurement)
        return self._state_to_box(corrected)


class OCRTrackingThread(QThread):
    """Fast tracking worker: Kalman/SORT-style state, optical-flow updates, OCR on demand.

    initial_roi: optional (x, y, w, h) in the start_frame. If supplied, the worker
    starts from the user's selection and does not need full-frame OCR to initialize.
    tracking_data format remains {absolute_frame_index: [(x, y, w, h), ...]}.
    """
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict, int, bool)
    monitor_frame_signal = pyqtSignal(object, int, str)
    monitor_data_signal = pyqtSignal(dict)

    # OCR runs only on sparse keyframes or after tracking loss, never every frame.
    GLOBAL_OCR_INTERVAL = 45          # full-frame target search about every 1.5 s at 30 FPS
    LOCAL_RECOVERY_INTERVAL = 12      # local ROI retry (~0.4 s at 30 FPS), with backoff
    MAX_OCR_ROI_DIM = 640             # bound local OCR input dimensions for speed
    MAX_GLOBAL_OCR_DIM = 960          # downscale full-frame keyframes to this max dimension
    MIN_FEATURE_POINTS = 4
    MAX_LOST_FRAMES = 3  # 達 3 幀仍失追就停止輸出該軌跡的馬賽克框
    CONSERVATIVE_MASK_FRAMES = 5  # 短暫失追時使用擴大的 Kalman 預測框，降低漏遮罩風險
    RECOVERY_SCALE = 3.0
    DETECT_CANVAS_SIZE = 1280
    MIN_OCR_CONFIDENCE = 0.20
    MONITOR_INTERVAL = 2
    INTERPOLATE_MAX_GAP = 2
    RETRO_OCR_MAX_GAP = 60  # 最多回查 60 幀的缺口；約 2 秒@30FPS，可依效能調整
    RETRO_ROI_EXPAND = 2.2  # 回查時以插值框為中心擴大局部搜尋區域
    USE_MSER_PREFILTER = False  # Avoid MSER false negatives blocking OCR.

    def __init__(self, video_path, start_frame, total_frames, target_text, initial_roi=None):
        super().__init__()
        self.video_path = video_path
        self.start_frame = max(0, int(start_frame))
        self.total_frames = max(0, int(total_frames))
        self.target_text = str(target_text or '')
        self.initial_roi = tuple(initial_roi) if initial_roi is not None else None

    def _report(self, stage, message, frame_no=None, **extra):
        self.monitor_data_signal.emit({'stage': stage, 'message': message,
            'frame': frame_no, 'clock': time.monotonic(), **extra})

    @staticmethod
    def _clip(frame, roi):
        h, w = frame.shape[:2]
        x, y, rw, rh = [int(round(float(v))) for v in roi]
        x = max(0, min(x, w - 1)); y = max(0, min(y, h - 1))
        rw = max(1, min(rw, w - x)); rh = max(1, min(rh, h - y))
        return x, y, rw, rh

    @staticmethod
    def _center(roi):
        return roi[0] + roi[2] / 2.0, roi[1] + roi[3] / 2.0

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _normalize(text):
        return ''.join(ch for ch in str(text).casefold() if ch.isalnum())

    @staticmethod
    def _create_reader():
        """Load RapidOCR only. Avoid importing EasyOCR/PyTorch, which can fail on Windows DLLs."""
        try:
            from rapidocr_onnxruntime import RapidOCR
            return {"backend": "rapidocr-onnxruntime", "engine": RapidOCR()}
        except Exception as exc:
            raise RuntimeError(
                "RapidOCR 無法載入。請在目前執行 main.py 的同一個虛擬環境安裝：\n"
                "python -m pip install --upgrade rapidocr-onnxruntime onnxruntime\n"
                "若目前使用 Python 3.14，建議改用 64-bit Python 3.11/3.12 建立新 venv；"
                "本版本不再載入 EasyOCR/PyTorch 作為備援。\n"
                f"詳細錯誤：{type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _run_ocr_engine(reader, image):
        """Return normalized [(quad, text, confidence)] from RapidOCR."""
        backend = reader.get("backend")
        engine = reader.get("engine")
        if str(backend).startswith("rapidocr"):
            output = engine(image)
            results = output[0] if isinstance(output, tuple) else output
            normalized = []
            for item in results or []:
                if len(item) >= 3:
                    normalized.append((item[0], str(item[1]), float(item[2])))
            return normalized
        results = engine.readtext(image, detail=1, paragraph=False, decoder='greedy',
            min_size=6, text_threshold=0.55, low_text=0.30, link_threshold=0.30,
            canvas_size=640, mag_ratio=1.0, batch_size=1, workers=0)
        return [(item[0], str(item[1]), float(item[2])) for item in (results or []) if len(item) >= 3]

    @staticmethod
    def _gray(frame):
        # Optical-flow points are stored in original-frame coordinates, so the gray
        # image must retain the original resolution (resizing here breaks coordinates).
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _features(gray_frame, roi):
        x, y, w, h = roi
        crop = gray_frame[y:y+h, x:x+w]
        if crop.size == 0:
            return None
        points = cv2.goodFeaturesToTrack(crop, maxCorners=60, qualityLevel=0.01,
                                         minDistance=3, blockSize=3)
        if points is None:
            return None
        points[:, 0, 0] += x
        points[:, 0, 1] += y
        return points.astype(np.float32)

    @staticmethod
    def _appearance(frame, roi):
        x, y, w, h = roi
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 24), interpolation=cv2.INTER_AREA)
        return cv2.equalizeHist(gray)

    @staticmethod
    def _appearance_score(a, b):
        if a is None or b is None or a.shape != b.shape:
            return 0.0
        return float(max(0.0, 1.0 - np.mean(cv2.absdiff(a, b)) / 255.0))

    @staticmethod
    def _estimate_global_motion(previous_gray, current_gray):
        """Estimate dominant frame-to-frame camera/background motion with sparse LK flow.

        This is a diagnostic heuristic, not a calibrated camera-motion estimator.
        Returns median displacement magnitude and median dx/dy in original pixels.
        """
        try:
            h, w = previous_gray.shape[:2]
            scale = min(1.0, 640.0 / max(h, w))
            if scale < 1.0:
                prev = cv2.resize(previous_gray, (max(1, int(w*scale)), max(1, int(h*scale))))
                curr = cv2.resize(current_gray, (max(1, int(w*scale)), max(1, int(h*scale))))
            else:
                prev, curr = previous_gray, current_gray
            points = cv2.goodFeaturesToTrack(prev, maxCorners=250, qualityLevel=0.01,
                                             minDistance=8, blockSize=5)
            if points is None or len(points) < 8:
                # Keep the return contract consistent with the caller: (magnitude, dx, dy, point_count).
                return 0.0, 0.0, 0.0, 0
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                prev, curr, points, None, winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
            if nxt is None or status is None:
                return 0.0, 0.0, 0.0, 0
            good = status.reshape(-1) == 1
            good_count = int(np.count_nonzero(good))
            if good_count < 8:
                return 0.0, 0.0, 0.0, good_count
            delta = (nxt[good] - points[good]).reshape(-1, 2)
            dx, dy = np.median(delta, axis=0) / scale
            mag = float(np.median(np.linalg.norm(delta, axis=1)) / scale)
            return mag, float(dx), float(dy), good_count
        except Exception:
            return 0.0, 0.0, 0.0, 0

    @staticmethod
    def _blur_score(frame, roi):
        """Laplacian variance inside ROI. Low values can indicate blur/low texture."""
        try:
            x, y, w, h = [int(v) for v in roi]
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
                return 0.0
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            return 0.0

    @staticmethod
    def _create_visual_tracker(frame, roi):
        # 建立每個目標獨立的 CSRT tracker；不可用時退回 Kalman + 光流。
        try:
            creator = getattr(cv2, 'TrackerCSRT_create', None)
            if creator is None and hasattr(cv2, 'legacy'):
                creator = getattr(cv2.legacy, 'TrackerCSRT_create', None)
            if creator is None:
                return None
            tracker = creator()
            x, y, w, h = [int(v) for v in roi]
            tracker.init(frame, (x, y, w, h))
            return tracker
        except Exception:
            return None

    def _new_track(self, frame, roi, track_id, frame_no, text='', confidence=1.0):
        roi = self._clip(frame, roi)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        appearance = self._appearance(frame, roi)
        return {'id': track_id, 'roi': roi, 'predicted_roi': roi,
                'kalman': SortMotionModel(roi), 'points': self._features(gray, roi),
                'visual_tracker': self._create_visual_tracker(frame, roi),
                'appearance': appearance, 'anchor_appearance': appearance, 'lost': 0,
                'speed': 0.0, 'text': text, 'confidence': float(confidence),
                'state': 'MANUAL_INIT' if self.initial_roi is not None else 'OCR_CONFIRMED',
                'last_seen_frame': int(frame_no), 'last_ocr_frame': int(frame_no),
                'last_local_search_frame': int(frame_no), 'recovery_attempts': 0,
                'loss_reason': '', 'last_diagnostics': {}}

    @staticmethod
    def _expand_roi(frame, roi, scale):
        x, y, w, h = [float(v) for v in roi]
        cx, cy = x + w / 2, y + h / 2
        nw, nh = max(12, w * scale), max(12, h * scale)
        return OCRTrackingThread._clip(frame, (cx - nw/2, cy - nh/2, nw, nh))

    def _ocr(self, reader, frame, frame_no, roi=None, full_frame=False, robust=False):
        """OCR a local ROI, or a downscaled full frame on sparse recovery keyframes only."""
        started = time.monotonic()
        if full_frame:
            fh, fw = frame.shape[:2]
            x, y, w, h = 0, 0, fw, fh
            crop = frame
            max_dim = self.MAX_GLOBAL_OCR_DIM
        else:
            if roi is None:
                self._report('ocr_skipped', '局部 OCR 缺少 ROI', frame_no)
                return [], 0.0
            x, y, w, h = self._clip(frame, roi)
            crop = frame[y:y+h, x:x+w]
            max_dim = self.MAX_OCR_ROI_DIM
        if crop.size == 0:
            return [], 0.0

        # Keep small text at native size; downscale only oversized inputs.
        ch, cw = crop.shape[:2]
        scale = min(1.0, max_dim / float(max(ch, cw)))
        ocr_crop = crop if scale >= 1.0 else cv2.resize(
            crop, (max(1, int(cw*scale)), max(1, int(ch*scale))), interpolation=cv2.INTER_AREA)
        target = self._normalize(self.target_text)
        found, raw = [], []
        gray = cv2.cvtColor(ocr_crop, cv2.COLOR_BGR2GRAY) if ocr_crop.ndim == 3 else ocr_crop
        variants = [("original", ocr_crop)]
        if robust:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 31, 11)
            variants.extend([("clahe_gray", clahe), ("adaptive_threshold", adaptive),
                             ("inverted_threshold", cv2.bitwise_not(adaptive))])

        for variant_name, variant_image in variants:
            try:
                results = self._run_ocr_engine(reader, variant_image)
            except Exception as exc:
                self._report('error', f'OCR {variant_name} 失敗：{type(exc).__name__}: {exc}', frame_no)
                results = []
            for pts, text, conf in results:
                normalized = self._normalize(text)
                raw.append({'text': text, 'confidence': conf, 'variant': variant_name})
                # If no target text is supplied, don't let OCR create new identities.
                if conf < self.MIN_OCR_CONFIDENCE or not target or target not in normalized:
                    continue
                pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
                x1, y1 = np.floor(pts.min(axis=0)).astype(int)
                x2, y2 = np.ceil(pts.max(axis=0)).astype(int)
                # Map OCR coordinates from resized ROI back to original frame coordinates.
                x1, x2 = int(x1 / scale), int(x2 / scale)
                y1, y2 = int(y1 / scale), int(y2 / scale)
                bw, bh = x2-x1, y2-y1
                if bw <= 0 or bh <= 0:
                    continue
                px, py = max(2, int(bw*.12)), max(2, int(bh*.20))
                box = self._clip(frame, (x+x1-px, y+y1-py, bw+2*px, bh+2*py))
                found.append((box, text, conf))
            if found:
                break  # only pay for preprocessing variants if the original failed

        found.sort(key=lambda d: d[2], reverse=True)
        unique = []
        for item in found:
            if not any(self._iou(item[0], prev[0]) > .82 for prev in unique):
                unique.append(item)
        elapsed = time.monotonic() - started
        mode = 'GLOBAL_KEYFRAME' if full_frame else 'LOCAL_ROI'
        self._report('ocr_done', f'OCR {mode} 完成：{elapsed:.2f}s，目標 {len(unique)} 個',
                     frame_no, elapsed_seconds=elapsed, target_count=len(unique), ocr_results=raw[:20],
                     roi=(x, y, w, h), mode=mode)
        return unique, elapsed

    def _draw_monitor(self, frame, frame_no, tracks, mode):
        shown = frame.copy()
        colors = {'OCR_CONFIRMED': (0, 220, 0), 'MANUAL_INIT': (0, 220, 0),
                  'TRACKING': (255, 210, 0), 'FAST_MOTION': (255, 0, 255),
                  'RECOVERY': (0, 190, 255), 'LOST': (0, 0, 255)}
        for tid, track in tracks.items():
            # Hide stale visualization after three consecutive lost frames; keep the
            # internal track state so local OCR may still recover it later.
            if int(track.get('lost', 0)) > self.MAX_LOST_FRAMES:
                continue
            roi = track.get('roi', (0, 0, 1, 1))
            x, y, w, h = roi
            state = track.get('state', 'TRACKING')
            color = colors.get(state, (255, 255, 255))
            cv2.rectangle(shown, (x, y), (x+w, y+h), color, 2)
            reason = track.get('loss_reason', '')
            label = f'ID {tid} {state} v={track.get("speed", 0):.1f}' + (f' [{reason}]' if reason else '')
            cv2.putText(shown, label, (x, max(18, y-5)), cv2.FONT_HERSHEY_SIMPLEX,
                        .48, color, 2, cv2.LINE_AA)
        cv2.putText(shown, f'Frame {frame_no} | {mode} | tracks={len(tracks)}', (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2, cv2.LINE_AA)
        self.monitor_frame_signal.emit(shown, int(frame_no), mode)

    def _retroactively_verify_gaps(self, cap, history, start_frame, end_frame,
                                   frame_w, frame_h, reader):
        """回溯檢查由 None 標記的缺失影格。

        只在同一追蹤軌跡的前後都有有效框時，才以兩端位置估算局部 OCR ROI。
        缺口上限由 RETRO_OCR_MAX_GAP 控制；回溯成功時以 OCR 實際框覆寫該影格。
        這個函式不會對每一幀都執行 OCR，只處理 history 中明確標成 None 的缺口。
        """
        checked = 0
        matched = 0
        max_gap = max(0, int(self.RETRO_OCR_MAX_GAP))
        if max_gap == 0:
            self._report('retroactive_check', '回溯已停用：RETRO_OCR_MAX_GAP=0', end_frame,
                         retro_checked=0, retro_matched=0, max_gap=max_gap)
            return checked

        target = self._normalize(self.target_text)
        if not target:
            return checked

        for track_id, track_history in history.items():
            valid_frames = sorted(f for f, roi in track_history.items() if roi is not None)
            for left, right in zip(valid_frames, valid_frames[1:]):
                gap = right - left - 1
                if gap <= 0 or gap > max_gap:
                    continue
                a, b = track_history[left], track_history[right]
                missing = [f for f in range(left + 1, right)
                           if track_history.get(f) is None]
                if not missing:
                    continue
                self._report('retroactive_check',
                    f'回溯軌跡 ID {track_id}：F{left}–F{right}，缺失 {len(missing)} 幀',
                    right, track_id=track_id, gap=gap, retro_checked=checked,
                    retro_matched=matched, max_gap=max_gap)
                for frame_no in missing:
                    ratio = (frame_no - left) / float(right - left)
                    predicted = tuple(int(round(a[i] * (1-ratio) + b[i] * ratio)) for i in range(4))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    pred = self._clip(frame, predicted)
                    search_roi = self._expand_roi(frame, pred, self.RETRO_ROI_EXPAND)
                    detections, _ = self._ocr(reader, frame, frame_no,
                                              roi=search_roi, full_frame=False, robust=True)
                    checked += 1
                    candidates = [d for d in detections
                                  if target in self._normalize(d[1])]
                    if candidates:
                        # 選擇最接近前後座標插值位置的同文字候選框。
                        candidates.sort(key=lambda d: math.hypot(
                            self._center(d[0])[0] - self._center(pred)[0],
                            self._center(d[0])[1] - self._center(pred)[1]))
                        track_history[frame_no] = tuple(map(int, self._clip(frame, candidates[0][0])))
                        matched += 1
                        self._report('retroactive_match',
                            f'回溯補到目標：F{frame_no}，ID {track_id}', frame_no,
                            track_id=track_id, roi=track_history[frame_no],
                            retro_checked=checked, retro_matched=matched)

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(end_frame))
        self._report('retroactive_check',
            f'回溯完成：檢查 {checked} 幀，OCR 確認並修正 {matched} 幀',
            end_frame, retro_checked=checked, retro_matched=matched, max_gap=max_gap)
        return checked

    def _retroactively_search_before_first_detection(self, cap, history, start_frame,
                                                     end_frame, frame_w, frame_h, reader):
        """對每條軌跡的「第一次已知位置」之前做反向追蹤 + 局部 OCR。

        這是補救固定頻率全畫面 OCR 漏掉初始出現影格的第二階段。
        由最早的可靠框向前逐幀追蹤特徵點，預測 ROI，再用 OCR 確認目標。
        只有 OCR 確認目標的影格才寫入 history，避免把不確定的預測框當成真實目標。
        注意：若目標在這段期間有鏡頭切換、嚴重遮擋或離開畫面，仍無法保證找回。
        """
        target = self._normalize(self.target_text)
        max_back = max(0, int(self.RETRO_OCR_MAX_GAP))
        checked = matched = tracks_checked = 0
        if reader is None or not target or max_back <= 0:
            return {"checked": 0, "matched": 0, "tracks": 0}

        for track_id, track_history in history.items():
            known = sorted(f for f, roi in track_history.items() if roi is not None)
            if not known:
                continue
            first_frame = known[0]
            scan_start = max(int(start_frame), first_frame - max_back)
            if first_frame <= scan_start:
                continue
            first_roi = tuple(map(int, track_history[first_frame]))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(first_frame))
            ok, anchor_frame = cap.read()
            if not ok or anchor_frame is None:
                continue
            current_frame = anchor_frame
            current_roi = self._clip(current_frame, first_roi)
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            tracks_checked += 1
            self._report('retroactive_first_detection',
                f'首次偵測回溯：ID {track_id}，從 F{first_frame} 向前搜尋至 F{scan_start}',
                first_frame, track_id=track_id, from_frame=first_frame,
                to_frame=scan_start, max_back=max_back)

            # 向前逐幀移動；每次從目前框內取特徵點，反向追到前一幀。
            for frame_no in range(first_frame - 1, scan_start - 1, -1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
                ok, prev_frame = cap.read()
                if not ok or prev_frame is None:
                    break
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                x, y, w, h = self._clip(current_frame, current_roi)
                mask = np.zeros(current_gray.shape, dtype=np.uint8)
                mask[y:y+h, x:x+w] = 255
                points = cv2.goodFeaturesToTrack(current_gray, maxCorners=50,
                    qualityLevel=0.01, minDistance=3, mask=mask, blockSize=5)
                predicted = current_roi
                motion_ok = False
                if points is not None and len(points) >= 3:
                    moved, status, _ = cv2.calcOpticalFlowPyrLK(
                        current_gray, prev_gray, points, None,
                        winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
                    if moved is not None and status is not None:
                        good = status.reshape(-1).astype(bool)
                        old_pts = points.reshape(-1, 2)[good]
                        new_pts = moved.reshape(-1, 2)[good]
                        if len(new_pts) >= 3:
                            delta = np.median(new_pts - old_pts, axis=0)
                            # Reject implausible jumps that likely indicate bad optical flow.
                            if np.isfinite(delta).all() and np.linalg.norm(delta) < max(40.0, 2.5 * max(w, h)):
                                predicted = self._clip(prev_frame, (current_roi[0] + delta[0],
                                    current_roi[1] + delta[1], current_roi[2], current_roi[3]))
                                motion_ok = True
                # 若光流失敗，仍嘗試在原位置附近搜尋；只以 OCR 確認結果填入 history。
                search_roi = self._expand_roi(prev_frame, predicted,
                    self.RETRO_ROI_EXPAND * (1.0 if motion_ok else 1.35))
                detections, _ = self._ocr(reader, prev_frame, frame_no,
                    roi=search_roi, full_frame=False, robust=True)
                checked += 1
                candidates = [d for d in detections if target in self._normalize(d[1])]
                if candidates:
                    pcx, pcy = self._center(predicted)
                    candidates.sort(key=lambda d: math.hypot(
                        self._center(d[0])[0] - pcx, self._center(d[0])[1] - pcy))
                    measured = tuple(map(int, self._clip(prev_frame, candidates[0][0])))
                    track_history[frame_no] = measured
                    matched += 1
                    current_roi = measured
                    self._report('retroactive_first_match',
                        f'首次偵測回溯補到目標：F{frame_no}，ID {track_id}', frame_no,
                        track_id=track_id, roi=measured, retro_checked=checked,
                        retro_matched=matched)
                else:
                    # 保留光流預測供下一幀追蹤，但不把未經 OCR 確認的框寫入馬賽克資料。
                    current_roi = predicted
                current_frame = prev_frame
                current_gray = prev_gray

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(end_frame))
        self._report('retroactive_first_detection',
            f'首次偵測前回溯完成：檢查 {checked} 幀，OCR 確認補到 {matched} 幀，軌跡 {tracks_checked} 條',
            end_frame, retro_checked=checked, retro_matched=matched,
            tracks_checked=tracks_checked, max_back=max_back)
        return {"checked": checked, "matched": matched, "tracks": tracks_checked}

    @staticmethod
    def _interpolate_short_gaps(history, max_gap=2):
        for track_history in history.values():
            frames = sorted(k for k, v in track_history.items() if v is not None)
            for left, right in zip(frames, frames[1:]):
                gap = right-left-1
                if 0 < gap <= max_gap:
                    a, b = track_history[left], track_history[right]
                    for frame_no in range(left+1, right):
                        ratio = (frame_no-left)/(right-left)
                        track_history[frame_no] = tuple(int(round(a[i]*(1-ratio)+b[i]*ratio)) for i in range(4))

    def run(self):
        cap = None
        data = {}
        last_frame = self.start_frame
        success = False
        try:
            self.progress_signal.emit(self.start_frame, '載入影片與追蹤模組...')
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError('無法開啟影片檔案。')
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if not math.isfinite(source_fps) or source_fps <= 0:
                source_fps = 0.0
            self._report('video_info', f'影片資訊：{source_fps:.2f} FPS，{frame_count} frames',
                         self.start_frame, fps=source_fps, frame_count=frame_count)
            end_frame = min(self.total_frames, frame_count) if self.total_frames > 0 else frame_count
            if end_frame <= self.start_frame:
                raise RuntimeError('起始影格超出影片範圍。')
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            ret, first_frame = cap.read()
            if not ret:
                raise RuntimeError('無法讀取起始影格。')

            # Strict ROI-only design: the user must draw the target box; no full-frame OCR.
            tracks = {}
            next_id = 1
            hit_frame = self.start_frame
            reader = None
            if self.target_text.strip():
                self.progress_signal.emit(hit_frame, '正在載入 RapidOCR（僅 ONNX Runtime，不載入 PyTorch）...')
                reader = self._create_reader()
                self._report('ocr_engine', f"OCR 引擎已載入：{reader.get('backend', 'unknown')}", hit_frame)
            if self.initial_roi is None:
                raise RuntimeError('此效能版禁止全畫面 OCR。請先在預覽畫面框選單一目標，再開始追蹤。')
            roi = self._clip(first_frame, self.initial_roi)
            if roi[2] < 3 or roi[3] < 3:
                raise RuntimeError('使用者選取的 ROI 太小或超出畫面。')
            tracks[next_id] = self._new_track(first_frame, roi, next_id, hit_frame,
                                              self.target_text, 1.0)
            self._report('locked', f'使用人工選取 ROI 初始化追蹤：{roi}', hit_frame,
                         roi=roi, mode='INIT')
            if self.target_text.strip():
                self.progress_signal.emit(hit_frame, '已框選目標；RapidOCR 會定期掃描縮小後的關鍵影格，失追時另做局部恢復...')

            frame_h, frame_w = first_frame.shape[:2]

            # Start output history at the initialization frame.
            history = {tid: {hit_frame: tuple(t['roi'])} for tid, t in tracks.items()}
            data[hit_frame] = [tuple(t['roi']) for t in tracks.values()]
            self._draw_monitor(first_frame, hit_frame, tracks, 'TRACK MODE')
            previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            cached_global_motion = cached_camera_dx = cached_camera_dy = 0.0
            cached_global_points = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, hit_frame + 1)

            for frame_no in range(hit_frame+1, end_frame):
                if self.isInterruptionRequested():
                    self._report('cancelled', '使用者取消追蹤', frame_no, mode='TRACK MODE')
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                last_frame = frame_no
                current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_h, frame_w = frame.shape[:2]
                detections = []
                need_local_recovery = []
                # Per-frame flag used by monitoring/reporting below.
                ran_ocr = False
                # Global motion is diagnostic only, so estimate it every 3 frames rather than
                # spending extra CPU on full-frame feature detection on every frame.
                if frame_no % 10 == 0:
                    (cached_global_motion, cached_camera_dx, cached_camera_dy,
                     cached_global_points) = self._estimate_global_motion(previous_gray, current_gray)
                global_motion, camera_dx, camera_dy, global_points = (
                    cached_global_motion, cached_camera_dx, cached_camera_dy, cached_global_points)
                # Heuristic only: large coherent image motion may indicate camera movement.
                camera_moving = global_motion >= 6.0 and global_points >= 8

                # SORT-style prediction for each track. Optical flow provides a cheap measurement.
                for tid, track in tracks.items():
                    if track['lost'] > 0:
                        track['lost'] += 1
                    try:
                        predicted = self._clip(frame, track['kalman'].predict())
                    except Exception:
                        predicted = self._clip(frame, track['roi'])
                    track['predicted_roi'] = predicted
                    previous_roi = tuple(track['roi'])
                    points = track.get('points')
                    flow_ok = False
                    candidate = predicted
                    flow_count = 0
                    csrt_flow_delta = None
                    # CSRT 每個影格更新一次，避免只靠光流造成框逐漸漂移。
                    visual_tracker = track.get('visual_tracker')
                    csrt_ok = False
                    csrt_candidate = predicted
                    if visual_tracker is not None:
                        try:
                            ok_csrt, box_csrt = visual_tracker.update(frame)
                            if ok_csrt:
                                csrt_candidate = self._clip(frame, box_csrt)
                                if csrt_candidate[2] >= 3 and csrt_candidate[3] >= 3:
                                    candidate = csrt_candidate
                                    flow_ok = True
                                    csrt_ok = True
                        except Exception:
                            track['visual_tracker'] = None
                    if points is not None and len(points) >= self.MIN_FEATURE_POINTS:
                        try:
                            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                                previous_gray, current_gray, points, None,
                                winSize=(21,21), maxLevel=3,
                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
                            if new_points is not None and status is not None:
                                good_old = points[status.reshape(-1) == 1]
                                good_new = new_points[status.reshape(-1) == 1]
                                flow_count = len(good_new)
                                if flow_count >= self.MIN_FEATURE_POINTS:
                                    delta = np.median(good_new - good_old, axis=0).reshape(-1)
                                    dx, dy = float(delta[0]), float(delta[1])
                                    flow_candidate = self._clip(frame, (previous_roi[0]+dx, previous_roi[1]+dy,
                                                                          previous_roi[2], previous_roi[3]))
                                    distance = math.hypot(dx, dy)
                                    track['speed'] = .7*float(track['speed']) + .3*distance
                                    if csrt_ok:
                                        # 兩種視覺訊號接近時融合；差異大時以 CSRT 為主並交由外觀檢查把關。
                                        cc, fc = self._center(csrt_candidate), self._center(flow_candidate)
                                        csrt_flow_delta = math.hypot(cc[0]-fc[0], cc[1]-fc[1])
                                        disagreement_gate = max(25.0, math.hypot(previous_roi[2], previous_roi[3]) * 0.5)
                                        if csrt_flow_delta <= disagreement_gate:
                                            candidate = tuple(int(round((a+b)/2.0)) for a,b in zip(csrt_candidate, flow_candidate))
                                        else:
                                            # When trackers disagree, prefer the candidate closer to
                                            # Kalman's prediction; if both are implausible, reject it.
                                            pred_center = self._center(predicted)
                                            csrt_center = self._center(csrt_candidate)
                                            flow_center = self._center(flow_candidate)
                                            d_csrt = math.hypot(csrt_center[0]-pred_center[0], csrt_center[1]-pred_center[1])
                                            d_flow = math.hypot(flow_center[0]-pred_center[0], flow_center[1]-pred_center[1])
                                            candidate = csrt_candidate if d_csrt <= d_flow else flow_candidate
                                            if min(d_csrt, d_flow) > max(32.0, math.hypot(previous_roi[2], previous_roi[3]) * 1.25):
                                                flow_ok = False
                                                track['loss_reason'] = 'tracker_disagreement'
                                    else:
                                        candidate = flow_candidate
                                        flow_ok = True
                                    track['points'] = good_new.reshape(-1,1,2).astype(np.float32)
                        except cv2.error:
                            flow_ok = csrt_ok
                    diag = max(20.0, math.hypot(previous_roi[2], previous_roi[3]))
                    # High-speed movement is allowed; the gate grows with measured velocity.
                    max_move = max(45.0, diag*2.5, float(track['speed'])*2.2 + diag)
                    old_c, new_c = self._center(previous_roi), self._center(candidate)
                    pred_c = self._center(predicted)
                    move = math.hypot(new_c[0]-old_c[0], new_c[1]-old_c[1])
                    pred_distance = math.hypot(new_c[0]-pred_c[0], new_c[1]-pred_c[1])
                    # Repeated identical objects can attract CSRT/KLT to the wrong instance.
                    # Gate candidates against the motion prediction, not only the previous box.
                    prediction_gate = max(32.0, diag * 1.25,
                                          float(track['speed']) * 2.5 + diag * 0.5)
                    if flow_ok and pred_distance > prediction_gate:
                        flow_ok = False
                        track['loss_reason'] = 'prediction_gate_rejected_possible_identity_switch'
                    # 固定錨點避免錯誤框逐幀污染外觀模板。
                    anchor = track.get('anchor_appearance')
                    candidate_appearance = self._appearance(frame, candidate) if flow_ok else None
                    anchor_score = self._appearance_score(anchor, candidate_appearance)
                    blur_score = self._blur_score(frame, candidate)
                    diag_info = {
                        'fps': source_fps, 'target_displacement_px_frame': round(move, 2),
                        'target_speed_px_sec': round(move * source_fps, 2) if source_fps else None,
                        'global_motion_px_frame': round(global_motion, 2),
                        'global_motion_dx': round(camera_dx, 2), 'global_motion_dy': round(camera_dy, 2),
                        'camera_motion_suspected': bool(camera_moving),
                        'flow_points': int(flow_count), 'csrt_ok': bool(csrt_ok),
                        'csrt_flow_disagreement_px': round(csrt_flow_delta, 2) if csrt_flow_delta is not None else None,
                        'blur_laplacian_variance': round(blur_score, 2),
                        'appearance_score': round(anchor_score, 3),
                    }
                    track['last_diagnostics'] = diag_info
                    if flow_ok and anchor is not None and anchor_score < 0.24:
                        flow_ok = False
                        track['loss_reason'] = 'appearance_mismatch'
                    if flow_ok and move > max_move:
                        track['loss_reason'] = 'large_displacement_gate'
                    if not csrt_ok and flow_count < self.MIN_FEATURE_POINTS:
                        track['loss_reason'] = 'few_features_or_motion_blur'
                    if csrt_flow_delta is not None and csrt_flow_delta > max(25.0, diag * 0.5):
                        track['loss_reason'] = 'tracker_disagreement'
                    if camera_moving and move < global_motion * 0.35:
                        track['loss_reason'] = 'possible_camera_motion'
                    if source_fps and source_fps < 20.0 and move > diag * 0.8 and (flow_count < 8 or not csrt_ok):
                        track['loss_reason'] = 'low_source_fps_plus_fast_motion'
                    elif move > diag * 0.8 and (flow_count < 8 or not csrt_ok):
                        track['loss_reason'] = 'possible_fast_target_motion'
                    rx, ry, rw, rh = candidate
                    if rx <= 1 or ry <= 1 or rx + rw >= frame_w - 1 or ry + rh >= frame_h - 1:
                        track['loss_reason'] = 'roi_near_frame_edge_or_target_leaving_view'
                    if blur_score < 35.0 and not csrt_ok:
                        track['loss_reason'] = 'possible_motion_blur_or_low_texture'
                    if flow_ok and move <= max_move:
                        track['roi'] = candidate
                        try:
                            track['kalman'].correct(candidate)
                        except Exception:
                            track['kalman'] = SortMotionModel(candidate)
                        track['lost'] = 0
                        track['loss_reason'] = ''
                        track['state'] = 'FAST_MOTION' if move > diag*.8 else 'TRACKING'
                        track['appearance'] = self._appearance(frame, candidate)
                    else:
                        track['lost'] = max(1, track['lost'])
                        track['state'] = 'LOST' if track['lost'] >= 3 else 'RECOVERY'
                        reason = track.get('loss_reason') or 'tracker_measurement_failed'
                        if track['lost'] == 1 or track['lost'] % 10 == 0:
                            self._report('tracking_lost',
                                f'ID {tid} 追蹤失敗，初步原因={reason}；位移={move:.1f}px/frame，'
                                f'光流點={flow_count}，模糊分數={blur_score:.1f}，全畫面位移={global_motion:.1f}px/frame',
                                frame_no, target_text=self.target_text, confidence=anchor_score,
                                bbox=candidate, lost_count=track['lost'], patience=self.MAX_LOST_FRAMES,
                                diagnosis=reason, diagnostics=diag_info, fps=source_fps,
                                global_motion=global_motion, flow_points=flow_count,
                                blur_score=blur_score, csrt_ok=csrt_ok,
                                csrt_flow_disagreement=csrt_flow_delta)
                        # Do not keep feeding a stale ROI back into CSRT: it can lock onto
                        # another visually similar target and leave a misleading tracking box.
                        track['visual_tracker'] = None
                        track['points'] = None
                        # Move the displayed recovery box with the Kalman prediction instead
                        # of leaving a blue/stale box on the previously rejected object.
                        track['roi'] = predicted
                        need_local_recovery.append(tid)

                    # Replenish feature points in the current ROI to reduce drift / point depletion.
                    if track['lost'] == 0 and (track.get('points') is None or len(track['points']) < 12 or frame_no % 5 == 0):
                        track['points'] = self._features(current_gray, track['roi'])

                # OCR scheduling: periodic downscaled full-frame keyframes find targets that
                # enter elsewhere in the image; local recovery handles fast short-term reacquisition.
                should_global_ocr = (
                    bool(self.target_text.strip())
                    and (frame_no - hit_frame) % self.GLOBAL_OCR_INTERVAL == 0
                )
                recovery_due = []
                for tid in need_local_recovery:
                    tr = tracks[tid]
                    attempts = int(tr.get('recovery_attempts', 0))
                    retry_interval = min(120, self.LOCAL_RECOVERY_INTERVAL * (2 ** min(attempts // 2, 3)))
                    if frame_no - int(tr.get('last_local_search_frame', -10**9)) >= retry_interval:
                        recovery_due.append(tid)

                if reader is not None:
                    # Full-frame OCR only at sparse keyframes (default every 45 frames),
                    # resized to at most 960 px. This is necessary to find a target that
                    # appears outside the selected object's local search area.
                    if should_global_ocr:
                        global_detections, _ = self._ocr(
                            reader, frame, frame_no, full_frame=True, robust=False)
                        detections.extend(global_detections)
                        ran_ocr = True
                        # If the requested text is confidently found far away from an
                        # active visual track, do not let a stale CSRT box remain blue
                        # on a visually similar but text-mismatched object. The OCR hit
                        # will be associated or become a fresh track below.
                        if global_detections:
                            for tid, tr in tracks.items():
                                if tr.get('lost', 0) > 0:
                                    continue
                                tracked_box = tr.get('predicted_roi') or tr['roi']
                                tc = self._center(tracked_box)
                                td = math.hypot(tracked_box[2], tracked_box[3])
                                far_match = True
                                for det_box, _det_text, _det_conf in global_detections:
                                    dc = self._center(det_box)
                                    if math.hypot(tc[0]-dc[0], tc[1]-dc[1]) <= max(100.0, td * 1.5):
                                        far_match = False
                                        break
                                if far_match:
                                    tr['lost'] = 1
                                    tr['state'] = 'RECOVERY'
                                    tr['loss_reason'] = 'global_ocr_found_target_elsewhere'
                                    tr['visual_tracker'] = None
                                    tr['points'] = None
                                    tr['roi'] = tracked_box
                                    if tid not in need_local_recovery:
                                        need_local_recovery.append(tid)
                    # Avoid duplicate local OCR on a frame that already had a global scan.
                    if recovery_due and not should_global_ocr:
                        for tid in recovery_due:
                            track = tracks[tid]
                            pred = track.get('predicted_roi') or track['roi']
                            local_roi = self._expand_roi(frame, pred, self.RECOVERY_SCALE)
                            track['last_local_search_frame'] = frame_no
                            track['recovery_attempts'] = int(track.get('recovery_attempts', 0)) + 1
                            robust_ocr = track['recovery_attempts'] % 3 == 0
                            local, _ = self._ocr(reader, frame, frame_no, roi=local_roi, robust=robust_ocr)
                            detections.extend(local)
                            ran_ocr = True

                # Hungarian matching (SORT association) between OCR boxes and predicted tracks.
                if detections and tracks:
                    tids = list(tracks.keys())
                    cost = np.full((len(tids), len(detections)), 1e6, dtype=np.float64)
                    for i, tid in enumerate(tids):
                        tr = tracks[tid]
                        pred = tr.get('predicted_roi') or tr['roi']
                        pc = self._center(pred)
                        gate = max(35.0, math.hypot(pred[2], pred[3])*1.75 + tr['speed']*1.5)
                        for j, det in enumerate(detections):
                            box, text, conf = det
                            dc = self._center(box)
                            dist = math.hypot(pc[0]-dc[0], pc[1]-dc[1])
                            iou = self._iou(pred, box)
                            app = self._appearance_score(tr.get('appearance'), self._appearance(frame, box))
                            text_match = (not self.target_text.strip() or
                                self._normalize(self.target_text) in self._normalize(text))
                            # Do not associate a distant same-text detection solely because
                            # the text looks alike. It must be close to the motion prediction
                            # or have meaningful box overlap.
                            if (dist <= gate and (iou >= .02 or dist <= gate * .70 or app >= .72)) or iou >= .15:
                                cost[i,j] = .55*(1-iou) + .30*min(1.0, dist/gate) + .10*(1-app) + .05*(1-conf)
                                if not text_match:
                                    cost[i,j] += .5
                    if linear_sum_assignment is not None:
                        rows, cols = linear_sum_assignment(cost)
                        pairs = list(zip(rows.tolist(), cols.tolist()))
                    else:
                        pairs = sorted(((cost[i,j],i,j) for i in range(cost.shape[0]) for j in range(cost.shape[1])))
                        used_i, used_j, greedy = set(), set(), []
                        for c,i,j in pairs:
                            if c < .82 and i not in used_i and j not in used_j:
                                greedy.append((i,j)); used_i.add(i); used_j.add(j)
                        pairs = greedy
                    used_dets = set()
                    for i,j in pairs:
                        if cost[i,j] >= .82:
                            continue
                        tid = tids[i]
                        box, text, conf = detections[j]
                        tr = tracks[tid]
                        old_center, new_center = self._center(tr['roi']), self._center(box)
                        moved = math.hypot(new_center[0]-old_center[0], new_center[1]-old_center[1])
                        corrected_box = self._clip(frame, box)
                        tr.update({'roi': corrected_box, 'predicted_roi': corrected_box,
                                   'visual_tracker': self._create_visual_tracker(frame, corrected_box),
                                   'lost': 0, 'state': 'OCR_CONFIRMED', 'text': text,
                                   'confidence': conf, 'speed': .7*tr['speed']+.3*moved,
                                   'last_seen_frame': frame_no, 'last_ocr_frame': frame_no,
                                   'last_local_search_frame': frame_no, 'recovery_attempts': 0,
                                   'appearance': self._appearance(frame, box),
                                   'points': self._features(current_gray, self._clip(frame, box))})
                        try:
                            tr['kalman'].correct(box)
                        except Exception:
                            tr['kalman'] = SortMotionModel(box)
                        used_dets.add(j)
                    # Additional OCR instances become new tracks, but manual single ROI mode
                    # only creates new IDs when OCR matches the requested target text.
                    for j, det in enumerate(detections):
                        if j in used_dets:
                            continue
                        if any(self._iou(det[0], tr['roi']) > .80 for tr in tracks.values()):
                            continue
                        if self.target_text.strip() and self._normalize(self.target_text) not in self._normalize(det[1]):
                            continue
                        tracks[next_id] = self._new_track(frame, det[0], next_id, frame_no, det[1], det[2])
                        history[next_id] = {}
                        next_id += 1

                # Export masks for live tracks. During a short loss, use a growing
                # conservative mask around the Kalman prediction instead of exposing
                # the target immediately. Long losses are marked for manual review.
                frame_rois, monitor_tracks = [], {}
                for tid, tr in tracks.items():
                    if tr['lost'] > self.MAX_LOST_FRAMES:
                        continue
                    roi = tuple(map(int, self._clip(frame, tr['roi'])))
                    monitor_tracks[tid] = tr
                    mask_roi = None
                    if tr['lost'] == 0:
                        mask_roi = roi
                    elif tr['lost'] <= self.CONSERVATIVE_MASK_FRAMES:
                        predicted_roi = tr.get('predicted_roi') or tr['roi']
                        px, py, pw, ph = self._clip(frame, predicted_roi)
                        cx, cy = px + pw / 2.0, py + ph / 2.0
                        scale = 1.0 + 0.18 * tr['lost']
                        mask_roi = self._clip(frame, (cx-pw*scale/2, cy-ph*scale/2,
                                                      pw*scale, ph*scale))
                    history.setdefault(tid, {})[frame_no] = mask_roi
                    if mask_roi is not None and not any(self._iou(mask_roi, old) > .88 for old in frame_rois):
                        frame_rois.append(tuple(map(int, mask_roi)))
                data[frame_no] = frame_rois
                if frame_no % self.MONITOR_INTERVAL == 0 or ran_ocr or need_local_recovery:
                    self._draw_monitor(frame, frame_no, monitor_tracks, 'RECOVERY MODE' if need_local_recovery else 'TRACK MODE')
                if ran_ocr or need_local_recovery:
                    diagnostic_tracks = [t for t in tracks.values() if t.get('lost', 0) > 0]
                    diagnosis = ', '.join(f"ID {t['id']}:{t.get('loss_reason') or 'recovering'}" for t in diagnostic_tracks[:4])
                    summary = f'；失追診斷={diagnosis}' if diagnosis else ''
                    self._report('tracking', f'影格 {frame_no}：追蹤 {len(tracks)}，局部恢復 {len(need_local_recovery)}，OCR 框 {len(detections)}{summary}',
                                 frame_no, target_count=len(tracks), lost_count=len(need_local_recovery),
                                 detection_count=len(detections), diagnosis=diagnosis,
                                 global_motion=global_motion, camera_motion_suspected=camera_moving,
                                 fps=source_fps, mode='RECOVERY MODE' if need_local_recovery else 'TRACK MODE')
                self.progress_signal.emit(frame_no, f'TRACK F{frame_no}/{end_frame}，有效框 {len(frame_rois)}')
                previous_gray = current_gray

            # Stage 1: fill missing frames between known positions.
            retro_checked = 0
            if reader is not None and self.target_text.strip() and last_frame >= hit_frame:
                retro_checked = self._retroactively_verify_gaps(
                    cap, history, hit_frame, last_frame, frame_w, frame_h, reader)

                # Stage 2: search BEFORE each track's first known position. This handles
                # targets first discovered only at a sparse global OCR keyframe.
                first_result = self._retroactively_search_before_first_detection(
                    cap, history, hit_frame, last_frame, frame_w, frame_h, reader)
                self._report('retroactive_check',
                    f'回溯總結：已知缺口檢查 {retro_checked} 幀；首次偵測前檢查 '
                    f'{first_result["checked"]} 幀、OCR 確認補到 {first_result["matched"]} 幀',
                    last_frame, retro_checked=retro_checked,
                    first_detection_checked=first_result['checked'],
                    first_detection_matched=first_result['matched'],
                    max_gap=self.RETRO_OCR_MAX_GAP)

            self._interpolate_short_gaps(history, self.INTERPOLATE_MAX_GAP)
            for f in range(hit_frame, last_frame+1):
                rois = list(data.get(f, []))
                for th in history.values():
                    roi = th.get(f)
                    if roi is not None and not any(self._iou(roi, old) > .88 for old in rois):
                        rois.append(tuple(map(int, roi)))
                data[f] = rois
            success = any(bool(v) for v in data.values())
            self._report('finished', '追蹤完成', last_frame, tracked_frames=len(data),
                         target_count=max((len(v) for v in data.values()), default=0), success=success)
        except Exception as exc:
            self._report('error', f'執行錯誤：{type(exc).__name__}: {exc}', last_frame)
            success = any(bool(v) for v in data.values())
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
