"""Multi-target tracking + sparse full-frame/local RapidOCR verification + mosaic export.

Requires: opencv-contrib-python, rapidocr-onnxruntime, PyQt5. EasyOCR/PyTorch is not used.
tracking_data format: {absolute_frame_index: [(x, y, w, h), ...]}
"""
import math
import time
from difflib import SequenceMatcher
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from frame_buffer import FrameRingBuffer


try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


class IntuitionisticFuzzyDecision:
    """Intuitionistic-fuzzy qualification for OCR candidates only.

    Fuzzy inference answers: "Does this OCR result look like the requested target?"
    It is deliberately NOT used to draw mosaics or to assign an existing track ID.
    Track association uses explicit geometric/motion costs and one-to-one assignment.
    Scores are heuristic and should be calibrated against the user's video samples.
    """

    @staticmethod
    def evaluate_target(ocr_confidence, text_similarity):
        confidence = float(np.clip(ocr_confidence, 0.0, 1.0))
        similarity = float(np.clip(text_similarity, 0.0, 1.0))

        # Text agreement is the primary evidence; OCR confidence is supporting evidence.
        support = 0.72 * similarity + 0.28 * confidence
        opposition = 0.82 * (1.0 - similarity) + 0.18 * (1.0 - confidence)
        mu = float(np.clip(support * (1.0 - opposition), 0.0, 1.0))
        nu = float(np.clip(opposition * (1.0 - support), 0.0, 1.0 - mu))
        pi = float(np.clip(1.0 - mu - nu, 0.0, 1.0))

        # Conservative target qualification: fuzzy logic filters OCR candidates;
        # it does not decide whether a frame should be mosaicked.
        if similarity >= 0.72 and mu >= 0.30 and mu > nu:
            decision = "ACCEPT"
        elif similarity >= 0.48 and (pi >= 0.12 or abs(mu - nu) < 0.12):
            decision = "UNCERTAIN"
        else:
            decision = "REJECT"
        return {"mu": mu, "nu": nu, "pi": pi, "decision": decision,
                "text_similarity": similarity, "ocr_confidence": confidence}


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

    initial_roi: required (x, y, w, h) in the start_frame. The user's selection is kept,
    while a one-time overlapping-tile OCR scan initializes other targets in the same frame.
    tracking_data format remains {absolute_frame_index: [(x, y, w, h), ...]}.
    """
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict, int, bool)
    monitor_frame_signal = pyqtSignal(object, int, str)
    monitor_data_signal = pyqtSignal(dict)

    # OCR runs only on sparse keyframes or after tracking loss, never every frame.
    GLOBAL_OCR_INTERVAL = 5           # 快速位移時更頻繁全畫面 OCR，降低錯過新位置的時間。
    LOCAL_RECOVERY_INTERVAL = 2       # 追蹤遺失時每 2 幀重試局部 OCR，縮短重新定位延遲
    MAX_OCR_ROI_DIM = 800             # 局部 OCR 搜尋區域的最大尺寸。
    MAX_GLOBAL_OCR_DIM = 1280         # 全畫面 OCR 的縮放尺寸上限。
    MIN_FEATURE_POINTS = 4
    MAX_LOST_FRAMES = 3  # 保留舊版相容設定；實際遮罩恢復另由狀態控制。
    CONSERVATIVE_MASK_FRAMES = 3  # 只短暫使用預測框；失去追蹤後不長時間遮住錯誤位置。
    RECOVERY_SCALE = 3.0
    DETECT_CANVAS_SIZE = 1280
    MIN_OCR_CONFIDENCE = 0.20
    MONITOR_INTERVAL = 2  # Update the live preview about 15 times/second at 30 FPS.
    RETROACTIVE_SEGMENT_INTERVAL = 30  # 更頻繁檢查近期追蹤路徑。
    INTERPOLATE_MAX_GAP = 2  # 保留舊的短缺口設定，路徑補遮另有獨立上限
    MAX_PATH_INTERPOLATION_GAP = 120  # OCR 仍無法確認時，最多以較寬路徑遮罩補 120 幀。
    PATH_MASK_EXPAND = 1.20  # 路徑推估 ROI 稍微擴大，降低位置估計誤差造成的漏遮
    # 自適應框穩定：小幅抖動時加強平滑，大幅移動時提高反應速度。
    STABLE_CENTER_DEADBAND_RATIO = 0.012
    STABLE_SIZE_DEADBAND_RATIO = 0.04
    # 通用尺寸先驗，不針對特定影片/物件。以使用者初始框和近期可信框共同估計尺寸。
    SIZE_PRIOR_MIN_SCALE = 0.50
    SIZE_PRIOR_MAX_SCALE = 2.00
    SIZE_JUMP_LIMIT = 1.80
    SIZE_HISTORY_LENGTH = 9
    IMMEDIATE_RETRACE_ON_REACQUISITION = True  # 重新找到目標時，立即回頭檢查剛才的缺口
    RETRO_OCR_MAX_GAP = 120  # 回溯 OCR 檢查的最大缺口幀數。
    RETRO_ROI_EXPAND = 2.2  # 回溯時擴大局部搜尋範圍。
    USE_MSER_PREFILTER = False  # Avoid MSER false negatives blocking OCR.
    ENABLE_END_REVERSE_CHECK = False  # 不做整部影片反掃；重新取得目標時會強制 OCR 回查前段。
    FRAME_CACHE_MAX_FRAMES = 12
    FRAME_CACHE_MAX_BYTES = 96 * 1024 * 1024  # Hard cap: 96 MiB of cached source frames.

    def __init__(self, video_path, start_frame, total_frames, target_text, initial_roi=None):
        super().__init__()
        self.video_path = video_path
        self.start_frame = max(0, int(start_frame))
        self.total_frames = max(0, int(total_frames))
        self.target_text = str(target_text or '')
        self.initial_roi = tuple(initial_roi) if initial_roi is not None else None
        self.frame_buffer = None  # Created in run() after video metadata is available.
        self.was_cancelled = False
        self.cancel_frame = self.start_frame
        # Remember completed OCR-anchor pairs. Without this guard, each periodic segment
        # can repeatedly OCR the same gap and appear to be stuck in backtracking.
        self._processed_retro_pairs = set()
        # Per-frame visual diagnostics consumed by the live monitor overlay.
        self._monitor_overlay = {'frame': -1, 'search_roi': None, 'candidates': [],
                                 'ocr_mode': '', 'ocr_elapsed': 0.0, 'target_count': 0,
                                 'recovery_note': ''}

    def _read_frame_at(self, cap, frame_no):
        """Read a frame from the bounded recent-frame cache or seek/read the source video.

        Cache misses do not populate the cache: old backfill reads therefore cannot evict
        the most recent forward frames from the Ring Buffer. The decoder position is restored.
        """
        if self.frame_buffer is not None:
            return self.frame_buffer.get_frame(int(frame_no), cap)
        resume = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
        ok, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, resume)
        return (frame.copy() if ok and frame is not None else None)

    def _report(self, stage, message, frame_no=None, **extra):
        self.monitor_data_signal.emit({'stage': stage, 'message': message,
            'frame': frame_no, 'clock': time.monotonic(), **extra})

    def _pending_anchor_pair_histories(self, ocr_history, start_frame, end_frame):
        """Build only unprocessed gaps between adjacent OCR-confirmed anchors.

        Tracker/Kalman prediction boxes are deliberately NOT used as anchors. A recovery
        box can be orange and visibly misplaced; treating it as a trusted endpoint makes
        the backfill confidently follow the wrong path. Each pair is checked at most once.
        """
        pending = {}
        max_gap = max(0, int(self.RETRO_OCR_MAX_GAP))
        if max_gap <= 0:
            return pending
        for track_id, trusted in ocr_history.items():
            anchors = sorted(
                int(f) for f, roi in trusted.items()
                if start_frame <= int(f) <= end_frame and roi is not None
            )
            for left, right in zip(anchors, anchors[1:]):
                gap = right - left - 1
                if gap <= 0 or gap > max_gap:
                    continue
                pair_key = (int(track_id), int(left), int(right))
                if pair_key in self._processed_retro_pairs:
                    continue
                # Synthetic key lets one batch contain several gaps for the same ID
                # while sharing a single OCR result for any frame used by multiple IDs.
                local = {f: None for f in range(left, right + 1)}
                local[left] = tuple(map(int, trusted[left]))
                local[right] = tuple(map(int, trusted[right]))
                pending[pair_key] = local
                self._processed_retro_pairs.add(pair_key)
        return pending

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
    def _text_similarity(target, detected):
        """Compare normalized OCR text with the requested target, tolerating small OCR errors."""
        target = ''.join(ch for ch in str(target).casefold() if ch.isalnum())
        detected = ''.join(ch for ch in str(detected).casefold() if ch.isalnum())
        if not target or not detected:
            return 0.0
        if target in detected:
            return 1.0
        # Very short identifiers are too ambiguous for fuzzy matching; require exact containment.
        if len(target) < 3:
            return 0.0
        best = SequenceMatcher(None, target, detected).ratio()
        # OCR can attach one or two neighbouring characters to a text box. Compare local
        # windows around target length as well as the entire detected string.
        low = max(1, len(target) - 2)
        high = min(len(detected), len(target) + 2)
        for window_len in range(low, high + 1):
            for offset in range(0, len(detected) - window_len + 1):
                candidate = detected[offset:offset + window_len]
                best = max(best, SequenceMatcher(None, target, candidate).ratio())
        return float(best)

    def _target_text_evidence(self, text, confidence):
        target = self._normalize(self.target_text)
        detected = self._normalize(text)
        if not target:
            return {"decision": "REJECT", "mu": 0.0, "nu": 1.0, "pi": 0.0,
                    "text_similarity": 0.0, "ocr_confidence": float(confidence)}
        similarity = self._text_similarity(target, detected)
        result = IntuitionisticFuzzyDecision.evaluate_target(confidence, similarity)
        # Do not let fuzzy matching loosen short codes or low-quality OCR too much.
        if len(target) < 3 and target not in detected:
            result["decision"] = "REJECT"
        return result

    @staticmethod
    def _create_reader():
        """Load RapidOCR only. Avoid importing EasyOCR/PyTorch, which can fail on Windows DLLs."""
        try:
            from rapidocr_onnxruntime import RapidOCR
            return {"backend": "rapidocr-onnxruntime", "engine": RapidOCR()}
        except Exception as exc:
            raise RuntimeError(
                "RapidOCR 初始化失敗。請確認已安裝 rapidocr-onnxruntime 與 onnxruntime。\n"
                "python -m pip install --upgrade rapidocr-onnxruntime onnxruntime\n"
                "若目前使用 Python 3.14，建議改用 64-bit Python 3.11 或 3.12 建立虛擬環境。\n"
                "此版本使用 RapidOCR，不需要 EasyOCR 或 PyTorch。\n"
                f"錯誤詳情：{type(exc).__name__}: {exc}"
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
        # 建立 CSRT 視覺追蹤器；若 OpenCV 未提供該 API，則回傳 None。
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
                # The user-selected box is a size prior, not a permanently fixed box.
                # Trusted observations may adapt it gradually as perspective/distance changes.
                'initial_roi': tuple(map(int, roi)),
                'size_reference': (float(roi[2]), float(roi[3])),
                'size_history': [(float(roi[2]), float(roi[3]))],
                'ocr_size_reference_ready': False,
                'size_rejections': 0,
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

    def _stabilize_roi(self, frame, track, measured_roi, source="visual", confidence=0.5):
        """Stabilize position and size using the user's initial ROI plus trusted history.

        The initial dimensions are a prior, not a hard-coded output size. The accepted
        size history lets the box adapt gradually when the target genuinely changes scale,
        while rejecting abrupt tracker/OCR boxes that are likely to include neighbours.
        """
        measured = tuple(map(int, self._clip(frame, measured_roi)))
        previous = track.get("roi")
        if previous is None:
            return measured

        px, py, pw, ph = [float(v) for v in previous]
        mx, my, mw, mh = [float(v) for v in measured]
        if min(pw, ph, mw, mh) <= 1:
            return measured

        # Build a robust size reference from the user selection and recent accepted boxes.
        # Median is resistant to one oversized detection in a dense scene.
        initial_w, initial_h = track.get('size_reference', (pw, ph))
        # The first OCR-confirmed box calibrates the text/target geometry. This matters
        # when the user's initial ROI contains a little context around the text itself.
        # Subsequent OCR boxes are checked against this learned baseline and recent history.
        if source == 'ocr' and not track.get('ocr_size_reference_ready', False):
            initial_w, initial_h = float(mw), float(mh)
            track['size_reference'] = (initial_w, initial_h)
            track['size_history'] = [(initial_w, initial_h)]
            track['ocr_size_reference_ready'] = True
        samples = track.setdefault('size_history', [(initial_w, initial_h)])
        valid_samples = [(float(w), float(h)) for w, h in samples if w > 1 and h > 1]
        if not valid_samples:
            valid_samples = [(float(initial_w), float(initial_h))]
        recent = valid_samples[-self.SIZE_HISTORY_LENGTH:]
        median_w = float(np.median([item[0] for item in recent]))
        median_h = float(np.median([item[1] for item in recent]))

        # Combine the initial selection with the recent median. As history grows, the
        # recent measurements can adapt the reference without forgetting initialization.
        history_weight = min(0.75, len(recent) / float(self.SIZE_HISTORY_LENGTH + 1))
        ref_w = (1.0 - history_weight) * float(initial_w) + history_weight * median_w
        ref_h = (1.0 - history_weight) * float(initial_h) + history_weight * median_h
        ref_w, ref_h = max(2.0, ref_w), max(2.0, ref_h)

        # Treat a large one-frame size jump as unreliable. Instead of freezing the whole
        # box, retain the new center but use a plausible size based on the current reference.
        min_w = max(2.0, ref_w * self.SIZE_PRIOR_MIN_SCALE)
        max_w = max(min_w, ref_w * self.SIZE_PRIOR_MAX_SCALE)
        min_h = max(2.0, ref_h * self.SIZE_PRIOR_MIN_SCALE)
        max_h = max(min_h, ref_h * self.SIZE_PRIOR_MAX_SCALE)
        candidate_w = float(np.clip(mw, min_w, max_w))
        candidate_h = float(np.clip(mh, min_h, max_h))
        size_jump = max(candidate_w / max(1.0, pw), pw / max(1.0, candidate_w),
                        candidate_h / max(1.0, ph), ph / max(1.0, candidate_h))
        if size_jump > self.SIZE_JUMP_LIMIT:
            # Keep the previous dimensions for this measurement; still follow its center.
            candidate_w, candidate_h = pw, ph
            track['size_rejections'] = int(track.get('size_rejections', 0)) + 1
        else:
            track['size_history'].append((candidate_w, candidate_h))
            del track['size_history'][:-self.SIZE_HISTORY_LENGTH]

        pcx, pcy = px + pw / 2.0, py + ph / 2.0
        mcx, mcy = mx + mw / 2.0, my + mh / 2.0
        diagonal = max(20.0, math.hypot(pw, ph))
        displacement = math.hypot(mcx - pcx, mcy - pcy)
        relative_motion = displacement / diagonal
        if relative_motion <= self.STABLE_CENTER_DEADBAND_RATIO:
            center_alpha = 0.18
        else:
            overshoot = relative_motion - self.STABLE_CENTER_DEADBAND_RATIO
            center_alpha = 1.0 - 0.82 * math.exp(-4.5 * overshoot)
        if source == "ocr":
            center_alpha = min(1.0, center_alpha + 0.15 * float(np.clip(confidence, 0.0, 1.0)))

        relative_size_change = max(
            abs(candidate_w - pw) / max(1.0, pw),
            abs(candidate_h - ph) / max(1.0, ph))
        if relative_size_change <= self.STABLE_SIZE_DEADBAND_RATIO:
            size_alpha = 0.16
        else:
            overshoot = relative_size_change - self.STABLE_SIZE_DEADBAND_RATIO
            size_alpha = 1.0 - 0.84 * math.exp(-3.5 * overshoot)
        if source == "ocr":
            size_alpha = min(1.0, size_alpha + 0.15 * float(np.clip(confidence, 0.0, 1.0)))

        cx = pcx + center_alpha * (mcx - pcx)
        cy = pcy + center_alpha * (mcy - pcy)
        width = max(2.0, pw + size_alpha * (candidate_w - pw))
        height = max(2.0, ph + size_alpha * (candidate_h - ph))
        stabilized = (cx - width / 2.0, cy - height / 2.0, width, height)
        result = tuple(map(int, self._clip(frame, stabilized)))
        track['last_size_diagnostics'] = {
            'source': source,
            'initial_size': (int(initial_w), int(initial_h)),
            'reference_size': (round(ref_w, 2), round(ref_h, 2)),
            'measured_size': (int(mw), int(mh)),
            'accepted_size': (int(result[2]), int(result[3])),
            'size_rejections': int(track.get('size_rejections', 0)),
        }
        return result

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
                self._report('ocr_skipped', '未提供 OCR 搜尋區域，略過本次 OCR', frame_no)
                return [], 0.0
            x, y, w, h = self._clip(frame, roi)
            crop = frame[y:y+h, x:x+w]
            max_dim = self.MAX_OCR_ROI_DIM
        if crop.size == 0:
            return [], 0.0

        # Keep small text at native size; downscale only oversized inputs.
        ch, cw = crop.shape[:2]
        if full_frame:
            scale = min(1.0, max_dim / float(max(ch, cw)))
        else:
            # 針對局部 OCR，如果太小則強制放大以利於密集微小字體辨識
            target_h = max(36.0, float(ch))
            scale = min(max_dim / float(max(ch, cw)), target_h / float(max(1, ch)))
            
        if scale == 1.0:
            ocr_crop = crop
        else:
            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            ocr_crop = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))), interpolation=interp)
            
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
                self._report('error', f'OCR 前處理 {variant_name} 失敗：{type(exc).__name__}: {exc}', frame_no)
                results = []
            for pts, text, conf in results:
                normalized = self._normalize(text)
                pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
                x1, y1 = np.floor(pts.min(axis=0)).astype(int)
                x2, y2 = np.ceil(pts.max(axis=0)).astype(int)
                # Convert OCR coordinates from resized crop to original frame coordinates.
                x1, x2 = int(x1 / scale), int(x2 / scale)
                y1, y2 = int(y1 / scale), int(y2 / scale)
                bw, bh = x2-x1, y2-y1
                if bw <= 0 or bh <= 0:
                    continue
                # Small proportional OCR margin: scales with the detected text rather than
                # using a large fixed expansion that merges neighbouring targets in dense scenes.
                px, py = max(1, int(round(bw * 0.04))), max(1, int(round(bh * 0.08)))
                box = self._clip(frame, (x+x1-px, y+y1-py, bw+2*px, bh+2*py))
                target_evidence = self._target_text_evidence(text, conf)
                matched = bool(
                    conf >= self.MIN_OCR_CONFIDENCE
                    and target_evidence['decision'] == 'ACCEPT'
                )
                raw.append({'text': text, 'confidence': float(conf), 'variant': variant_name,
                            'bbox': tuple(map(int, box)), 'matched': matched,
                            'target_similarity': target_evidence['text_similarity'],
                            'fuzzy_mu': target_evidence['mu'],
                            'fuzzy_nu': target_evidence['nu'],
                            'fuzzy_pi': target_evidence['pi'],
                            'target_decision': target_evidence['decision']})
                # Fuzzy inference is used only to qualify OCR candidates as target text.
                # Accepted target boxes are passed to the tracking/data-association stage.
                if matched:
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
        self._monitor_overlay = {
            'frame': int(frame_no), 'search_roi': (x, y, w, h),
            'candidates': raw[:30], 'ocr_mode': mode,
            'ocr_elapsed': float(elapsed), 'target_count': len(unique),
        }
        self._report('ocr_done', f'OCR {mode} 完成：耗時 {elapsed:.2f} 秒，找到 {len(unique)} 個符合目標的候選框；原始文字框 {len(raw)} 個',
                     frame_no, elapsed_seconds=elapsed, target_count=len(unique), result_count=len(raw),
                     ocr_results=raw[:20], roi=(x, y, w, h), mode=mode,
                     target_text=self.target_text)
        return unique, elapsed

    def _initial_tiled_ocr_scan(self, reader, frame, frame_no):
        """One-time dense-scene initialization scan using overlapping, native-resolution tiles.

        A single downscaled full-frame OCR pass can miss small/repeated text in dense scenes.
        This bounded scan runs only at initialization. Each tile uses the existing OCR path,
        which tries alternate preprocessing only when the original tile has no target match.
        Overlapping tile detections are merged conservatively so nearby repeated targets keep
        separate identities.
        """
        if reader is None or not self.target_text.strip():
            return []

        frame_h, frame_w = frame.shape[:2]
        tile_limit = max(160, min(int(self.MAX_OCR_ROI_DIM), 800))
        tile_w = min(tile_limit, frame_w)
        tile_h = min(tile_limit, frame_h)
        overlap_ratio = 0.20

        def positions(length, tile_length):
            if length <= tile_length:
                return [0]
            step = max(1, int(round(tile_length * (1.0 - overlap_ratio))))
            values = list(range(0, max(1, length - tile_length + 1), step))
            values.append(length - tile_length)
            return sorted(set(values))

        xs = positions(frame_w, tile_w)
        ys = positions(frame_h, tile_h)
        tile_count = len(xs) * len(ys)
        self._report(
            'initial_dense_scan',
            f'起始影格密集目標掃描：{len(xs)}×{len(ys)} 個重疊區塊，避免只縮小全畫面而漏掉小字。',
            frame_no, tile_count=tile_count, initialization_scan=True)

        candidates = []
        checked = 0
        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                if self.isInterruptionRequested():
                    return self._merge_initial_detections(candidates)
                roi = (x, y, min(tile_w, frame_w - x), min(tile_h, frame_h - y))
                found, _elapsed = self._ocr(
                    reader, frame, frame_no, roi=roi, full_frame=False, robust=True)
                candidates.extend(found)
                checked += 1
                self._report(
                    'initial_dense_scan_tile',
                    f'起始影格掃描區塊 {checked}/{tile_count}；目前找到 {len(self._merge_initial_detections(candidates))} 個候選目標。',
                    frame_no, tile_index=checked, tile_count=tile_count,
                    candidate_count=len(candidates), initialization_scan=True)

        merged = self._merge_initial_detections(candidates)
        self._report(
            'initial_dense_scan_done',
            f'起始影格密集掃描完成：共檢查 {checked} 個區塊，合併後有 {len(merged)} 個目標候選。',
            frame_no, tile_count=checked, target_count=len(merged), initialization_scan=True)
        return merged

    def _merge_initial_detections(self, detections):
        """Merge duplicate OCR boxes from overlapping tiles without collapsing neighbours."""
        ranked = sorted(detections, key=lambda item: float(item[2]), reverse=True)
        merged = []
        for detection in ranked:
            box, text, confidence = detection
            duplicate_index = None
            for index, existing in enumerate(merged):
                old_box, old_text, old_confidence = existing
                iou = self._iou(box, old_box)
                # Strong overlap is needed: a mere close center is not enough because dense
                # rows may contain multiple identical target strings side by side.
                if iou >= 0.58:
                    duplicate_index = index
                    break
                x1 = max(box[0], old_box[0]); y1 = max(box[1], old_box[1])
                x2 = min(box[0] + box[2], old_box[0] + old_box[2])
                y2 = min(box[1] + box[3], old_box[1] + old_box[3])
                intersection = max(0, x2 - x1) * max(0, y2 - y1)
                min_area = max(1, min(box[2] * box[3], old_box[2] * old_box[3]))
                center_distance = math.hypot(
                    self._center(box)[0] - self._center(old_box)[0],
                    self._center(box)[1] - self._center(old_box)[1])
                if (intersection / min_area >= 0.88 and
                        center_distance <= max(4.0, min(box[2], box[3], old_box[2], old_box[3]) * 0.22)):
                    duplicate_index = index
                    break
            if duplicate_index is None:
                merged.append((tuple(map(int, box)), str(text), float(confidence)))
            elif float(confidence) > float(merged[duplicate_index][2]):
                merged[duplicate_index] = (tuple(map(int, box)), str(text), float(confidence))
        merged.sort(key=lambda item: (self._center(item[0])[1], self._center(item[0])[0]))
        return merged

    def _draw_monitor(self, frame, frame_no, tracks, mode):
        shown = frame.copy()
        colors = {'OCR_CONFIRMED': (0, 220, 0), 'MANUAL_INIT': (0, 220, 0),
                  'TRACKING': (255, 210, 0), 'FAST_MOTION': (255, 0, 255),
                  'RECOVERY': (0, 190, 255), 'LOST': (0, 0, 255)}
        for tid, track in tracks.items():
            lost = int(track.get('lost', 0))
            state = track.get('state', 'TRACKING')
            roi = track.get('roi', (0, 0, 1, 1))
            x, y, w, h = map(int, roi)
            color = colors.get(state, (255, 255, 255))

            # A visual tracker result is not trusted once the track is lost.
            # Show only a short-lived dashed prediction, never a solid blue/stale box.
            if lost == 0:
                cv2.rectangle(shown, (x, y), (x+w, y+h), color, 2)
                label = f'ID {tid} {state} v={track.get("speed", 0):.1f}'
                cv2.putText(shown, label, (x, max(18, y-5)), cv2.FONT_HERSHEY_SIMPLEX,
                            .48, color, 2, cv2.LINE_AA)
                continue

            # Keep the target ID internally for OCR recovery, but stop drawing a box
            # at an unverified location after the short prediction grace period.
            if lost <= 2:
                predicted_roi = track.get('predicted_roi')
                if predicted_roi is not None:
                    px, py, pw, ph = map(int, self._clip(frame, predicted_roi))
                    dash = 8
                    for xx in range(px, px+pw, dash*2):
                        cv2.line(shown, (xx, py), (min(xx+dash, px+pw), py), (0, 165, 255), 2)
                        cv2.line(shown, (xx, py+ph), (min(xx+dash, px+pw), py+ph), (0, 165, 255), 2)
                    for yy in range(py, py+ph, dash*2):
                        cv2.line(shown, (px, yy), (px, min(yy+dash, py+ph)), (0, 165, 255), 2)
                        cv2.line(shown, (px+pw, yy), (px+pw, min(yy+dash, py+ph)), (0, 165, 255), 2)
                    cv2.putText(shown, f'ID {tid} SEARCHING', (px, max(18, py-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, .46, (0, 165, 255), 2, cv2.LINE_AA)
            # For lost > 2, intentionally draw no object box. The ID remains available
            # to local/global OCR recovery, but the monitor does not imply a false lock.
        overlay = getattr(self, '_monitor_overlay', {})
        # Visualize OCR search ROI only on the exact frame where OCR ran, avoiding stale boxes.
        if overlay.get('frame') == int(frame_no):
            search_roi = overlay.get('search_roi')
            if search_roi is not None:
                sx, sy, sw, sh = map(int, self._clip(frame, search_roi))
                # Orange = area sent into OCR (whole frame for global OCR keyframes).
                cv2.rectangle(shown, (sx, sy), (sx+sw, sy+sh), (0, 165, 255), 2)
                cv2.putText(shown, f"OCR SEARCH: {overlay.get('ocr_mode', '')}",
                            (sx, min(shown.shape[0]-8, sy+20)),
                            cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 165, 255), 2, cv2.LINE_AA)
            for cand in overlay.get('candidates', []):
                bbox = cand.get('bbox')
                if not bbox:
                    continue
                cx, cy, cw, ch = map(int, self._clip(frame, bbox))
                matched = bool(cand.get('matched'))
                conf = float(cand.get('confidence', 0.0) or 0.0)
                # Green = text matches target; red = OCR text found but not a target match.
                color = (0, 220, 0) if matched else (40, 40, 230)
                cv2.rectangle(shown, (cx, cy), (cx+cw, cy+ch), color, 2 if matched else 1)
                text = str(cand.get('text', '')).replace('\n', ' ')[:24]
                label = f"{'MATCH' if matched else 'OCR'} {conf:.2f}: {text}"
                label_y = cy-5 if cy > 24 else min(shown.shape[0]-6, cy+ch+18)
                cv2.putText(shown, label, (cx, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                            .42, color, 1, cv2.LINE_AA)
            info = (f"OCR={overlay.get('ocr_mode', '--')}  candidates={len(overlay.get('candidates', []))}  "
                    f"matches={overlay.get('target_count', 0)}  {overlay.get('ocr_elapsed', 0.0):.2f}s")
            cv2.rectangle(shown, (0, 0), (shown.shape[1], 48), (18, 18, 18), -1)
            cv2.putText(shown, info, (12, 19), cv2.FONT_HERSHEY_SIMPLEX,
                        .48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(shown, f'Frame {frame_no} | {mode} | tracks={len(tracks)}', (12, 41),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1, cv2.LINE_AA)
        else:
            cv2.putText(shown, f'Frame {frame_no} | {mode} | tracks={len(tracks)}', (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2, cv2.LINE_AA)
        # Fixed legend makes it easier to distinguish real OCR results from predictions.
        legend_y = max(18, shown.shape[0] - 12)
        cv2.rectangle(shown, (0, max(0, legend_y - 22)), (shown.shape[1], shown.shape[0]), (18, 18, 18), -1)
        cv2.putText(shown, 'GREEN=OCR MATCH  RED=OTHER OCR  ORANGE=SHORT PREDICTION  BLUE=VERIFIED TRACK',
                    (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, .43, (245, 245, 245), 1, cv2.LINE_AA)
        mh, mw = shown.shape[:2]
        monitor_scale = min(1.0, 960.0 / max(mh, mw))
        if monitor_scale < 1.0:
            shown = cv2.resize(shown, (max(1, int(mw * monitor_scale)), max(1, int(mh * monitor_scale))), interpolation=cv2.INTER_AREA)
        self.monitor_frame_signal.emit(shown, int(frame_no), mode)

    def _draw_reverse_check(self, frame, frame_no, track_id, predicted_roi=None,
                            measured_roi=None, note="回溯檢查"):
        """在監控視窗顯示預測框與實際量測框，協助觀察追蹤品質。

        預測框與量測框使用不同顏色，方便比較兩者的位置差異。
        frame_no 為目前影格編號；此方法只負責繪製診斷畫面。
        """
        shown = frame.copy()
        if predicted_roi is not None:
            x, y, w, h = map(int, self._clip(frame, predicted_roi))
            cv2.rectangle(shown, (x, y), (x+w, y+h), (0, 165, 255), 2)
            cv2.putText(shown, "PREDICTED ROI", (x, max(18, y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 2, cv2.LINE_AA)
        if measured_roi is not None:
            x, y, w, h = map(int, self._clip(frame, measured_roi))
            cv2.rectangle(shown, (x, y), (x+w, y+h), (0, 220, 0), 2)
            cv2.putText(shown, "OCR MATCH", (x, min(shown.shape[0]-6, y+h+18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 0), 2, cv2.LINE_AA)
        overlay = getattr(self, '_monitor_overlay', {})
        if overlay.get('frame') == int(frame_no):
            sx_roi = overlay.get('search_roi')
            if sx_roi is not None:
                sx, sy, sw, sh = map(int, self._clip(frame, sx_roi))
                cv2.rectangle(shown, (sx, sy), (sx+sw, sy+sh), (255, 180, 0), 2)
                cv2.putText(shown, 'LOCAL OCR SEARCH', (sx, min(shown.shape[0]-8, sy+18)),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 180, 0), 1, cv2.LINE_AA)
            for cand in overlay.get('candidates', []):
                bbox = cand.get('bbox')
                if not bbox:
                    continue
                cx, cy, cw, ch = map(int, self._clip(frame, bbox))
                is_match = bool(cand.get('matched'))
                col = (0, 220, 0) if is_match else (40, 40, 230)
                cv2.rectangle(shown, (cx, cy), (cx+cw, cy+ch), col, 2 if is_match else 1)
                cv2.putText(shown, f"{'MATCH' if is_match else 'OCR'} {float(cand.get('confidence', 0)):.2f}",
                            (cx, max(16, cy-4)), cv2.FONT_HERSHEY_SIMPLEX, .42, col, 1, cv2.LINE_AA)
        cv2.rectangle(shown, (0, 0), (shown.shape[1], 42), (20, 20, 20), -1)
        cv2.putText(shown, f"REVERSE CHECK | F{frame_no} | ID {track_id} | {note}",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        mh, mw = shown.shape[:2]
        monitor_scale = min(1.0, 960.0 / max(mh, mw))
        if monitor_scale < 1.0:
            shown = cv2.resize(shown, (max(1, int(mw * monitor_scale)), max(1, int(mh * monitor_scale))), interpolation=cv2.INTER_AREA)
        self.monitor_frame_signal.emit(shown, int(frame_no), "分段回溯 OCR")
        self._report('reverse_check_frame',
            f'回溯檢查影格 F{frame_no}，ID {track_id}：{note}',
            frame_no, track_id=track_id, predicted_roi=predicted_roi,
            measured_roi=measured_roi, reverse_check=True)

    def _draw_path_backfill(self, cap, frame_no, track_id, roi, left_frame, right_frame,
                            left_roi, right_roi, resume_frame=None):
        """Show a sampled path-interpolated ROI in the live monitor window."""
        try:
            frame = self._read_frame_at(cap, frame_no)
            if frame is None:
                return
            shown = frame.copy()
            x, y, w, h = self._clip(shown, roi)
            ax, ay = self._center(left_roi)
            bx, by = self._center(right_roi)
            cv2.line(shown, (int(ax), int(ay)), (int(bx), int(by)), (255, 0, 220), 2, cv2.LINE_AA)
            cv2.rectangle(shown, (x, y), (x+w, y+h), (255, 0, 220), 2)
            cv2.putText(shown, f"PATH MASK | F{frame_no} | ID {track_id} | anchors F{left_frame}-F{right_frame}",
                        (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            mh, mw = shown.shape[:2]
            scale = min(1.0, 960.0 / max(mh, mw))
            if scale < 1.0:
                shown = cv2.resize(shown, (max(1, int(mw*scale)), max(1, int(mh*scale))),
                                   interpolation=cv2.INTER_AREA)
            self.monitor_frame_signal.emit(shown, int(frame_no), "路徑補遮（推估 ROI）")
            self._report('path_mask_visualized',
                f'路徑補遮視覺化：F{frame_no}，ID {track_id}，參考錨點 F{left_frame}/F{right_frame}',
                frame_no, track_id=track_id, roi=roi, left_frame=left_frame,
                right_frame=right_frame, path_mask=True)
        except Exception:
            pass
        finally:
            if resume_frame is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(resume_frame))

    def _retroactively_verify_gaps(self, cap, history, start_frame, end_frame,
                                   frame_w, frame_h, reader, force_check=False):
        """跨影格批次 OCR 回溯 + 自適應路徑最佳化。

        核心設計：
        1. 先按影格收集所有 ID 的工作，同一影格只解碼一次、只執行一次全畫面 OCR。
        2. 不用固定半徑直接淘汰遠距離候選；大位移候選仍可進入路徑評分。
        3. 每個錨點區間用動態規劃比較整段候選路徑，考慮位置、位移、加速度、框尺寸與 OCR 信心。
        4. 線性插值只作為弱先驗，不會把搜尋限制在插值框附近。
        5. 未確認的影格保留 None，交給既有的路徑補遮策略；不把推估位置冒充 OCR 確認。
        """
        checked = matched = 0
        max_gap = max(0, int(self.RETRO_OCR_MAX_GAP))
        target = self._normalize(self.target_text)
        if max_gap <= 0 or not target or reader is None:
            self._report('retroactive_check', '批次回溯略過：OCR、目標文字或回溯上限未設定',
                         end_frame, retro_checked=0, retro_matched=0, max_gap=max_gap,
                         batch_backfill=True)
            return 0

        # 一段 segment 是同一 ID 的兩個可信錨點及其中間缺口。
        segments = []
        frame_jobs = {}
        for track_id, track_history in history.items():
            valid_frames = sorted(f for f, roi in track_history.items() if roi is not None)
            for left, right in zip(valid_frames, valid_frames[1:]):
                gap = int(right) - int(left) - 1
                if gap <= 0 or gap > max_gap:
                    continue
                frames = [f for f in range(int(left) + 1, int(right))
                          if int(start_frame) <= f <= int(end_frame)]
                if not frames:
                    continue
                segment = {
                    'track_id': track_id, 'history': track_history,
                    'left': int(left), 'right': int(right),
                    'left_roi': tuple(track_history[left]), 'right_roi': tuple(track_history[right]),
                    'frames': frames,
                }
                segments.append(segment)
                for f in frames:
                    if force_check:
                        track_history[f] = None
                    elif track_history.get(f) is not None:
                        continue
                    frame_jobs.setdefault(f, True)

        if not frame_jobs:
            self._report('retroactive_check', '自適應批次回溯完成：沒有符合條件的缺口影格',
                         end_frame, retro_checked=0, retro_matched=0,
                         max_gap=max_gap, batch_backfill=True)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(end_frame))
            return 0

        # 每個獨立影格只 OCR 一次，保存該影格全部符合目標文字的候選框。
        detections_by_frame = {}
        frames_to_scan = sorted(frame_jobs)
        for frame_no in frames_to_scan:
            if self.was_cancelled:
                break
            frame = self._read_frame_at(cap, frame_no)
            if frame is None:
                detections_by_frame[frame_no] = []
                continue
            try:
                detections, _ = self._ocr(reader, frame, frame_no, full_frame=True, robust=True)
            except Exception as exc:
                detections = []
                self._report('retroactive_ocr_error',
                    f'批次回溯 F{frame_no} OCR 失敗：{exc}', frame_no,
                    batch_backfill=True)
            detections_by_frame[frame_no] = list(detections or [])
            checked += 1
            self._report('retroactive_frame',
                f'自適應回溯 OCR：F{frame_no}，候選 {len(detections_by_frame[frame_no])}',
                frame_no, detections=len(detections_by_frame[frame_no]), batch_backfill=True)

        frame_diag = max(1.0, math.hypot(float(frame_w), float(frame_h)))

        def center(box):
            return self._center(box)

        def box_size_cost(box, anchor):
            # 對數比例讓尺寸變化的懲罰平滑，不會因尺度變化就直接拒絕候選。
            bw, bh = max(2.0, float(box[2])), max(2.0, float(box[3]))
            aw, ah = max(2.0, float(anchor[2])), max(2.0, float(anchor[3]))
            return abs(math.log(bw / aw)) + abs(math.log(bh / ah))

        def candidate_conf(det):
            try:
                return max(0.0, min(1.0, float(det[2]))) if len(det) > 2 else 0.5
            except Exception:
                return 0.5

        # 逐段做 Viterbi/動態規劃。每個中間影格都有 OCR 候選；None 是允許的空缺狀態。
        for segment in segments:
            if self.was_cancelled:
                break
            track_history = segment['history']
            raw_track_id = segment['track_id']
            display_track_id = (raw_track_id[0]
                                if isinstance(raw_track_id, tuple) and len(raw_track_id) == 3
                                else raw_track_id)
            left, right = segment['left'], segment['right']
            left_roi, right_roi = segment['left_roi'], segment['right_roi']
            frames = [f for f in segment['frames']
                      if force_check or track_history.get(f) is None]
            if not frames:
                continue

            # 每一層的狀態為候選 OCR 框索引；-1 代表該影格不採信 OCR，留給補遮策略。
            layers = []
            for f in frames:
                dets = detections_by_frame.get(f, [])
                frame = self._read_frame_at(cap, f)
                if frame is None:
                    layers.append([{'roi': None, 'det': None, 'emission': 1.05, 'confidence': 0.0}])
                    continue
                candidates = []
                ratio = (f - left) / float(max(1, right - left))
                interp = tuple(left_roi[i] * (1-ratio) + right_roi[i] * ratio for i in range(4))
                ix, iy, iw, ih = interp
                interp_box = (ix, iy, max(2.0, iw), max(2.0, ih))
                interp_cx, interp_cy = center(interp_box)
                # 保留全部目標文字候選，不用固定搜尋框/固定最大距離截斷。
                for det in dets:
                    try:
                        roi = tuple(map(float, self._clip(frame, det[0])))
                        if roi[2] < 2 or roi[3] < 2:
                            continue
                        dcx, dcy = center(roi)
                        distance = math.hypot(dcx - interp_cx, dcy - interp_cy) / frame_diag
                        confidence = candidate_conf(det)
                        # 插值距離只佔較低權重，遠距離不會被硬性淘汰。
                        emission = 0.90 * min(2.0, distance * 5.0)
                        emission += 0.16 * min(3.0, box_size_cost(roi, left_roi))
                        emission += 0.16 * min(3.0, box_size_cost(roi, right_roi))
                        emission += 0.55 * (1.0 - confidence)
                        candidates.append({'roi': roi, 'det': det, 'emission': emission,
                                           'confidence': confidence})
                    except Exception:
                        continue
                # 不強迫每幀一定要配對，降低錯誤 OCR 框被硬塞進軌跡的風險。
                candidates.append({'roi': None, 'det': None, 'emission': 1.05, 'confidence': 0.0})
                layers.append(candidates)

            if not layers:
                continue

            # 每一層的累積成本與回溯指標。
            costs = []
            parents = []
            previous_states = [{'roi': left_roi, 'emission': 0.0, 'confidence': 1.0}]
            previous_costs = [0.0]
            previous_frame = left
            for layer_index, (frame_no, layer) in enumerate(zip(frames, layers)):
                current_costs, current_parents = [], []
                dt = max(1, int(frame_no) - int(previous_frame))
                for ci, cand in enumerate(layer):
                    best_cost, best_parent = float('inf'), 0
                    for pi, prev in enumerate(previous_states):
                        prev_cost = previous_costs[pi]
                        if cand['roi'] is None:
                            transition = 0.10 if prev.get('roi') is None else 0.22
                        elif prev.get('roi') is None:
                            transition = 0.18
                        else:
                            pcx, pcy = center(prev['roi'])
                            ccx, ccy = center(cand['roi'])
                            displacement = math.hypot(ccx-pcx, ccy-pcy) / frame_diag
                            # 飽滿型位移成本：大幅移動增加成本，但不會被硬門檻淘汰。
                            transition = 0.72 * min(2.0, displacement * 5.0 / max(1.0, math.sqrt(dt)))
                            transition += 0.12 * min(3.0, box_size_cost(cand['roi'], prev['roi']))
                            # 用上一段速度做柔性加速度懲罰；只在已有前一個移動狀態時生效。
                            if pi < len(previous_states) and prev.get('prev_center') is not None:
                                vx = (ccx-pcx) / dt
                                vy = (ccy-pcy) / dt
                                pvx, pvy = prev['velocity']
                                accel = math.hypot(vx-pvx, vy-pvy) / frame_diag
                                transition += 0.12 * min(2.0, accel * 8.0)
                        value = prev_cost + float(cand['emission']) + transition
                        if value < best_cost:
                            best_cost, best_parent = value, pi
                    current_costs.append(best_cost)
                    current_parents.append(best_parent)
                costs.append(current_costs)
                parents.append(current_parents)

                # 為下一層狀態附上速度資訊，以便評估加速度；空缺狀態不虛構速度。
                next_states = []
                for ci, cand in enumerate(layer):
                    state = {'roi': cand['roi'], 'emission': cand['emission'],
                             'confidence': cand['confidence'], 'prev_center': None,
                             'velocity': (0.0, 0.0)}
                    parent_state = previous_states[current_parents[ci]]
                    if cand['roi'] is not None and parent_state.get('roi') is not None:
                        cc = center(cand['roi'])
                        pc = center(parent_state['roi'])
                        state['prev_center'] = pc
                        state['velocity'] = ((cc[0]-pc[0]) / dt, (cc[1]-pc[1]) / dt)
                    next_states.append(state)
                previous_states, previous_costs = next_states, current_costs
                previous_frame = frame_no

            # 將右錨點納入終端成本，避免只看前向局部最優。
            terminal_costs = []
            for state, base_cost in zip(previous_states, previous_costs):
                if state.get('roi') is None:
                    terminal_costs.append(base_cost + 0.35)
                else:
                    c1, c2 = center(state['roi']), center(right_roi)
                    d = math.hypot(c1[0]-c2[0], c1[1]-c2[1]) / frame_diag
                    terminal_costs.append(base_cost + 0.80 * min(2.0, d * 5.0))
            idx = min(range(len(terminal_costs)), key=terminal_costs.__getitem__)
            selected = [None] * len(layers)
            for li in range(len(layers)-1, -1, -1):
                selected[li] = idx
                idx = parents[li][idx]

            for f, layer, choice in zip(frames, layers, selected):
                cand = layer[choice]
                if cand['roi'] is None:
                    continue
                frame = self._read_frame_at(cap, f)
                if frame is None:
                    continue
                measured = tuple(map(int, self._clip(frame, cand['roi'])))
                track_history[f] = measured
                matched += 1
                self._draw_reverse_check(frame, f, display_track_id, measured, measured,
                                         '自適應路徑回溯：OCR 候選經跨影格評分')
                self._report('retroactive_match',
                    f'自適應路徑回溯補到目標：F{f}，ID {display_track_id}', f,
                    track_id=display_track_id, roi=measured, retro_checked=checked,
                    retro_matched=matched, batch_backfill=True, adaptive_path=True)

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(end_frame))
        self._report('retroactive_check',
            f'自適應多目標批次回溯完成：共 OCR {checked} 個不同影格，確認 {matched} 個目標影格',
            end_frame, retro_checked=checked, retro_matched=matched,
            max_gap=max_gap, batch_backfill=True, adaptive_path=True)
        return checked

    def _retroactively_search_before_first_detection(self, cap, history, start_frame,
                                                     end_frame, frame_w, frame_h, reader):
        """以影格為單位批次回溯多個目標，避免每個 ID 重複解碼與 OCR。

        每個反向影格只讀取一次，並對該影格執行一次全畫面 OCR；再將符合目標文字的
        OCR 框，依各 ID 的光流預測位置進行一對一配對。沒有 OCR 確認時，仍保留
        原本的反向預測遮罩行為。只有 OCR 確認的框會作為可靠量測位置。
        """
        target = self._normalize(self.target_text)
        max_back = max(0, int(self.RETRO_OCR_MAX_GAP))
        checked = matched = tracks_checked = 0
        if reader is None or not target or max_back <= 0:
            return {"checked": 0, "matched": 0, "tracks": 0}

        # 每個目標各自維護反向追蹤狀態，但共用影格讀取與 OCR 結果。
        states = {}
        for track_id, track_history in history.items():
            known = sorted(f for f, roi in track_history.items() if roi is not None)
            if not known:
                continue
            first_frame = known[0]
            scan_start = max(int(start_frame), first_frame - max_back)
            if first_frame <= scan_start:
                continue
            anchor_frame = self._read_frame_at(cap, first_frame)
            if anchor_frame is None:
                continue
            states[track_id] = {
                "history": track_history,
                "first_frame": first_frame,
                "scan_start": scan_start,
                "current_frame_no": first_frame,
                "current_frame": anchor_frame,
                "current_gray": cv2.cvtColor(anchor_frame, cv2.COLOR_BGR2GRAY),
                "current_roi": self._clip(anchor_frame, track_history[first_frame]),
            }
            tracks_checked += 1
            self._report(
                'retroactive_first_detection',
                f'首次偵測前回溯排程：ID {track_id}，F{first_frame} 至 F{scan_start}',
                first_frame, track_id=track_id, from_frame=first_frame,
                to_frame=scan_start, max_back=max_back, batch_backfill=True)

        if not states:
            return {"checked": 0, "matched": 0, "tracks": 0}

        earliest_frame = min(state["scan_start"] for state in states.values())
        latest_frame = max(state["first_frame"] for state in states.values()) - 1

        for frame_no in range(latest_frame, earliest_frame - 1, -1):
            active = []
            for track_id, state in states.items():
                if (state["scan_start"] <= frame_no < state["first_frame"]
                        and state["current_frame_no"] == frame_no + 1):
                    active.append((track_id, state))
            if not active:
                continue

            # 同一個反向影格只解碼一次，所有正在回溯的 ID 共用此影格。
            prev_frame = self._read_frame_at(cap, frame_no)
            if prev_frame is None:
                continue
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            predictions = {}
            search_rois = {}

            for track_id, state in active:
                current_frame = state["current_frame"]
                current_gray = state["current_gray"]
                current_roi = state["current_roi"]
                x, y, w, h = self._clip(current_frame, current_roi)
                mask = np.zeros(current_gray.shape, dtype=np.uint8)
                mask[y:y+h, x:x+w] = 255
                points = cv2.goodFeaturesToTrack(
                    current_gray, maxCorners=50, qualityLevel=0.01,
                    minDistance=3, mask=mask, blockSize=5)
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
                            if (np.isfinite(delta).all()
                                    and np.linalg.norm(delta) < max(40.0, 2.5 * max(w, h))):
                                predicted = self._clip(prev_frame, (
                                    current_roi[0] + delta[0], current_roi[1] + delta[1],
                                    current_roi[2], current_roi[3]))
                                motion_ok = True
                predictions[track_id] = (predicted, motion_ok)
                search_rois[track_id] = self._expand_roi(
                    prev_frame, predicted,
                    self.RETRO_ROI_EXPAND * (1.0 if motion_ok else 1.35))

            # 關鍵優化：本影格只跑一次全畫面 OCR，結果供所有 active ID 配對。
            detections, _ = self._ocr(
                reader, prev_frame, frame_no, full_frame=True, robust=True)
            checked += 1

            # 以距離由近到遠配對，避免同一個 OCR 框被多個 ID 重複使用。
            pairs = []
            for track_id, _state in active:
                predicted, _motion_ok = predictions[track_id]
                sx, sy, sw, sh = search_rois[track_id]
                pcx, pcy = self._center(predicted)
                for det_index, detection in enumerate(detections):
                    box = detection[0]
                    dcx, dcy = self._center(box)
                    distance = math.hypot(dcx - pcx, dcy - pcy)
                    predicted_iou = self._iou(box, predicted)
                    # A broad expanded search window is for OCR discovery only, not proof of
                    # identity. Require a plausible one-frame motion distance or real overlap.
                    diag = math.hypot(max(2, predicted[2]), max(2, predicted[3]))
                    motion_gate = max(18.0, diag * 1.25)
                    size_ratio = max(
                        box[2] / max(1.0, predicted[2]), predicted[2] / max(1.0, box[2]),
                        box[3] / max(1.0, predicted[3]), predicted[3] / max(1.0, box[3]))
                    plausible_size = size_ratio <= 2.5
                    if plausible_size and (distance <= motion_gate or predicted_iou >= 0.10):
                        # Mild size penalty helps prefer the same physical text box when
                        # neighbouring targets are close and have similar OCR strings.
                        size_penalty = max(0.0, math.log(max(1.0, size_ratio))) * diag * 0.15
                        pairs.append((distance + size_penalty, track_id, det_index))
            pairs.sort(key=lambda item: item[0])
            assigned_tracks = set()
            assigned_detections = set()
            assignment = {}
            for _distance, track_id, det_index in pairs:
                if track_id in assigned_tracks or det_index in assigned_detections:
                    continue
                assigned_tracks.add(track_id)
                assigned_detections.add(det_index)
                assignment[track_id] = detections[det_index]

            for track_id, state in active:
                predicted, motion_ok = predictions[track_id]
                detection = assignment.get(track_id)
                if detection is not None:
                    measured = tuple(map(int, self._clip(prev_frame, detection[0])))
                    state["history"][frame_no] = measured
                    state["current_roi"] = measured
                    matched += 1
                    self._draw_reverse_check(
                        prev_frame, frame_no, track_id, predicted, measured,
                        "批次首次回溯：OCR 已確認")
                    self._report(
                        'retroactive_first_match',
                        f'批次回溯找到目標：F{frame_no}，ID {track_id}', frame_no,
                        track_id=track_id, roi=measured, retro_checked=checked,
                        retro_matched=matched, batch_backfill=True)
                else:
                    # 維持既有策略：未被 OCR 確認時，仍將反向預測框寫入遮罩歷史。
                    px, py, pw, ph = self._clip(prev_frame, predicted)
                    ew = max(2, int(round(pw * self.PATH_MASK_EXPAND)))
                    eh = max(2, int(round(ph * self.PATH_MASK_EXPAND)))
                    predicted_mask = self._clip(
                        prev_frame,
                        (px + pw / 2.0 - ew / 2.0,
                         py + ph / 2.0 - eh / 2.0, ew, eh))
                    state["history"][frame_no] = tuple(map(int, predicted_mask))
                    state["current_roi"] = predicted
                    self._draw_reverse_check(
                        prev_frame, frame_no, track_id, predicted, None,
                        f"批次首次回溯，光流={'成功' if motion_ok else '備援'}")

                state["current_frame_no"] = frame_no
                state["current_frame"] = prev_frame
                state["current_gray"] = prev_gray

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(end_frame))
        self._report(
            'retroactive_first_detection',
            f'批次首次偵測前回溯完成：共檢查 {checked} 個影格，'
            f'OCR 確認 {matched} 個目標影格，處理 {tracks_checked} 個目標',
            end_frame, retro_checked=checked, retro_matched=matched,
            tracks_checked=tracks_checked, max_back=max_back, batch_backfill=True)
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

    def _interpolate_path_gaps(self, history, max_gap=None, expand=None):
        """Fill short/medium missing spans between two known ROIs with per-frame path masks.

        This is a privacy-oriented fallback, not a claim that the estimated box is the exact
        object position. OCR backfill runs first; interpolation only fills remaining gaps.
        """
        max_gap = self.MAX_PATH_INTERPOLATION_GAP if max_gap is None else max(0, int(max_gap))
        expand = self.PATH_MASK_EXPAND if expand is None else max(1.0, float(expand))
        filled = []
        if max_gap <= 0:
            return filled
        for track_id, track_history in history.items():
            known = sorted(f for f, roi in track_history.items() if roi is not None)
            for left, right in zip(known, known[1:]):
                gap = right - left - 1
                if gap <= 0 or gap > max_gap:
                    continue
                a, b = track_history[left], track_history[right]
                for frame_no in range(left + 1, right):
                    if track_history.get(frame_no) is not None:
                        continue
                    ratio = (frame_no - left) / float(right - left)
                    x = a[0] * (1-ratio) + b[0] * ratio
                    y = a[1] * (1-ratio) + b[1] * ratio
                    w = a[2] * (1-ratio) + b[2] * ratio
                    h = a[3] * (1-ratio) + b[3] * ratio
                    # Longer gaps imply more position uncertainty, so use a gap-dependent
                    # privacy margin instead of expanding every interpolated box equally.
                    gap_uncertainty = min(0.20, 0.20 * gap / float(max(1, max_gap)))
                    adaptive_expand = max(1.0, expand + gap_uncertainty)
                    ew, eh = w * adaptive_expand, h * adaptive_expand
                    track_history[frame_no] = (int(round(x + w/2 - ew/2)),
                                               int(round(y + h/2 - eh/2)),
                                               max(2, int(round(ew))), max(2, int(round(eh))))
                    filled.append((track_id, frame_no, tuple(map(int, track_history[frame_no])), left, right))
        return filled

    def run(self):
        cap = None
        data = {}
        last_frame = self.start_frame
        success = False
        try:
            self.progress_signal.emit(self.start_frame, '正在開啟影片並初始化追蹤...')
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError('無法開啟影片檔案。')
            self.frame_buffer = FrameRingBuffer(
                max_frames=self.FRAME_CACHE_MAX_FRAMES,
                max_bytes=self.FRAME_CACHE_MAX_BYTES)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if not math.isfinite(source_fps) or source_fps <= 0:
                source_fps = 0.0
            self._report('video_info', f'影片資訊：{source_fps:.2f} FPS，共 {frame_count} 幀',
                         self.start_frame, fps=source_fps, frame_count=frame_count)
            end_frame = min(self.total_frames, frame_count) if self.total_frames > 0 else frame_count
            if end_frame <= self.start_frame:
                raise RuntimeError('影片結束影格早於追蹤起始影格。')
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            ret, first_frame = cap.read()
            if not ret:
                raise RuntimeError('無法讀取追蹤起始影格。')
            self.frame_buffer.put(self.start_frame, first_frame)

            # Dense-scene initialization: preserve the user's ROI, but also scan the entire
            # first frame once in overlapping tiles so small/repeated targets are not ignored.
            tracks = {}
            next_id = 1
            hit_frame = self.start_frame
            reader = None
            if self.target_text.strip():
                self.progress_signal.emit(hit_frame, '正在初始化 RapidOCR（ONNX Runtime）...')
                reader = self._create_reader()
                self._report('ocr_engine', f"OCR 引擎已載入：{reader.get('backend', 'unknown')}", hit_frame)
            if self.initial_roi is None:
                raise RuntimeError('目前模式需要先在影片預覽畫面框選一個追蹤目標。')
            manual_roi = self._clip(first_frame, self.initial_roi)
            if manual_roi[2] < 3 or manual_roi[3] < 3:
                raise RuntimeError('框選的 ROI 太小，請重新框選較完整的目標。')

            initial_detections = self._initial_tiled_ocr_scan(reader, first_frame, hit_frame)
            matched_initial_index = None
            if initial_detections:
                mcx, mcy = self._center(manual_roi)
                manual_diag = math.hypot(manual_roi[2], manual_roi[3])
                candidates = []
                for idx, det in enumerate(initial_detections):
                    box = det[0]
                    dcx, dcy = self._center(box)
                    distance = math.hypot(mcx - dcx, mcy - dcy)
                    overlap = self._iou(manual_roi, box)
                    # The manually selected ROI remains the size/position prior. Match only
                    # the nearest plausible OCR box; do not suppress other nearby targets.
                    if overlap >= 0.08 or distance <= max(14.0, min(48.0, manual_diag * 0.45)):
                        score = distance / max(1.0, manual_diag) - 0.70 * overlap
                        candidates.append((score, idx))
                if candidates:
                    matched_initial_index = min(candidates)[1]

            # ID 1 always represents the user's explicitly selected target. If OCR sees it,
            # store its OCR box as the trusted anchor while retaining the user's ROI for masks.
            manual_text, manual_conf = self.target_text, 1.0
            if matched_initial_index is not None:
                _box, manual_text, manual_conf = initial_detections[matched_initial_index]
            tracks[next_id] = self._new_track(
                first_frame, manual_roi, next_id, hit_frame, manual_text, manual_conf)
            tracks[next_id]['state'] = 'MANUAL_INIT'
            manual_id = next_id
            next_id += 1

            # Every distinct first-frame OCR detection becomes its own stable ID. Only the
            # single OCR box paired with the manual ROI is skipped; neighbouring repeated text
            # is intentionally not discarded just because it is close to the manual selection.
            initial_id_count = 1
            for idx, det in enumerate(initial_detections):
                if idx == matched_initial_index:
                    continue
                box, text, confidence = det
                if any(self._iou(box, tr['roi']) >= 0.82 for tr in tracks.values()):
                    continue
                tracks[next_id] = self._new_track(first_frame, box, next_id, hit_frame, text, confidence)
                tracks[next_id]['state'] = 'OCR_CONFIRMED'
                next_id += 1
                initial_id_count += 1

            self._report(
                'locked',
                f'起始影格初始化完成：手動目標 ID {manual_id}，第一幀 OCR 找到 {len(initial_detections)} 個候選，建立 {len(tracks)} 個獨立 ID。',
                hit_frame, roi=manual_roi, mode='INIT',
                initial_detection_count=len(initial_detections), initial_track_count=len(tracks),
                initialization_scan=True)
            if self.target_text.strip():
                self.progress_signal.emit(
                    hit_frame,
                    f'起始影格已建立 {len(tracks)} 個目標 ID；後續若仍漏檢，會在新目標被找到後回溯補查前段。')

            frame_h, frame_w = first_frame.shape[:2]

            # Start output history at the initialization frame. The manual ROI is the mask
            # geometry; its matched OCR box, when available, is the trusted text anchor.
            history = {tid: {hit_frame: tuple(t['roi'])} for tid, t in tracks.items()}
            ocr_history = {tid: {hit_frame: tuple(t['roi'])} for tid, t in tracks.items()}
            if matched_initial_index is not None:
                ocr_history[manual_id][hit_frame] = tuple(map(int, initial_detections[matched_initial_index][0]))
            for tid, track in tracks.items():
                if tid != manual_id:
                    ocr_history[tid][hit_frame] = tuple(map(int, track['roi']))
            data[hit_frame] = [tuple(t['roi']) for t in tracks.values()]
            self._draw_monitor(first_frame, hit_frame, tracks, 'TRACK MODE')
            previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            cached_global_motion = cached_camera_dx = cached_camera_dy = 0.0
            cached_global_points = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, hit_frame + 1)
            all_created_track_ids = []

            for frame_no in range(hit_frame+1, end_frame):
                if self.isInterruptionRequested():
                    self.was_cancelled = True
                    self.cancel_frame = last_frame
                    self._report('cancelled', '使用者中止追蹤；保留已處理影格，並停止後續處理。', last_frame, mode='TRACK MODE', cancelled=True)
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                self.frame_buffer.put(frame_no, frame)
                last_frame = frame_no
                current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_h, frame_w = frame.shape[:2]
                detections = []
                need_local_recovery = []
                reacquired_events = []  # (track_id, prior_lost_frames, current_frame)
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
                    # 優先使用 CSRT 結果，再由光流結果交叉驗證。
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
                                        # 兩種追蹤結果一致時取平均；差異大時選擇較接近預測框者。
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
                    # 針對密集小字優化：縮小預測容忍度，避免 CSRT/KLT 鎖死在相鄰的靜止干擾物上
                    prediction_gate = max(18.0, diag * 1.1, float(track['speed']) * 2.0 + diag * 0.4)
                    if flow_ok and pred_distance > prediction_gate:
                        flow_ok = False
                        track['loss_reason'] = 'prediction_gate_rejected_possible_identity_switch'
                    # 根據外觀、位移與影格品質驗證候選框。
                    anchor = track.get('anchor_appearance')
                    previous_appearance = track.get('appearance')
                    candidate_appearance = self._appearance(frame, candidate) if flow_ok else None
                    anchor_score = self._appearance_score(anchor, candidate_appearance)
                    recent_appearance_score = self._appearance_score(previous_appearance, candidate_appearance)
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
                        'recent_appearance_score': round(recent_appearance_score, 3),
                    }
                    track['last_diagnostics'] = diag_info
                    # Repetitive neighbouring text can look trackable to CSRT. Require
                    # agreement with both the initial appearance and the most recent trusted
                    # crop; CSRT-only updates get a stricter threshold than corroborated flow.
                    if flow_ok and anchor is not None and anchor_score < 0.28:
                        flow_ok = False
                        track['loss_reason'] = 'anchor_appearance_mismatch'
                    if flow_ok and previous_appearance is not None:
                        recent_gate = 0.34 if csrt_ok and flow_count < self.MIN_FEATURE_POINTS else 0.25
                        if recent_appearance_score < recent_gate:
                            flow_ok = False
                            track['loss_reason'] = 'recent_appearance_mismatch'
                    if flow_ok and move > max_move:
                        flow_ok = False
                        track['loss_reason'] = 'large_displacement_gate'
                    if not csrt_ok and flow_count < self.MIN_FEATURE_POINTS:
                        flow_ok = False
                        track['loss_reason'] = 'few_features_or_motion_blur'
                    if csrt_flow_delta is not None and csrt_flow_delta > max(25.0, diag * 0.5):
                        flow_ok = False
                        track['loss_reason'] = 'tracker_disagreement'
                    if camera_moving and move < global_motion * 0.35:
                        flow_ok = False
                        track['loss_reason'] = 'possible_camera_motion'
                    if source_fps and source_fps < 20.0 and move > diag * 0.8 and (flow_count < 8 or not csrt_ok):
                        flow_ok = False
                        track['loss_reason'] = 'low_source_fps_plus_fast_motion'
                    elif move > diag * 0.8 and (flow_count < 8 or not csrt_ok):
                        flow_ok = False
                        track['loss_reason'] = 'possible_fast_target_motion'
                    rx, ry, rw, rh = candidate
                    if rx <= 1 or ry <= 1 or rx + rw >= frame_w - 1 or ry + rh >= frame_h - 1:
                        flow_ok = False
                        track['loss_reason'] = 'roi_near_frame_edge_or_target_leaving_view'
                    if blur_score < 35.0 and not csrt_ok:
                        flow_ok = False
                        track['loss_reason'] = 'possible_motion_blur_or_low_texture'
                    if flow_ok and move <= max_move:
                        stable_candidate = self._stabilize_roi(frame, track, candidate, source='visual')
                        track['roi'] = stable_candidate
                        try:
                            track['kalman'].correct(stable_candidate)
                        except Exception:
                            track['kalman'] = SortMotionModel(stable_candidate)
                        track['lost'] = 0
                        track['loss_reason'] = ''
                        track['state'] = 'FAST_MOTION' if move > diag*.8 else 'TRACKING'
                        track['appearance'] = self._appearance(frame, stable_candidate)
                    else:
                        track['lost'] = max(1, track['lost'])
                        track['state'] = 'LOST' if track['lost'] >= 3 else 'RECOVERY'
                        reason = track.get('loss_reason') or 'tracker_measurement_failed'
                        if track['lost'] == 1 or track['lost'] % 10 == 0:
                            self._report('tracking_lost',
                                f'ID {tid} 追蹤遺失：原因={reason}，移動量={move:.1f}px/frame；'
                                f'光流點={flow_count}，模糊分數={blur_score:.1f}，全域移動={global_motion:.1f}px/frame',
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
                    # Recovery uses a short fixed retry interval. Exponential backoff can
                    # leave fast-moving targets unsearched for several seconds.
                    retry_interval = max(1, int(self.LOCAL_RECOVERY_INTERVAL))
                    if frame_no - int(tr.get('last_local_search_frame', -10**9)) >= retry_interval:
                        recovery_due.append(tid)

                if reader is not None:
                    # Full-frame OCR only at sparse keyframes (default every 5 frames),
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
                            track['last_local_search_frame'] = frame_no
                            track['recovery_attempts'] = int(track.get('recovery_attempts', 0)) + 1
                            attempts = int(track['recovery_attempts'])
                            # Expand local search progressively as the target may have moved
                            # farther than the motion model predicted.限制膨脹比例避免鄰居干擾。
                            base_scale = self.RECOVERY_SCALE * (1.0 + 0.35 * max(0, attempts - 1))
                            scale = min(base_scale, 4.5)
                            local_roi = self._expand_roi(frame, pred, scale)
                            robust_ocr = attempts >= 2
                            local, _ = self._ocr(reader, frame, frame_no, roi=local_roi, robust=robust_ocr)
                            detections.extend(local)
                            ran_ocr = True
                            track['recovery_note'] = f'局部恢復第 {attempts} 次，搜尋倍率 {scale:.1f}x'
                            self._report('recovery_search',
                                f'ID {tid} 局部恢復搜尋第 {attempts} 次，搜尋倍率 {scale:.1f}x，候選 {len(local)}',
                                frame_no, track_id=tid, attempt=attempts, search_roi=local_roi,
                                detections=len(local), recovery_scale=scale)
                            # If several progressively larger local searches fail, fall back
                            # to a robust full-frame OCR pass. This catches targets that moved
                            # outside the predicted ROI after a sudden motion/occlusion.
                            if not local and attempts >= 2 and (attempts - 2) % 2 == 0:
                                global_recovery, _ = self._ocr(
                                    reader, frame, frame_no, full_frame=True, robust=True)
                                detections.extend(global_recovery)
                                track['recovery_note'] = f'全畫面恢復搜尋（第 {attempts} 次嘗試）'
                                self._report('recovery_global_fallback',
                                    f'ID {tid} 局部搜尋未找到目標，改做全畫面 OCR；候選 {len(global_recovery)}',
                                    frame_no, track_id=tid, attempt=attempts,
                                    detections=len(global_recovery), mode='GLOBAL_RECOVERY')

                # Keep track of brand-new OCR tracks so we can immediately trace them backward.
                created_track_ids_this_frame = []

                # Hungarian matching (SORT association) between OCR boxes and predicted tracks.
                if detections and tracks:
                    tids = list(tracks.keys())
                    cost = np.full((len(tids), len(detections)), 1e6, dtype=np.float64)
                    
                    for i, tid in enumerate(tids):
                        tr = tracks[tid]
                        pred = tr.get('predicted_roi') or tr['roi']
                        pc = self._center(pred)
                        gate = max(35.0, math.hypot(pred[2], pred[3])*1.75 + tr['speed']*1.5)

                        old_center = self._center(tr['roi'])
                        pred_center = self._center(pred)
                        expected_vec = (pred_center[0] - old_center[0], pred_center[1] - old_center[1])
                        expected_speed = math.hypot(expected_vec[0], expected_vec[1])

                        for j, det in enumerate(detections):
                            box, text, conf = det
                            dc = self._center(box)
                            dist = math.hypot(pc[0]-dc[0], pc[1]-dc[1])
                            iou = self._iou(pred, box)
                            app = self._appearance_score(tr.get('appearance'), self._appearance(frame, box))
                            text_match = (not self.target_text.strip() or
                                          self._target_text_evidence(text, conf)['decision'] == 'ACCEPT')
                            
                            dir_penalty = 0.0
                            if expected_speed > 3.0:
                                actual_vec = (dc[0] - old_center[0], dc[1] - old_center[1])
                                actual_mag = math.hypot(actual_vec[0], actual_vec[1])
                                if actual_mag > 0:
                                    cos_sim = (expected_vec[0]*actual_vec[0] + expected_vec[1]*actual_vec[1]) / (expected_speed * actual_mag)
                                    dir_penalty = 0.20 * (1.0 - cos_sim)
                                    
                            # 尺寸變化懲罰 (Size Penalty)
                            pw, ph = max(2.0, pred[2]), max(2.0, pred[3])
                            bw, bh = max(2.0, box[2]), max(2.0, box[3])
                            size_change = max(abs(bw - pw) / pw, abs(bh - ph) / ph)
                            size_penalty = min(2.0, size_change * 1.5) if size_change > 0.5 else 0.0

                            lost_track = int(tr.get('lost', 0)) > 0
                            lost_gate = max(gate, min(math.hypot(frame_w, frame_h) * 0.45, 900.0))
                            effective_gate = lost_gate if (lost_track and text_match) else gate
                            
                            # Explicit association cost: geometry + appearance + motion.
                            # Fuzzy inference is intentionally NOT involved in ID assignment.
                            # A candidate outside the allowed motion gate cannot claim this ID.
                            if dist > effective_gate:
                                cost[i, j] = 1e6
                                continue

                            motion_cost = min(1.0, dist / max(1.0, effective_gate))
                            appearance_cost = 1.0 - float(np.clip(app, 0.0, 1.0))
                            iou_cost = 1.0 - float(np.clip(iou, 0.0, 1.0))
                            direction_cost = min(1.0, max(0.0, dir_penalty) / 0.40)
                            normalized_size_change = min(1.0, max(0.0, size_change))

                            base_cost = (
                                0.42 * motion_cost
                                + 0.24 * iou_cost
                                + 0.20 * appearance_cost
                                + 0.09 * direction_cost
                                + 0.05 * normalized_size_change
                            )
                            if not text_match:
                                base_cost += 0.25
                            cost[i, j] = base_cost
                                    
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
                        old_lost_count = int(tr.get('lost', 0))
                        if old_lost_count > 0:
                            reacquired_events.append((tid, old_lost_count, frame_no))
                        old_center, new_center = self._center(tr['roi']), self._center(box)
                        moved = math.hypot(new_center[0]-old_center[0], new_center[1]-old_center[1])
                        raw_ocr_box = self._clip(frame, box)
                        corrected_box = self._stabilize_roi(
                            frame, tr, raw_ocr_box, source='ocr', confidence=conf)
                        tr.update({'roi': corrected_box, 'predicted_roi': corrected_box,
                                   'visual_tracker': self._create_visual_tracker(frame, corrected_box),
                                   'lost': 0, 'state': 'OCR_CONFIRMED', 'text': text,
                                   'confidence': conf, 'speed': .7*tr['speed']+.3*moved,
                                   'last_seen_frame': frame_no, 'last_ocr_frame': frame_no,
                                   'last_local_search_frame': frame_no, 'recovery_attempts': 0,
                                   'appearance': self._appearance(frame, corrected_box),
                                   'points': self._features(current_gray, corrected_box)})
                        try:
                            tr['kalman'].correct(corrected_box)
                        except Exception:
                            tr['kalman'] = SortMotionModel(corrected_box)
                        history.setdefault(tid, {})[frame_no] = tuple(map(int, corrected_box))
                        ocr_history.setdefault(tid, {})[frame_no] = tuple(map(int, corrected_box))
                        used_dets.add(j)
                    # Additional OCR instances become new tracks, but manual single ROI mode
                    # only creates new IDs when OCR matches the requested target text.
                    for j, det in enumerate(detections):
                        if j in used_dets:
                            continue
                        # Only suppress a near-identical duplicate if another track was OCR-updated
                        # on THIS frame. A stale/predicted neighbouring track must not block a
                        # new instance of the same text in a row of identical targets.
                        if any(self._iou(det[0], tr['roi']) > .90 and
                               int(tr.get('last_ocr_frame', -10**9)) == frame_no
                               for tr in tracks.values()):
                            continue
                        if self.target_text.strip():
                            target_evidence = self._target_text_evidence(det[1], det[2])
                            if target_evidence['decision'] != 'ACCEPT':
                                continue
                        tracks[next_id] = self._new_track(frame, det[0], next_id, frame_no, det[1], det[2])
                        history[next_id] = {}
                        ocr_history[next_id] = {frame_no: tuple(map(int, det[0]))}
                        created_track_ids_this_frame.append(next_id)
                        next_id += 1

                # Export masks for live tracks. During a short loss, use a growing
                # conservative mask around the Kalman prediction instead of exposing
                # the target immediately. Long losses are marked for manual review.
                frame_rois, monitor_tracks = [], {}
                for tid, tr in tracks.items():
                    if tr['lost'] > self.CONSERVATIVE_MASK_FRAMES:
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
                        scale = min(2.4, 1.0 + 0.14 * tr['lost'])
                        mask_roi = self._clip(frame, (cx-pw*scale/2, cy-ph*scale/2,
                                                      pw*scale, ph*scale))
                    # A conservative prediction may still be used for immediate masking, but it is
                    # NOT considered OCR/measurement-confirmed history. Mark it as a gap so the
                    # segment backfill can re-check it instead of silently skipping it.
                    history.setdefault(tid, {})[frame_no] = mask_roi if tr['lost'] == 0 else None
                    # Preserve separate masks for adjacent identical targets. Exact duplicate
                    # boxes are already controlled by one-to-one OCR assignment; overlapping
                    # masks are safer than silently dropping a neighbouring target.
                    if mask_roi is not None:
                        if not any(self._iou(mask_roi, old) > .985 for old in frame_rois):
                            frame_rois.append(tuple(map(int, mask_roi)))
                data[frame_no] = frame_rois

                # 新目標的首次偵測前回溯延後到正向追蹤結束後批次執行，
                # 讓不同影格發現的多個 ID 可以共用反向影格與 OCR 結果。
                if reader is not None and self.target_text.strip() and created_track_ids_this_frame:
                    all_created_track_ids.extend(created_track_ids_this_frame)
                    self._report(
                        'first_detection_backfill_queued',
                        f'已排入批次首次回溯：新增 {len(created_track_ids_this_frame)} 個目標，'
                        f'目前累計 {len(set(all_created_track_ids))} 個 ID。',
                        frame_no, track_ids=list(created_track_ids_this_frame),
                        queued_track_count=len(set(all_created_track_ids)),
                        first_detection_backfill=True, batch_backfill=True)

                # 若同一影格重新找回多個目標，將它們合併成一次批次回溯，
                # 讓同一個 frame 的全畫面 OCR 結果可由所有 ID 共用。
                if (self.IMMEDIATE_RETRACE_ON_REACQUISITION and reader is not None
                        and reacquired_events and self.target_text.strip()):
                    _batch_local_history = {}
                    _batch_meta = {}
                    _batch_start = frame_no
                    _batch_end = frame_no
                    for _tid, _lost_count, _reacquired_frame in reacquired_events:
                        _track_history = history.get(_tid, {})
                        _trusted_history = ocr_history.get(_tid, {})
                        _anchors = [f for f, r in _trusted_history.items()
                                    if f < _reacquired_frame and r is not None]
                        if not _anchors:
                            continue
                        _left_anchor = max(_anchors)
                        _gap = _reacquired_frame - _left_anchor - 1
                        if _gap <= 0 or _gap > self.RETRO_OCR_MAX_GAP:
                            continue
                        _batch_local_history[_tid] = {
                            f: (_trusted_history.get(f) if f == _left_anchor else
                                _trusted_history.get(f) if f == _reacquired_frame else None)
                            for f in range(_left_anchor, _reacquired_frame + 1)
                        }
                        # The current reacquisition must be OCR-confirmed before it can
                        # serve as the right anchor; do not use a predicted tracker box.
                        if _trusted_history.get(_reacquired_frame) is None:
                            _batch_local_history.pop(_tid, None)
                            continue
                        self._processed_retro_pairs.add((int(_tid), int(_left_anchor), int(_reacquired_frame)))
                        _batch_meta[_tid] = (_track_history, _left_anchor, _reacquired_frame, _gap)
                        _batch_start = min(_batch_start, _left_anchor)
                        _batch_end = max(_batch_end, _reacquired_frame)
                        self._report('reacquisition_backfill_start',
                            f'目標重新找到：加入批次回溯 ID {_tid}，F{_left_anchor} 至 F{_reacquired_frame}，缺口 {_gap} 幀。',
                            _reacquired_frame, track_id=_tid, gap=_gap,
                            from_frame=_left_anchor, to_frame=_reacquired_frame,
                            immediate_backfill=True, batch_backfill=True)

                    if _batch_local_history:
                        _checked = self._retroactively_verify_gaps(
                            cap, _batch_local_history, _batch_start, _batch_end,
                            frame_w, frame_h, reader, force_check=True)
                        _all_path_filled = 0
                        for _tid, _local_history in _batch_local_history.items():
                            _track_history, _left_anchor, _reacquired_frame, _gap = _batch_meta[_tid]
                            for _f in range(_left_anchor + 1, _reacquired_frame):
                                _track_history[_f] = None
                            for _f, _roi in _local_history.items():
                                if _roi is not None:
                                    _track_history[_f] = tuple(map(int, _roi))
                            _path_filled = self._interpolate_path_gaps(
                                {_tid: _local_history}, max_gap=self.MAX_PATH_INTERPOLATION_GAP)
                            _all_path_filled += len(_path_filled)
                            for _idx, (_path_tid, _path_frame, _path_roi, _left, _right) in enumerate(_path_filled):
                                if _idx % 3 == 0:
                                    _left_roi = _local_history.get(_left)
                                    _right_roi = _local_history.get(_right)
                                    if _left_roi is not None and _right_roi is not None:
                                        self._draw_path_backfill(cap, _path_frame, _tid, _path_roi,
                                            _left, _right, _left_roi, _right_roi, resume_frame=frame_no + 1)
                            for _f, _roi in _local_history.items():
                                if _roi is not None:
                                    _track_history[_f] = tuple(map(int, _roi))
                            for _f in range(_left_anchor, _reacquired_frame + 1):
                                data[_f] = [tuple(map(int, _h[_f])) for _h in history.values()
                                            if _h.get(_f) is not None]
                            self._report('reacquisition_backfill_done',
                                f'批次立即回溯完成：ID {_tid}，路徑補遮 {_path_filled.__len__()} 幀。',
                                _reacquired_frame, track_id=_tid, retro_checked=_checked,
                                path_filled=len(_path_filled), gap=_gap,
                                immediate_backfill=True, batch_backfill=True)
                        self._report('reacquisition_batch_done',
                            f'多目標重新取得批次回溯完成：{len(_batch_local_history)} 個目標共用 {_checked} 個影格 OCR。',
                            _batch_end, tracks_checked=len(_batch_local_history),
                            retro_checked=_checked, path_filled=_all_path_filled,
                            immediate_backfill=True, batch_backfill=True)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no + 1)

                # Publish live frames regularly, plus every OCR/recovery event.
                if frame_no % self.MONITOR_INTERVAL == 0 or ran_ocr or need_local_recovery:
                    self._draw_monitor(frame, frame_no, monitor_tracks, 'RECOVERY MODE' if need_local_recovery else 'TRACK MODE')

                # Every segment, perform OCR checks on recent gaps while forward tracking continues.
                # The backward scan is bounded to this segment plus the configured gap context.
                if (reader is not None and self.target_text.strip()
                        and self.RETROACTIVE_SEGMENT_INTERVAL > 0
                        and (frame_no - hit_frame + 1) % self.RETROACTIVE_SEGMENT_INTERVAL == 0):
                    segment_start = max(hit_frame, frame_no - self.RETROACTIVE_SEGMENT_INTERVAL + 1 - self.RETRO_OCR_MAX_GAP)
                    # Only adjacent OCR-confirmed anchors define a valid backfill gap.
                    # Never fall back to orange/predicted tracker boxes as anchor points.
                    segment_history = self._pending_anchor_pair_histories(
                        ocr_history, segment_start, frame_no)
                    self._report('retroactive_check',
                        f'開始分段回溯：F{segment_start} 至 F{frame_no}，檢查最近一段的追蹤缺口。',
                        frame_no, segment_start=segment_start, segment_end=frame_no,
                        retroactive_segment=True)
                    checked_now = self._retroactively_verify_gaps(
                        cap, segment_history, segment_start, frame_no, frame_w, frame_h, reader,
                        force_check=False)
                    # Merge only the pair's gap into its original ID history.
                    for _pair_key, _segment_track_history in segment_history.items():
                        _tid, _left_anchor, _right_anchor = _pair_key
                        _full_history = history.setdefault(_tid, {})
                        for _f, _roi in _segment_track_history.items():
                            _full_history[_f] = tuple(map(int, _roi)) if _roi is not None else None
                    # Reflect retroactive detections in the data that will be exported as mosaic.
                    for _f in range(segment_start, frame_no + 1):
                        data[_f] = [tuple(map(int, _h[_f])) for _h in history.values()
                                    if _h.get(_f) is not None]
                    self._report('retroactive_segment_done',
                        f'分段回溯結束：本段檢查 {checked_now} 個缺漏影格；已更新馬賽克追蹤資料。',
                        frame_no, retro_checked=checked_now, segment_start=segment_start,
                        segment_end=frame_no, retroactive_segment=True)
                    # The backward OCR helper seeks around the file; restore the next forward frame.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no + 1)
                if ran_ocr or need_local_recovery:
                    diagnostic_tracks = [t for t in tracks.values() if t.get('lost', 0) > 0]
                    diagnosis = ', '.join(f"ID {t['id']}:{t.get('loss_reason') or 'recovering'}" for t in diagnostic_tracks[:4])
                    summary = f'追蹤診斷={diagnosis}' if diagnosis else ''
                    self._report('tracking', f'影格 {frame_no}：追蹤數 {len(tracks)}，待恢復 {len(need_local_recovery)}，OCR 候選 {len(detections)}；{summary}',
                                 frame_no, target_count=len(tracks), lost_count=len(need_local_recovery),
                                 detection_count=len(detections), diagnosis=diagnosis,
                                 global_motion=global_motion, camera_motion_suspected=camera_moving,
                                 fps=source_fps, mode='RECOVERY MODE' if need_local_recovery else 'TRACK MODE')
                self.progress_signal.emit(frame_no, f'追蹤中 F{frame_no}/{end_frame}；本幀遮罩 {len(frame_rois)} 個')
                previous_gray = current_gray

            # 批次回溯所有正向追蹤期間新建立的目標；每個反向影格只 OCR 一次。
            unique_created_ids = list(dict.fromkeys(all_created_track_ids))
            if reader is not None and self.target_text.strip() and unique_created_ids:
                batch_history = {
                    _tid: history.get(_tid, {}) for _tid in unique_created_ids
                    if history.get(_tid)
                }
                self._report(
                    'first_detection_backfill_start',
                    f'開始批次首次偵測前回溯：{len(batch_history)} 個目標，'
                    f'從各目標首次偵測位置回溯，最多 {self.RETRO_OCR_MAX_GAP} 幀。',
                    last_frame, track_ids=list(batch_history.keys()),
                    first_detection_backfill=True, batch_backfill=True)
                _backfill_result = self._retroactively_search_before_first_detection(
                    cap, batch_history, hit_frame, last_frame,
                    frame_w, frame_h, reader)
                for _tid, _local_track_history in batch_history.items():
                    _full_history = history.setdefault(_tid, {})
                    for _f, _roi in _local_track_history.items():
                        if _roi is None:
                            continue
                        _full_history[_f] = tuple(map(int, _roi))
                        ocr_history.setdefault(_tid, {})[_f] = tuple(map(int, _roi))
                        _rois = list(data.get(_f, []))
                        if not any(self._iou(_roi, _old) > .985 for _old in _rois):
                            _rois.append(tuple(map(int, _roi)))
                        data[_f] = _rois
                self._report(
                    'first_detection_backfill_done',
                    f'批次首次回溯完成：檢查 {_backfill_result["checked"]} 個影格，'
                    f'OCR 確認 {_backfill_result["matched"]} 個目標影格；'
                    f'共處理 {_backfill_result["tracks"]} 個目標。',
                    last_frame, first_detection_backfill=True,
                    retro_checked=_backfill_result["checked"],
                    retro_matched=_backfill_result["matched"],
                    tracks_checked=_backfill_result["tracks"], batch_backfill=True)
                cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame + 1)

            # Process a final partial segment as well (e.g. if the user cancels mid-segment).
            # This remains a bounded local pass, not a full-video end reverse scan.
            frames_in_run = max(0, last_frame - hit_frame + 1)
            if (reader is not None and self.target_text.strip()
                    and self.RETROACTIVE_SEGMENT_INTERVAL > 0 and frames_in_run > 0
                    and frames_in_run % self.RETROACTIVE_SEGMENT_INTERVAL != 0):
                segment_start = max(hit_frame, last_frame - self.RETROACTIVE_SEGMENT_INTERVAL + 1 - self.RETRO_OCR_MAX_GAP)
                # Final pass uses only still-unprocessed OCR-anchor pairs; it does not
                # repeat gaps already handled during periodic or reacquisition backfill.
                segment_history = self._pending_anchor_pair_histories(
                    ocr_history, segment_start, last_frame)
                self._report('retroactive_check',
                    f'開始最後一段回溯：F{segment_start} 至 F{last_frame}。',
                    last_frame, segment_start=segment_start, segment_end=last_frame,
                    retroactive_segment=True, final_partial_segment=True)
                checked_now = self._retroactively_verify_gaps(
                    cap, segment_history, segment_start, last_frame, frame_w, frame_h, reader,
                    force_check=False)
                for _pair_key, _segment_track_history in segment_history.items():
                    _tid, _left_anchor, _right_anchor = _pair_key
                    _full_history = history.setdefault(_tid, {})
                    for _f, _roi in _segment_track_history.items():
                        _full_history[_f] = tuple(map(int, _roi)) if _roi is not None else None
                for _f in range(segment_start, last_frame + 1):
                    data[_f] = [tuple(map(int, _h[_f])) for _h in history.values()
                                if _h.get(_f) is not None]
                self._report('retroactive_segment_done',
                    f'最後一段回溯完成：檢查 {checked_now} 個缺漏影格。', last_frame,
                    retro_checked=checked_now, segment_start=segment_start,
                    segment_end=last_frame, retroactive_segment=True, final_partial_segment=True)

            # Do not run a second full reverse scan after the whole video ends.
            self._report('retroactive_check',
                         '分段回溯補查已完成；僅處理尚未檢查的 OCR 錨點缺口，避免重複回朔。',
                         last_frame, retroactive_mode='per_segment',
                         segment_interval=self.RETROACTIVE_SEGMENT_INTERVAL,
                         end_reverse_check=False, cancelled=self.was_cancelled)

            # First run OCR backfill; then use bounded path interpolation for residual gaps.
            # This creates an ROI for each frame between known endpoints when the gap is short
            # enough, so export applies mosaic frame-by-frame rather than blurring a whole path strip.
            path_filled = self._interpolate_path_gaps(
                history, max_gap=self.MAX_PATH_INTERPOLATION_GAP)
            # At end-of-run, visualize a sample of the path masks that were inferred after OCR backfill.
            for _idx, (_tid, _path_frame, _path_roi, _left, _right) in enumerate(path_filled):
                if _idx % 3 == 0:
                    _left_roi = history.get(_tid, {}).get(_left)
                    _right_roi = history.get(_tid, {}).get(_right)
                    if _left_roi is not None and _right_roi is not None:
                        self._draw_path_backfill(cap, _path_frame, _tid, _path_roi,
                            _left, _right, _left_roi, _right_roi, resume_frame=last_frame + 1)
            self._interpolate_short_gaps(history, self.INTERPOLATE_MAX_GAP)
            if path_filled:
                self._report('path_backfill_done',
                    f'路徑補遮完成：依前後有效 ROI 補上 {len(path_filled)} 個影格的逐幀遮罩。',
                    last_frame, path_filled=len(path_filled),
                    max_path_gap=self.MAX_PATH_INTERPOLATION_GAP,
                    path_mask_expand=self.PATH_MASK_EXPAND)
            for f in range(hit_frame, last_frame+1):
                # Rebuild the export masks from the final corrected history. Do not retain
                # stale boxes that were produced before retroactive OCR corrected the path.
                data[f] = [tuple(map(int, th[f])) for th in history.values()
                           if th.get(f) is not None]
            success = any(bool(v) for v in data.values())
            finish_message = '追蹤已中止；已保留中止前的追蹤資料' if self.was_cancelled else '追蹤完成'
            self._report('finished', finish_message, last_frame, tracked_frames=len(data),
                         target_count=max((len(v) for v in data.values()), default=0),
                         success=success, cancelled=self.was_cancelled)
        except Exception as exc:
            self._report('error', f'追蹤發生錯誤：{type(exc).__name__}: {exc}', last_frame)
            success = any(bool(v) for v in data.values())
        finally:
            if cap is not None:
                cap.release()
            if self.frame_buffer is not None:
                self.frame_buffer.clear()
                self.frame_buffer = None
            self.finished_signal.emit(data, last_frame, success)

# Backwards-compatible import used by main.py.
from export_worker import VideoProcessorThread