"""Multi-target tracking + sparse full-frame/local RapidOCR verification + mosaic export.

Requires: opencv-contrib-python, rapidocr-onnxruntime, PyQt5. EasyOCR/PyTorch is not used.
tracking_data format: {absolute_frame_index: [(x, y, w, h), ...]}
"""
import math
import time
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from frame_buffer import FrameRingBuffer


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
    GLOBAL_OCR_INTERVAL = 15          # ��函�恍�� OCR 瘥� 15 撟�銝�甈∴��30 FPS 蝝� 0.5 蝘�嚗�
    LOCAL_RECOVERY_INTERVAL = 3       # 追蹤遺失時約每 3 幀重試一次局部 OCR，不採用長時間指數退避
    MAX_OCR_ROI_DIM = 800             # 撅���� OCR 靽����頛�憭�撠�摮�蝝啁��
    MAX_GLOBAL_OCR_DIM = 1280         # 皜�撠�撠�摮���典�函�恍�Ｙ葬��暹�����蝝啁�����憭�
    MIN_FEATURE_POINTS = 4
    MAX_LOST_FRAMES = 3  # ��� 3 撟�隞�憭梯蕭撠勗��甇Ｚ撓��箄府頠�頝∠��擐祈魚���獢�
    CONSERVATIVE_MASK_FRAMES = 10  # ��剜�怠仃餈賣��雿輻�冽�游之��� Kalman ���皜祆��嚗����雿�瞍���桃蔗憸券��
    RECOVERY_SCALE = 3.0
    DETECT_CANVAS_SIZE = 1280
    MIN_OCR_CONFIDENCE = 0.20
    MONITOR_INTERVAL = 2  # Update the live preview about 15 times/second at 30 FPS.
    RETROACTIVE_SEGMENT_INTERVAL = 90  # 每 90 幀進行一次分段回溯
    INTERPOLATE_MAX_GAP = 2  # 保留舊的短缺口設定，路徑補遮另有獨立上限
    MAX_PATH_INTERPOLATION_GAP = 45  # 兩個可靠位置之間最多以路徑補遮 45 幀（30 FPS 約 1.5 秒）
    PATH_MASK_EXPAND = 1.20  # 路徑推估 ROI 稍微擴大，降低位置估計誤差造成的漏遮
    IMMEDIATE_RETRACE_ON_REACQUISITION = True  # 重新找到目標時，立即回頭檢查剛才的缺口
    RETRO_OCR_MAX_GAP = 120  # ���憭������� 60 撟����蝻箏��嚗�蝝� 2 蝘�@30FPS嚗���臭�������質矽���
    RETRO_ROI_EXPAND = 2.2  # �����交��隞交����潭����箔葉敹���游之撅���冽��撠�������
    USE_MSER_PREFILTER = False  # Avoid MSER false negatives blocking OCR.
    ENABLE_END_REVERSE_CHECK = False  # Disable post-run reverse OCR; keep forward tracking and local recovery.
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
        # 撱箇��瘥������格����函����� CSRT tracker嚗�銝���舐�冽�������� Kalman + ���瘚����
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
                self._report('ocr_skipped', '未提供 OCR 搜尋區域，略過本次 OCR', frame_no)
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
                px, py = max(2, int(bw*.12)), max(2, int(bh*.20))
                box = self._clip(frame, (x+x1-px, y+y1-py, bw+2*px, bh+2*py))
                matched = bool(conf >= self.MIN_OCR_CONFIDENCE and target and target in normalized)
                raw.append({'text': text, 'confidence': float(conf), 'variant': variant_name,
                            'bbox': tuple(map(int, box)), 'matched': matched})
                # Draw every OCR candidate in the monitor, including non-matching text.
                # Only target matches may create or update tracked identities.
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

    def _draw_monitor(self, frame, frame_no, tracks, mode):
        shown = frame.copy()
        colors = {'OCR_CONFIRMED': (0, 220, 0), 'MANUAL_INIT': (0, 220, 0),
                  'TRACKING': (255, 210, 0), 'FAST_MOTION': (255, 0, 255),
                  'RECOVERY': (0, 190, 255), 'LOST': (0, 0, 255)}
        for tid, track in tracks.items():
            # Hide stale visualization after three consecutive lost frames; keep the
            # internal track state so local OCR may still recover it later.
            if int(track.get('lost', 0)) > self.CONSERVATIVE_MASK_FRAMES:
                continue
            roi = track.get('roi', (0, 0, 1, 1))
            x, y, w, h = roi
            state = track.get('state', 'TRACKING')
            color = colors.get(state, (255, 255, 255))
            cv2.rectangle(shown, (x, y), (x+w, y+h), color, 2)
            # Dashed-style predicted box shows where motion model expects the target.
            predicted_roi = track.get('predicted_roi')
            if predicted_roi is not None and (track.get('lost', 0) > 0 or state in ('RECOVERY', 'LOST')):
                px, py, pw, ph = map(int, self._clip(frame, predicted_roi))
                for xx in range(px, px+pw, 14):
                    cv2.line(shown, (xx, py), (min(xx+7, px+pw), py), (0, 165, 255), 2)
                    cv2.line(shown, (xx, py+ph), (min(xx+7, px+pw), py+ph), (0, 165, 255), 2)
                for yy in range(py, py+ph, 14):
                    cv2.line(shown, (px, yy), (px, min(yy+7, py+ph)), (0, 165, 255), 2)
                    cv2.line(shown, (px+pw, yy), (px+pw, min(yy+7, py+ph)), (0, 165, 255), 2)
            reason = track.get('loss_reason', '')
            recovery_note = str(track.get('recovery_note', ''))
            label = f'ID {tid} {state} v={track.get("speed", 0):.1f}' + (f' [{reason}]' if reason else '')
            if recovery_note and int(track.get('lost', 0)) > 0:
                label += ' | ' + recovery_note[:32]
            cv2.putText(shown, label, (x, max(18, y-5)), cv2.FONT_HERSHEY_SIMPLEX,
                        .48, color, 2, cv2.LINE_AA)
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
        cv2.putText(shown, 'GREEN=OCR MATCH  RED=OTHER OCR  ORANGE=SEARCH/PREDICTION  BLUE=TRACK',
                    (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, .43, (245, 245, 245), 1, cv2.LINE_AA)
        mh, mw = shown.shape[:2]
        monitor_scale = min(1.0, 960.0 / max(mh, mw))
        if monitor_scale < 1.0:
            shown = cv2.resize(shown, (max(1, int(mw * monitor_scale)), max(1, int(mh * monitor_scale))), interpolation=cv2.INTER_AREA)
        self.monitor_frame_signal.emit(shown, int(frame_no), mode)

    def _draw_reverse_check(self, frame, frame_no, track_id, predicted_roi=None,
                            measured_roi=None, note="回溯檢查"):
        """��� GUI ��單�������折＊蝷箏�����瑼Ｘ�亙祕���霈�������敶望�潦��

        璈�獢�=������餈質馱���皜祆��撠�雿�蝵殷��蝬�獢�=OCR 蝣箄��雿�蝵柴��
        frame_no �����典�����瑼Ｘ�仿��皜�嚗���嫣噶雿輻�刻��蝣箄����恍�ＹⅡ撖血����啗��敶望�潦��
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
                                   frame_w, frame_h, reader):
        """���皞舀炎��亦�� None 璅�閮����蝻箏仃敶望�潦��

        ��芸�典��銝�餈質馱頠�頝∠�����敺���賣��������獢����嚗����隞亙�拍垢雿�蝵桐摯蝞�撅���� OCR ROI���
        蝻箏��銝������� RETRO_OCR_MAX_GAP ��批�塚�����皞舀��������隞� OCR 撖阡��獢�閬�撖怨府敶望�潦��
        ��������賢��銝����撠�瘥�銝�撟���賢�瑁�� OCR嚗���芾����� history 銝剜��蝣箸����� None ���蝻箏�����
        """
        checked = 0
        matched = 0
        max_gap = max(0, int(self.RETRO_OCR_MAX_GAP))
        if max_gap == 0:
            self._report('retroactive_check', '回溯補查已停用：RETRO_OCR_MAX_GAP 設為 0', end_frame,
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
                    f'分段回溯：ID {track_id}，F{left} 到 F{right}，檢查缺漏 {len(missing)} 幀',
                    right, track_id=track_id, gap=gap, retro_checked=checked,
                    retro_matched=matched, max_gap=max_gap)
                for frame_no in missing:
                    ratio = (frame_no - left) / float(right - left)
                    predicted = tuple(int(round(a[i] * (1-ratio) + b[i] * ratio)) for i in range(4))
                    frame = self._read_frame_at(cap, frame_no)
                    if frame is None:
                        continue
                    pred = self._clip(frame, predicted)
                    search_roi = self._expand_roi(frame, pred, self.RETRO_ROI_EXPAND)
                    self._draw_reverse_check(frame, frame_no, track_id, pred, None,
                                             f"缺漏影格 OCR 補查 #{checked+1}")
                    detections, _ = self._ocr(reader, frame, frame_no,
                                              roi=search_roi, full_frame=False, robust=True)
                    checked += 1
                    candidates = [d for d in detections
                                  if target in self._normalize(d[1])]
                    if not candidates:
                        self._report('retroactive_global_fallback',
                            f'F{frame_no} 局部回溯未匹配，改做全畫面 OCR', frame_no,
                            track_id=track_id, search_roi=search_roi)
                        global_detections, _ = self._ocr(
                            reader, frame, frame_no, full_frame=True, robust=True)
                        checked += 1
                        detections.extend(global_detections)
                        candidates = [d for d in global_detections
                                      if target in self._normalize(d[1])]
                    self._draw_reverse_check(
                        frame, frame_no, track_id, pred, None,
                        f"補查完成：候選 {len(self._monitor_overlay.get('candidates', []))}，匹配 {len(candidates)}")
                    if candidates:
                        # ��豢�������亥�����敺�摨扳�������潔��蝵桃��������摮������豢�����
                        candidates.sort(key=lambda d: math.hypot(
                            self._center(d[0])[0] - self._center(pred)[0],
                            self._center(d[0])[1] - self._center(pred)[1]))
                        track_history[frame_no] = tuple(map(int, self._clip(frame, candidates[0][0])))
                        matched += 1
                        self._draw_reverse_check(frame, frame_no, track_id, pred,
                                                 track_history[frame_no], "OCR 已確認")
                        self._report('retroactive_match',
                            f'回溯補到目標：F{frame_no}，ID {track_id}', frame_no,
                            track_id=track_id, roi=track_history[frame_no],
                            retro_checked=checked, retro_matched=matched)

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(end_frame))
        self._report('retroactive_check',
            f'分段回溯完成：檢查 {checked} 幀，OCR 成功補到 {matched} 幀',
            end_frame, retro_checked=checked, retro_matched=matched, max_gap=max_gap)
        return checked

    def _retroactively_search_before_first_detection(self, cap, history, start_frame,
                                                     end_frame, frame_w, frame_h, reader):
        """撠�瘥�璇�頠�頝∠�����蝚砌��甈∪歇��乩��蝵柴��銋�������������餈質馱 + 撅���� OCR���

        �����航�������箏����餌����函�恍�� OCR 瞍�������憪���箇�曉蔣��潛��蝚砌�����畾萸��
        ��望����拍����舫��獢����������撟�餈質馱��孵噩暺�嚗����皜� ROI嚗������� OCR 蝣箄����格�����
        ��芣�� OCR 蝣箄����格�����敶望�潭��撖怠�� history嚗���踹�����銝�蝣箏��������皜祆����嗆�����撖衣�格�����
        瘜冽��嚗���亦�格����券��畾菜����������⊿�剖����������湧����格�������ａ����恍�ｇ��隞���⊥��靽�霅���曉�����
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
            anchor_frame = self._read_frame_at(cap, first_frame)
            if anchor_frame is None:
                continue
            current_frame = anchor_frame
            current_roi = self._clip(current_frame, first_roi)
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            tracks_checked += 1
            self._report('retroactive_first_detection',
                f'首次偵測前回溯：ID {track_id}，從 F{first_frame} 往前檢查至 F{scan_start}',
                first_frame, track_id=track_id, from_frame=first_frame,
                to_frame=scan_start, max_back=max_back)

            # ���������撟�蝘餃��嚗�瘥�甈∪����桀��獢���批����孵噩暺�嚗�������餈賢�啣��銝�撟����
            for frame_no in range(first_frame - 1, scan_start - 1, -1):
                prev_frame = self._read_frame_at(cap, frame_no)
                if prev_frame is None:
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
                # ��亙��瘚�憭望��嚗�隞����閰血�典��雿�蝵桅��餈����撠�嚗���芯誑 OCR 蝣箄��蝯����憛怠�� history���
                search_roi = self._expand_roi(prev_frame, predicted,
                    self.RETRO_ROI_EXPAND * (1.0 if motion_ok else 1.35))
                self._draw_reverse_check(prev_frame, frame_no, track_id, predicted, None,
                    f"首次偵測前回溯 #{checked+1}，光流={'成功' if motion_ok else '備援'}")
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
                    self._draw_reverse_check(prev_frame, frame_no, track_id, predicted,
                                             measured, "首次偵測前回溯：OCR 已確認")
                    self._report('retroactive_first_match',
                        f'首次偵測前回溯找到目標：F{frame_no}，ID {track_id}', frame_no,
                        track_id=track_id, roi=measured, retro_checked=checked,
                        retro_matched=matched)
                else:
                    # OCR 未確認時，仍將反向光流推估框寫入逐幀 history。
                    # 否則只更新 current_roi、卻不寫 history，匯出時就完全不會遮這一幀。
                    px, py, pw, ph = self._clip(prev_frame, predicted)
                    ew = max(2, int(round(pw * self.PATH_MASK_EXPAND)))
                    eh = max(2, int(round(ph * self.PATH_MASK_EXPAND)))
                    predicted_mask = self._clip(
                        prev_frame,
                        (px + pw / 2.0 - ew / 2.0,
                         py + ph / 2.0 - eh / 2.0, ew, eh))
                    track_history[frame_no] = tuple(map(int, predicted_mask))
                    current_roi = predicted
                current_frame = prev_frame
                current_gray = prev_gray

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(end_frame))
        self._report('retroactive_first_detection',
            f'首次偵測前回溯完成：檢查 {checked} 幀、補到 {matched} 幀、檢查 {tracks_checked} 個目標',
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
                    # Keep the interpolated center, but add a small privacy margin.
                    ew, eh = w * expand, h * expand
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

            # Strict ROI-only design: the user must draw the target box; no full-frame OCR.
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
            roi = self._clip(first_frame, self.initial_roi)
            if roi[2] < 3 or roi[3] < 3:
                raise RuntimeError('框選的 ROI 太小，請重新框選較完整的目標。')
            tracks[next_id] = self._new_track(first_frame, roi, next_id, hit_frame,
                                              self.target_text, 1.0)
            self._report('locked', f'已鎖定手動框選的追蹤目標：ROI={roi}', hit_frame,
                         roi=roi, mode='INIT')
            if self.target_text.strip():
                self.progress_signal.emit(hit_frame, '追蹤開始；將持續顯示即時監控，並定期回溯補查缺漏影格...')

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
                    # CSRT 瘥����敶望�潭�湔�唬��甈∴����踹����芷�����瘚�������獢����瞍豢��蝘颯��
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
                                        # ��拍車閬�閬箄�������亥�����������嚗�撌桃�啣之���隞� CSRT ��箔蜓銝虫漱��勗��閫�瑼Ｘ�交��������
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
                    # ��箏����券����踹����航炊獢����撟�瘙⊥��憭�閫�璅⊥�踴��
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
                    # Full-frame OCR only at sparse keyframes (default every 15 frames),
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
                            # farther than the motion model predicted.
                            scale = min(self.RECOVERY_SCALE * (1.0 + 0.45 * max(0, attempts - 1)), 8.0)
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
                            if not local and attempts >= 3 and (attempts - 3) % 3 == 0:
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
                            lost_track = int(tr.get('lost', 0)) > 0
                            # A reacquired exact OCR text may be far from the stale motion
                            # prediction after a sudden move or occlusion. Permit a wider,
                            # bounded association gate for lost tracks; Hungarian matching
                            # still assigns each detection to at most one track.
                            lost_gate = max(gate, min(math.hypot(frame_w, frame_h) * 0.45, 900.0))
                            geometrically_plausible = (
                                (dist <= gate and (iou >= .02 or dist <= gate * .70 or app >= .72))
                                or iou >= .15
                            )
                            recovery_plausible = lost_track and text_match and (dist <= lost_gate or app >= .78)
                            if geometrically_plausible or recovery_plausible:
                                effective_gate = lost_gate if recovery_plausible else gate
                                cost[i,j] = .55*(1-iou) + .30*min(1.0, dist/effective_gate) + .10*(1-app) + .05*(1-conf)
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
                        old_lost_count = int(tr.get('lost', 0))
                        if old_lost_count > 0:
                            reacquired_events.append((tid, old_lost_count, frame_no))
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
                        # Only suppress a near-identical duplicate if another track was OCR-updated
                        # on THIS frame. A stale/predicted neighbouring track must not block a
                        # new instance of the same text in a row of identical targets.
                        if any(self._iou(det[0], tr['roi']) > .90 and
                               int(tr.get('last_ocr_frame', -10**9)) == frame_no
                               for tr in tracks.values()):
                            continue
                        if self.target_text.strip() and self._normalize(self.target_text) not in self._normalize(det[1]):
                            continue
                        tracks[next_id] = self._new_track(frame, det[0], next_id, frame_no, det[1], det[2])
                        history[next_id] = {}
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

                # Critical fix: a newly OCR-detected target has no history before its first
                # detection. Trace its path backward immediately to the tracking start frame.
                # This is separate from reacquisition-gap repair and runs for first detections too.
                if reader is not None and self.target_text.strip() and created_track_ids_this_frame:
                    new_track_history = {
                        _tid: history.get(_tid, {})
                        for _tid in created_track_ids_this_frame
                    }
                    self._report(
                        'first_detection_backfill_start',
                        f'首次 OCR 偵測後立即反向追蹤：{len(created_track_ids_this_frame)} 個新目標，'
                        f'從 F{frame_no} 回溯至 F{hit_frame}。',
                        frame_no, track_ids=created_track_ids_this_frame,
                        from_frame=frame_no, to_frame=hit_frame,
                        first_detection_backfill=True)
                    _backfill_result = self._retroactively_search_before_first_detection(
                        cap, new_track_history, hit_frame, frame_no,
                        frame_w, frame_h, reader)
                    # Merge every predicted/OCR-confirmed ROI into the full per-frame history
                    # and the actual export dictionary. The exporter reads only tracking_data.
                    for _tid, _local_track_history in new_track_history.items():
                        _full_history = history.setdefault(_tid, {})
                        for _f, _roi in _local_track_history.items():
                            if _roi is not None:
                                _full_history[_f] = tuple(map(int, _roi))
                                _rois = list(data.get(_f, []))
                                if not any(self._iou(_roi, _old) > .985 for _old in _rois):
                                    _rois.append(tuple(map(int, _roi)))
                                data[_f] = _rois
                    self._report(
                        'first_detection_backfill_done',
                        f'首次偵測前回溯完成：檢查 {_backfill_result["checked"]} 幀，'
                        f'OCR 確認 {_backfill_result["matched"]} 幀；反向推估框已寫入匯出資料。',
                        frame_no, first_detection_backfill=True,
                        retro_checked=_backfill_result["checked"],
                        retro_matched=_backfill_result["matched"],
                        backfilled_frames=sum(
                            1 for _tid in created_track_ids_this_frame
                            for _f, _roi in history.get(_tid, {}).items()
                            if hit_frame <= _f < frame_no and _roi is not None))
                    # The backward helper seeks to old frames; restore forward decoder position.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no + 1)

                # If OCR reacquires a previously lost track, immediately backfill the gap
                # instead of waiting for the next 90-frame scheduled segment check.
                if (self.IMMEDIATE_RETRACE_ON_REACQUISITION and reader is not None
                        and reacquired_events and self.target_text.strip()):
                    for _tid, _lost_count, _reacquired_frame in reacquired_events:
                        _track_history = history.get(_tid, {})
                        _anchors = [f for f, r in _track_history.items()
                                    if f < _reacquired_frame and r is not None]
                        if not _anchors:
                            continue
                        _left_anchor = max(_anchors)
                        _gap = _reacquired_frame - _left_anchor - 1
                        if _gap <= 0 or _gap > self.RETRO_OCR_MAX_GAP:
                            continue
                        _local_history = {
                            _tid: {f: _track_history.get(f) for f in range(_left_anchor, _reacquired_frame + 1)}
                        }
                        self._report('reacquisition_backfill_start',
                            f'目標重新找到：立即回溯 ID {_tid}，F{_left_anchor} 至 F{_reacquired_frame}，缺口 {_gap} 幀。',
                            _reacquired_frame, track_id=_tid, gap=_gap,
                            from_frame=_left_anchor, to_frame=_reacquired_frame,
                            immediate_backfill=True)
                        _checked = self._retroactively_verify_gaps(
                            cap, _local_history, _left_anchor, _reacquired_frame,
                            frame_w, frame_h, reader)
                        for _f, _roi in _local_history.get(_tid, {}).items():
                            if _roi is not None:
                                _track_history[_f] = tuple(map(int, _roi))
                        _path_filled = self._interpolate_path_gaps(
                            _local_history, max_gap=self.MAX_PATH_INTERPOLATION_GAP)
                        # Display sampled path masks so the monitor distinguishes measured OCR boxes
                        # from interpolated privacy masks. Avoid flooding the GUI with every frame.
                        for _idx, (_path_tid, _path_frame, _path_roi, _left, _right) in enumerate(_path_filled):
                            if _idx % 3 == 0:
                                _left_roi = _local_history[_tid].get(_left)
                                _right_roi = _local_history[_tid].get(_right)
                                if _left_roi is not None and _right_roi is not None:
                                    self._draw_path_backfill(cap, _path_frame, _tid, _path_roi,
                                        _left, _right, _left_roi, _right_roi, resume_frame=frame_no + 1)
                        for _f, _roi in _local_history.get(_tid, {}).items():
                            if _roi is not None:
                                _track_history[_f] = tuple(map(int, _roi))
                        for _f in range(_left_anchor, _reacquired_frame + 1):
                            _rois = list(data.get(_f, []))
                            _roi = _track_history.get(_f)
                            if _roi is not None and not any(self._iou(_roi, _old) > .985 for _old in _rois):
                                _rois.append(tuple(map(int, _roi)))
                            data[_f] = _rois
                        self._report('reacquisition_backfill_done',
                            f'立即回溯完成：ID {_tid}，OCR 檢查 {_checked} 次，路徑補遮 {_path_filled.__len__()} 幀。',
                            _reacquired_frame, track_id=_tid, retro_checked=_checked,
                            path_filled=len(_path_filled), gap=_gap, immediate_backfill=True)
                        # Retro OCR seeks backward; restore forward reading position.
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
                    segment_history = {}
                    for _tid, _track_history in history.items():
                        segment_history[_tid] = {
                            _f: _roi for _f, _roi in _track_history.items()
                            if segment_start <= _f <= frame_no
                        }
                    self._report('retroactive_check',
                        f'開始分段回溯：F{segment_start} 至 F{frame_no}，檢查最近一段的追蹤缺口。',
                        frame_no, segment_start=segment_start, segment_end=frame_no,
                        retroactive_segment=True)
                    checked_now = self._retroactively_verify_gaps(
                        cap, segment_history, segment_start, frame_no, frame_w, frame_h, reader)
                    # Merge only newly OCR-verified boxes back into the full history.
                    for _tid, _segment_track_history in segment_history.items():
                        _full_history = history.setdefault(_tid, {})
                        for _f, _roi in _segment_track_history.items():
                            if _roi is not None:
                                _full_history[_f] = tuple(map(int, _roi))
                    # Reflect retroactive detections in the data that will be exported as mosaic.
                    for _f in range(segment_start, frame_no + 1):
                        _rois = list(data.get(_f, []))
                        for _track_history in history.values():
                            _roi = _track_history.get(_f)
                            if _roi is not None and not any(self._iou(_roi, _old) > .985 for _old in _rois):
                                _rois.append(tuple(map(int, _roi)))
                        data[_f] = _rois
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

            # Process a final partial segment as well (e.g. if the user cancels mid-segment).
            # This remains a bounded local pass, not a full-video end reverse scan.
            frames_in_run = max(0, last_frame - hit_frame + 1)
            if (reader is not None and self.target_text.strip()
                    and self.RETROACTIVE_SEGMENT_INTERVAL > 0 and frames_in_run > 0
                    and frames_in_run % self.RETROACTIVE_SEGMENT_INTERVAL != 0):
                segment_start = max(hit_frame, last_frame - self.RETROACTIVE_SEGMENT_INTERVAL + 1 - self.RETRO_OCR_MAX_GAP)
                segment_history = {
                    _tid: {_f: _roi for _f, _roi in _track_history.items()
                           if segment_start <= _f <= last_frame}
                    for _tid, _track_history in history.items()
                }
                self._report('retroactive_check',
                    f'開始最後一段回溯：F{segment_start} 至 F{last_frame}。',
                    last_frame, segment_start=segment_start, segment_end=last_frame,
                    retroactive_segment=True, final_partial_segment=True)
                checked_now = self._retroactively_verify_gaps(
                    cap, segment_history, segment_start, last_frame, frame_w, frame_h, reader)
                for _tid, _segment_track_history in segment_history.items():
                    _full_history = history.setdefault(_tid, {})
                    for _f, _roi in _segment_track_history.items():
                        if _roi is not None:
                            _full_history[_f] = tuple(map(int, _roi))
                for _f in range(segment_start, last_frame + 1):
                    _rois = list(data.get(_f, []))
                    for _track_history in history.values():
                        _roi = _track_history.get(_f)
                        if _roi is not None and not any(self._iou(_roi, _old) > .985 for _old in _rois):
                            _rois.append(tuple(map(int, _roi)))
                    data[_f] = _rois
                self._report('retroactive_segment_done',
                    f'最後一段回溯完成：檢查 {checked_now} 個缺漏影格。', last_frame,
                    retro_checked=checked_now, segment_start=segment_start,
                    segment_end=last_frame, retroactive_segment=True, final_partial_segment=True)

            # Do not run a second full reverse scan after the whole video ends.
            self._report('retroactive_check',
                         '分段回溯補查已完成；未啟用整段影片結束後的額外反向掃描。',
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
                rois = list(data.get(f, []))
                for th in history.values():
                    roi = th.get(f)
                    if roi is not None and not any(self._iou(roi, old) > .985 for old in rois):
                        rois.append(tuple(map(int, roi)))
                data[f] = rois
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
