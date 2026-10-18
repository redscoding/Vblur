"""Helper models and methods for the original OCR tracking worker.

This module only relocates existing code; it does not change tracking policy.
"""
import math
import time
import cv2
import numpy as np
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



class TrackingHelpersMixin:
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
        return TrackingHelpersMixin._clip(frame, (cx - nw/2, cy - nh/2, nw, nh))

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
