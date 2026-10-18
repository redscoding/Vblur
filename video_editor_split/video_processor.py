"""Original OCR tracking worker, with helper methods moved to tracking_helpers.py.

Tracking behavior and the public OCRTrackingThread import are preserved.
"""
import math
import time
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from frame_buffer import FrameRingBuffer
from tracking_helpers import (
    IntuitionisticFuzzyDecision,
    SortMotionModel,
    TrackingHelpersMixin,
)

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


class OCRTrackingThread(TrackingHelpersMixin, QThread):
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
    GLOBAL_OCR_INTERVAL = 15  # 原始註解編碼損壞；此處僅保留程式邏輯。
    LOCAL_RECOVERY_INTERVAL = 3       # 追蹤遺失時約每 3 幀重試一次局部 OCR，不採用長時間指數退避
    MAX_OCR_ROI_DIM = 800  # 原始註解編碼損壞；此處僅保留程式邏輯。
    MAX_GLOBAL_OCR_DIM = 1280  # 原始註解編碼損壞；此處僅保留程式邏輯。
    MIN_FEATURE_POINTS = 4
    MAX_LOST_FRAMES = 3  # 原始註解編碼損壞；此處僅保留程式邏輯。
    CONSERVATIVE_MASK_FRAMES = 10  # 原始註解編碼損壞；此處僅保留程式邏輯。
    RECOVERY_SCALE = 3.0
    DETECT_CANVAS_SIZE = 1280
    MIN_OCR_CONFIDENCE = 0.20
    MONITOR_INTERVAL = 2  # Update the live preview about 15 times/second at 30 FPS.
    RETROACTIVE_SEGMENT_INTERVAL = 90  # 每 90 幀進行一次分段回溯
    INTERPOLATE_MAX_GAP = 2  # 保留舊的短缺口設定，路徑補遮另有獨立上限
    MAX_PATH_INTERPOLATION_GAP = 45  # 兩個可靠位置之間最多以路徑補遮 45 幀（30 FPS 約 1.5 秒）
    PATH_MASK_EXPAND = 1.20  # 路徑推估 ROI 稍微擴大，降低位置估計誤差造成的漏遮
    IMMEDIATE_RETRACE_ON_REACQUISITION = True  # 重新找到目標時，立即回頭檢查剛才的缺口
    RETRO_OCR_MAX_GAP = 120  # 原始註解編碼損壞；此處僅保留程式邏輯。
    RETRO_ROI_EXPAND = 2.2  # 原始註解編碼損壞；此處僅保留程式邏輯。
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
  # 使用 CSRT 視覺追蹤器更新目標位置。
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
  # 比較 CSRT 與光流追蹤結果，避免兩者差異過大時直接採用錯誤位置。
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
  # 檢查候選位置，避免目標身分切換。
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
