import time
import math
import cv2
import easyocr
from PyQt5.QtCore import QThread, pyqtSignal


class OCRTrackingThread(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict, int, bool)
    monitor_frame_signal = pyqtSignal(object, int, str)
    monitor_data_signal = pyqtSignal(dict)

    # 可調參數：OCR 越頻繁通常越慢；先用這組作為速度與穩定度的折衷。
    VERIFY_INTERVAL = 5            # 正常追蹤時每 5 幀驗證一次
    VERIFY_MISSES_TO_LOSE = 1      # 驗證未命中就進入搜尋，避免再等一個完整週期
    SEARCH_EVERY_N_FRAMES = 2      # 失去目標後每 2 幀做一次 OCR 搜尋
    FULL_FRAME_AFTER_ATTEMPTS = 4  # 前 3 次局部搜尋，第 4 次全畫面搜尋
    LOST_RETRY_INTERVAL = 15       # 累積 30 次失敗後，每 15 幀才重試一次
    MAX_FAILED_SEARCHES = 30       # 累積 30 次搜尋失敗後進入低頻重試，不結束整段影片
    INITIAL_SEARCH_PAD = 80
    MAX_LOCAL_SEARCH_PAD = 350
    MIN_OCR_CONFIDENCE = 0.20

    def __init__(self, video_path, start_frame, total_frames, target_text):
        super().__init__()
        self.video_path = video_path
        self.start_frame = max(0, int(start_frame))
        self.total_frames = max(0, int(total_frames))
        self.target_text = str(target_text or "")

    def _report(self, stage, message, frame_no=None, **extra):
        self.monitor_data_signal.emit({
            "stage": stage,
            "message": message,
            "frame": frame_no,
            "clock": time.monotonic(),
            **extra,
        })

    @staticmethod
    def _create_tracker():
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        raise RuntimeError("目前的 OpenCV 沒有 CSRT tracker。請安裝 opencv-contrib-python，並避免同時安裝多個 OpenCV 套件。")

    @staticmethod
    def _normalize(text):
        return "".join(str(text).split()).casefold()

    @staticmethod
    def _clip_roi(frame, roi):
        h, w = frame.shape[:2]
        x, y, rw, rh = [int(round(v)) for v in roi]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        rw = max(1, min(rw, w - x))
        rh = max(1, min(rh, h - y))
        return x, y, rw, rh

    @classmethod
    def _make_roi(cls, frame, bbox, offset_x=0, offset_y=0):
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x = offset_x + int(min(xs))
        y = offset_y + int(min(ys))
        w = max(1, int(max(xs) - min(xs)))
        h = max(1, int(max(ys) - min(ys)))
        px, py = max(2, int(w * 0.20)), max(2, int(h * 0.40))
        return cls._clip_roi(frame, (x - px, y - py, w + 2 * px, h + 2 * py))

    @staticmethod
    def _center(roi):
        x, y, w, h = roi
        return x + w / 2.0, y + h / 2.0

    @staticmethod
    def _expanded_roi(frame, roi, lost_frames):
        h, w = frame.shape[:2]
        x, y, rw, rh = roi
        # 先快速擴大，最多保護到畫面約三分之一；這是保守遮罩，不是精準位置預測。
        px = min(max(12, int(rw * 0.15)) * max(1, lost_frames), max(20, w // 3))
        py = min(max(12, int(rh * 0.20)) * max(1, lost_frames), max(20, h // 3))
        return OCRTrackingThread._clip_roi(frame, (x - px, y - py, rw + 2 * px, rh + 2 * py))

    def _emit_frame(self, frame, frame_no, mode, bbox=None, search_roi=None, note=None):
        shown = frame.copy()
        if search_roi:
            x, y, w, h = map(int, search_roi)
            cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 165, 255), 2)
        if bbox:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 220, 0), 2)
        label = f"Frame {frame_no} | {mode}" + (f" | {note}" if note else "")
        cv2.putText(shown, label[:150], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 2, cv2.LINE_AA)
        self.monitor_frame_signal.emit(shown, int(frame_no), str(mode))

    def _read_ocr(self, reader, image):
        # greedy decoder 通常比 beamsearch 快；不使用 paragraph 合併，方便逐項比對目標字串。
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        started = time.monotonic()
        results = reader.readtext(
            rgb, detail=1, paragraph=False, decoder="greedy",
            batch_size=1, workers=0, mag_ratio=1.0, canvas_size=1280
        )
        return results, time.monotonic() - started

    def _find_match(self, results, crop, crop_origin=(0, 0), expected_roi=None, allow_far=False):
        target = self._normalize(self.target_text)
        if not target:
            return None
        ox, oy = crop_origin
        candidates = []
        for bbox, text, prob in results:
            norm = self._normalize(text)
            confidence = float(prob)
            if target not in norm or confidence < self.MIN_OCR_CONFIDENCE:
                continue
            # 先在 crop 座標建立 ROI，再平移至原始影片座標，避免局部 crop 座標被錯誤裁切。
            lx, ly, rw, rh = self._make_roi(crop, bbox)
            roi = (lx + ox, ly + oy, rw, rh)
            distance = 0.0
            if expected_roi is not None:
                ax, ay = self._center(roi)
                bx, by = self._center(expected_roi)
                distance = math.hypot(ax - bx, ay - by)
            score = confidence if allow_far or expected_roi is None else confidence - min(distance / 600.0, 1.0) * 0.20
            candidates.append((score, roi, str(text), confidence, distance))
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[0], reverse=True)
        _, roi, text, confidence, distance = candidates[0]
        if expected_roi is not None and not allow_far:
            _, _, ew, eh = expected_roi
            if distance > max(120.0, math.hypot(ew, eh) * 3.0):
                return None
        return {"roi": roi, "text": text, "confidence": confidence, "distance": distance}

    def _ocr_crop(self, reader, frame, crop, crop_origin, frame_no, stage,
                  expected_roi=None, allow_far=False, search_roi=None):
        if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None, 0.0
        self._report("ocr_running", stage, frame_no, search_roi=search_roi)
        results, elapsed = self._read_ocr(reader, crop)
        ox, oy = crop_origin
        match = self._find_match(results, crop, (ox, oy), expected_roi, allow_far=allow_far)
        if match:
            match["roi"] = self._clip_roi(frame, match["roi"])
        self._report(
            "ocr_done", f"{stage}完成，耗時 {elapsed:.2f} 秒", frame_no,
            elapsed_seconds=elapsed, result_count=len(results),
            ocr_results=[{"text": t, "confidence": float(p)} for _, t, p in results[:8]],
            match=match,
        )
        return match, elapsed

    @staticmethod
    def _interpolate_gap(data, old_frame, old_roi, new_frame, new_roi, max_gap=12):
        # 只補很短的 gap；長 gap 的真實移動不可由兩端框可靠推知，需標記待確認。
        gap = new_frame - old_frame
        if gap <= 1 or gap > max_gap:
            return
        ox, oy, ow, oh = old_roi
        nx, ny, nw, nh = new_roi
        for f in range(old_frame + 1, new_frame):
            t = (f - old_frame) / float(gap)
            x = round(ox + (nx - ox) * t)
            y = round(oy + (ny - oy) * t)
            w = round(ow + (nw - ow) * t)
            h = round(oh + (nh - oh) * t)
            px, py = max(3, int(w * 0.15)), max(3, int(h * 0.15))
            data[f] = (max(0, x - px), max(0, y - py), max(1, w + 2 * px), max(1, h + 2 * py))

    def run(self):
        cap = None
        tracking_data = {}
        last_tracked_frame = self.start_frame
        last_verified_roi = None
        last_verified_frame = self.start_frame
        final_success = False
        stop_reason = "未知原因"
        is_tracking = False
        tracker = None
        verify_misses = 0
        lost_frames = 0
        failed_searches = 0
        frames_since_search = 0

        try:
            self.progress_signal.emit(self.start_frame, "正在載入 OCR 模型...")
            self._report("loading", "正在載入 EasyOCR 模型", self.start_frame)
            started = time.monotonic()
            reader = easyocr.Reader(["ch_tra", "en"], gpu=False)
            self._report("ready", f"OCR 模型載入完成（{time.monotonic() - started:.1f} 秒）",
                         self.start_frame, elapsed_seconds=time.monotonic() - started)

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError("無法開啟影片檔案。")
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            ret, frame = cap.read()
            if not ret:
                stop_reason = "無法讀取起始影格"
                self._report("error", stop_reason, self.start_frame)
                return

            self._emit_frame(frame, self.start_frame, "INITIAL OCR")
            match, _ = self._ocr_crop(reader, frame, frame, (0, 0), self.start_frame,
                                      "起始影格全畫面 OCR", allow_far=True,
                                      search_roi=(0, 0, frame.shape[1], frame.shape[0]))
            if not match:
                stop_reason = f"起始影格找不到目標文字：{self.target_text}"
                self._report("not_found", stop_reason, self.start_frame)
                return

            initial_roi = self._clip_roi(frame, match["roi"])
            tracker = self._create_tracker()
            tracker.init(frame, initial_roi)
            tracking_data[self.start_frame] = initial_roi
            last_tracked_frame = self.start_frame
            last_verified_roi = initial_roi
            last_verified_frame = self.start_frame
            is_tracking = True
            self._emit_frame(frame, self.start_frame, "TRACKING", bbox=initial_roi)
            self._report("locked", "初始目標已鎖定", self.start_frame,
                         target_text=match["text"], confidence=match["confidence"], bbox=initial_roi)
            self.progress_signal.emit(self.start_frame, "目標鎖定！開始逐影格分析...")
            stop_reason = "已處理至影片結尾"

            for i in range(self.start_frame + 1, self.total_frames):
                if self.isInterruptionRequested():
                    stop_reason = "使用者取消追蹤"
                    self._report("cancelled", stop_reason, i)
                    break
                ret, frame = cap.read()
                if not ret:
                    stop_reason = "影片讀取結束或讀取失敗"
                    break
                frames_since_search += 1
                current_roi = None
                h, w = frame.shape[:2]

                # 1) 每幀先用 CSRT 更新框；OCR 只在固定週期或框移動異常時執行。
                if is_tracking and tracker is not None:
                    t0 = time.monotonic()
                    ok, bbox = tracker.update(frame)
                    tracker_elapsed = time.monotonic() - t0
                    if ok and bbox[2] > 0 and bbox[3] > 0:
                        current_roi = self._clip_roi(frame, bbox)
                        # 位移/大小突變時立即 OCR，不必等固定驗證週期。
                        old_cx, old_cy = self._center(last_verified_roi)
                        new_cx, new_cy = self._center(current_roi)
                        displacement = math.hypot(new_cx - old_cx, new_cy - old_cy)
                        _, _, old_w, old_h = last_verified_roi
                        _, _, new_w, new_h = current_roi
                        size_jump = max(new_w / max(1, old_w), old_w / max(1, new_w),
                                        new_h / max(1, old_h), old_h / max(1, new_h))
                        suspicious = displacement > max(70, math.hypot(old_w, old_h) * 1.8) or size_jump > 2.0
                        should_verify = suspicious or ((i - self.start_frame) % self.VERIFY_INTERVAL == 0)

                        if should_verify:
                            bx, by, bw, bh = current_roi
                            mx, my = max(35, int(bw * 0.8)), max(25, int(bh * 0.8))
                            x1, y1 = max(0, bx - mx), max(0, by - my)
                            x2, y2 = min(w, bx + bw + mx), min(h, by + bh + my)
                            crop = frame[y1:y2, x1:x2]
                            vmatch, _ = self._ocr_crop(
                                reader, frame, crop, (x1, y1), i,
                                "異常位移 OCR 驗證" if suspicious else "定期 OCR 驗證",
                                expected_roi=current_roi, allow_far=False,
                                search_roi=(x1, y1, x2 - x1, y2 - y1)
                            )
                            if vmatch:
                                corrected = self._clip_roi(frame, vmatch["roi"])
                                self._interpolate_gap(tracking_data, last_verified_frame,
                                                      last_verified_roi, i, corrected)
                                tracker = self._create_tracker()
                                tracker.init(frame, corrected)
                                current_roi = corrected
                                last_verified_roi = corrected
                                last_verified_frame = i
                                verify_misses = 0
                                self._report("verified", "OCR 驗證成功並校正框", i,
                                             bbox=corrected, target_text=vmatch["text"],
                                             confidence=vmatch["confidence"])
                            else:
                                verify_misses += 1
                                self._report("verify_miss", f"OCR 驗證未命中 {verify_misses}", i,
                                             bbox=current_roi, displacement=displacement, suspicious=suspicious)
                                if suspicious or verify_misses >= self.VERIFY_MISSES_TO_LOSE:
                                    is_tracking = False
                                    lost_frames = 0
                                    failed_searches = 0
                                    frames_since_search = self.SEARCH_EVERY_N_FRAMES
                                    self._report("tracking_lost", "追蹤框異常或 OCR 未命中，立即進入搜尋", i,
                                                 bbox=current_roi, displacement=displacement)

                        if is_tracking:
                            tracking_data[i] = current_roi
                            last_tracked_frame = i
                            if i % 3 == 0:
                                self._emit_frame(frame, i, "TRACKING", bbox=current_roi)
                                self.progress_signal.emit(i, f"逐影格追蹤中：{i}/{self.total_frames}")
                    else:
                        is_tracking = False
                        lost_frames = 0
                        failed_searches = 0
                        frames_since_search = self.SEARCH_EVERY_N_FRAMES
                        self._report("tracking_lost", "CSRT 更新失敗，立即進入搜尋", i,
                                     tracker_elapsed_seconds=tracker_elapsed)

                # 2) LOST 狀態：每一幀都保留暫時保護框，但 OCR 搜尋按節奏執行。
                if not is_tracking:
                    lost_frames += 1
                    if last_verified_roi is not None:
                        current_roi = self._expanded_roi(frame, last_verified_roi, lost_frames)
                        tracking_data[i] = current_roi

                    # 前 30 次失敗是「30 次搜尋嘗試」，不是在同一幀重跑 30 次。
                    # 30 次後不結束整段影片，改成每 15 幀全畫面重試。
                    if failed_searches < self.MAX_FAILED_SEARCHES:
                        # 每兩幀最多搜尋一次；前 3 次局部，之後局部與全畫面交替。
                        should_search = frames_since_search >= self.SEARCH_EVERY_N_FRAMES
                    else:
                        # 累積 30 次 OCR 搜尋失敗後，不再密集搜尋，改成每 15 幀重試。
                        should_search = (lost_frames % self.LOST_RETRY_INTERVAL == 0)
                    if should_search and last_verified_roi is not None:
                        frames_since_search = 0
                        failed_searches += 1
                        # 第 4 次以及之後每第 5 次搜尋做全畫面 OCR，其餘使用局部 crop，
                        # 避免目標消失時每次都掃整張高解析度畫面而拖慢處理。
                        use_full = (
                            failed_searches >= self.FULL_FRAME_AFTER_ATTEMPTS
                            and (failed_searches - self.FULL_FRAME_AFTER_ATTEMPTS) % 5 == 0
                        )
                        if use_full:
                            crop = frame
                            origin = (0, 0)
                            search_roi = (0, 0, w, h)
                            mode = "FULL-FRAME OCR"
                        else:
                            pad = min(self.INITIAL_SEARCH_PAD * (2 ** (failed_searches - 1)),
                                      self.MAX_LOCAL_SEARCH_PAD)
                            ax, ay, aw, ah = last_verified_roi
                            x1, y1 = max(0, ax - pad), max(0, ay - pad)
                            x2, y2 = min(w, ax + aw + pad), min(h, ay + ah + pad)
                            crop = frame[y1:y2, x1:x2]
                            origin = (x1, y1)
                            search_roi = (x1, y1, x2 - x1, y2 - y1)
                            mode = "LOCAL OCR SEARCH"
                        self._emit_frame(frame, i, mode, bbox=current_roi,
                                         search_roi=search_roi,
                                         note=f"failed OCR attempts={failed_searches}")
                        smatch, elapsed = self._ocr_crop(
                            reader, frame, crop, origin, i,
                            "全畫面重新搜尋" if use_full else "局部重新搜尋",
                            expected_roi=last_verified_roi, allow_far=use_full,
                            search_roi=search_roi
                        )
                        if smatch:
                            new_roi = self._clip_roi(frame, smatch["roi"])
                            self._interpolate_gap(tracking_data, last_verified_frame,
                                                  last_verified_roi, i, new_roi)
                            tracker = self._create_tracker()
                            tracker.init(frame, new_roi)
                            tracking_data[i] = new_roi
                            last_tracked_frame = i
                            last_verified_roi = new_roi
                            last_verified_frame = i
                            current_roi = new_roi
                            is_tracking = True
                            verify_misses = 0
                            lost_frames = 0
                            failed_searches = 0
                            frames_since_search = 0
                            self._emit_frame(frame, i, "REACQUIRED", bbox=new_roi)
                            self._report("recovered", "OCR 找回目標，恢復追蹤", i,
                                         target_text=smatch["text"], confidence=smatch["confidence"],
                                         bbox=new_roi, elapsed_seconds=elapsed,
                                         search_mode="full_frame" if use_full else "local")
                            self.progress_signal.emit(i, "OCR 找回目標，恢復追蹤")
                        else:
                            self._report("lost", f"搜尋未命中：第 {failed_searches} 次失敗", i,
                                         failed_searches=failed_searches, lost_frames=lost_frames,
                                         search_mode="full_frame" if use_full else "local",
                                         fallback_roi=current_roi)
                            if failed_searches == self.MAX_FAILED_SEARCHES:
                                self._report("lost_cooldown", "已累積 30 次 OCR 搜尋失敗；影片分析繼續，每 15 幀全畫面重試", i,
                                             failed_searches=failed_searches, lost_frames=lost_frames)
                    if i % 3 == 0:
                        self.progress_signal.emit(i, f"搜尋中：{i}/{self.total_frames} 幀；OCR 失敗次數 {failed_searches}")
                        self._emit_frame(frame, i, "SEARCHING - PROTECTIVE ROI", bbox=current_roi)

            final_success = bool(tracking_data)
            self._report("finished", stop_reason, last_tracked_frame,
                         last_tracked_frame=last_tracked_frame,
                         tracked_frames=len(tracking_data), lost_frames=lost_frames,
                         failed_searches=failed_searches, success=final_success)
        except Exception as exc:
            stop_reason = f"執行錯誤：{type(exc).__name__}: {exc}"
            self._report("error", stop_reason, last_tracked_frame)
            final_success = bool(tracking_data)
        finally:
            if cap is not None:
                cap.release()
            self.finished_signal.emit(tracking_data, last_tracked_frame, final_success)


class VideoProcessorThread(QThread):
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
        x, y, w, h = [int(round(v)) for v in roi]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(0, min(w, width - x))
        h = max(0, min(h, height - y))
        if w <= 0 or h <= 0:
            return
        area = frame[y:y + h, x:x + w]
        strength = max(1, int(strength))
        sw, sh = max(1, w // strength), max(1, h // strength)
        mosaic = cv2.resize(cv2.resize(area, (sw, sh), interpolation=cv2.INTER_LINEAR),
                            (w, h), interpolation=cv2.INTER_NEAREST)
        frame[y:y + h, x:x + w] = mosaic

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished_signal.emit("")
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not out.isOpened():
            cap.release()
            self.finished_signal.emit("")
            return
        current_frame = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                for task in self.mosaic_tasks:
                    if task["start"] <= current_frame <= task["end"]:
                        if task.get("type") == "tracking":
                            roi = task.get("tracking_data", {}).get(current_frame)
                        else:
                            roi = task.get("roi")
                        if roi:
                            self._apply_mosaic(frame, roi, task.get("strength", 15))
                out.write(frame)
                current_frame += 1
                if current_frame % 15 == 0:
                    self.progress_signal.emit(current_frame, total)
        finally:
            cap.release()
            out.release()
        self.finished_signal.emit(self.output_path)
