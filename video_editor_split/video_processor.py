import time
import cv2
import easyocr
from PyQt5.QtCore import QThread, pyqtSignal


class OCRTrackingThread(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict, int, bool)  # tracking_data, last_frame, success
    monitor_frame_signal = pyqtSignal(object, int, str)  # annotated BGR frame, frame number, mode
    monitor_data_signal = pyqtSignal(dict)

    def __init__(self, video_path, start_frame, total_frames, target_text):
        super().__init__()
        self.video_path = video_path
        self.start_frame = start_frame
        self.total_frames = total_frames
        self.target_text = target_text

    def _report(self, stage, message, frame_no=None, **extra):
        data = {
            "stage": stage,
            "message": message,
            "frame": frame_no,
            "clock": time.monotonic(),
            **extra,
        }
        self.monitor_data_signal.emit(data)

    @staticmethod
    def _create_tracker():
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        raise RuntimeError(
            "目前的 OpenCV 沒有 CSRT tracker。請安裝 opencv-contrib-python，並避免同時安裝多個 OpenCV 套件。"
        )

    @staticmethod
    def _make_roi(frame, bbox, offset_x=0, offset_y=0):
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x = offset_x + int(min(xs))
        y = offset_y + int(min(ys))
        w = max(1, int(max(xs) - min(xs)))
        h = max(1, int(max(ys) - min(ys)))
        pad_x, pad_y = int(w * 0.2), int(h * 0.4)
        height, width = frame.shape[:2]
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        right = min(width, x + w + pad_x * 2)
        bottom = min(height, y + h + pad_y * 2)
        return (x, y, max(1, right - x), max(1, bottom - y))

    def _emit_frame(self, frame, frame_no, mode, bbox=None, search_roi=None, note=None):
        shown = frame.copy()
        if search_roi:
            x, y, w, h = map(int, search_roi)
            cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(shown, "OCR SEARCH AREA", (x, max(22, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)
        if bbox:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 220, 0), 2)
            cv2.putText(shown, "TRACKED TARGET", (x, max(22, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2, cv2.LINE_AA)
        label = f"Frame {frame_no} | {mode}"
        if note:
            label += f" | {note}"
        cv2.putText(shown, label[:150], (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2, cv2.LINE_AA)
        self.monitor_frame_signal.emit(shown, frame_no, mode)

    def run(self):
        cap = None
        tracking_data = {}
        last_tracked_frame = self.start_frame
        final_success = False
        stop_reason = "未知原因"
        lost_count = 0
        patience = 30

        try:
            self.progress_signal.emit(self.start_frame, "正在載入 OCR 模型...")
            self._report("loading", "正在載入 EasyOCR 模型", self.start_frame)
            started = time.monotonic()
            reader = easyocr.Reader(["ch_tra", "en"], gpu=False)
            self._report("ready", f"OCR 模型載入完成（{time.monotonic() - started:.1f} 秒）", self.start_frame,
                         elapsed_seconds=time.monotonic() - started)

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError("無法開啟影片檔案。")
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            ret, frame = cap.read()
            if not ret:
                stop_reason = "無法讀取起始影格"
                self._report("error", stop_reason, self.start_frame)
                return

            self.progress_signal.emit(self.start_frame, "正在全畫面尋找初始目標...")
            self._emit_frame(frame, self.start_frame, "INITIAL OCR")
            self._report("ocr_running", "正在對起始影格執行全畫面 OCR", self.start_frame,
                         target_text=self.target_text, lost_count=0, patience=patience)
            ocr_started = time.monotonic()
            results = reader.readtext(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ocr_elapsed = time.monotonic() - ocr_started
            self._report("ocr_done", f"初始 OCR 完成，耗時 {ocr_elapsed:.2f} 秒", self.start_frame,
                         elapsed_seconds=ocr_elapsed, result_count=len(results))

            initial_roi = None
            best_match = None
            for bbox, text, prob in results:
                if self.target_text in text:
                    initial_roi = self._make_roi(frame, bbox)
                    best_match = (text, float(prob))
                    break

            if initial_roi is None:
                stop_reason = f"起始影格找不到目標文字：{self.target_text}"
                self._report("not_found", stop_reason, self.start_frame,
                             ocr_results=[{"text": t, "confidence": float(p)} for _, t, p in results[:10]])
                return

            tracker = self._create_tracker()
            init_result = tracker.init(frame, initial_roi)
            tracking_data[self.start_frame] = initial_roi
            last_tracked_frame = self.start_frame
            # 這是最近一次 OCR 確認文字存在的位置；搜尋失敗時以它為基準，而非盲信 CSRT 框。
            last_verified_roi = initial_roi
            verify_interval = 10
            verify_misses = 0
            self._emit_frame(frame, self.start_frame, "TRACKING", bbox=initial_roi)
            self._report("locked", "初始目標已鎖定", self.start_frame,
                         target_text=best_match[0], confidence=best_match[1], bbox=initial_roi)
            self.progress_signal.emit(self.start_frame, "目標鎖定！開始計算 AI 追蹤軌跡...")

            is_tracking = True
            stop_reason = "已處理至影片結尾"

            for i in range(self.start_frame + 1, self.total_frames):
                if self.isInterruptionRequested():
                    stop_reason = "使用者取消追蹤"
                    self._report("cancelled", stop_reason, i, lost_count=lost_count, patience=patience)
                    break

                ret, frame = cap.read()
                if not ret:
                    stop_reason = "影片讀取結束或讀取失敗"
                    break

                current_bbox = None
                if is_tracking:
                    update_started = time.monotonic()
                    success, bbox = tracker.update(frame)
                    update_elapsed = time.monotonic() - update_started
                    if success and bbox[2] > 0 and bbox[3] > 0:
                        current_bbox = tuple(map(int, bbox))
                        tracking_data[i] = current_bbox
                        last_tracked_frame = i
                        lost_count = 0

                        # 每 10 幀用 OCR 驗證一次 CSRT 框，避免 CSRT 成功但實際已偏離文字。
                        if i % verify_interval == 0:
                            bx, by, bw, bh = current_bbox
                            margin_x = max(40, int(bw * 0.75))
                            margin_y = max(30, int(bh * 0.75))
                            frame_h, frame_w = frame.shape[:2]
                            vx1, vy1 = max(0, bx - margin_x), max(0, by - margin_y)
                            vx2, vy2 = min(frame_w, bx + bw + margin_x), min(frame_h, by + bh + margin_y)
                            verify_crop = frame[vy1:vy2, vx1:vx2]
                            self._report("verify_running", "定期 OCR 校正：確認 CSRT 框內仍有目標文字", i,
                                         bbox=current_bbox, verify_roi=(vx1, vy1, vx2-vx1, vy2-vy1))
                            verify_started = time.monotonic()
                            verify_results = reader.readtext(cv2.cvtColor(verify_crop, cv2.COLOR_BGR2RGB))
                            verify_elapsed = time.monotonic() - verify_started
                            normalized_target = "".join(self.target_text.split()).casefold()
                            verified_match = None
                            for vbbox, vtext, vprob in verify_results:
                                if normalized_target and normalized_target in "".join(vtext.split()).casefold():
                                    verified_match = (vbbox, vtext, float(vprob))
                                    break
                            self._report("verify_done", "定期 OCR 校正完成" if verified_match else "定期 OCR 未在追蹤框附近找到目標",
                                         i, elapsed_seconds=verify_elapsed, result_count=len(verify_results),
                                         bbox=current_bbox, ocr_results=[{"text": t, "confidence": float(p)} for _, t, p in verify_results[:10]])
                            if verified_match:
                                vbbox, vtext, vprob = verified_match
                                corrected_roi = self._make_roi(frame, vbbox, vx1, vy1)
                                tracker = self._create_tracker()
                                tracker.init(frame, corrected_roi)
                                current_bbox = corrected_roi
                                tracking_data[i] = corrected_roi
                                last_verified_roi = corrected_roi
                                verify_misses = 0
                                self._report("verified", "OCR 確認目標存在並校正框位置", i,
                                             target_text=vtext, confidence=vprob, bbox=corrected_roi,
                                             elapsed_seconds=verify_elapsed)
                            else:
                                verify_misses += 1
                                self._report("verify_miss", f"OCR 校正連續未命中 {verify_misses}/2", i,
                                             verify_misses=verify_misses, bbox=current_bbox,
                                             elapsed_seconds=verify_elapsed)
                                if verify_misses >= 2:
                                    is_tracking = False
                                    self._report("tracking_lost", "連續兩次 OCR 校正未命中，停止信任 CSRT 並啟動搜尋", i,
                                                 verify_misses=verify_misses, last_verified_roi=last_verified_roi)

                        if is_tracking and i % 3 == 0:
                            self._emit_frame(frame, i, "TRACKING", bbox=current_bbox)
                            self.progress_signal.emit(i, f"AI 追蹤中：{i}/{self.total_frames} 幀")
                            self._report("tracking", "CSRT 正常追蹤", i, bbox=current_bbox,
                                         lost_count=lost_count, patience=patience,
                                         tracker_elapsed_seconds=update_elapsed)
                    else:
                        is_tracking = False
                        self._report("tracking_lost", "CSRT 追蹤失敗，開始局部 OCR 搜尋", i,
                                     lost_count=lost_count, patience=patience,
                                     tracker_elapsed_seconds=update_elapsed)

                if not is_tracking:
                    # 使用最近一次 OCR 確認的位置當搜尋錨點，避免錯誤 CSRT 框把搜尋區帶偏。
                    last_x, last_y, last_w, last_h = last_verified_roi
                    frame_h, frame_w = frame.shape[:2]

                    # 隨著連續失敗逐步擴大搜尋區域，不再永遠只搜尋固定的 150px 範圍。
                    search_pad = min(150 + (lost_count // 3) * 100, max(frame_w, frame_h))
                    sx1 = max(0, last_x - search_pad)
                    sy1 = max(0, last_y - search_pad)
                    sx2 = min(frame_w, last_x + last_w + search_pad)
                    sy2 = min(frame_h, last_y + last_h + search_pad)
                    local_search_roi = (sx1, sy1, max(1, sx2 - sx1), max(1, sy2 - sy1))

                    # 每第 5 次未找回時，改做全畫面 OCR，避免目標已移出局部搜尋區卻一直找不到。
                    use_full_frame = ((lost_count + 1) % 5 == 0)
                    if use_full_frame:
                        crop_x, crop_y = 0, 0
                        ocr_frame = frame
                        visible_search_roi = (0, 0, frame_w - 1, frame_h - 1)
                        search_label = "FULL-FRAME OCR FALLBACK"
                    else:
                        crop_x, crop_y = sx1, sy1
                        ocr_frame = frame[sy1:sy2, sx1:sx2]
                        visible_search_roi = local_search_roi
                        search_label = "OCR SEARCH"

                    self._emit_frame(frame, i, search_label, search_roi=visible_search_roi,
                                     note=f"lost {lost_count}/{patience}")
                    found_again = False
                    if ocr_frame.shape[0] > 10 and ocr_frame.shape[1] > 10:
                        self._report("ocr_running", "正在全畫面重新搜尋" if use_full_frame else "正在局部搜尋目標文字", i,
                                     lost_count=lost_count, patience=patience,
                                     search_roi=visible_search_roi, search_mode="full_frame" if use_full_frame else "local",
                                     target_text=self.target_text)
                        ocr_started = time.monotonic()
                        re_results = reader.readtext(cv2.cvtColor(ocr_frame, cv2.COLOR_BGR2RGB))
                        ocr_elapsed = time.monotonic() - ocr_started
                        self._report("ocr_done", f"{'全畫面' if use_full_frame else '局部'} OCR 完成，耗時 {ocr_elapsed:.2f} 秒", i,
                                     lost_count=lost_count, patience=patience,
                                     elapsed_seconds=ocr_elapsed, result_count=len(re_results),
                                     search_mode="full_frame" if use_full_frame else "local",
                                     ocr_results=[{"text": t, "confidence": float(p)} for _, t, p in re_results[:10]])

                        # 文字比對忽略前後空白與大小寫，降低 OCR 輸出格式差異造成的漏判。
                        normalized_target = "".join(self.target_text.split()).casefold()
                        for new_bbox, text, prob in re_results:
                            normalized_text = "".join(text.split()).casefold()
                            if normalized_target and normalized_target in normalized_text:
                                new_roi = self._make_roi(frame, new_bbox, crop_x, crop_y)
                                tracker = self._create_tracker()
                                tracker.init(frame, new_roi)
                                tracking_data[i] = new_roi
                                last_tracked_frame = i
                                last_verified_roi = new_roi
                                verify_misses = 0
                                is_tracking = True
                                found_again = True
                                lost_count = 0
                                self._emit_frame(frame, i, "REACQUIRED", bbox=new_roi)
                                self._report("recovered", "OCR 找回目標，重新啟動 CSRT", i,
                                             target_text=text, confidence=float(prob), bbox=new_roi,
                                             lost_count=lost_count, patience=patience,
                                             search_mode="full_frame" if use_full_frame else "local")
                                self.progress_signal.emit(i, "OCR 找回目標，恢復追蹤")
                                break

                    if not found_again:
                        lost_count += 1
                        self._report("lost", f"尚未找回目標（{lost_count}/{patience}）", i,
                                     lost_count=lost_count, patience=patience,
                                     search_roi=visible_search_roi,
                                     search_mode="full_frame" if use_full_frame else "local")
                        self.progress_signal.emit(i, f"正在重新尋找：{lost_count}/{patience}")
                        if lost_count >= patience:
                            stop_reason = f"連續 {patience} 次搜尋未找回目標（每 5 次包含一次全畫面 OCR）"
                            self._report("stopped", stop_reason, i, lost_count=lost_count, patience=patience)
                            break

            final_success = bool(tracking_data)
            if self.isInterruptionRequested():
                stop_reason = "使用者取消追蹤"
            self._report("finished", stop_reason, last_tracked_frame,
                         last_tracked_frame=last_tracked_frame,
                         tracked_frames=len(tracking_data), lost_count=lost_count,
                         patience=patience, success=final_success)

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
                for m in self.mosaic_tasks:
                    if m["start"] <= current_frame <= m["end"]:
                        if m.get("type") == "tracking":
                            roi = m["tracking_data"].get(current_frame)
                        else:
                            roi = m.get("roi")
                        if roi:
                            x, y, w, h = map(int, roi)
                            x, y = max(0, x), max(0, y)
                            w, h = min(width - x, w), min(height - y, h)
                            if w > 0 and h > 0:
                                roi_frame = frame[y:y + h, x:x + w]
                                strength = max(1, int(m.get("strength", 15)))
                                s_w, s_h = max(1, w // strength), max(1, h // strength)
                                mosaic = cv2.resize(cv2.resize(roi_frame, (s_w, s_h), interpolation=cv2.INTER_LINEAR),
                                                    (w, h), interpolation=cv2.INTER_NEAREST)
                                frame[y:y + h, x:x + w] = mosaic
                out.write(frame)
                current_frame += 1
                if current_frame % 15 == 0:
                    self.progress_signal.emit(current_frame, total)
        finally:
            cap.release()
            out.release()
        self.finished_signal.emit(self.output_path)
