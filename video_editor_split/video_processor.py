import cv2
import easyocr
from PyQt5.QtCore import QThread, pyqtSignal

# ================= 混合 AI 追蹤執行緒 (高度容錯與智慧重連版) =================
class OCRTrackingThread(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict, int, bool) # 回傳: tracking_data, last_frame, 是否成功

    def __init__(self, video_path, start_frame, total_frames, target_text):
        super().__init__()
        self.video_path = video_path
        self.start_frame = start_frame
        self.total_frames = total_frames
        self.target_text = target_text

    def run(self):
        self.progress_signal.emit(self.start_frame, "正在載入 OCR 模型...")
        reader = easyocr.Reader(['ch_tra', 'en'], gpu=False) 

        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        ret, frame = cap.read()
        if not ret:
            self.finished_signal.emit({}, self.start_frame, False)
            return

        self.progress_signal.emit(self.start_frame, "正在全畫面尋找初始目標...")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = reader.readtext(rgb_frame)

        initial_roi = None
        for (bbox, text, prob) in results:
            if self.target_text in text:
                # 使用 min/max 確保傾斜字體也能抓到正確的矩形寬高
                xs = [pt[0] for pt in bbox]
                ys = [pt[1] for pt in bbox]
                x, y = int(min(xs)), int(min(ys))
                w, h = int(max(xs) - min(xs)), int(max(ys) - min(ys))
                
                pad_x, pad_y = int(w * 0.2), int(h * 0.4)
                x, y = max(0, x - pad_x), max(0, y - pad_y)
                w, h = min(frame.shape[1] - x, w + pad_x * 2), min(frame.shape[0] - y, h + pad_y * 2)
                
                initial_roi = (x, y, w, h)
                break

        if not initial_roi:
            cap.release()
            self.finished_signal.emit({}, self.start_frame, False)
            return

        self.progress_signal.emit(self.start_frame, "目標鎖定！開始計算 AI 追蹤軌跡...")
        try: tracker = cv2.TrackerCSRT_create()
        except AttributeError: tracker = cv2.legacy.TrackerCSRT_create()
            
        tracker.init(frame, initial_roi)
        tracking_data = {self.start_frame: initial_roi}
        last_tracked_frame = self.start_frame

        # 加入 Patience (容錯時間)
        is_tracking = True
        lost_count = 0
        patience = 30 # 允許目標因模糊或被遮擋而短暫消失 30 幀 (約 1 秒)

        for i in range(self.start_frame + 1, self.total_frames):
            if self.isInterruptionRequested(): break 
            
            ret, frame = cap.read()
            if not ret: break

            # 只有在追蹤狀態下，才呼叫 CSRT
            if is_tracking:
                success, bbox = tracker.update(frame)
                if success and bbox[2] > 0 and bbox[3] > 0:
                    tracking_data[i] = tuple(map(int, bbox))
                    last_tracked_frame = i
                    lost_count = 0 # 成功追蹤，重置遺失計數
                else:
                    is_tracking = False
            
            # 追蹤失敗時，進入「局部尋找」模式
            if not is_tracking:
                last_x, last_y, last_w, last_h = tracking_data[last_tracked_frame]
                search_pad = 150
                h_max, w_max = frame.shape[:2]
                
                sx1 = max(0, last_x - search_pad)
                sy1 = max(0, last_y - search_pad)
                sx2 = min(w_max, last_x + last_w + search_pad)
                sy2 = min(h_max, last_y + last_h + search_pad)
                
                local_frame = frame[sy1:sy2, sx1:sx2]
                
                # 防呆：確保切割出來的圖片夠大
                if local_frame.shape[0] > 10 and local_frame.shape[1] > 10:
                    rgb_local = cv2.cvtColor(local_frame, cv2.COLOR_BGR2RGB)
                    re_results = reader.readtext(rgb_local)
                    
                    found_again = False
                    for (new_bbox, text, prob) in re_results:
                        if self.target_text in text:
                            xs = [pt[0] for pt in new_bbox]
                            ys = [pt[1] for pt in new_bbox]
                            x = sx1 + int(min(xs))
                            y = sy1 + int(min(ys))
                            w = int(max(xs) - min(xs))
                            h = int(max(ys) - min(ys))
                            
                            pad_x, pad_y = int(w * 0.2), int(h * 0.4)
                            x, y = max(0, x - pad_x), max(0, y - pad_y)
                            w, h = min(w_max - x, w + pad_x * 2), min(h_max - y, h + pad_y * 2)
                            new_roi = (x, y, w, h)
                            
                            try: tracker = cv2.TrackerCSRT_create()
                            except AttributeError: tracker = cv2.legacy.TrackerCSRT_create()
                                
                            tracker.init(frame, new_roi)
                            tracking_data[i] = new_roi
                            last_tracked_frame = i
                            is_tracking = True
                            found_again = True
                            lost_count = 0 # 找回目標，重置遺失計數
                            break
                            
                    if not found_again:
                        lost_count += 1
                else:
                    lost_count += 1
                
                # 如果遺失超過容錯極限，才真正放棄
                if lost_count >= patience:
                    break 
            
            # 定期更新進度與狀態 (每 3 幀更新一次 UI，讓你看得到它在努力)
            if i % 3 == 0: 
                if is_tracking:
                    self.progress_signal.emit(i, f"AI 追蹤中: {i}/{self.total_frames} 幀")
                else:
                    self.progress_signal.emit(i, f"受動態模糊影響，尋找中... ({lost_count}/{patience})")

        cap.release()
        self.finished_signal.emit(tracking_data, last_tracked_frame, True)

# ================= 背景輸出執行緒 =================
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
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret: break

            for m in self.mosaic_tasks:
                if m['start'] <= current_frame <= m['end']:
                    if m.get('type') == 'tracking':
                        roi = m['tracking_data'].get(current_frame)
                    else:
                        roi = m.get('roi')

                    if roi:
                        x, y, w, h = roi
                        x, y = max(0, x), max(0, y)
                        w, h = min(width - x, w), min(height - y, h)
                        if w > 0 and h > 0:
                            roi_frame = frame[y:y+h, x:x+w]
                            s_w, s_h = max(1, w // m['strength']), max(1, h // m['strength'])
                            mosaic = cv2.resize(cv2.resize(roi_frame, (s_w, s_h), interpolation=cv2.INTER_LINEAR),
                                                (w, h), interpolation=cv2.INTER_NEAREST)
                            frame[y:y+h, x:x+w] = mosaic

            out.write(frame)
            current_frame += 1
            if current_frame % 15 == 0: self.progress_signal.emit(current_frame, total)

        cap.release()
        out.release()
        self.finished_signal.emit(self.output_path)
