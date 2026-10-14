import cv2
import easyocr
from PyQt5.QtCore import QThread, pyqtSignal


# ================= 混合 AI 追蹤執行緒 (OCR + CSRT) =================
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
        # 載入繁體中文與英文模型
        reader = easyocr.Reader(['ch_tra', 'en'], gpu=False) 

        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        ret, frame = cap.read()
        if not ret:
            self.finished_signal.emit({}, self.start_frame, False)
            return

        self.progress_signal.emit(self.start_frame, "正在全畫面掃描文字...")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = reader.readtext(rgb_frame)

        initial_roi = None
        # 比對 OCR 結果是否包含目標關鍵字
        for (bbox, text, prob) in results:
            if self.target_text in text:
                (tl, tr, br, bl) = bbox
                x, y = int(tl[0]), int(tl[1])
                w, h = int(br[0] - tl[0]), int(br[1] - tl[1])
                initial_roi = (max(0, x), max(0, y), max(1, w), max(1, h))
                break

        if not initial_roi:
            cap.release()
            self.finished_signal.emit({}, self.start_frame, False)
            return

        self.progress_signal.emit(self.start_frame, "目標鎖定！開始計算 AI 追蹤軌跡...")
        tracker = cv2.legacy.TrackerCSRT_create()
        tracker.init(frame, initial_roi)
        
        tracking_data = {self.start_frame: initial_roi}
        last_tracked_frame = self.start_frame

        # 啟動 CSRT 追蹤後續影格
        for i in range(self.start_frame + 1, self.total_frames):
            if self.isInterruptionRequested(): break 
            
            ret, frame = cap.read()
            if not ret: break

            success, bbox = tracker.update(frame)
            if success:
                tracking_data[i] = tuple(map(int, bbox))
                last_tracked_frame = i
            else:
                break # 文字移出畫面或嚴重變形
            
            if i % 10 == 0: 
                self.progress_signal.emit(i, f"追蹤進度: {i}/{self.total_frames} 幀")

        cap.release()
        self.finished_signal.emit(tracking_data, last_tracked_frame, True)

# ================= 背景輸出執行緒 (支援雙軌混合任務) =================
class VideoProcessorThread(QThread):
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(str)

    def __init__(self, video_path, output_path, mosaic_tasks):
        super().__init__()
        self.video_path, self.output_path, self.mosaic_tasks = video_path, output_path, mosaic_tasks 

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
                    # 判斷是靜態框選還是 AI 追蹤軌跡
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
