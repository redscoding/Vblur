import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

class AutoMosaicApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto-Mosaic Video Editor")
        self.root.geometry("350x250")
        
        # 狀態變數
        self.video_path = None
        self.output_path = None
        self.roi = None  # 儲存你圖片中的 [X, Y, W, H]

        # === GUI 介面佈局 (對應圖片左側面板) ===
        # 1. 匯入影片按鈕
        self.btn_import = tk.Button(root, text="Import Video", command=self.import_video, width=20, bg="#4a4a4a", fg="white")
        self.btn_import.pack(pady=15)

        # 2. 馬賽克強度滑桿
        self.lbl_strength = tk.Label(root, text="Mosaic Strength")
        self.lbl_strength.pack()
        self.scale_strength = tk.Scale(root, from_=5, to=50, orient=tk.HORIZONTAL, length=200)
        self.scale_strength.set(15) # 預設強度
        self.scale_strength.pack(pady=5)

        # 3. 開始處理按鈕
        self.btn_process = tk.Button(root, text="Start Processing", command=self.start_processing, state=tk.DISABLED, width=20)
        self.btn_process.pack(pady=10)

        # 4. 狀態提示文字
        self.lbl_status = tk.Label(root, text="Status: Waiting for video...", fg="gray")
        self.lbl_status.pack()

    def import_video(self):
        """匯入影片並讓使用者框選區域"""
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if not self.video_path:
            return

        self.lbl_status.config(text=f"Loaded: {self.video_path.split('/')[-1]}")
        
        # 讀取影片第一幀來進行框選
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "無法讀取影片")
            return
            
        messagebox.showinfo("操作提示", "請在彈出的視窗中，用滑鼠拖曳框選要打馬賽克的區域 (ROI)\n框選完成後請按 [SPACE] 或 [ENTER] 鍵確認。")
        
        # 呼叫 OpenCV 內建的 ROI 選擇器 (完美實現你圖片中的 Drawn Mosaic Area)
        self.roi = cv2.selectROI("Select Mosaic Area (Press ENTER to confirm)", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Mosaic Area (Press ENTER to confirm)")
        cap.release()

        # 檢查是否有成功框選 (寬度與高度 > 0)
        if self.roi and self.roi[2] > 0 and self.roi[3] > 0:
            x, y, w, h = self.roi
            self.lbl_status.config(text=f"ROI Selected: [X:{x}, Y:{y}, W:{w}, H:{h}]", fg="blue")
            self.btn_process.config(state=tk.NORMAL)
        else:
            self.lbl_status.config(text="取消框選", fg="red")
            self.btn_process.config(state=tk.DISABLED)

    def start_processing(self):
        """選擇輸出路徑並啟動背景處理執行緒"""
        self.output_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4")])
        if not self.output_path:
            return

        self.btn_process.config(state=tk.DISABLED)
        self.btn_import.config(state=tk.DISABLED)
        self.lbl_status.config(text="Processing... 請稍候", fg="orange")
        
        # 使用 Thread 避免處理影片時 GUI 卡死
        threading.Thread(target=self.process_video, daemon=True).start()

    def process_video(self):
        """核心影片處理與馬賽克演算法"""
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 設定影片輸出格式 (H.264 或 mp4v)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        strength = self.scale_strength.get()
        x, y, w, h = self.roi

        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. 擷取被框選的區域 (ROI)
            roi_frame = frame[y:y+h, x:x+w]
            
            # 2. 馬賽克演算法：先大幅縮小，再放大回原尺寸 (利用插值法產生像素區塊感)
            small_w, small_h = max(1, w // strength), max(1, h // strength)
            small = cv2.resize(roi_frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 3. 將打好馬賽克的區域貼回原圖
            frame[y:y+h, x:x+w] = mosaic

            # 寫入輸出影片
            out.write(frame)
            current_frame += 1
            
            # 在終端機印出進度
            if current_frame % 30 == 0:
                print(f"進度: {current_frame} / {total_frames} 幀")

        # 釋放資源
        cap.release()
        out.release()
        
        # 恢復 GUI 狀態
        self.lbl_status.config(text="Processing Complete!", fg="green")
        self.btn_process.config(state=tk.NORMAL)
        self.btn_import.config(state=tk.NORMAL)
        messagebox.showinfo("Success", f"影片已成功儲存至:\n{self.output_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoMosaicApp(root)
    root.mainloop()
