import sys
import cv2
import os
import PyQt5

plugin_path = os.path.join(
    os.path.dirname(PyQt5.__file__),
    "Qt5",
    "plugins"
)

os.environ["QT_PLUGIN_PATH"] = plugin_path
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
    plugin_path, "platforms"
)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, QMessageBox, QGroupBox, QSpinBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

class VideoProcessorThread(QThread):
    """獨立的背景處理執行緒，避免輸出影片時 GUI 卡死"""
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(str)

    def __init__(self, video_path, output_path, roi, start_frame, end_frame, strength):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.roi = roi
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.strength = strength

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        # 這裡使用最基礎的固定座標馬賽克作範例。若要追蹤可在此加入 Tracker
        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 只有在使用者設定的起訖範圍內才打馬賽克
            if self.start_frame <= current_frame <= self.end_frame and self.roi:
                x, y, w, h = self.roi
                if w > 0 and h > 0:
                    roi_frame = frame[y:y+h, x:x+w]
                    small_w, small_h = max(1, w // self.strength), max(1, h // self.strength)
                    small = cv2.resize(roi_frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                    frame[y:y+h, x:x+w] = mosaic

            out.write(frame)
            current_frame += 1
            
            if current_frame % 15 == 0:
                self.progress_signal.emit(current_frame, total)

        cap.release()
        out.release()
        self.finished_signal.emit(self.output_path)


class ProVideoEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Pro Video Editor")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;") # 深色專業主題

        # 核心變數
        self.video_path = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_playing = False
        
        self.total_frames = 0
        self.current_frame = 0
        self.roi = None
        self.mosaic_start = 0
        self.mosaic_end = 0

        self.init_ui()

    def init_ui(self):
        # 主佈局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ================= 左側控制面板 =================
        left_panel = QVBoxLayout()
        left_panel.setAlignment(Qt.AlignTop)
        left_panel.setContentsMargins(10, 10, 20, 10)

        # 1. 檔案操作區
        group_file = QGroupBox("File Management")
        group_file.setStyleSheet("QGroupBox { border: 1px solid #333; border-radius: 5px; margin-top: 10px; font-weight: bold; }")
        vbox_file = QVBoxLayout()
        btn_import = QPushButton("Import Video")
        btn_import.setStyleSheet("background-color: #2d5a88; padding: 8px; border-radius: 4px;")
        btn_import.clicked.connect(self.import_video)
        vbox_file.addWidget(btn_import)
        group_file.setLayout(vbox_file)
        left_panel.addWidget(group_file)

        # 2. 時間標記區 (設定馬賽克起訖時間)
        group_time = QGroupBox("Mosaic Timeline Control")
        group_time.setStyleSheet("QGroupBox { border: 1px solid #333; border-radius: 5px; margin-top: 10px; font-weight: bold; }")
        vbox_time = QVBoxLayout()
        
        self.lbl_time_info = QLabel("Target: 0s to 0s")
        vbox_time.addWidget(self.lbl_time_info)

        btn_set_in = QPushButton("Set Mark In (開始)")
        btn_set_in.clicked.connect(self.set_mark_in)
        btn_set_in.setStyleSheet("background-color: #444; padding: 5px;")
        
        btn_set_out = QPushButton("Set Mark Out (結束)")
        btn_set_out.clicked.connect(self.set_mark_out)
        btn_set_out.setStyleSheet("background-color: #444; padding: 5px;")
        
        vbox_time.addWidget(btn_set_in)
        vbox_time.addWidget(btn_set_out)
        group_time.setLayout(vbox_time)
        left_panel.addWidget(group_time)

        # 3. 馬賽克設定區
        group_mosaic = QGroupBox("Mosaic Settings")
        group_mosaic.setStyleSheet("QGroupBox { border: 1px solid #333; border-radius: 5px; margin-top: 10px; font-weight: bold; }")
        vbox_mosaic = QVBoxLayout()
        
        btn_draw_roi = QPushButton("Draw Region (框選)")
        btn_draw_roi.clicked.connect(self.draw_roi)
        btn_draw_roi.setStyleSheet("background-color: #006600; padding: 8px; border-radius: 4px;")
        
        self.lbl_roi_info = QLabel("ROI: Not Selected")
        
        self.spin_strength = QSpinBox()
        self.spin_strength.setRange(5, 50)
        self.spin_strength.setValue(15)
        self.spin_strength.setPrefix("Strength: ")
        
        vbox_mosaic.addWidget(btn_draw_roi)
        vbox_mosaic.addWidget(self.lbl_roi_info)
        vbox_mosaic.addWidget(self.spin_strength)
        group_mosaic.setLayout(vbox_mosaic)
        left_panel.addWidget(group_mosaic)

        # 4. 匯出區
        group_export = QGroupBox("Export")
        group_export.setStyleSheet("QGroupBox { border: 1px solid #333; border-radius: 5px; margin-top: 10px; font-weight: bold; }")
        vbox_export = QVBoxLayout()
        self.btn_export = QPushButton("Export Video")
        self.btn_export.clicked.connect(self.export_video)
        self.btn_export.setStyleSheet("background-color: #882d2d; padding: 10px; border-radius: 4px; font-weight: bold;")
        self.btn_export.setEnabled(False)
        vbox_export.addWidget(self.btn_export)
        
        self.lbl_status = QLabel("Status: Ready")
        vbox_export.addWidget(self.lbl_status)
        group_export.setLayout(vbox_export)
        left_panel.addWidget(group_export)

        main_layout.addLayout(left_panel, stretch=1)

        # ================= 右側影片預覽與時間軸 =================
        right_panel = QVBoxLayout()
        
        # 影片顯示區域
        self.video_display = QLabel("No Video Loaded")
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setStyleSheet("background-color: #000000; border: 2px solid #333;")
        right_panel.addWidget(self.video_display, stretch=1)

        # 播放控制與時間軸
        timeline_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setFixedWidth(80)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.sliderMoved.connect(self.set_position)
        
        self.lbl_time = QLabel("00:00 / 00:00")
        
        timeline_layout.addWidget(self.btn_play)
        timeline_layout.addWidget(self.slider)
        timeline_layout.addWidget(self.lbl_time)
        
        right_panel.addLayout(timeline_layout)
        main_layout.addLayout(right_panel, stretch=4)

    def import_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.video_path = file_name
            self.cap = cv2.VideoCapture(self.video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setRange(0, self.total_frames)
            self.mosaic_end = self.total_frames # 預設結束點為影片結尾
            
            self.update_time_label()
            self.read_and_display_frame()
            self.btn_export.setEnabled(True)

    def toggle_play(self):
        if not self.cap: return
        if self.is_playing:
            self.timer.stop()
            self.btn_play.setText("▶ Play")
        else:
            self.timer.start(30) # 大約 30 FPS
            self.btn_play.setText("⏸ Pause")
        self.is_playing = not self.is_playing

    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.slider.setValue(self.current_frame)
                self.display_frame(frame)
                self.update_time_label()
            else:
                self.toggle_play() # 影片結束

    def set_position(self, position):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            self.current_frame = position
            self.read_and_display_frame()
            self.update_time_label()

    def read_and_display_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)

    def display_frame(self, frame):
        # 轉換 OpenCV 的 BGR 格式為 PyQt 的 RGB 格式
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 縮放影片以適應介面
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.video_display.width(), self.video_display.height(), Qt.KeepAspectRatio)
        self.video_display.setPixmap(scaled_pixmap)

    def update_time_label(self):
        if not self.cap: return
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        current_sec = int(self.current_frame / fps)
        total_sec = int(self.total_frames / fps)
        self.lbl_time.setText(f"{current_sec // 60:02d}:{current_sec % 60:02d} / {total_sec // 60:02d}:{total_sec % 60:02d}")

    def set_mark_in(self):
        self.mosaic_start = self.current_frame
        self.lbl_time_info.setText(f"Target: F{self.mosaic_start} to F{self.mosaic_end}")

    def set_mark_out(self):
        self.mosaic_end = self.current_frame
        self.lbl_time_info.setText(f"Target: F{self.mosaic_start} to F{self.mosaic_end}")

    def draw_roi(self):
        if not self.cap: return
        # 暫停播放
        if self.is_playing:
            self.toggle_play()
            
        # 抓取當下這張影格讓使用者畫框
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            QMessageBox.information(self, "Instruction", "Press ENTER to confirm ROI, or 'C' to cancel.")
            self.roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select ROI")
            
            if self.roi and self.roi[2] > 0:
                self.lbl_roi_info.setText(f"ROI: X:{self.roi[0]} Y:{self.roi[1]}")
            else:
                self.lbl_roi_info.setText("ROI: Cancelled")

    def export_video(self):
        if not self.roi:
            QMessageBox.warning(self, "Warning", "Please Draw Region (ROI) first!")
            return
            
        output_path, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "Video Files (*.mp4)")
        if not output_path: return

        self.lbl_status.setText("Exporting...")
        self.btn_export.setEnabled(False)
        
        # 啟動背景處理執行緒
        self.worker = VideoProcessorThread(self.video_path, output_path, self.roi, 
                                           self.mosaic_start, self.mosaic_end, self.spin_strength.value())
        self.worker.progress_signal.connect(self.update_export_progress)
        self.worker.finished_signal.connect(self.export_finished)
        self.worker.start()

    def update_export_progress(self, current, total):
        self.lbl_status.setText(f"Exporting: {current} / {total} frames")

    def export_finished(self, path):
        self.lbl_status.setText("Ready")
        self.btn_export.setEnabled(True)
        QMessageBox.information(self, "Success", f"Video exported to:\n{path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = ProVideoEditor()
    editor.show()
    sys.exit(app.exec_())