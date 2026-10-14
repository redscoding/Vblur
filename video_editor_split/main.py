import sys 
import os
import cv2 
#需先載入避免dll被pyqt蓋掉
import easyocr 
from video_processor import OCRTrackingThread, VideoProcessorThread

import PyQt5 # ====== 設定 Windows Qt 外掛路徑 ======
plugin_path = os.path.join( os.path.dirname(PyQt5.__file__), "Qt5", "plugins" ) 
os.environ["QT_PLUGIN_PATH"] = plugin_path 
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join( plugin_path, "platforms" ) # 先載入 EasyOCR / PyTorch，再載入 GUI 元件 
from video_processor import OCRTrackingThread, VideoProcessorThread 
from PyQt5.QtWidgets import ( QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, QMessageBox, QGroupBox, QSpinBox, QFrame, QListWidget, QAbstractItemView, QLineEdit, QProgressDialog ) 
from PyQt5.QtCore import Qt, QTimer 
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor


# ================= 專業級深色主題 QSS =================
MODERN_THEME = """
QMainWindow { background-color: #1E1E1E; }
QWidget { color: #CCCCCC; font-family: "Segoe UI", "Microsoft JhengHei", sans-serif; font-size: 13px; }
QGroupBox { border: 1px solid #3A3A3A; border-radius: 6px; margin-top: 15px; font-weight: bold; color: #E0E0E0; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #4DAAFB; }
QPushButton { background-color: #333333; border: 1px solid #444444; border-radius: 4px; padding: 6px 12px; color: #FFFFFF; font-weight: bold; }
QPushButton:hover { background-color: #444444; border: 1px solid #555555; }
QPushButton:pressed { background-color: #222222; }
QPushButton#primary { background-color: #0E639C; border: none; }
QPushButton#primary:hover { background-color: #1177BB; }
QPushButton#danger { background-color: #983333; border: none; }
QPushButton#danger:hover { background-color: #B33D3D; }
QPushButton:disabled { background-color: #2D2D2D; color: #666666; }
QSlider::groove:horizontal { border: 1px solid #2B2B2B; height: 6px; background: #111111; margin: 2px 0; border-radius: 3px; }
QSlider::handle:horizontal { background: #4DAAFB; border: 1px solid #000000; width: 14px; margin: -5px 0; border-radius: 7px; }
QSlider::handle:horizontal:hover { background: #73C2FB; }
QLabel#TimeLabel { font-family: "Consolas", monospace; color: #4DAAFB; font-size: 14px; }
QSpinBox { background-color: #252526; border: 1px solid #3A3A3A; color: #FFFFFF; padding: 4px; border-radius: 3px; }
QListWidget { background-color: #252526; border: 1px solid #3A3A3A; outline: none; }
QListWidget::item:selected { background-color: #0E639C; color: white; }
QLineEdit { background-color: #252526; border: 1px solid #3A3A3A; color: #FFFFFF; padding: 6px; border-radius: 4px; }
"""


class VideoDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.setMinimumSize(400, 300)
        
    def set_image(self, pixmap):
        self.pixmap = pixmap
        self.update() 

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000")) 
        if self.pixmap:
            scaled_pixmap = self.pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (self.width() - scaled_pixmap.width()) // 2
            y = (self.height() - scaled_pixmap.height()) // 2
            painter.drawPixmap(x, y, scaled_pixmap)


# ================= 主程式介面 =================
class ProVideoEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Pro Video Editor (Hybrid: Static + OCR Tracking)")
        self.setGeometry(100, 100, 1200, 750)
        self.setStyleSheet(MODERN_THEME)

        self.video_path, self.cap, self.is_playing = None, None, False
        self.total_frames, self.current_frame = 0, 0
        self.current_raw_frame = None  
        self.mosaic_tasks = [] 

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)

        left_panel = QFrame()
        left_panel.setFixedWidth(340)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 1. 檔案區
        group_file = QGroupBox("1. Source Video")
        vbox_file = QVBoxLayout()
        self.btn_import = QPushButton("Import Video (匯入影片)")
        self.btn_import.setObjectName("primary")
        self.btn_import.clicked.connect(self.import_video)
        vbox_file.addWidget(self.btn_import)
        group_file.setLayout(vbox_file)

        # 2. 段落管理區
        group_mosaic = QGroupBox("2. Mosaic Segments (混合追蹤段落)")
        vbox_mosaic = QVBoxLayout()

        # [新增] OCR 文字追蹤區
        hbox_ocr = QHBoxLayout()
        self.txt_ocr = QLineEdit()
        self.txt_ocr.setPlaceholderText("輸入目標文字 (如: 車牌)")
        self.btn_ocr = QPushButton("🔤 掃描並追蹤")
        self.btn_ocr.setObjectName("primary")
        self.btn_ocr.setEnabled(False)
        self.btn_ocr.clicked.connect(self.start_ocr_tracking)
        hbox_ocr.addWidget(self.txt_ocr, stretch=2)
        hbox_ocr.addWidget(self.btn_ocr, stretch=1)

        # [新增] 傳統手動區
        hbox_add_rm = QHBoxLayout()
        self.btn_add_task = QPushButton("➕ 手動固定框選")
        self.btn_add_task.setEnabled(False) 
        self.btn_add_task.clicked.connect(self.add_new_task)
        self.btn_rm_task = QPushButton("🗑 刪除段落")
        self.btn_rm_task.setEnabled(False)
        self.btn_rm_task.clicked.connect(self.remove_task)
        hbox_add_rm.addWidget(self.btn_add_task)
        hbox_add_rm.addWidget(self.btn_rm_task)

        # 清單
        self.list_tasks = QListWidget()
        self.list_tasks.setFixedHeight(120)
        self.list_tasks.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_tasks.currentRowChanged.connect(self.on_task_selected)

        # 編輯面板
        self.edit_panel = QWidget()
        vbox_edit = QVBoxLayout(self.edit_panel)
        vbox_edit.setContentsMargins(0, 5, 0, 0)
        
        hbox_se = QHBoxLayout()
        self.btn_set_start = QPushButton("設定起點")
        self.btn_set_start.clicked.connect(self.set_task_start)
        self.lbl_start = QLabel("Start: --")
        self.btn_set_end = QPushButton("設定終點")
        self.btn_set_end.clicked.connect(self.set_task_end)
        self.lbl_end = QLabel("End: --")
        hbox_se.addWidget(self.btn_set_start)
        hbox_se.addWidget(self.lbl_start)
        hbox_se.addWidget(self.btn_set_end)
        hbox_se.addWidget(self.lbl_end)

        hbox_roi = QHBoxLayout()
        self.btn_redraw = QPushButton("重新畫框 (僅限手動)")
        self.btn_redraw.clicked.connect(self.redraw_roi)
        self.spin_strength = QSpinBox()
        self.spin_strength.setRange(5, 50)
        self.spin_strength.setPrefix("模糊強度: ")
        self.spin_strength.valueChanged.connect(self.update_strength)
        hbox_roi.addWidget(self.btn_redraw)
        hbox_roi.addWidget(self.spin_strength)

        vbox_edit.addLayout(hbox_se)
        vbox_edit.addLayout(hbox_roi)
        self.edit_panel.setEnabled(False)

        vbox_mosaic.addLayout(hbox_ocr)
        vbox_mosaic.addLayout(hbox_add_rm)
        vbox_mosaic.addWidget(self.list_tasks)
        vbox_mosaic.addWidget(self.edit_panel)
        group_mosaic.setLayout(vbox_mosaic)

        # 匯出區
        group_export = QGroupBox("3. Export")
        vbox_export = QVBoxLayout()
        self.btn_export = QPushButton("Export Video (匯出影片)")
        self.btn_export.setObjectName("danger")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_video)
        self.lbl_status = QLabel("Status: Ready")
        vbox_export.addWidget(self.btn_export)
        vbox_export.addWidget(self.lbl_status)
        group_export.setLayout(vbox_export)

        left_layout.addWidget(group_file)
        left_layout.addWidget(group_mosaic)
        left_layout.addWidget(group_export)
        left_layout.addStretch() 

        # 右側預覽區
        right_panel = QVBoxLayout()
        self.video_display = VideoDisplayWidget()
        
        control_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setFixedWidth(80)
        self.btn_play.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self.set_position)
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setObjectName("TimeLabel")
        self.lbl_time.setFixedWidth(110)
        self.lbl_time.setAlignment(Qt.AlignCenter)

        control_layout.addWidget(self.btn_play)
        control_layout.addWidget(self.slider)
        control_layout.addWidget(self.lbl_time)

        right_panel.addWidget(self.video_display, stretch=1)
        right_panel.addLayout(control_layout)

        main_layout.addWidget(left_panel)
        main_layout.addLayout(right_panel, stretch=1)

    # =============== 播放控制 ===============
    def import_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.video_path = file_name
            self.cap = cv2.VideoCapture(self.video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setRange(0, self.total_frames)
            
            self.btn_add_task.setEnabled(True)
            self.btn_ocr.setEnabled(True)
            self.btn_export.setEnabled(True)
            self.update_time_label()
            self.read_and_display_frame()

    def toggle_play(self):
        if not self.cap: return
        if self.is_playing:
            self.timer.stop()
            self.btn_play.setText("▶ Play")
        else:
            self.timer.start(33) 
            self.btn_play.setText("⏸ Pause")
        self.is_playing = not self.is_playing

    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.current_raw_frame = frame
                self.slider.setValue(self.current_frame)
                self.display_frame(frame)
                self.update_time_label()
            else:
                self.toggle_play()

    def set_position(self, position):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            self.current_frame = position
            self.read_and_display_frame()
            self.update_time_label()

    def read_and_display_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_raw_frame = frame
                self.display_frame(frame)

    # 混合即時渲染邏輯
    def apply_mosaics(self, frame):
        frame_copy = frame.copy()
        h_max, w_max = frame_copy.shape[:2]
        
        for m in self.mosaic_tasks:
            if m['start'] <= self.current_frame <= m['end']:
                if m.get('type') == 'tracking':
                    roi = m['tracking_data'].get(self.current_frame)
                else:
                    roi = m.get('roi')

                if roi:
                    x, y, w, h = roi
                    x, y = max(0, x), max(0, y)
                    w, h = min(w_max - x, w), min(h_max - y, h)
                    
                    if w > 0 and h > 0:
                        roi_frame = frame_copy[y:y+h, x:x+w]
                        s_w, s_h = max(1, w // m['strength']), max(1, h // m['strength'])
                        mosaic = cv2.resize(cv2.resize(roi_frame, (s_w, s_h), interpolation=cv2.INTER_LINEAR),
                                            (w, h), interpolation=cv2.INTER_NEAREST)
                        frame_copy[y:y+h, x:x+w] = mosaic
        return frame_copy

    def display_frame(self, frame):
        processed = self.apply_mosaics(frame) 
        rgb_img = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        q_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_display.set_image(QPixmap.fromImage(q_img))

    def refresh_display(self):
        if self.current_raw_frame is not None:
            self.display_frame(self.current_raw_frame)

    def update_time_label(self):
        if not self.cap: return
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        c_sec, t_sec = int(self.current_frame / fps), int(self.total_frames / fps)
        self.lbl_time.setText(f"{c_sec//60:02d}:{c_sec%60:02d} / {t_sec//60:02d}:{t_sec%60:02d}")

    # =============== AI 追蹤 (OCR + CSRT) ===============
    def start_ocr_tracking(self):
        target_text = self.txt_ocr.text().strip()
        if not target_text:
            QMessageBox.warning(self, "輸入錯誤", "請先輸入要追蹤的目標文字！")
            return
            
        if self.is_playing: self.toggle_play()

        # 防呆與 UI 互動進度條
        self.progress_dialog = QProgressDialog("正在啟動 AI 引擎...", "取消", self.current_frame, self.total_frames, self)
        self.progress_dialog.setWindowTitle("OCR Tracking")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(True)

        self.ocr_thread = OCRTrackingThread(self.video_path, self.current_frame, self.total_frames, target_text)
        self.progress_dialog.canceled.connect(self.ocr_thread.requestInterruption)
        self.ocr_thread.progress_signal.connect(self.update_ocr_progress)
        self.ocr_thread.finished_signal.connect(self.on_ocr_finished)
        
        self.ocr_thread.start()
        self.progress_dialog.show()

    def update_ocr_progress(self, val, msg):
        self.progress_dialog.setValue(val)
        self.progress_dialog.setLabelText(msg)

    def on_ocr_finished(self, tracking_data, last_tracked_frame, success):
        if not success:
            QMessageBox.warning(self, "掃描失敗", f"在當前畫面找不到文字「{self.txt_ocr.text()}」\n請確認文字是否清晰可見，或換一幀試試。")
            return
            
        task = {
            'type': 'tracking',
            'start': min(tracking_data.keys()),
            'end': last_tracked_frame,
            'tracking_data': tracking_data,
            'strength': 15
        }
        self.mosaic_tasks.append(task)
        self.update_task_list_display()
        self.list_tasks.setCurrentRow(len(self.mosaic_tasks) - 1)
        self.refresh_display()
        
        if last_tracked_frame < self.total_frames - 1 and not self.progress_dialog.wasCanceled():
            QMessageBox.information(self, "追蹤結束", f"目標在第 {last_tracked_frame} 幀後離開畫面或變形，已自動設為終點。")

    # =============== 傳統手動段落管理 ===============
    def add_new_task(self):
        if not self.cap or self.current_raw_frame is None: return
        if self.is_playing: self.toggle_play()
        
        QMessageBox.information(self, "操作提示", "請框選要『固定打碼』的區域，畫完按【ENTER】。")
        roi = cv2.selectROI("Select ROI", self.current_raw_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        
        if roi and roi[2] > 0 and roi[3] > 0:
            task = {
                'type': 'static',
                'start': self.current_frame,
                'end': self.total_frames,
                'roi': roi,
                'strength': 15
            }
            self.mosaic_tasks.append(task)
            self.update_task_list_display()
            self.list_tasks.setCurrentRow(len(self.mosaic_tasks) - 1)
            self.refresh_display()

    def update_task_list_display(self):
        self.list_tasks.clear()
        for i, m in enumerate(self.mosaic_tasks):
            t_type = "AI追蹤" if m.get('type') == 'tracking' else "固定"
            self.list_tasks.addItem(f"[{t_type}] 段落 {i+1}: F{m['start']} ➔ F{m['end']}")

    def on_task_selected(self, row):
        if row < 0 or row >= len(self.mosaic_tasks):
            self.edit_panel.setEnabled(False)
            self.btn_rm_task.setEnabled(False)
            return
            
        self.edit_panel.setEnabled(True)
        self.btn_rm_task.setEnabled(True)
        task = self.mosaic_tasks[row]
        
        self.lbl_start.setText(f"Start: {task['start']}")
        self.lbl_end.setText(f"End: {task['end']}")
        
        # AI追蹤無法「重新畫框」，自動停用該按鈕防呆
        self.btn_redraw.setEnabled(task.get('type') != 'tracking')
        
        self.spin_strength.blockSignals(True)
        self.spin_strength.setValue(task['strength'])
        self.spin_strength.blockSignals(False)

    def set_task_start(self):
        row = self.list_tasks.currentRow()
        if row < 0: return
        task = self.mosaic_tasks[row]
        if self.current_frame > task['end']: task['end'] = self.current_frame
        task['start'] = self.current_frame
        self.update_task_list_display()
        self.list_tasks.setCurrentRow(row)
        self.refresh_display()

    def set_task_end(self):
        row = self.list_tasks.currentRow()
        if row < 0: return
        task = self.mosaic_tasks[row]
        if self.current_frame < task['start']: task['start'] = self.current_frame
        task['end'] = self.current_frame
        self.update_task_list_display()
        self.list_tasks.setCurrentRow(row)
        self.refresh_display()

    def redraw_roi(self):
        row = self.list_tasks.currentRow()
        if row < 0 or self.current_raw_frame is None: return
        if self.is_playing: self.toggle_play()
        roi = cv2.selectROI("Redraw ROI", self.current_raw_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        if roi and roi[2] > 0 and roi[3] > 0:
            self.mosaic_tasks[row]['roi'] = roi
            self.refresh_display()

    def update_strength(self, value):
        row = self.list_tasks.currentRow()
        if row >= 0:
            self.mosaic_tasks[row]['strength'] = value
            self.refresh_display()

    def remove_task(self):
        row = self.list_tasks.currentRow()
        if row >= 0:
            del self.mosaic_tasks[row]
            self.update_task_list_display()
            self.refresh_display()

    # =============== 匯出 ===============
    def export_video(self):
        if not self.mosaic_tasks:
            QMessageBox.warning(self, "Warning", "目前沒有任何馬賽克段落，無須匯出！")
            return
            
        output_path, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "Video Files (*.mp4)")
        if not output_path: return

        self.btn_export.setEnabled(False)
        self.worker = VideoProcessorThread(self.video_path, output_path, self.mosaic_tasks)
        self.worker.progress_signal.connect(lambda c, t: self.lbl_status.setText(f"Exporting: {c}/{t} frames"))
        self.worker.finished_signal.connect(self.export_finished)
        self.worker.start()

    def export_finished(self, path):
        self.lbl_status.setText("Status: Ready")
        self.btn_export.setEnabled(True)
        QMessageBox.information(self, "Success", f"影片已成功輸出！\n{path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = ProVideoEditor()
    editor.show()
    sys.exit(app.exec_())