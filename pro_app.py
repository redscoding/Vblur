import sys
import cv2
import os

# ====== 解決 Windows 虛擬環境 Qt 外掛遺失問題 ======
import PyQt5
plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
os.environ["QT_PLUGIN_PATH"] = plugin_path
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(plugin_path, "platforms")

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QSlider,
    QFileDialog, QMessageBox, QGroupBox, QSpinBox,
    QFrame, QListWidget, QAbstractItemView
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
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
"""

# ================= 自訂防膨脹影片渲染引擎 =================
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

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret: break

            for m in self.mosaic_tasks:
                if m['start'] <= current_frame <= m['end']:
                    x, y, w, h = m['roi']
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

# ================= 主程式介面 =================
class ProVideoEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Pro Video Editor (Segment Management)")
        self.setGeometry(100, 100, 1150, 700)
        self.setStyleSheet(MODERN_THEME)

        self.video_path, self.cap, self.is_playing = None, None, False
        self.total_frames, self.current_frame = 0, 0
        self.current_raw_frame = None  
        self.mosaic_tasks = [] # 儲存所有的段落資料

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # ====== 左側面板 ======
        left_panel = QFrame()
        left_panel.setFixedWidth(320)
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

        # 2. 段落管理區 (核心重構)
        group_mosaic = QGroupBox("2. Mosaic Segments (馬賽克段落管理)")
        vbox_mosaic = QVBoxLayout()

        # 新增與刪除按鈕
        hbox_add_rm = QHBoxLayout()
        self.btn_add_task = QPushButton("➕ 新增段落")
        self.btn_add_task.setObjectName("primary")
        self.btn_add_task.setEnabled(False) # 防呆：沒影片不能點
        self.btn_add_task.clicked.connect(self.add_new_task)
        
        self.btn_rm_task = QPushButton("🗑 刪除段落")
        self.btn_rm_task.setEnabled(False)
        self.btn_rm_task.clicked.connect(self.remove_task)
        
        hbox_add_rm.addWidget(self.btn_add_task)
        hbox_add_rm.addWidget(self.btn_rm_task)

        # 段落清單
        self.list_tasks = QListWidget()
        self.list_tasks.setFixedHeight(100)
        self.list_tasks.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_tasks.currentRowChanged.connect(self.on_task_selected)

        # 單一段落編輯面板 (選中清單才會啟用)
        self.edit_panel = QWidget()
        vbox_edit = QVBoxLayout(self.edit_panel)
        vbox_edit.setContentsMargins(0, 10, 0, 0)
        
        lbl_edit_title = QLabel("📝 編輯所選段落:")
        lbl_edit_title.setStyleSheet("color: #73C2FB;")
        
        # 設定起訖點按鈕
        hbox_se = QHBoxLayout()
        self.btn_set_start = QPushButton("設定為起點")
        self.btn_set_start.clicked.connect(self.set_task_start)
        self.lbl_start = QLabel("Start: --")
        
        self.btn_set_end = QPushButton("設定為終點")
        self.btn_set_end.clicked.connect(self.set_task_end)
        self.lbl_end = QLabel("End: --")
        
        hbox_se.addWidget(self.btn_set_start)
        hbox_se.addWidget(self.lbl_start)
        hbox_se.addWidget(self.btn_set_end)
        hbox_se.addWidget(self.lbl_end)

        # 重新框選與強度
        hbox_roi = QHBoxLayout()
        self.btn_redraw = QPushButton("重新畫框 (ROI)")
        self.btn_redraw.clicked.connect(self.redraw_roi)
        
        self.spin_strength = QSpinBox()
        self.spin_strength.setRange(5, 50)
        self.spin_strength.setPrefix("模糊強度: ")
        self.spin_strength.valueChanged.connect(self.update_strength)

        hbox_roi.addWidget(self.btn_redraw)
        hbox_roi.addWidget(self.spin_strength)

        vbox_edit.addWidget(lbl_edit_title)
        vbox_edit.addLayout(hbox_se)
        vbox_edit.addLayout(hbox_roi)
        self.edit_panel.setEnabled(False) # 預設停用

        vbox_mosaic.addLayout(hbox_add_rm)
        vbox_mosaic.addWidget(self.list_tasks)
        vbox_mosaic.addWidget(self.edit_panel)
        group_mosaic.setLayout(vbox_mosaic)

        # 3. 匯出區
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

        # ====== 右側預覽區 ======
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

    # =============== 影片播放控制 ===============
    def import_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.video_path = file_name
            self.cap = cv2.VideoCapture(self.video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setRange(0, self.total_frames)
            
            # 解除按鈕封印 (防呆機制解除)
            self.btn_add_task.setEnabled(True)
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

    def apply_mosaics(self, frame):
        frame_copy = frame.copy()
        for m in self.mosaic_tasks:
            if m['start'] <= self.current_frame <= m['end']:
                x, y, w, h = m['roi']
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

    # =============== 馬賽克段落管理 (核心防呆與編輯邏輯) ===============
    def add_new_task(self):
        if not self.cap: return
        if self.is_playing: self.toggle_play()
        
        QMessageBox.information(self, "操作提示", "請在彈出的視窗中框選區域\n畫完後按【ENTER】確認，或按【C】取消。")
        
        if self.current_raw_frame is not None:
            roi = cv2.selectROI("Select ROI", self.current_raw_frame, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()
            
            # 防呆：確認使用者真的有畫框 (寬度與高度 > 0)
            if roi and roi[2] > 0 and roi[3] > 0:
                task = {
                    'start': self.current_frame,
                    'end': self.total_frames,
                    'roi': roi,
                    'strength': 15
                }
                self.mosaic_tasks.append(task)
                self.update_task_list_display()
                self.list_tasks.setCurrentRow(len(self.mosaic_tasks) - 1)
                self.refresh_display()
            else:
                QMessageBox.warning(self, "取消新增", "您沒有框選任何區域，段落未新增。")

    def update_task_list_display(self):
        self.list_tasks.clear()
        for i, m in enumerate(self.mosaic_tasks):
            self.list_tasks.addItem(f"段落 {i+1}: Frame {m['start']} ➔ {m['end']}")

    def on_task_selected(self, row):
        if row < 0 or row >= len(self.mosaic_tasks):
            self.edit_panel.setEnabled(False)
            self.btn_rm_task.setEnabled(False)
            return
            
        # 啟用編輯面板
        self.edit_panel.setEnabled(True)
        self.btn_rm_task.setEnabled(True)
        task = self.mosaic_tasks[row]
        
        self.lbl_start.setText(f"Start: {task['start']}")
        self.lbl_end.setText(f"End: {task['end']}")
        
        # 暫時阻斷 Signal，避免在讀取資料時觸發數值變更的 Event
        self.spin_strength.blockSignals(True)
        self.spin_strength.setValue(task['strength'])
        self.spin_strength.blockSignals(False)

    def set_task_start(self):
        row = self.list_tasks.currentRow()
        if row < 0: return
        task = self.mosaic_tasks[row]
        
        # 防呆：起點不能大於終點
        if self.current_frame > task['end']:
            QMessageBox.warning(self, "時間錯誤", "起點不能晚於終點！\n系統已自動將「終點」對齊。")
            task['end'] = self.current_frame
            
        task['start'] = self.current_frame
        self.update_task_list_display()
        self.list_tasks.setCurrentRow(row) # 保持選取狀態
        self.refresh_display()

    def set_task_end(self):
        row = self.list_tasks.currentRow()
        if row < 0: return
        task = self.mosaic_tasks[row]
        
        # 防呆：終點不能小於起點
        if self.current_frame < task['start']:
            QMessageBox.warning(self, "時間錯誤", "終點不能早於起點！\n系統已自動將「起點」對齊。")
            task['start'] = self.current_frame
            
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
