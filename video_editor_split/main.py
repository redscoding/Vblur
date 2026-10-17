
import sys
import os
import time

# ==================================================
# 1. 先載入 OCR / ONNX Runtime
#    必須放在 PyQt5 前面
# ==================================================
import onnxruntime
from rapidocr_onnxruntime import RapidOCR

# ==================================================
# 2. 再載入 OpenCV
# ==================================================
import cv2

# ==================================================
# 3. 設定 PyQt5 的 Qt 外掛路徑
#    設定好環境變數後再匯入 Qt Widgets
# ==================================================
import PyQt5

plugin_path = os.path.join(
    os.path.dirname(PyQt5.__file__),
    "Qt5",
    "plugins"
)

os.environ["QT_PLUGIN_PATH"] = plugin_path
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
    plugin_path,
    "platforms"
)

# ==================================================
# 4. 最後載入自己的影片處理模組與 PyQt5
# ==================================================
from video_processor import OCRTrackingThread, VideoProcessorThread

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QFileDialog, QMessageBox,
    QGroupBox, QSpinBox, QFrame,
    QListWidget, QAbstractItemView, QLineEdit,
    QProgressDialog, QDialog, QProgressBar,
    QSplitter
)

from PyQt5.QtCore import Qt, QTimer, pyqtSignal

from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPen
)



MODERN_THEME = """
QMainWindow, QDialog { background-color: #1E1E1E; }
QWidget { color: #CCCCCC; font-family: "Segoe UI", "Microsoft JhengHei", sans-serif; font-size: 13px; }
QGroupBox { border: 1px solid #3A3A3A; border-radius: 6px; margin-top: 15px; font-weight: bold; color: #E0E0E0; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #4DAAFB; }
QPushButton { background-color: #333333; border: 1px solid #444444; border-radius: 4px; padding: 6px 12px; color: #FFFFFF; font-weight: bold; }
QPushButton:hover { background-color: #444444; border: 1px solid #555555; }
QPushButton#primary { background-color: #0E639C; border: none; }
QPushButton#danger { background-color: #983333; border: none; }
QPushButton:disabled { background-color: #2D2D2D; color: #666666; }
QSlider::groove:horizontal { border: 1px solid #2B2B2B; height: 6px; background: #111111; margin: 2px 0; border-radius: 3px; }
QSlider::handle:horizontal { background: #4DAAFB; border: 1px solid #000000; width: 14px; margin: -5px 0; border-radius: 7px; }
QLabel#TimeLabel { font-family: "Consolas", monospace; color: #4DAAFB; font-size: 14px; }
QLabel#Metric { background: #252526; border: 1px solid #3A3A3A; border-radius: 5px; padding: 8px; }
QSpinBox, QListWidget, QLineEdit { background-color: #252526; border: 1px solid #3A3A3A; color: #FFFFFF; padding: 5px; }
QListWidget::item:selected { background-color: #0E639C; color: white; }
QProgressBar { border: 1px solid #3A3A3A; border-radius: 4px; text-align: center; background: #252526; }
QProgressBar::chunk { background-color: #0E639C; }
"""


class VideoDisplayWidget(QWidget):
    """影片預覽元件，支援在縮放後的預覽畫面上框選原始影格 ROI。"""
    roi_selected = pyqtSignal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.selection_enabled = False
        self.drag_start = None
        self.drag_end = None
        self.selected_roi = None  # 原始影片影格座標 (x, y, w, h)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)

    def set_image(self, pixmap):
        self.pixmap = pixmap
        self.update()

    def set_selection_mode(self, enabled=True):
        self.selection_enabled = bool(enabled)
        self.drag_start = None
        self.drag_end = None
        if enabled:
            self.selected_roi = None
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def clear_selection(self):
        self.selection_enabled = False
        self.drag_start = None
        self.drag_end = None
        self.selected_roi = None
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def _display_geometry(self):
        """回傳預覽影像在 widget 中的實際顯示矩形 (left, top, width, height)。"""
        if self.pixmap is None or self.pixmap.width() <= 0 or self.pixmap.height() <= 0:
            return 0, 0, 0, 0
        scale = min(self.width() / self.pixmap.width(), self.height() / self.pixmap.height())
        width = max(1, int(round(self.pixmap.width() * scale)))
        height = max(1, int(round(self.pixmap.height() * scale)))
        return (self.width() - width) // 2, (self.height() - height) // 2, width, height

    def _clamp_to_image(self, point):
        left, top, width, height = self._display_geometry()
        if width <= 0 or height <= 0:
            return None
        x = max(left, min(int(point[0]), left + width - 1))
        y = max(top, min(int(point[1]), top + height - 1))
        return x, y

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))
        if self.pixmap is None:
            return

        left, top, width, height = self._display_geometry()
        scaled = self.pixmap.scaled(width, height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(left, top, scaled)

        # 已確認的 ROI 以綠框顯示；座標保存在原始影格座標，因此縮放視窗也不會偏移。
        if self.selected_roi is not None:
            x, y, rw, rh = self.selected_roi
            sx, sy = width / self.pixmap.width(), height / self.pixmap.height()
            painter.setPen(QPen(QColor("#00E676"), 2))
            painter.drawRect(int(left + x * sx), int(top + y * sy),
                             max(1, int(rw * sx)), max(1, int(rh * sy)))

        if self.selection_enabled and self.drag_start is not None and self.drag_end is not None:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_end
            painter.setPen(QPen(QColor("#FFD54F"), 2, Qt.DashLine))
            painter.drawRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

    def mousePressEvent(self, event):
        if not self.selection_enabled or self.pixmap is None or event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)
        point = self._clamp_to_image((event.x(), event.y()))
        left, top, width, height = self._display_geometry()
        if point is None or not (left <= event.x() < left + width and top <= event.y() < top + height):
            return
        self.drag_start = point
        self.drag_end = point
        self.selected_roi = None
        self.update()

    def mouseMoveEvent(self, event):
        if self.selection_enabled and self.drag_start is not None:
            self.drag_end = self._clamp_to_image((event.x(), event.y()))
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if not self.selection_enabled or self.drag_start is None or event.button() != Qt.LeftButton:
            return super().mouseReleaseEvent(event)
        self.drag_end = self._clamp_to_image((event.x(), event.y()))
        left, top, width, height = self._display_geometry()
        if self.drag_end is None or width <= 0 or height <= 0:
            self.clear_selection()
            return
        x1, y1 = self.drag_start
        x2, y2 = self.drag_end
        display_x, display_y = min(x1, x2), min(y1, y2)
        display_w, display_h = abs(x2 - x1), abs(y2 - y1)
        if display_w < 4 or display_h < 4:
            self.clear_selection()
            return
        sx, sy = self.pixmap.width() / width, self.pixmap.height() / height
        x = int(round((display_x - left) * sx))
        y = int(round((display_y - top) * sy))
        rw = int(round(display_w * sx))
        rh = int(round(display_h * sy))
        x = max(0, min(x, self.pixmap.width() - 1))
        y = max(0, min(y, self.pixmap.height() - 1))
        rw = max(1, min(rw, self.pixmap.width() - x))
        rh = max(1, min(rh, self.pixmap.height() - y))
        self.selected_roi = (x, y, rw, rh)
        self.selection_enabled = False
        self.drag_start = None
        self.drag_end = None
        self.setCursor(Qt.ArrowCursor)
        self.update()
        self.roi_selected.emit(self.selected_roi)


def frame_to_pixmap(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(image)


class OCRMonitorDialog(QDialog):
    """顯示 OCR 執行緒實際送來的畫面、狀態、耗時與事件。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI OCR 即時監控")
        self.resize(1050, 720)
        self.setStyleSheet(MODERN_THEME)
        self.last_clock = None
        self.stage_started_clock = None
        self.stage_text = "等待 AI 回報..."

        root = QVBoxLayout(self)
        metrics = QHBoxLayout()
        self.lbl_stage = QLabel("階段：等待啟動")
        self.lbl_stage.setObjectName("Metric")
        self.lbl_frame = QLabel("影格：--")
        self.lbl_frame.setObjectName("Metric")
        self.lbl_lost = QLabel("遺失次數：0 / 30")
        self.lbl_lost.setObjectName("Metric")
        self.lbl_elapsed = QLabel("目前階段耗時：0.0 秒")
        self.lbl_elapsed.setObjectName("Metric")
        for label in (self.lbl_stage, self.lbl_frame, self.lbl_lost, self.lbl_elapsed):
            label.setWordWrap(True)
            metrics.addWidget(label, 1)
        root.addLayout(metrics)

        splitter = QSplitter(Qt.Horizontal)
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        self.image_display = VideoDisplayWidget()
        self.lbl_legend = QLabel("綠色：OCR 確認｜青色：逐幀追蹤且可遮罩｜黃色：追蹤遺失（不遮罩舊位置）｜紅色：位置跳躍被拒絕")
        self.lbl_legend.setWordWrap(True)
        self.lbl_legend.setStyleSheet("padding: 6px; background: #202938; color: #e5e7eb; border-radius: 5px;")
        image_layout.addWidget(self.image_display, 1)
        image_layout.addWidget(self.lbl_legend)
        splitter.addWidget(image_panel)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.lbl_target = QLabel("目標文字：--")
        self.lbl_confidence = QLabel("信心分數：--")
        self.lbl_ocr_elapsed = QLabel("最近一次 OCR 耗時：--")
        self.lbl_search_roi = QLabel("搜尋區域：--")
        self.lbl_detail = QLabel("詳細狀態：等待資料")
        for label in (self.lbl_target, self.lbl_confidence, self.lbl_ocr_elapsed, self.lbl_search_roi, self.lbl_detail):
            label.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 30)
        self.progress.setValue(0)
        self.progress.setFormat("遺失次數 %v / %m")
        self.events = QListWidget()
        right_layout.addWidget(self.lbl_target)
        right_layout.addWidget(self.lbl_confidence)
        right_layout.addWidget(self.lbl_ocr_elapsed)
        right_layout.addWidget(self.lbl_search_roi)
        right_layout.addWidget(self.progress)
        right_layout.addWidget(QLabel("事件紀錄（最新事件在最下方）"))
        right_layout.addWidget(self.events, 1)
        right_layout.addWidget(self.lbl_detail)
        splitter.addWidget(right)
        splitter.setSizes([650, 350])
        root.addWidget(splitter, 1)

        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_elapsed_clock)
        self.clock_timer.start(200)

    def update_frame(self, frame, frame_no, mode):
        if frame is not None:
            self.image_display.set_image(frame_to_pixmap(frame))
        self.lbl_frame.setText(f"影格：{frame_no}")
        self.lbl_detail.setText(f"畫面模式：{mode}")

    def update_data(self, data):
        stage = str(data.get("stage", "unknown"))
        message = str(data.get("message", ""))
        frame_no = data.get("frame")
        clock = data.get("clock", time.monotonic())
        self.last_clock = clock
        self.stage_text = message or stage
        if stage in ("ocr_running", "verify_running", "loading", "tracking_lost", "ready", "locked", "recovered"):
            self.stage_started_clock = clock
        elif stage in ("ocr_done", "verify_done", "verified", "verify_miss", "finished", "error", "not_found", "cancelled", "stopped"):
            self.stage_started_clock = None

        self.lbl_stage.setText(f"階段：{self.stage_text}")
        if frame_no is not None:
            self.lbl_frame.setText(f"影格：{frame_no}")
        lost = int(data.get("lost_count", 0) or 0)
        patience = int(data.get("patience", 30) or 30)
        self.progress.setRange(0, max(1, patience))
        self.progress.setValue(min(lost, patience))
        self.lbl_lost.setText(f"遺失次數：{lost} / {patience}")
        if "target_text" in data:
            self.lbl_target.setText(f"目標文字：{data['target_text']}")
        if "confidence" in data:
            self.lbl_confidence.setText(f"信心分數：{float(data['confidence']):.3f}")
        if "elapsed_seconds" in data:
            self.lbl_ocr_elapsed.setText(f"最近一次 OCR 耗時：{float(data['elapsed_seconds']):.2f} 秒")
        if "search_roi" in data:
            self.lbl_search_roi.setText(f"搜尋區域 (x, y, w, h)：{data['search_roi']}")
        if "bbox" in data:
            self.lbl_search_roi.setText(f"追蹤框 (x, y, w, h)：{data['bbox']}")
        if "result_count" in data:
            self.lbl_detail.setText(f"本次 OCR 偵測到 {data['result_count']} 個文字區塊")
        elif "tracked_frames" in data:
            self.lbl_detail.setText(f"追蹤影格數：{data['tracked_frames']}；結束原因：{message}")
        elif stage == "error":
            self.lbl_detail.setText(message)

        # 對使用者最有幫助的事件才寫入紀錄，避免每一幀都刷滿清單。
        if stage not in ("tracking",):
            self.events.addItem(f"{time.strftime('%H:%M:%S')} | F{frame_no if frame_no is not None else '--'} | {stage} | {message}")
            self.events.scrollToBottom()
            while self.events.count() > 500:
                self.events.takeItem(0)

        for result in data.get("ocr_results", [])[:10]:
            text = result.get("text", "")
            confidence = result.get("confidence", 0)
            self.events.addItem(f"    OCR：{text} (信心 {float(confidence):.3f})")
        self.events.scrollToBottom()

    def update_elapsed_clock(self):
        if self.stage_started_clock is not None:
            elapsed = max(0.0, time.monotonic() - self.stage_started_clock)
            self.lbl_elapsed.setText(f"目前階段耗時：{elapsed:.1f} 秒")
        else:
            self.lbl_elapsed.setText("目前階段耗時：--")


class ProVideoEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Pro Video Editor (Hybrid: Static + OCR Tracking)")
        self.setGeometry(100, 100, 1200, 750)
        self.setStyleSheet(MODERN_THEME)
        self.video_path, self.cap, self.is_playing = None, None, False
        self.total_frames, self.current_frame = 0, 0
        self.current_raw_frame = None
        self.selected_tracking_roi = None
        self.selected_tracking_frame = None
        self.mosaic_tasks = []
        self.ocr_thread = None
        self.monitor_dialog = None
        self.progress_dialog = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(15, 15, 15, 15)
        left_panel = QFrame()
        left_panel.setFixedWidth(340)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        group_file = QGroupBox("1. Source Video")
        file_layout = QVBoxLayout(group_file)
        self.btn_import = QPushButton("Import Video (匯入影片)")
        self.btn_import.setObjectName("primary")
        self.btn_import.clicked.connect(self.import_video)
        file_layout.addWidget(self.btn_import)

        group_mosaic = QGroupBox("2. Mosaic Segments (混合追蹤段落)")
        mosaic_layout = QVBoxLayout(group_mosaic)
        self.btn_select_tracking = QPushButton("🎯 框選 AI 追蹤目標")
        self.btn_select_tracking.setEnabled(False)
        self.btn_select_tracking.clicked.connect(self.begin_tracking_selection)
        mosaic_layout.addWidget(self.btn_select_tracking)
        self.lbl_tracking_selection = QLabel("AI 追蹤目標：尚未框選（請先框選；不執行全畫面 OCR）")
        self.lbl_tracking_selection.setWordWrap(True)
        mosaic_layout.addWidget(self.lbl_tracking_selection)

        ocr_row = QHBoxLayout()
        self.txt_ocr = QLineEdit()
        self.txt_ocr.setPlaceholderText("輸入目標文字 (如：車牌)")
        self.btn_ocr = QPushButton("🔤 掃描並追蹤")
        self.btn_ocr.setObjectName("primary")
        self.btn_ocr.setEnabled(False)
        self.btn_ocr.clicked.connect(self.start_ocr_tracking)
        ocr_row.addWidget(self.txt_ocr, 2)
        ocr_row.addWidget(self.btn_ocr, 1)

        task_row = QHBoxLayout()
        self.btn_add_task = QPushButton("➕ 手動固定框選")
        self.btn_add_task.setEnabled(False)
        self.btn_add_task.clicked.connect(self.add_new_task)
        self.btn_rm_task = QPushButton("🗑 刪除段落")
        self.btn_rm_task.setEnabled(False)
        self.btn_rm_task.clicked.connect(self.remove_task)
        task_row.addWidget(self.btn_add_task)
        task_row.addWidget(self.btn_rm_task)
        self.list_tasks = QListWidget()
        self.list_tasks.setFixedHeight(120)
        self.list_tasks.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_tasks.currentRowChanged.connect(self.on_task_selected)

        self.edit_panel = QWidget()
        edit_layout = QVBoxLayout(self.edit_panel)
        edit_layout.setContentsMargins(0, 5, 0, 0)
        se_row = QHBoxLayout()
        self.btn_set_start = QPushButton("設定起點")
        self.btn_set_start.clicked.connect(self.set_task_start)
        self.lbl_start = QLabel("Start: --")
        self.btn_set_end = QPushButton("設定終點")
        self.btn_set_end.clicked.connect(self.set_task_end)
        self.lbl_end = QLabel("End: --")
        for widget in (self.btn_set_start, self.lbl_start, self.btn_set_end, self.lbl_end):
            se_row.addWidget(widget)
        roi_row = QHBoxLayout()
        self.btn_redraw = QPushButton("重新畫框 (僅限手動)")
        self.btn_redraw.clicked.connect(self.redraw_roi)
        self.spin_strength = QSpinBox()
        self.spin_strength.setRange(5, 50)
        self.spin_strength.setPrefix("模糊強度: ")
        self.spin_strength.valueChanged.connect(self.update_strength)
        roi_row.addWidget(self.btn_redraw)
        roi_row.addWidget(self.spin_strength)
        edit_layout.addLayout(se_row)
        edit_layout.addLayout(roi_row)
        self.edit_panel.setEnabled(False)
        mosaic_layout.addLayout(ocr_row)
        mosaic_layout.addLayout(task_row)
        mosaic_layout.addWidget(self.list_tasks)
        mosaic_layout.addWidget(self.edit_panel)

        group_export = QGroupBox("3. Export")
        export_layout = QVBoxLayout(group_export)
        self.btn_export = QPushButton("Export Video (匯出影片)")
        self.btn_export.setObjectName("danger")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_video)
        self.lbl_status = QLabel("Status: Ready")
        export_layout.addWidget(self.btn_export)
        export_layout.addWidget(self.lbl_status)

        left_layout.addWidget(group_file)
        left_layout.addWidget(group_mosaic)
        left_layout.addWidget(group_export)
        left_layout.addStretch()

        right_panel = QVBoxLayout()
        self.video_display = VideoDisplayWidget()
        self.video_display.roi_selected.connect(self.on_tracking_roi_selected)
        control = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setFixedWidth(80)
        self.btn_play.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self.set_position)
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setObjectName("TimeLabel")
        self.lbl_time.setFixedWidth(110)
        self.lbl_time.setAlignment(Qt.AlignCenter)
        control.addWidget(self.btn_play)
        control.addWidget(self.slider)
        control.addWidget(self.lbl_time)
        right_panel.addWidget(self.video_display, 1)
        right_panel.addLayout(control)
        main_layout.addWidget(left_panel)
        main_layout.addLayout(right_panel, 1)

    def import_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if not file_name:
            return
        if self.cap:
            self.cap.release()
        self.video_path = file_name
        self.clear_tracking_selection()
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "讀取失敗", "無法開啟這個影片檔案。")
            self.cap = None
            return
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.slider.setRange(0, max(0, self.total_frames - 1))
        self.btn_add_task.setEnabled(True)
        self.btn_select_tracking.setEnabled(True)
        self.btn_ocr.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.update_time_label()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.read_and_display_frame()

    def toggle_play(self):
        if not self.cap:
            return
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
                new_frame_no = max(0, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                if self.selected_tracking_frame is not None and new_frame_no != self.selected_tracking_frame:
                    self.clear_tracking_selection()
                self.current_frame = new_frame_no
                self.current_raw_frame = frame.copy()
                self.slider.blockSignals(True)
                self.slider.setValue(self.current_frame)
                self.slider.blockSignals(False)
                self.display_frame(frame)
                self.update_time_label()
            else:
                self.timer.stop()
                self.is_playing = False
                self.btn_play.setText("▶ Play")

    def set_position(self, position):
        if self.cap:
            if self.selected_tracking_frame is not None and int(position) != self.selected_tracking_frame:
                self.clear_tracking_selection()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            self.current_frame = position
            self.read_and_display_frame()
            self.update_time_label()

    def read_and_display_frame(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if ret:
                self.current_raw_frame = frame.copy()
                self.display_frame(frame)

    def apply_mosaics(self, frame):
        frame_copy = frame.copy()
        h_max, w_max = frame_copy.shape[:2]
        for m in self.mosaic_tasks:
            if m["start"] <= self.current_frame <= m["end"]:
                value = m.get("tracking_data", {}).get(self.current_frame) if m.get("type") == "tracking" else m.get("roi")
                if value is None:
                    rois = []
                elif isinstance(value, (tuple, list)) and len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
                    rois = [value]  # 相容舊版單一 ROI
                elif isinstance(value, (tuple, list)):
                    # 新格式為每影格 list[ROI]；只接受合法四元素 ROI，避免格式異常。
                    rois = [r for r in value if isinstance(r, (tuple, list)) and len(r) == 4]
                else:
                    rois = []
                for roi in rois:
                    if not roi or len(roi) != 4:
                        continue
                    x, y, w, h = map(int, roi)
                    x, y = max(0, x), max(0, y)
                    w, h = min(w_max - x, w), min(h_max - y, h)
                    if w > 0 and h > 0:
                        roi_frame = frame_copy[y:y + h, x:x + w]
                        strength = max(1, int(m.get("strength", 15)))
                        sw, sh = max(1, w // strength), max(1, h // strength)
                        mosaic = cv2.resize(cv2.resize(roi_frame, (sw, sh), interpolation=cv2.INTER_LINEAR),
                                            (w, h), interpolation=cv2.INTER_NEAREST)
                        frame_copy[y:y + h, x:x + w] = mosaic
        return frame_copy

    def display_frame(self, frame):
        self.video_display.set_image(frame_to_pixmap(self.apply_mosaics(frame)))

    def refresh_display(self):
        if self.current_raw_frame is not None:
            self.display_frame(self.current_raw_frame)

    def update_time_label(self):
        if not self.cap:
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        current_seconds = int(self.current_frame / fps)
        total_seconds = int(self.total_frames / fps)
        self.lbl_time.setText(f"{current_seconds // 60:02d}:{current_seconds % 60:02d} / {total_seconds // 60:02d}:{total_seconds % 60:02d}")

    def begin_tracking_selection(self):
        """暫停影片並在右側預覽畫面啟用滑鼠拖曳框選。"""
        if self.current_raw_frame is None or not self.video_path:
            QMessageBox.information(self, "尚未載入影片", "請先匯入影片，再選擇要追蹤的影格。")
            return
        if self.is_playing:
            self.toggle_play()
        self.selected_tracking_roi = None
        self.selected_tracking_frame = None
        self.video_display.set_selection_mode(True)
        self.lbl_tracking_selection.setText("請在右側影片預覽畫面按住滑鼠左鍵拖曳，框住目標後放開。")
        self.lbl_status.setText(f"Status: 請在影格 F{self.current_frame} 框選 AI 追蹤目標")

    def on_tracking_roi_selected(self, roi):
        self.selected_tracking_roi = tuple(int(v) for v in roi)
        self.selected_tracking_frame = int(self.current_frame)
        self.lbl_tracking_selection.setText(
            f"AI 追蹤目標：F{self.selected_tracking_frame}，ROI={self.selected_tracking_roi}（原始影格座標）"
        )
        self.lbl_status.setText("Status: 已框選目標；輸入目標文字後按「掃描並追蹤」開始。")

    def clear_tracking_selection(self):
        self.selected_tracking_roi = None
        self.selected_tracking_frame = None
        if hasattr(self, "video_display"):
            self.video_display.clear_selection()
        if hasattr(self, "lbl_tracking_selection"):
            self.lbl_tracking_selection.setText("AI 追蹤目標：尚未框選（請先框選；不執行全畫面 OCR）")

    def start_ocr_tracking(self):
        target = self.txt_ocr.text().strip()
        if not self.video_path:
            QMessageBox.warning(self, "尚未匯入影片", "請先匯入影片。")
            return
        if self.ocr_thread and self.ocr_thread.isRunning():
            QMessageBox.information(self, "正在執行", "目前已有一個 OCR 追蹤工作在執行。")
            return
        if self.is_playing:
            self.toggle_play()

        initial_roi = None
        if self.selected_tracking_roi is not None:
            if self.selected_tracking_frame != self.current_frame:
                self.clear_tracking_selection()
                QMessageBox.warning(self, "框選影格已變更", "你已切換到其他影格，請重新框選目標後再開始追蹤。")
                return
            initial_roi = self.selected_tracking_roi
        if initial_roi is None:
            QMessageBox.warning(self, "尚未框選目標", "此效能版禁止全畫面 OCR。請先按框選追蹤目標，在預覽畫面拖曳框住單一目標，再開始追蹤。")
            return
        if not target:
            answer = QMessageBox.question(
                self, "未輸入文字", "未輸入目標文字時，系統只使用 CSRT／光流／Kalman 追蹤，不會執行 OCR 驗證。要繼續嗎？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return

        self.monitor_dialog = OCRMonitorDialog(self)
        self.monitor_dialog.show()
        self.progress_dialog = QProgressDialog("正在啟動 AI 引擎...", "取消追蹤", self.current_frame,
                                                max(self.current_frame + 1, self.total_frames), self)
        self.progress_dialog.setWindowTitle("OCR Tracking")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(False)
        self.ocr_thread = OCRTrackingThread(
            self.video_path, self.current_frame, self.total_frames, target,
            initial_roi=initial_roi
        )
        self.progress_dialog.canceled.connect(self.ocr_thread.requestInterruption)
        self.ocr_thread.progress_signal.connect(self.update_ocr_progress)
        self.ocr_thread.monitor_frame_signal.connect(self.monitor_dialog.update_frame)
        self.ocr_thread.monitor_data_signal.connect(self.monitor_dialog.update_data)
        self.ocr_thread.finished_signal.connect(self.on_ocr_finished)
        self.btn_ocr.setEnabled(False)
        self.ocr_thread.start()
        self.progress_dialog.show()

    def update_ocr_progress(self, val, msg):
        if self.progress_dialog:
            self.progress_dialog.setValue(val)
            self.progress_dialog.setLabelText(msg)
        self.lbl_status.setText(msg)

    def on_ocr_finished(self, tracking_data, last_frame, success):
        if self.progress_dialog:
            self.progress_dialog.close()
        self.btn_ocr.setEnabled(bool(self.video_path))
        if not success or not tracking_data:
            QMessageBox.warning(self, "追蹤未完成", "沒有取得可用的追蹤資料。請查看「AI OCR 即時監控」中的結束原因與錯誤紀錄。")
            self.lbl_status.setText("Status: OCR tracking failed")
            return
        task = {
            "type": "tracking",
            "start": min(tracking_data.keys()),
            "end": last_frame,
            "tracking_data": tracking_data,
            "strength": 15,
        }
        self.mosaic_tasks.append(task)
        self.update_task_list_display()
        self.list_tasks.setCurrentRow(len(self.mosaic_tasks) - 1)
        self.refresh_display()
        max_targets = max((len(v) if isinstance(v, (list, tuple)) and not (len(v) == 4 and all(isinstance(x, (int, float)) for x in v)) else (1 if v else 0) for v in tracking_data.values()), default=0)
        self.lbl_status.setText(f"Status: 多目標追蹤完成，最多同時 {max_targets} 個框；最後影格 F{last_frame}")
        if self.monitor_dialog:
            self.monitor_dialog.raise_()
            self.monitor_dialog.activateWindow()

    def add_new_task(self):
        if not self.cap or self.current_raw_frame is None:
            return
        if self.is_playing:
            self.toggle_play()
        QMessageBox.information(self, "操作提示", "請框選要固定打碼的區域，畫完按 ENTER。")
        roi = cv2.selectROI("Select ROI", self.current_raw_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        if roi and roi[2] > 0 and roi[3] > 0:
            self.mosaic_tasks.append({"type": "static", "start": self.current_frame,
                                     "end": max(0, self.total_frames - 1), "roi": tuple(map(int, roi)), "strength": 15})
            self.update_task_list_display()
            self.list_tasks.setCurrentRow(len(self.mosaic_tasks) - 1)
            self.refresh_display()

    def update_task_list_display(self):
        self.list_tasks.clear()
        for i, task in enumerate(self.mosaic_tasks):
            label = "AI追蹤" if task.get("type") == "tracking" else "固定"
            self.list_tasks.addItem(f"[{label}] 段落 {i + 1}: F{task['start']} ➔ F{task['end']}")

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
        self.btn_redraw.setEnabled(task.get("type") != "tracking")
        self.spin_strength.blockSignals(True)
        self.spin_strength.setValue(task["strength"])
        self.spin_strength.blockSignals(False)

    def set_task_start(self):
        row = self.list_tasks.currentRow()
        if row < 0:
            return
        task = self.mosaic_tasks[row]
        task["start"] = min(self.current_frame, task["end"])
        self.update_task_list_display()
        self.list_tasks.setCurrentRow(row)
        self.refresh_display()

    def set_task_end(self):
        row = self.list_tasks.currentRow()
        if row < 0:
            return
        task = self.mosaic_tasks[row]
        task["end"] = max(self.current_frame, task["start"])
        self.update_task_list_display()
        self.list_tasks.setCurrentRow(row)
        self.refresh_display()

    def redraw_roi(self):
        row = self.list_tasks.currentRow()
        if row < 0 or self.current_raw_frame is None:
            return
        if self.is_playing:
            self.toggle_play()
        roi = cv2.selectROI("Redraw ROI", self.current_raw_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        if roi and roi[2] > 0 and roi[3] > 0:
            self.mosaic_tasks[row]["roi"] = tuple(map(int, roi))
            self.refresh_display()

    def update_strength(self, value):
        row = self.list_tasks.currentRow()
        if row >= 0:
            self.mosaic_tasks[row]["strength"] = value
            self.refresh_display()

    def remove_task(self):
        row = self.list_tasks.currentRow()
        if row >= 0:
            del self.mosaic_tasks[row]
            self.update_task_list_display()
            self.refresh_display()

    def export_video(self):
        if not self.mosaic_tasks:
            QMessageBox.warning(self, "Warning", "目前沒有任何馬賽克段落，無須匯出！")
            return
        output, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "Video Files (*.mp4)")
        if not output:
            return
        if not output.lower().endswith(".mp4"):
            output += ".mp4"
        self.btn_export.setEnabled(False)
        self.worker = VideoProcessorThread(self.video_path, output, list(self.mosaic_tasks))
        self.worker.progress_signal.connect(lambda current, total: self.lbl_status.setText(f"Exporting: {current}/{total} frames"))
        self.worker.finished_signal.connect(self.export_finished)
        self.worker.start()

    def export_finished(self, path):
        self.btn_export.setEnabled(True)
        if not path:
            self.lbl_status.setText("Status: Export failed")
            QMessageBox.warning(self, "匯出失敗", "無法建立輸出影片。請確認影片編碼器與輸出路徑。")
            return
        self.lbl_status.setText("Status: Ready")
        QMessageBox.information(self, "Success", f"影片已成功輸出！\n{path}")

    def closeEvent(self, event):
        if self.ocr_thread and self.ocr_thread.isRunning():
            self.ocr_thread.requestInterruption()
            self.ocr_thread.wait(1500)
        if self.cap:
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = ProVideoEditor()
    editor.show()
    sys.exit(app.exec_())
