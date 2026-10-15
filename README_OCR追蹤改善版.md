# OCR 多目標追蹤改善版

此版本保留 `main.py` 的 GUI 介面，主要替換 `video_processor.py` 的初始化搜尋與 OCR 排程。

## 改善內容

1. **快速搜尋階段**：影片尚未出現目標時，將畫面縮小至長邊最多 960 px，並每 3 幀執行一次 OCR，降低長時間空掃描成本。
2. **多目標初始化 burst**：首次命中後，對命中前 2 幀到後 5 幀進行全解析度 OCR，收集每一幀的所有文字框，而不是第一個命中就立刻停止初始化。
3. **初始影格回補**：保留 burst 掃描期間每一幀直接 OCR 的 ROI，讓匯出時可使用這些高畫質偵測框。
4. **追蹤期間 OCR 降頻**：穩定時每 6 幀執行 OCR；追蹤器開始失敗時提高至每 2 幀執行一次。CSRT 仍逐幀更新。
5. **保留所有命中框**：相同文字出現在多個位置時，不會只保留一個框；僅移除高度重疊、近乎相同的框。

## 安裝依賴

建議在專案目前使用的 Python 虛擬環境中執行：

```bash
python -m pip install PyQt5 opencv-contrib-python easyocr supervision numpy
```

若環境已安裝 `opencv-python`，請避免同時安裝互相衝突的 OpenCV wheel；CSRT 需要 `opencv-contrib-python`。

## 使用方式

將 ZIP 解壓到專案目錄，備份後以其中的 `main.py` 和 `video_processor.py` 覆蓋原檔，再以原本方式啟動 GUI。

## 可調整參數

在 `video_processor.py` 的 `OCRTrackingThread` 類別最上方：

- `SEARCH_STEP = 3`：搜尋每隔幾幀抽樣一次。設為 1 最完整但較慢；設為 4 或 5 會更快，但可能略過只出現很短時間的目標。
- `SEARCH_MAX_DIM = 960`：搜尋階段影像長邊上限。提高可改善小字辨識，但速度較慢。
- `INIT_BURST_BEFORE = 2`、`INIT_BURST_AFTER = 5`：首次命中前後的全解析度複查範圍。提高可收集更多目標，但初始化時間增加。
- `OCR_INTERVAL = 6`：穩定追蹤時的 OCR 間隔。
- `ADAPTIVE_OCR_INTERVAL = 2`：追蹤框不穩時的 OCR 間隔。

## 注意事項

- 此版本使用 EasyOCR 的縮圖搜尋與分階段排程，並非換成另一個獨立的小型 OCR 神經網路。
- 每 3 幀抽樣一次可能漏掉只存在 1–2 幀的短暫目標。若必須逐幀零漏檢，需使用更快的文字偵測器/GPU，或對每幀執行偵測。
- 本執行環境缺少 `easyocr` 套件，已完成 Python 語法編譯檢查，但無法在此環境進行實際影片端到端測試。請在安裝上述依賴的專案環境中測試。
