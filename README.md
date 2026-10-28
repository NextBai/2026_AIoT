# 物聯網聲紋辨識與邊緣控制系統 (AIoT Voice Identification & Edge Control System)

[![GitHub Pages](https://img.shields.io/badge/Demo-Interactive%20GitHub%20Page-00f2fe?style=for-the-badge&logo=github)](https://nextbai.github.io/2026_AIoT/)
[![SpeechBrain](https://img.shields.io/badge/SpeechBrain-ECAPA--TDNN%20192D-7928ca?style=for-the-badge)](https://speechbrain.github.io/)
[![IoTtalk Integrated](https://img.shields.io/badge/IoTtalk-Edge%20Trigger%20(!0Voice)-ff0080?style=for-the-badge)](test.py)
[![Test Accuracy](https://img.shields.io/badge/Blind%20Test-100%25%20Accuracy-10b981?style=for-the-badge)](reports/03_消融實驗_去噪前後對ECAPA-TDNN聲紋辨識成效報告.md)
[![License](https://img.shields.io/badge/License-MIT-38bdf8?style=for-the-badge)](LICENSE)

本專案是一套專為**邊緣物聯網（AIoT）**環境設計的高精度語者聲紋辨識與智慧控制系統。核心採用 **SpeechBrain ECAPA-TDNN** 預訓練骨幹網絡，結合「高保真抗混疊降取樣、RMS 能量正規化、雙門檻 VAD 黃金片段截取、六維資料增強與譜減去噪消融」的端到端音訊工程管線。

系統提供**雙軌部署方案**：
1. **雲端/前端互動展示**：免安裝即可在 [GitHub Pages 互動網頁](https://nextbai.github.io/2026_AIoT/) 體驗即時 Web Audio 麥克風示波器、四階段音訊播放器與論文級消融圖譜。
2. **邊緣物聯網平台介接**：透過 Python 回呼（Callback）機制與多執行緒架構，無縫對接 **IoTtalk 物聯網平台**，將聲紋識別結果轉換為智慧開關控制訊號（`500` / `0`）。

---

## 🌐 線上互動體驗（GitHub Pages）

👉 **[https://nextbai.github.io/2026_AIoT/](https://nextbai.github.io/2026_AIoT/)**

- 🎙️ **瀏覽器即時收音室**：透過 Web Audio API 截取 3.5 秒語音，實時繪製時域示波圖與 FFT 頻譜，動態提取 RMS、預估基頻 $F_0$ 與訊噪比 (SNR)，並即時模擬聲紋預測。
- 🎵 **四階段音訊比對播放器**：線上直接試聽 4 位講者（吳、甯、白、蕭）從「原始錄音」、「16kHz 正規化」、「3.5s VAD 黃金段」到「資料增強變體」的音質變化。
- 🔬 **消融實驗指標儀表板**：有降噪 vs 未降噪核心指標對照（同人相似度、輪廓係數、分離邊際）。
- 📊 **六大高解析聲學圖譜 Lightbox**：全螢幕高解析檢視時頻梅爾頻譜、PSD 功率譜、收斂曲線、盲測混淆矩陣與 t-SNE 降維空間。
- 📚 **四大技術文檔閱讀器**：線上分頁查閱完整前處理診斷、增強演算法與消融實驗報告。

---

## 一、 專案目錄結構

全專案腳本均已嚴格重構為**純相對路徑（Relative Path）**，相容 Windows、Linux (Ubuntu) 與 macOS。

```text
Voice identify/
│
├── README.md                               # 專案總覽、架構說明與完整使用文檔
├── requirements.txt                        # Python 環境相依套件清單 (一鍵安裝)
├── inference.py                            # [頂層推論核心] 支援終端互動選單、麥克風錄音、單檔/批次推論與 Callback 回呼
├── test.py                                 # [IoTtalk 介接模組] 聲紋辨識結果與 IoTtalk 平台開關訊號轉換器
├── IoTtalk_info.ipynb                      # [IoTtalk 配置指南] 記錄 IoTtalk 設備模型名稱 (!0Voice) 與密碼輸入規範
├── index.html                              # [GitHub Pages 互動展示頁] 網頁版即時收音、頻譜示波與消融儀表板
├── .gitignore                              # Git 版本控制過濾清單
│
├── assets/                                 # [前端靜態資源庫]
│   ├── css/
│   │   └── styles.css                      # Cyber Dark Theme 設計系統與 Glassmorphism 樣式
│   └── js/
│       └── app.js                          # Web Audio API 示波器繪製與動態推論邏輯
│
├── data/                                   # [音訊數據層級]
│   ├── 01_raw/                             # 原始錄音音檔 (48kHz Stereo M4A, 12 檔)
│   ├── 02_normalized/                      # 16kHz Mono, RMS -20 dBFS 全長正規化音檔 (12 檔)
│   ├── 03_valid_3.5s/                      # VAD 靜音切除之 3.5s 黃金發音段 (12 檔, 20ms Hann 窗)
│   ├── 04_augmented/                       # 6 種維度增強音訊 (共 72 檔)
│   ├── 05_speaker_dataset/                 # 按說話者分類之聲紋庫 (吳/甯/白/蕭)
│   │   ├── denoised/                       # 有降噪 (84 檔，正式訓練推薦)
│   │   └── raw/                            # 未降噪 (84 檔，消融實驗對照)
│   └── recordings/                         # 即時麥克風錄音與黃金段自動備份目錄
│
├── models/                                 # [微調模型權重庫]
│   ├── ecapa_classifier_有降噪_Denoised.pth# 正式生產模型 (獨立盲測集 100% 準確率)
│   └── ecapa_classifier_未降噪_Raw.pth     # 消融對照模型
│
├── figures/                                # [高解析度視覺化圖表庫]
│   ├── audio_comparison_waveform_spec.png  # 原音檔 vs 前處理音檔時頻對比圖
│   ├── audio_psd_energy_comparison.png     # 功率頻譜密度 (PSD) 與能量分佈圖
│   ├── training_curves_comparison.png      # 訓練損失與準確率收斂曲線圖
│   ├── confusion_matrices_comparison.png   # 獨立盲測集混淆矩陣熱力圖 (26/26 正確)
│   ├── tsne_speaker_embeddings.png         # 192 維聲紋 Embedding 空間 t-SNE 2D 投影圖
│   └── cosine_similarity_matrix.png        # 4 位說話者聲紋中心餘弦相似度熱力圖
│
├── reports/                                # [工程與實驗技術報告庫]
│   ├── 01_前處理問題分析與改進報告.md
│   ├── 02_前處理正規化與資料增強技術文檔.md
│   ├── 03_消融實驗_去噪前後對ECAPA-TDNN聲紋辨識成效報告.md
│   └── 04_原音與前處理音頻聲學對比視覺化報告.md
│
└── scripts/                                # [自動化工程腳本庫]
    ├── pipeline_process.py                 # 重新執行標準化、VAD 切割與資料擴增
    ├── run_ecapa_ablation.py               # 重新訓練 ECAPA-TDNN 分類器與執行消融評估
    ├── plot_training_results.py            # 重新繪製收斂曲線、混淆矩陣與 t-SNE 圖
    └── plot_audio_comparison.py            # 重新繪製時頻對比圖與 PSD 能量圖
```

---

## 二、 IoTtalk 物聯網平台介接與智慧聯動

本專案支援將聲紋特徵辨識結果作為物聯網系統的輸入觸發源（Input Device Feature, IDF），透過 [`test.py`](test.py) 與 [`inference.py`](inference.py) 的 `on_result` 回呼函式串接，實現**特定說話者身份開關控制**。

### 1. 介接架構原理

```text
  [ 說話者朗讀 ]
        │
        ▼ (麥克風收音 4.5s)
 [ inference.py ]  ──> [ 前處理 + 譜減降噪 + 3.5s VAD 切片 ]
        │
        ▼ (ECAPA-TDNN 192D 嵌入向量分類)
 [ on_result 回呼 ] ──> 回傳 {"speaker": "蕭", "confidence": 0.998, ...}
        │
        ▼
   [ test.py ]     ──> 身份驗證判定 (若為目標授權者 "蕭")
        │
        ▼ (狀態反轉 State Toggle)
  [ IoTtalk 訊號 ] ──> 輸出 500 (開啟設備 / ON) 或 0 (關閉設備 / OFF)
        │
        ▼
 [ Actuator / 智慧插座 / 門禁開鎖 ]
```

### 2. IoTtalk 設備模型資訊

根據 [`IoTtalk_info.ipynb`](IoTtalk_info.ipynb) 之配置規範：
- **Device Model Name**：`!0Voice`
- **功能特性**：
  - 輸入特徵（IDF）：音訊聲紋識別訊號。
  - 控制邏輯：當辨識到目標使用者（預設為「蕭」）時，觸發狀態變更，回傳數值控制燈光亮度、伺服馬達或門禁繼電器：
    - 啟動狀態（ON）：輸出數值 `500`
    - 關閉狀態（OFF）：輸出數值 `0`

### 3. 測試執行模式

在 [`test.py`](test.py) 中提供了兩種執行機制：

- **單次即時識別與狀態回傳（`pre2()`）**：
  ```bash
  python test.py
  ```
  直接呼叫 `interactive_mic_mode` 錄音並由 `handle_result` 捕獲結果字典，若預測說話者為目標講者，則翻轉當前狀態並回傳對應的控制代碼（500 / 0）。

- **背景多執行緒佇列輪詢（`pre()`）**：
  使用 `threading.Thread` 與 `queue.Queue` 在背景非同步執行收音推論，主線程可持續處理 IoTtalk 通訊心跳包（Heartbeat）或感測器輪詢。

---

## 三、 本機推論操作指南（Inference CLI）

本專案將推論入口完全整合於單一可執行腳本 [`inference.py`](inference.py)。

### 1. 互動式終端選單（推薦初次使用）
直接執行腳本，開啟互動選單：
```bash
python inference.py
```

選單預覽：
```text
============================================================
      物聯網聲紋辨識終端交互系統 - 主選單      
============================================================
  [1] 🎙️  啟動麥克風即時錄音辨識 (支援自訂語音輸入)
  [2] 📁  輸入或拖曳音訊檔案進行辨識
  [3] 📂  執行原始 12 首音檔自動批次評估
  [4] ❌  退出程式
============================================================
👉 請選擇功能編號 (1/2/3/4) [預設: 1]:
```

- **[1] 麥克風即時辨識**：按下 `Enter` 進入 3 秒倒數，朗讀任意語句（4.5 秒），系統自動完成重取樣、降噪、VAD 截取與聲紋預測，並將音訊保存於 `data/recordings/`。
- **[2] 音訊檔案推論**：直接輸入音訊檔名或路徑（支援拖曳檔案進視窗）。
- **[3] 批次自動評估**：針對 `data/01_raw/` 內全部 12 條基準音檔進行全自動預測與正確率統計。

### 2. 命令列快捷參數（適合腳本串接）
```bash
# 1. 直接啟動麥克風辨識
python inference.py --mic

# 2. 推論單一音訊檔案
python inference.py data/01_raw/吳1.m4a

# 3. 指定資料夾進行批次推論
python inference.py data/01_raw/

# 4. 指定消融模型（切換至未降噪模型對照組）
python inference.py data/01_raw/吳1.m4a --model models/ecapa_classifier_未降噪_Raw.pth --no-denoise
```

---

## 四、 開發者 API 呼叫範例（Python SDK & Callback）

外部系統可直接引用 `VoiceIdentifier` 類別，透過程式化介面取得聲紋特徵與預測分數：

```python
from inference import VoiceIdentifier

# 1. 初始化辨識器 (自動載入 SpeechBrain ECAPA-TDNN 與微調權重)
identifier = VoiceIdentifier(
    model_path="models/ecapa_classifier_有降噪_Denoised.pth",
    use_denoise=True
)

# 2. 定義自訂回呼處理函式
def my_iot_callback(result):
    speaker = result["speaker"]
    confidence = result["confidence"]
    probs = result["probabilities"]
    print(f"辨識完成 -> 說話者: {speaker}, 信心度: {confidence:.2f}%")
    return result

# 3. 呼叫麥克風錄音並傳入回呼
result = identifier.interactive_mic_mode(on_result=my_iot_callback)

# 4. 或是直接傳入本機音訊路徑推論
# speaker, confidence, probs = identifier.predict_file("path/to/audio.wav")
```

---

## 五、 消融實驗量化成果（Ablation Benchmark）

本專案深入探討「在進入 ECAPA-TDNN 前是否實施頻譜門限降噪（Spectral Gating）」之差異，並在 26 條包含強白噪聲、粉紅噪聲、時域隨機遮罩與帶通濾波的獨立盲測集上進行評估：

| 評估維度 | 量化指標 | Condition A (未降噪) | Condition B (有降噪) | 效益與機制解讀 |
| :--- | :--- | :---: | :---: | :--- |
| **分類性能** | **盲測集準確率 (Accuracy)** | **`100.00%`** | **`100.00%`** | 26 條強擾動測試樣本全部零失誤預測 |
| **分類性能** | **Macro F1-Score** | **`1.000`** | **`1.000`** | 各類別精準度與召回率均達 1.000 |
| **聲紋空間表徵** | **同說話者平均餘弦相似度 (Intra-sim)** | `0.6493` | **`0.6512`** | **提升 +0.0019**（同人聲紋特徵更緊湊集中） |
| **聲紋空間表徵** | **異說話者平均餘弦相似度 (Inter-sim)** | `0.2650` | `0.2665` | 持平（不同說話者間保持優異區隔度） |
| **分離能力** | **分離邊際 (Margin = Intra - Inter)** | `0.3843` | **`0.3847`** | **提升 +0.0004**（開放集門禁驗證容錯率更高） |
| **空間聚類品質** | **餘弦輪廓係數 (Silhouette Score)** | `0.4802` | **`0.4825`** | **提升 +0.0023**（在 192 維空間中邊界更乾淨） |
| **訓練動態** | **收斂速度 (Loss < 0.05)** | Epoch 14 | **Epoch 5** | **收斂速度提升近 3 倍**（去除底噪加速特徵學習） |

> 詳細實驗論證與聲學機制請參閱：[`reports/03_消融實驗_去噪前後對ECAPA-TDNN聲紋辨識成效報告.md`](reports/03_消融實驗_去噪前後對ECAPA-TDNN聲紋辨識成效報告.md)

---

## 六、 重現訓練與資料擴增流程

若欲加入新講者或使用自訂語音資料集重新訓練模型，可循以下步驟一鍵執行管線：

```bash
# 步驟 1: 執行標準正規化、VAD 切割與六維資料擴增 (生成 data/02, data/03, data/04)
python scripts/pipeline_process.py

# 步驟 2: 重新訓練 ECAPA-TDNN 分類頭並執行「未降噪 vs 有降噪」消融實驗
python scripts/run_ecapa_ablation.py

# 步驟 3: 重新繪製並匯出所有訓練曲線、t-SNE 空間投影與混淆矩陣
python scripts/plot_training_results.py

# 步驟 4: 重新繪製原音檔 vs 前處理音檔時頻對比圖與 PSD 能量分佈圖
python scripts/plot_audio_comparison.py
```

---

## 七、 依賴環境安裝（Installation）

本專案使用標準 Python 虛擬環境管理套件，可直接透過 `requirements.txt` 完成配置：

```bash
# 1. 建立虛擬環境 (可選)
python -m venv venv
# Windows 啟動:
.\venv\Scripts\activate
# Linux/macOS 啟動:
source venv/bin/activate

# 2. 一鍵安裝依賴套件
pip install -r requirements.txt
```

*註：若使用 NVIDIA GPU 加速，建議至 [PyTorch 官網](https://pytorch.org/get-started/locally/) 依據當前 CUDA 驅動版本安裝對應之 `torch` 與 `torchaudio`（例如 CUDA 12.8: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`）。*

---

## 八、 授權協議與鳴謝

- 本專案採用 **MIT License** 開源協議。
- 聲學特徵提取骨幹網絡基於 [SpeechBrain](https://speechbrain.github.io/) 之 `speechbrain/spkrec-ecapa-voxceleb` 預訓練模型。
- 物聯網整合架構參考國立陽明交通大學 [IoTtalk](https://iottalk.tw/) 平台規格。
