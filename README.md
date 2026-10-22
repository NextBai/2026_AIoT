# 物聯網聲紋辨識專案 (IoT Voice Identification Project)

[![GitHub Pages](https://img.shields.io/badge/Demo-Interactive%20GitHub%20Page-00f2fe?style=for-the-badge&logo=github)](https://nextbai.github.io/2026_AIoT/)
[![SpeechBrain](https://img.shields.io/badge/SpeechBrain-ECAPA--TDNN-7928ca?style=for-the-badge)](https://speechbrain.github.io/)
[![Test Accuracy](https://img.shields.io/badge/Accuracy-100%25%20(Blind%20Test)-10b981?style=for-the-badge)](reports/03_消融實驗_去噪前後對ECAPA-TDNN聲紋辨識成效報告.md)

本專案基於 **SpeechBrain ECAPA-TDNN** 預訓練架構，構建了一套完整的物聯網語音前處理、頻譜降噪、VAD 去頭去尾切片、自監督學習資料增強、模型訓練與推論評估管線。

🌐 **專案互動體驗網頁（GitHub Pages）**：
👉 **[https://nextbai.github.io/2026_AIoT/](https://nextbai.github.io/2026_AIoT/)**
- 🎙️ **瀏覽器即時收音室**：透過 Web Audio API 進行 3.5 秒即時錄音、時域示波圖 / 頻譜可視化與即時語者辨識。
- 🎵 **四階段音訊比對播放器**：線上直接試聽原音、16kHz 正規化、3.5s VAD 黃金片段與 6 種資料增強音訊。
- 🔬 **消融實驗指標儀表板**：有降噪 vs 未降噪核心指標對比（同人相似度、輪廓係數、分離邊際）。
- 📊 **聲學特徵與模型圖庫**：高解析度 Lightbox 檢視 6 大時頻與空間流形投影圖。
- 📚 **技術文檔瀏覽器**：線上查閱 4 份完整前處理分析與消融研究報告。

全專案腳本均已轉換為**相對路徑（Relative Path）**，不依賴任何特定作業系統之絕對磁碟機代號，可直接在不同 Windows / Linux / macOS 機器上跨平台無縫執行。

---

## 一、 專案目錄結構

```text
Voice identify/
│
├── README.md                               # 專案總覽與使用說明
├── inference.py                            # [頂層推論入口] 支援單檔與批次目錄推論
│
├── data/                                   # [音訊數據層級]
│   ├── 01_raw/                             # 原始錄音音檔 (物聯網音檔，48kHz Stereo M4A)
│   ├── 02_normalized/                      # 16kHz Mono，RMS -20 dBFS，保留全句長度 (12 檔)
│   ├── 03_valid_3.5s/                      # VAD 靜音切除之 3.5s 黃金段 (12 檔，零爆音)
│   ├── 04_augmented/                       # 6 種維度增強音訊 (72 檔)
│   └── 05_speaker_dataset/                 # 按說話者 (吳/甯/白/蕭) 分類
│       ├── denoised/                       # 有降噪 (84 檔，訓練推薦)
│       └── raw/                            # 未降噪 (84 檔，消融對照)
│
├── models/                                 # [模型權重庫]
│   ├── ecapa_classifier_有降噪_Denoised.pth# 正式生產模型
│   └── ecapa_classifier_未降噪_Raw.pth     # 消融對照模型
│
├── figures/                                # [視覺化圖表庫]
│   ├── audio_comparison_waveform_spec.png  # 原音檔 vs 前處理音檔時頻對比圖
│   ├── audio_psd_energy_comparison.png     # 功率頻譜密度 (PSD) 與能量包絡對比圖
│   ├── training_curves_comparison.png      # 訓練損失與準確率收斂曲線圖
│   ├── confusion_matrices_comparison.png   # 獨立盲測集混淆矩陣熱力圖 (100% Acc)
│   ├── tsne_speaker_embeddings.png         # 192 維聲紋 Embedding 空間 t-SNE 2D 投影圖
│   └── cosine_similarity_matrix.png        # 4 位說話者聲紋中心餘弦相似度熱力圖
│
├── reports/                                # [分析與技術報告庫]
│   ├── 01_前處理問題分析與改進報告.md
│   ├── 02_前處理正規化與資料增強技術文檔.md
│   ├── 03_消融實驗_去噪前後對ECAPA-TDNN聲紋辨識成效報告.md
│   └── 04_原音與前處理音頻聲學對比視覺化報告.md
│
└── scripts/                                # [工程腳本庫]
    ├── pipeline_process.py                 # 重新執行正規化與資料擴增
    ├── run_ecapa_ablation.py               # 重新訓練 ECAPA-TDNN 分類器與消融評估
    ├── plot_training_results.py            # 重新繪製訓練收斂曲線與混淆矩陣
    └── plot_audio_comparison.py            # 重新繪製原音 vs 前處理音頻對比圖
```

---

## 二、 快速開始：終端交互推論（Inference）

本專案將所有推論功能完全整合於單一核心腳本 [`inference.py`](inference.py) 中，提供友善直覺的**終端交互介面**。只要執行以下指令即可開啟互動選單：

```bash
python inference.py
```

### 終端主選單功能：
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

- **選擇 [1] 🎙️ 麥克風即時辨識**：
  按下 `Enter` 啟動 3 秒倒數，對著麥克風朗讀一句話（4.5 秒），系統自動完成前處理、降噪與 ECAPA-TDNN 特徵辨識，印出最高置信度與說話者機率條形圖，並自動將錄音備份至 `data/recordings/`。
- **選擇 [2] 📁 音訊檔案推論**：
  直接在終端輸入音訊路徑或將音訊檔案拖曳進終端窗口即可即時辨識。
- **選擇 [3] 📂 批次自動評估**：
  自動檢測 `data/01_raw/` 內的 12 條原始音檔並輸出矩陣評估報告。

---

### CLI 快捷指令（直接指定參數）：
- **直接啟動麥克風錄音**：`python inference.py --mic`
- **直接指定單一檔案**：`python inference.py data/01_raw/吳1.m4a`
- **直接執行 12 檔批次評估**：`python inference.py --demo`

### 2. 單一音檔推論
```bash
python inference.py data/01_raw/吳1.m4a
```
輸出範例：
```text
[推論結果]
音檔路徑: data/01_raw/吳1.m4a
預測說話者: 【 吳 】
預測置信度: 40.88%
各說話者機率: {'吳': 40.88%, '甯': 21.32%, '白': 21.52%, '蕭': 16.28%}
```

### 3. 指定資料夾進行批次推論
```bash
python inference.py data/01_raw/
```

### 4. 關閉頻譜降噪（可選）
若希望以未降噪的原始特徵進行推論：
```bash
python inference.py data/01_raw/吳1.m4a --model models/ecapa_classifier_未降噪_Raw.pth --no-denoise
```

---

## 三、 重現訓練與消融實驗（Training & Ablation）

若新增了說話者或更新了音檔，可透過以下步驟一鍵重新生成資料集、微調模型與繪製圖表：

```bash
# 1. 執行標準正規化、VAD 切割與 SSL 資料擴增 (產出 data/02, data/03, data/04)
python scripts/pipeline_process.py

# 2. 重新訓練 ECAPA-TDNN 分類頭並執行「未降噪 vs 有降噪」消融實驗
python scripts/run_ecapa_ablation.py

# 3. 重新繪製並匯出所有訓練曲線、t-SNE 空間投影與混淆矩陣
python scripts/plot_training_results.py

# 4. 重新繪製原音檔 vs 前處理音檔時頻對比圖
python scripts/plot_audio_comparison.py
```

---

## 四、 依賴環境安裝

```bash
pip install torch torchaudio soundfile av scipy noisereduce speechbrain scikit-learn matplotlib seaborn
```
*註：若使用 NVIDIA GPU，建議依官方指引安裝對應 CUDA 版本的 `torch` 與 `torchaudio`。*
