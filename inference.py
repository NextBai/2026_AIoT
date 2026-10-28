import os
import sys
import time
import datetime
import argparse
import numpy as np
import scipy.signal
import soundfile as sf
import sounddevice as sd
import noisereduce as nr
import av
import torch
import torch.nn as nn
from speechbrain.inference.speaker import EncoderClassifier

sys.stdout.reconfigure(encoding='utf-8')

# Ensure relative path based on script location for complete portability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "ecapa_classifier_有降噪_Denoised.pth")
DEFAULT_AUDIO_DIR = os.path.join(BASE_DIR, "data", "01_raw")
RECORDINGS_DIR = os.path.join(BASE_DIR, "data", "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

SPEAKERS = ["吳", "甯", "白", "蕭"]
SPK2IDX = {s: i for i, s in enumerate(SPEAKERS)}
IDX2SPK = {i: s for i, s in enumerate(SPEAKERS)}

class ECAPAClassifier(nn.Module):
    def __init__(self, emb_dim=192, num_classes=4):
        super().__init__()
        self.bn = nn.BatchNorm1d(emb_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = self.bn(x)
        x = self.dropout(x)
        return self.fc(x)

class VoiceIdentifier:
    def __init__(self, classifier_path=None, device=None, use_denoise=True, model_path=None):
        if classifier_path is None:
            classifier_path = model_path if model_path is not None else DEFAULT_MODEL_PATH
        self.use_denoise = use_denoise
        
        if device is None:
            self.device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = device
            
        print("\n" + "="*65)
        print("   物聯網聲紋辨識終端交互系統 (SpeechBrain ECAPA-TDNN)   ")
        print("="*65)
        print(f"[*] 運算裝置 (Device): {self.device_str}")
        print(f"[*] 載入 ECAPA-TDNN 聲紋預訓練骨幹網絡...")
        self.ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device_str}
        )
        
        self.classifier = ECAPAClassifier(emb_dim=192, num_classes=len(SPEAKERS)).to(self.device_str)
        if os.path.exists(classifier_path):
            state_dict = torch.load(classifier_path, map_location=self.device_str)
            self.classifier.load_state_dict(state_dict)
            self.classifier.eval()
            print(f"[*] 成功載入分類器權重: {os.path.relpath(classifier_path, BASE_DIR)}")
        else:
            raise FileNotFoundError(f"找不到模型權重檔: {classifier_path}")

    @staticmethod
    def load_audio(filepath, target_sr=16000):
        """讀取任何格式音訊 (.m4a, .wav, .mp3 等)，轉單聲道並以多相濾波降取樣至 target_sr"""
        container = av.open(filepath)
        stream = container.streams.audio[0]
        sr_orig = stream.rate
        frames = [f.to_ndarray() for f in container.decode(stream)]
        container.close()

        audio = np.concatenate(frames, axis=-1)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)

        # 多相濾波抗混疊降取樣
        if sr_orig != target_sr:
            gcd = np.gcd(sr_orig, target_sr)
            up = target_sr // gcd
            down = sr_orig // gcd
            audio = scipy.signal.resample_poly(audio, up, down).astype(np.float32)

        audio = audio - np.mean(audio)
        return audio, target_sr

    @staticmethod
    def denoise_audio(audio, sr=16000, noise_lead_sec=0.3, prop_decrease=0.75):
        """以音訊前導 300ms 靜音段作為噪聲輪廓實施頻譜門限降噪"""
        n_lead = int(sr * noise_lead_sec)
        noise_clip = audio[:n_lead] if len(audio) > n_lead else audio
        denoised = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_clip, prop_decrease=prop_decrease, stationary=True)
        return denoised.astype(np.float32)

    @staticmethod
    def normalize_loudness(audio, target_rms_db=-20.0, peak_limit_db=-1.0):
        """RMS 響度正規化至 target_rms_db，並設定 Peak 限制防止數位破音"""
        rms = np.sqrt(np.mean(audio**2))
        target_rms = 10 ** (target_rms_db / 20.0)
        if rms > 1e-7:
            audio = audio * (target_rms / rms)
        peak = np.max(np.abs(audio))
        peak_limit = 10 ** (peak_limit_db / 20.0)
        if peak > peak_limit:
            audio = audio * (peak_limit / peak)
        return audio

    @staticmethod
    def vad_and_extract_golden_segment(audio, sr=16000, target_sec=3.5, fade_ms=20):
        """VAD 雙門檻靜音切除 + 3.5 秒能量最大黃金段截取 + 20ms Hann 窗平滑淡入淡出（零爆音）"""
        frame_len = int(sr * 0.03)
        hop_len = int(sr * 0.01)
        n_frames = (len(audio) - frame_len) // hop_len + 1
        energies = np.array([np.sum(audio[i*hop_len : i*hop_len + frame_len]**2) for i in range(n_frames)])
        thresh = np.max(energies) * 0.035
        active = np.where(energies > thresh)[0]

        if len(active) == 0:
            speech = audio
        else:
            buf = int(0.08 * sr / hop_len)
            s = max(0, active[0] - buf) * hop_len
            e = min(len(audio), (min(n_frames-1, active[-1] + buf)) * hop_len + frame_len)
            speech = audio[s:e]

        target_len = int(target_sec * sr)
        if len(speech) <= target_len:
            segment = speech.copy()
            if len(segment) < target_len:
                pad = np.zeros(target_len - len(segment), dtype=np.float32)
                segment = np.concatenate([segment, pad])
        else:
            best_e = -1.0
            best_s = 0
            step = sr // 10
            for st in range(0, len(speech) - target_len, step):
                e = np.sum(speech[st : st + target_len]**2)
                if e > best_e:
                    best_e = e
                    best_s = st
            segment = speech[best_s : best_s + target_len].copy()

        # 20ms Hann 窗平滑加窗
        fade_len = int(sr * fade_ms / 1000)
        if len(segment) > 2 * fade_len:
            hann_in = 0.5 * (1 - np.cos(np.pi * np.arange(fade_len) / fade_len))
            hann_out = 0.5 * (1 + np.cos(np.pi * np.arange(fade_len) / fade_len))
            segment[:fade_len] *= hann_in
            segment[-fade_len:] *= hann_out

        return segment

    def preprocess_signal(self, raw_audio, sr=16000, apply_denoise=True):
        """完整前處理管線（直流校正 -> 頻譜降噪 -> RMS -20dBFS -> VAD 3.5s Hann 窗加窗）"""
        audio = raw_audio.flatten().astype(np.float32)
        audio = audio - np.mean(audio)
        
        if np.max(np.abs(audio)) < 0.005:
            return None, "音訊振幅過低（可能未發音或麥克風靜音）"

        if apply_denoise:
            audio = self.denoise_audio(audio, sr=sr)
            
        audio = self.normalize_loudness(audio, target_rms_db=-20.0, peak_limit_db=-1.0)
        segment = self.vad_and_extract_golden_segment(audio, sr=sr, target_sec=3.5, fade_ms=20)
        return segment, "OK"

    def predict_segment(self, segment):
        """輸入 3.5s 前處理音訊，透過 ECAPA-TDNN 輸出說話者預測結果"""
        wav_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(self.device_str)
        with torch.no_grad():
            emb = self.ecapa.encode_batch(wav_tensor).squeeze()
            emb = emb / (torch.norm(emb, p=2) + 1e-7)
            emb = emb.unsqueeze(0)
            logits = self.classifier(emb)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_spk = IDX2SPK[pred_idx]
        confidence = float(probs[pred_idx]) * 100.0
        spk_probs = {IDX2SPK[i]: float(probs[i]) * 100.0 for i in range(len(SPEAKERS))}

        return pred_spk, confidence, spk_probs

    def predict_file(self, filepath, apply_denoise=True):
        """預測單一音訊檔案"""
        raw_audio, sr = self.load_audio(filepath, target_sr=16000)
        segment, status = self.preprocess_signal(raw_audio, sr=sr, apply_denoise=apply_denoise)
        if segment is None:
            return None, status
        pred_spk, conf, probs = self.predict_segment(segment)
        return {
            'filepath': filepath,
            'speaker': pred_spk,
            'confidence': conf,
            'probabilities': probs
        }, "OK"

    def print_result_card(self, pred_spk, confidence, spk_probs, extra_info=None):
        """格式化輸出美觀之預測卡片"""
        print("\n" + "="*52)
        print(f"🎯 預測說話者辨識結果: 【 {pred_spk} 】")
        print(f"🔥 最高辨識置信度:    {confidence:6.2f} %")
        print("="*52)
        print("📊 各說話者 Softmax 機率分佈:")
        for spk in SPEAKERS:
            p = spk_probs[spk]
            bar = "█" * int(p / 4)
            marker = " 👈 [命中目標]" if spk == pred_spk else ""
            print(f"   {spk}: {p:6.2f} %  | {bar:<25}{marker}")
        print("="*52)
        if extra_info:
            for k, v in extra_info.items():
                print(f"ℹ️ {k}: {v}")
        print()

    # -------------------------------------------------------------
    # 程式化 API: 麥克風直接錄音並推論 (適合 IoT / 背景線程)
    # -------------------------------------------------------------
    def record_and_predict(self, duration_sec=4.5, sr=16000, apply_denoise=True, save_recording=True):
        """
        純程式化非互動麥克風收音與推論 API (適合 IoT 邊緣裝置 / 背景線程直接調用)
        回傳結構:
        {
            "success": bool,
            "speaker": str or None,
            "confidence": float,
            "probabilities": dict,
            "raw_path": str or None,
            "processed_path": str or None,
            "status": str
        }
        """
        n_samples = int(duration_sec * sr)
        raw_recording = sd.rec(n_samples, samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = None
        clean_path = None
        
        if save_recording:
            raw_path = os.path.join(RECORDINGS_DIR, f"rec_{timestamp}_raw.wav")
            sf.write(raw_path, raw_recording, sr, subtype='PCM_16')
            
        clean_segment, status = self.preprocess_signal(raw_recording, sr=sr, apply_denoise=apply_denoise)
        if clean_segment is None:
            return {
                "success": False,
                "speaker": None,
                "confidence": 0.0,
                "probabilities": {spk: 0.0 for spk in self.speakers},
                "raw_path": raw_path,
                "processed_path": None,
                "status": f"failed: {status}"
            }
            
        if save_recording:
            clean_path = os.path.join(RECORDINGS_DIR, f"rec_{timestamp}_processed.wav")
            sf.write(clean_path, clean_segment, sr, subtype='PCM_16')
            
        pred_spk, conf, probs = self.predict_segment(clean_segment)
        return {
            "success": True,
            "speaker": pred_spk,
            "confidence": conf,
            "probabilities": probs,
            "raw_path": raw_path,
            "processed_path": clean_path,
            "status": "success"
        }

    # -------------------------------------------------------------
    # 終端交互功能 1: 麥克風即時錄音辨識
    # -------------------------------------------------------------
    def interactive_mic_mode(self, duration_sec=4.5, sr=16000, on_result=None, auto_start=False):
        """
        終端麥克風錄音模式。
        - 若 auto_start=True，將跳過 Enter 鍵等待，適合非互動或邊緣腳本。
        - 若提供 on_result 回呼函式，將在取得結果後返回。
        """
        print("\n" + "-"*65)
        print("🎙️ 【麥克風即時錄音辨識模式】")
        print("💡 請準備說話（例如朗讀：『故人西辭黃鶴樓』或『紅豆生南國』）")
        
        while True:
            if not auto_start:
                try:
                    cmd = input("👉 按下 【Enter】 鍵立即開始倒數錄音 (輸入 'q' 退出): ").strip()
                    if cmd.lower() == 'q':
                        break
                except (EOFError, KeyboardInterrupt):
                    break
            
            for c in [3, 2, 1]:
                print(f"⏱️ 倒數計時: {c} 秒...", end="\r", flush=True)
                time.sleep(0.8)
                
            print("\n🎤 【● 正在錄音中】請對著麥克風說話...", flush=True)
            
            n_samples = int(duration_sec * sr)
            raw_recording = sd.rec(n_samples, samplerate=sr, channels=1, dtype='float32')
            
            steps = 20
            for i in range(steps):
                time.sleep(duration_sec / steps)
                prog = "█" * (i + 1) + "░" * (steps - i - 1)
                pct = int((i + 1) / steps * 100)
                print(f"[{prog}] {pct}% 錄製中...", end="\r", flush=True)
                
            sd.wait()
            print("\n✅ 錄音完成！正在執行前處理與聲紋辨識...", flush=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_path = os.path.join(RECORDINGS_DIR, f"rec_{timestamp}_raw.wav")
            sf.write(raw_path, raw_recording, sr, subtype='PCM_16')
            
            clean_segment, status = self.preprocess_signal(raw_recording, sr=sr, apply_denoise=True)
            if clean_segment is None:
                print(f"\n⚠️ 辨識失敗：{status}\n")
                if on_result is not None:
                    fail_dict = {
                        "success": False,
                        "speaker": None,
                        "confidence": 0.0,
                        "probabilities": {spk: 0.0 for spk in self.speakers},
                        "status": f"failed: {status}"
                    }
                    cb_res = on_result(fail_dict)
                    return cb_res if cb_res is not None else fail_dict
                if auto_start:
                    break
                continue
                
            clean_path = os.path.join(RECORDINGS_DIR, f"rec_{timestamp}_processed.wav")
            sf.write(clean_path, clean_segment, sr, subtype='PCM_16')
            
            pred_spk, conf, probs = self.predict_segment(clean_segment)

            result_dict = {
                "success": True,
                "speaker": pred_spk,
                "confidence": conf,
                "probabilities": probs,
                "raw_path": raw_path,
                "processed_path": clean_path,
                "status": "success"
            }

            if on_result is not None:
                cb_res = on_result(result_dict)
                return cb_res if cb_res is not None else result_dict

            extra = {
                "原始錄音備份": os.path.relpath(raw_path, BASE_DIR),
                "前處理黃金段": os.path.relpath(clean_path, BASE_DIR)
            }
            self.print_result_card(pred_spk, conf, probs, extra)
            
            if auto_start:
                return result_dict

    # -------------------------------------------------------------
    # 終端交互功能 2: 手動輸入音訊檔案辨識
    # -------------------------------------------------------------
    def interactive_file_mode(self):
        print("\n" + "-"*65)
        print("📁 【音訊檔案推論模式】")
        while True:
            path_input = input("👉 請輸入音訊檔案路徑或直接拖曳檔案進來 (輸入 'q' 返回): ").strip(' "\'')
            if path_input.lower() == 'q':
                break
            if not os.path.exists(path_input):
                # 嘗試在 data/01_raw 下尋找
                candidate = os.path.join(BASE_DIR, "data", "01_raw", path_input)
                if os.path.exists(candidate):
                    path_input = candidate
                else:
                    print(f"❌ 找不到該檔案: {path_input}，請重新輸入！")
                    continue
                    
            res, status = self.predict_file(path_input, apply_denoise=True)
            if res is None:
                print(f"❌ 處理失敗: {status}")
                continue
            extra = {"音檔路徑": os.path.relpath(res['filepath'], BASE_DIR)}
            self.print_result_card(res['speaker'], res['confidence'], res['probabilities'], extra)

    # -------------------------------------------------------------
    # 終端交互功能 3: 原始 12 首音檔自動批次評估
    # -------------------------------------------------------------
    def batch_demo_mode(self):
        print("\n" + "="*75)
        print(f"📂 【原始音檔自動批次評估】 目錄: {os.path.relpath(DEFAULT_AUDIO_DIR, BASE_DIR)}")
        print("="*75)
        if not os.path.exists(DEFAULT_AUDIO_DIR):
            print(f"❌ 找不到目錄: {DEFAULT_AUDIO_DIR}")
            return
            
        files = sorted([f for f in os.listdir(DEFAULT_AUDIO_DIR) if f.endswith(('.m4a', '.wav', '.mp3'))])
        if not files:
            print("目錄內無音訊檔案！")
            return
            
        print(f"{'檔案名稱':<12} | {'真實說話者':<8} | {'預測說話者':<8} | {'置信度 (%)':<10} | {'各說話者機率分佈 (%)'}")
        print("-" * 75)
        correct = 0
        for fn in files:
            fp = os.path.join(DEFAULT_AUDIO_DIR, fn)
            true_spk = fn[0]
            res, status = self.predict_file(fp, apply_denoise=True)
            if res:
                is_correct = (res['speaker'] == true_spk)
                if is_correct:
                    correct += 1
                p_str = " | ".join([f"{k}: {v:5.1f}%" for k, v in res['probabilities'].items()])
                flag = "✓" if is_correct else "✗"
                print(f"{fn:<12} | {true_spk:<10} | {res['speaker']:<10} | {res['confidence']:6.2f}% {flag} | {p_str}")
        print("-" * 75)
        print(f"批次評估總計：{correct} / {len(files)} 正確 (準確率: {correct/len(files)*100:.2f}%)\n")

# -------------------------------------------------------------
# 主程式入口：終端交互主選單
# -------------------------------------------------------------
def interactive_menu(identifier):
    while True:
        print("\n" + "="*60)
        print("      物聯網聲紋辨識終端交互系統 - 主選單      ")
        print("="*60)
        print("  [1] 🎙️  啟動麥克風即時錄音辨識 (支援自訂語音輸入)")
        print("  [2] 📁  輸入或拖曳音訊檔案進行辨識")
        print("  [3] 📂  執行原始 12 首音檔自動批次評估")
        print("  [4] ❌  退出程式")
        print("="*60)
        choice = input("👉 請選擇功能編號 (1/2/3/4) [預設: 1]: ").strip()
        
        if choice == "2":
            identifier.interactive_file_mode()
        elif choice == "3":
            identifier.batch_demo_mode()
        elif choice == "4":
            print("\n👋 感謝使用，已退出聲紋辨識系統。")
            break
        else:
            identifier.interactive_mic_mode(duration_sec=4.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Identification Inference CLI")
    parser.add_argument("input", nargs="?", help="可選：音檔路徑或目錄")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="指定分類器模型權重路徑 (.pth)")
    parser.add_argument("--mic", action="store_true", help="直接啟動麥克風錄音模式")
    parser.add_argument("--demo", action="store_true", help="直接執行 12 檔批次評估")
    parser.add_argument("--no-denoise", action="store_true", help="停用頻譜門限降噪")
    args = parser.parse_args()

    identifier = VoiceIdentifier(classifier_path=args.model)

    if args.mic:
        identifier.interactive_mic_mode(duration_sec=4.5)
    elif args.demo:
        identifier.batch_demo_mode()
    elif args.input:
        if os.path.isfile(args.input):
            res, status = identifier.predict_file(args.input, apply_denoise=not args.no_denoise)
            if res:
                extra = {"音檔路徑": os.path.relpath(res['filepath'], BASE_DIR)}
                identifier.print_result_card(res['speaker'], res['confidence'], res['probabilities'], extra)
        elif os.path.isdir(args.input):
            print(f"\n批次推論目錄: {args.input}")
            files = sorted([f for f in os.listdir(args.input) if f.endswith(('.m4a', '.wav', '.mp3', '.flac'))])
            for fn in files:
                fp = os.path.join(args.input, fn)
                res, _ = identifier.predict_file(fp, apply_denoise=not args.no_denoise)
                if res:
                    print(f"{fn:<15} -> 預測: {res['speaker']} (置信度: {res['confidence']:.2f}%)")
    else:
        # 預設直接進入終端交互主選單
        interactive_menu(identifier)
