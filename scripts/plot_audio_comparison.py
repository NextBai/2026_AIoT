import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal
import av

sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_dir = os.path.join(base_dir, "data", "01_raw")
valid_dir = os.path.join(base_dir, "data", "03_valid_3.5s")
figures_dir = os.path.join(base_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

artifact_dir = os.environ.get("ANTIGRAVITY_ARTIFACT_DIR")
if artifact_dir:
    os.makedirs(artifact_dir, exist_ok=True)

def load_m4a_raw(filepath):
    container = av.open(filepath)
    stream = container.streams.audio[0]
    sr = stream.rate
    frames = [f.to_ndarray() for f in container.decode(stream)]
    container.close()
    audio = np.concatenate(frames, axis=-1)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    return audio, sr

# Pick two representative files:
# 1. '吳1' (Standard speech)
# 2. '蕭1' (Had 12.7s duration, long silence, and previous noise failure)
demo_files = [("吳1", "吳1.m4a", "吳1_valid_3.5s.wav"),
              ("蕭1", "蕭1.m4a", "蕭1_valid_3.5s.wav")]

# =========================================================================
# Figure 1: Waveforms and Mel-Spectrograms (Original vs Preprocessed)
# =========================================================================
fig, axes = plt.subplots(4, 2, figsize=(16, 12), dpi=200)

for col_idx, (spk_title, raw_fn, proc_fn) in enumerate(demo_files):
    raw_path = os.path.join(raw_dir, raw_fn)
    proc_path = os.path.join(valid_dir, proc_fn)
    
    # Load raw
    raw_audio, sr_raw = load_m4a_raw(raw_path)
    raw_mono = np.mean(raw_audio, axis=0) if raw_audio.ndim > 1 else raw_audio[0]
    raw_t = np.linspace(0, len(raw_mono) / sr_raw, len(raw_mono))
    
    # Load preprocessed
    proc_mono, sr_proc = sf.read(proc_path)
    proc_t = np.linspace(0, len(proc_mono) / sr_proc, len(proc_mono))
    
    # 1. Raw Waveform
    axes[0, col_idx].plot(raw_t, raw_mono, color='#1f77b4', linewidth=0.8, alpha=0.9)
    axes[0, col_idx].set_title(f"【原音檔】{spk_title} 時域波形 (48kHz, {len(raw_mono)/sr_raw:.2f}s, 含前後長靜音)", fontsize=11.5, fontweight='bold')
    axes[0, col_idx].set_xlabel("時間 (秒)", fontsize=10)
    axes[0, col_idx].set_ylabel("振幅 (Amplitude)", fontsize=10)
    axes[0, col_idx].grid(True, linestyle=':', alpha=0.6)
    axes[0, col_idx].set_ylim(-1.0, 1.0)
    # Highlight leading and trailing silence
    axes[0, col_idx].axvspan(0, 0.5, color='gray', alpha=0.2, label='前導靜音')
    axes[0, col_idx].legend(loc='upper right', fontsize=9)
    
    # 2. Preprocessed Waveform
    axes[1, col_idx].plot(proc_t, proc_mono, color='#2ca02c', linewidth=1.0)
    axes[1, col_idx].set_title(f"【前處理後】{spk_title} 時域波形 (16kHz, 3.50s, VAD黃金段 + 20ms Hann淡入淡出)", fontsize=11.5, fontweight='bold')
    axes[1, col_idx].set_xlabel("時間 (秒)", fontsize=10)
    axes[1, col_idx].set_ylabel("振幅 (Amplitude)", fontsize=10)
    axes[1, col_idx].grid(True, linestyle=':', alpha=0.6)
    axes[1, col_idx].set_ylim(-1.0, 1.0)
    # Highlight zero-crossing boundary
    axes[1, col_idx].axvspan(0, 0.02, color='orange', alpha=0.3, label='20ms Hann淡入')
    axes[1, col_idx].axvspan(3.48, 3.50, color='orange', alpha=0.3, label='20ms Hann淡出')
    axes[1, col_idx].legend(loc='upper right', fontsize=9)
    
    # 3. Raw Spectrogram
    f_raw, t_raw_s, Sxx_raw = scipy.signal.spectrogram(raw_mono, fs=sr_raw, nperseg=1024, noverlap=512)
    Sxx_raw_db = 10 * np.log10(Sxx_raw + 1e-9)
    im0 = axes[2, col_idx].pcolormesh(t_raw_s, f_raw, Sxx_raw_db, shading='gouraud', cmap='magma', vmin=-100, vmax=-20)
    axes[2, col_idx].set_title(f"【原音檔】{spk_title} 聲譜圖 (頻率範圍 0~24kHz, 含高頻雜訊與環境底噪)", fontsize=11.5, fontweight='bold')
    axes[2, col_idx].set_xlabel("時間 (秒)", fontsize=10)
    axes[2, col_idx].set_ylabel("頻率 (Hz)", fontsize=10)
    axes[2, col_idx].set_ylim(0, 20000)
    
    # 4. Preprocessed Spectrogram
    f_proc, t_proc_s, Sxx_proc = scipy.signal.spectrogram(proc_mono, fs=sr_proc, nperseg=512, noverlap=256)
    Sxx_proc_db = 10 * np.log10(Sxx_proc + 1e-9)
    im1 = axes[3, col_idx].pcolormesh(t_proc_s, f_proc, Sxx_proc_db, shading='gouraud', cmap='magma', vmin=-90, vmax=-10)
    axes[3, col_idx].set_title(f"【前處理後】{spk_title} 聲譜圖 (標準語音 0~8kHz, 清晰母音共振峰與純淨特徵)", fontsize=11.5, fontweight='bold')
    axes[3, col_idx].set_xlabel("時間 (秒)", fontsize=10)
    axes[3, col_idx].set_ylabel("頻率 (Hz)", fontsize=10)
    axes[3, col_idx].set_ylim(0, 8000)

plt.tight_layout()
fig1_path = os.path.join(figures_dir, "audio_comparison_waveform_spec.png")
fig.savefig(fig1_path)
plt.close(fig)
print(f"Generated: {fig1_path}")

# =========================================================================
# Figure 2: Power Spectral Density (PSD) & Energy Envelope Comparison
# =========================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 8.5), dpi=200)

for col_idx, (spk_title, raw_fn, proc_fn) in enumerate(demo_files):
    raw_path = os.path.join(raw_dir, raw_fn)
    proc_path = os.path.join(valid_dir, proc_fn)
    
    raw_audio, sr_raw = load_m4a_raw(raw_path)
    raw_mono = np.mean(raw_audio, axis=0) if raw_audio.ndim > 1 else raw_audio[0]
    proc_mono, sr_proc = sf.read(proc_path)
    
    # Resample raw to 16k just for direct PSD comparison
    raw_16k = scipy.signal.resample_poly(raw_mono, 1, sr_raw // 16000).astype(np.float32)
    
    # 1. Welch Power Spectral Density
    f_raw_psd, p_raw = scipy.signal.welch(raw_16k, fs=16000, nperseg=1024)
    f_proc_psd, p_proc = scipy.signal.welch(proc_mono, fs=16000, nperseg=1024)
    
    p_raw_db = 10 * np.log10(p_raw + 1e-12)
    p_proc_db = 10 * np.log10(p_proc + 1e-12)
    
    axes2[0, col_idx].plot(f_raw_psd, p_raw_db, label='原音檔 (含環境低頻噪聲與底噪)', color='#1f77b4', linewidth=1.8, alpha=0.85)
    axes2[0, col_idx].plot(f_proc_psd, p_proc_db, label='前處理後 (去噪+正規化+共振峰增益)', color='#d62728', linewidth=2.0)
    axes2[0, col_idx].set_title(f"{spk_title} 功率頻譜密度 (PSD) 能量分佈對比", fontsize=12, fontweight='bold', pad=10)
    axes2[0, col_idx].set_xlabel("頻率 (Hz)", fontsize=10.5)
    axes2[0, col_idx].set_ylabel("功率譜密度 (dB/Hz)", fontsize=10.5)
    axes2[0, col_idx].set_xlim(50, 8000)
    axes2[0, col_idx].grid(True, linestyle=':', alpha=0.6)
    axes2[0, col_idx].legend(loc='upper right', fontsize=10)
    
    # 2. Short-Time RMS Energy Envelope
    hop = 160 # 10ms
    win = 320 # 20ms
    n_f_raw = (len(raw_16k) - win) // hop
    rms_raw = np.array([20 * np.log10(np.sqrt(np.mean(raw_16k[i*hop:i*hop+win]**2)) + 1e-6) for i in range(n_f_raw)])
    t_rms_raw = np.arange(n_f_raw) * (hop / 16000)
    
    n_f_proc = (len(proc_mono) - win) // hop
    rms_proc = np.array([20 * np.log10(np.sqrt(np.mean(proc_mono[i*hop:i*hop+win]**2)) + 1e-6) for i in range(n_f_proc)])
    t_rms_proc = np.arange(n_f_proc) * (hop / 16000)
    
    axes2[1, col_idx].plot(t_rms_raw, rms_raw, label=f'原音檔短時能量 (全長 {len(raw_mono)/sr_raw:.1f}s)', color='#1f77b4', linewidth=1.5, alpha=0.7)
    axes2[1, col_idx].plot(t_rms_proc, rms_proc, label=f'前處理音檔短時能量 (3.5s 黃金段)', color='#2ca02c', linewidth=2.2)
    axes2[1, col_idx].axhline(-20.0, color='r', linestyle='--', linewidth=1.5, label='目標標準 RMS (-20 dBFS)')
    axes2[1, col_idx].set_title(f"{spk_title} 短時能量包絡 (Short-Time RMS Energy)", fontsize=12, fontweight='bold', pad=10)
    axes2[1, col_idx].set_xlabel("時間 (秒)", fontsize=10.5)
    axes2[1, col_idx].set_ylabel("能量 (dBFS)", fontsize=10.5)
    axes2[1, col_idx].set_ylim(-65, -5)
    axes2[1, col_idx].grid(True, linestyle=':', alpha=0.6)
    axes2[1, col_idx].legend(loc='lower right', fontsize=9.5)

plt.tight_layout()
fig2_path = os.path.join(figures_dir, "audio_psd_energy_comparison.png")
fig2.savefig(fig2_path)
plt.close(fig2)
print(f"Generated: {fig2_path}")

# Copy to artifact dir for UI embedding if set
if artifact_dir:
    shutil.copy2(fig1_path, os.path.join(artifact_dir, os.path.basename(fig1_path)))
    shutil.copy2(fig2_path, os.path.join(artifact_dir, os.path.basename(fig2_path)))
    print("Copied both figures to artifact directory.")
