import os
import sys
import numpy as np
import scipy.signal
import soundfile as sf
import av

sys.stdout.reconfigure(encoding='utf-8')

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
orig_dir = os.path.join(base_dir, "data", "01_raw")

dir_normalized = os.path.join(base_dir, "data", "02_normalized")
dir_trimmed_valid = os.path.join(base_dir, "data", "03_valid_3.5s")
dir_augmented = os.path.join(base_dir, "data", "04_augmented")

for d in [dir_normalized, dir_trimmed_valid, dir_augmented]:
    os.makedirs(d, exist_ok=True)

def load_and_normalize(filepath, target_sr=16000, target_rms_db=-20.0, peak_limit_db=-1.0):
    container = av.open(filepath)
    stream = container.streams.audio[0]
    sr_orig = stream.rate
    frames = [f.to_ndarray() for f in container.decode(stream)]
    container.close()
    
    audio = np.concatenate(frames, axis=-1)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0) # Dual-mono to Mono
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)
        
    # High-quality polyphase anti-aliasing resampling
    if sr_orig != target_sr:
        gcd = np.gcd(sr_orig, target_sr)
        up = target_sr // gcd
        down = sr_orig // gcd
        audio = scipy.signal.resample_poly(audio, up, down).astype(np.float32)
        
    # Remove DC offset
    audio = audio - np.mean(audio)
    
    # RMS Loudness Normalization
    current_rms = np.sqrt(np.mean(audio**2))
    target_rms = 10 ** (target_rms_db / 20.0)
    if current_rms > 1e-7:
        scale = target_rms / current_rms
        audio = audio * scale
        
    # Peak limit check to avoid clipping
    peak = np.max(np.abs(audio))
    peak_limit = 10 ** (peak_limit_db / 20.0)
    if peak > peak_limit:
        audio = audio * (peak_limit / peak)
        
    return audio, target_sr

def vad_trim(audio, sr=16000, frame_ms=30, hop_ms=10, energy_thresh_ratio=0.035):
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    n_frames = (len(audio) - frame_len) // hop_len + 1
    
    energies = np.array([
        np.sum(audio[i*hop_len : i*hop_len + frame_len]**2)
        for i in range(n_frames)
    ])
    
    max_e = np.max(energies)
    thresh = max_e * energy_thresh_ratio
    active = np.where(energies > thresh)[0]
    
    if len(active) == 0:
        return audio
        
    # Add 80ms buffer on both sides
    buf_frames = int(80 / hop_ms)
    s_frame = max(0, active[0] - buf_frames)
    e_frame = min(n_frames - 1, active[-1] + buf_frames)
    
    s_sample = s_frame * hop_len
    e_sample = min(len(audio), e_frame * hop_len + frame_len)
    
    return audio[s_sample:e_sample]

def extract_valid_segment(speech_audio, sr=16000, target_sec=3.5, fade_ms=20):
    target_samples = int(target_sec * sr)
    n_samples = len(speech_audio)
    
    if n_samples <= target_samples:
        segment = speech_audio.copy()
    else:
        # Pick the most energetic / central valid 3.5s segment
        # Using sliding window to find max energy 3.5s window within speech
        step = sr // 10 # 100ms step
        best_energy = -1.0
        best_start = 0
        for st in range(0, n_samples - target_samples, step):
            win = speech_audio[st : st + target_samples]
            e = np.sum(win**2)
            if e > best_energy:
                best_energy = e
                best_start = st
        segment = speech_audio[best_start : best_start + target_samples].copy()
        
    # Apply 20ms Hann fade-in and fade-out to guarantee zero-crossing & no pop clicks
    fade_len = int(sr * fade_ms / 1000)
    if len(segment) > 2 * fade_len:
        hann_in = 0.5 * (1 - np.cos(np.pi * np.arange(fade_len) / fade_len))
        hann_out = 0.5 * (1 + np.cos(np.pi * np.arange(fade_len) / fade_len))
        segment[:fade_len] *= hann_in
        segment[-fade_len:] *= hann_out
        
    return segment

# Augmentation functions
def add_noise(audio, noise_type="white", snr_db=15.0):
    sig_power = np.mean(audio**2)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    if noise_type == "white":
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)
    elif noise_type == "pink":
        # Approximate pink noise via 1/f filter on white noise
        white = np.random.normal(0, 1, len(audio))
        # Filter with IIR pinking filter (Paul Kellet's method)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        pink = scipy.signal.lfilter(b, a, white)
        pink = pink / (np.std(pink) + 1e-9) * np.sqrt(noise_power)
        noise = pink.astype(np.float32)
    else:
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)
    augmented = audio + noise
    # peak normalize if overflow
    if np.max(np.abs(augmented)) > 0.95:
        augmented = augmented * (0.95 / np.max(np.abs(augmented)))
    return augmented

def ssl_time_mask(audio, sr=16000, n_masks=2, mask_ms_range=(80, 150)):
    # Wav2Vec 2.0 / HuBERT / WavLM style temporal chunk masking
    aug = audio.copy()
    dur_samples = len(aug)
    for _ in range(n_masks):
        mask_len = int(sr * np.random.uniform(*mask_ms_range) / 1000)
        if dur_samples > mask_len:
            # avoid masking the exact first/last 100ms
            start = np.random.randint(int(sr*0.1), dur_samples - mask_len - int(sr*0.1))
            # Smooth mask edge
            edge = int(sr * 0.01) # 10ms edge
            aug[start + edge : start + mask_len - edge] = 0.0
            aug[start : start + edge] *= np.linspace(1, 0, edge)
            aug[start + mask_len - edge : start + mask_len] *= np.linspace(0, 1, edge)
    return aug

def ssl_bandpass_filter(audio, sr=16000, lowcut=300, highcut=3400):
    # Telephony / IoT Microphone bandwidth limitation
    sos = scipy.signal.butter(4, [lowcut, highcut], btype='bandpass', fs=sr, output='sos')
    filtered = scipy.signal.sosfilt(sos, audio).astype(np.float32)
    # maintain energy
    scale = (np.std(audio) + 1e-9) / (np.std(filtered) + 1e-9)
    return np.clip(filtered * scale, -0.95, 0.95)

def volume_jitter(audio, gain_db_range=(-3.0, 3.0)):
    gain_db = np.random.uniform(*gain_db_range)
    scale = 10 ** (gain_db / 20.0)
    jittered = audio * scale
    return np.clip(jittered, -0.95, 0.95)

def time_shift(audio, sr=16000, shift_ms_range=(-80, 80)):
    shift_samples = int(sr * np.random.uniform(*shift_ms_range) / 1000)
    shifted = np.roll(audio, shift_samples)
    if shift_samples > 0:
        shifted[:shift_samples] = 0.0
    elif shift_samples < 0:
        shifted[shift_samples:] = 0.0
    return shifted

print("==================== STARTING AUDIO NORMALIZATION & AUGMENTATION PIPELINE ====================")

summary_records = []

for fn in sorted(os.listdir(orig_dir)):
    if not fn.endswith('.m4a'):
        continue
    base_name = os.path.splitext(fn)[0]
    orig_path = os.path.join(orig_dir, fn)
    
    # 1. Standard Normalize
    norm_audio, sr = load_and_normalize(orig_path, target_sr=16000, target_rms_db=-20.0)
    norm_out_path = os.path.join(dir_normalized, f"{base_name}_normalized.wav")
    sf.write(norm_out_path, norm_audio, sr, subtype='PCM_16')
    
    # 2. VAD Trim and Extract 3.5s Valid Speech
    speech_audio = vad_trim(norm_audio, sr=sr)
    valid_segment = extract_valid_segment(speech_audio, sr=sr, target_sec=3.5, fade_ms=20)
    valid_out_path = os.path.join(dir_trimmed_valid, f"{base_name}_valid_3.5s.wav")
    sf.write(valid_out_path, valid_segment, sr, subtype='PCM_16')
    
    # Verify boundary click-free
    start_v = valid_segment[0]
    end_v = valid_segment[-1]
    
    # 3. Augmentations
    # (a) White noise SNR=15dB
    white_aug = add_noise(valid_segment, noise_type="white", snr_db=15.0)
    sf.write(os.path.join(dir_augmented, f"{base_name}_aug_white_snr15.wav"), white_aug, sr, subtype='PCM_16')
    
    # (b) Pink noise SNR=15dB (IoT fan/ambient low-frequency noise)
    pink_aug = add_noise(valid_segment, noise_type="pink", snr_db=15.0)
    sf.write(os.path.join(dir_augmented, f"{base_name}_aug_pink_snr15.wav"), pink_aug, sr, subtype='PCM_16')
    
    # (c) SSL Time Masking (Wav2Vec 2.0 / HuBERT Masking)
    ssl_mask_aug = ssl_time_mask(valid_segment, sr=sr, n_masks=2)
    sf.write(os.path.join(dir_augmented, f"{base_name}_aug_ssl_timemask.wav"), ssl_mask_aug, sr, subtype='PCM_16')
    
    # (d) SSL Telephony / Bandpass Filter (300Hz-3400Hz)
    bandpass_aug = ssl_bandpass_filter(valid_segment, sr=sr)
    sf.write(os.path.join(dir_augmented, f"{base_name}_aug_ssl_bandpass.wav"), bandpass_aug, sr, subtype='PCM_16')
    
    # (e) Volume Gain Jitter (+-3dB)
    gain_aug = volume_jitter(valid_segment, gain_db_range=(-3.0, 3.0))
    sf.write(os.path.join(dir_augmented, f"{base_name}_aug_volume_jitter.wav"), gain_aug, sr, subtype='PCM_16')
    
    # (f) Time Shift
    shift_aug = time_shift(valid_segment, sr=sr)
    sf.write(os.path.join(dir_augmented, f"{base_name}_aug_timeshift.wav"), shift_aug, sr, subtype='PCM_16')
    
    summary_records.append({
        'name': base_name,
        'orig_dur': len(norm_audio)/sr,
        'speech_dur': len(speech_audio)/sr,
        'seg_dur': len(valid_segment)/sr,
        'start_val': start_v,
        'end_val': end_v,
        'rms_db': 20 * np.log10(np.sqrt(np.mean(valid_segment**2))),
        'peak_db': 20 * np.log10(np.max(np.abs(valid_segment)))
    })
    print(f"Processed {base_name}: Orig={len(norm_audio)/sr:.2f}s -> Speech={len(speech_audio)/sr:.2f}s -> ValidSegment={len(valid_segment)/sr:.2f}s | StartVal={start_v:.5f}, EndVal={end_v:.5f}")

print("\nPipeline execution complete successfully!")
