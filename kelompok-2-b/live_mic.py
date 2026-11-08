import os
import sys
import time
import csv
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Pastikan encoding UTF-8 di Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
import win32com.client

# =====================================================================
#   KONFIGURASI UTAMA SISTEM
# =====================================================================
THRESHOLD_KERAS_SEC  = 10.0   # Kran KERAS > 10 detik berturut-turut -> peringatan
THRESHOLD_PELAN_SEC  = 20.0   # Kran PELAN > 20 detik berturut-turut -> peringatan
CHUNK_SEC            = 0.8    # Durasi setiap potongan audio yang dianalisis
BUFFER_SEC           = 2.0    # Panjang jendela analisis
SR                   = 22050  # Sample rate
TARGET_RMS           = 0.08   # Target normalisasi gain
MAX_GAIN             = 30.0   # Batas atas gain amplifikasi
# =====================================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "output", "model_aliran_air.pkl")
LOG_PATH   = os.path.join(BASE_DIR, "output", "log_peringatan.csv")

# Inisialisasi Text-to-Speech Windows
speaker = win32com.client.Dispatch("SAPI.SpVoice")

BANNER = r"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║    ███████╗ ██████╗███╗   ███╗ █████╗ ██████╗ ████████╗         ║
║    ██╔════╝██╔════╝████╗ ████║██╔══██╗██╔══██╗╚══██╔══╝         ║
║    ███████╗██║     ██╔████╔██║███████║██████╔╝   ██║            ║
║    ╚════██║██║     ██║╚██╔╝██║██╔══██║██╔══██╗   ██║            ║
║    ███████║╚██████╗██║ ╚═╝ ██║██║  ██║██║  ██║   ██║            ║
║    ╚══════╝ ╚═════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝            ║
║                                                                  ║
║        SISTEM CERDAS HEMAT AIR WUDHU - MASJID                    ║
║        Berbasis Machine Learning & Klasifikasi Audio             ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Kelas yang Dikenali:                                            ║
║   [0] Suara Ngobrol / Kebisingan  ->  Tidak memicu alarm         ║
║   [1] Kran Pelan / Normal         ->  Hemat air                  ║
║   [2] Kran Keras / Deras          ->  Boros, pantau terus        ║
║                                                                  ║
║  Peringatan dipicu jika:                                         ║
║   - Kran KERAS  > 10 detik berturut-turut -> minta kecilkan      ║
║   - Kran PELAN  > 20 detik berturut-turut -> minta matikan       ║
╚══════════════════════════════════════════════════════════════════╝
"""

def speak_warning(message):
    """Mengeluarkan output suara peringatan melalui speaker laptop"""
    try:
        speaker.Speak(message)
    except Exception as e:
        print(f"[Audio Error]: {e}")

def log_peringatan(jenis, durasi, pesan):
    """Mencatat setiap peringatan ke file CSV log"""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file_baru = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if file_baru:
            writer.writerow(["Waktu", "Jenis Peringatan", "Durasi (detik)", "Pesan"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), jenis, f"{durasi:.1f}", pesan])

def create_volume_meter(rms, max_scale=0.03, bar_length=15):
    """Membuat visualisasi bar volume di terminal"""
    ratio = min(1.0, max(0.0, rms / max_scale))
    filled = int(ratio * bar_length)
    return "█" * filled + "░" * (bar_length - filled)

def extract_features(seg, sr=SR):
    """Ekstrak semua fitur audio yang konsisten dengan train.py."""
    rms_frames = librosa.feature.rms(y=seg)
    rms_val = float(np.mean(rms_frames))
    spec_cent = float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)))
    spec_bw = float(np.mean(librosa.feature.spectral_bandwidth(y=seg, sr=sr)))
    spec_ro85 = float(np.mean(librosa.feature.spectral_rolloff(y=seg, sr=sr, roll_percent=0.85)))
    spec_ro25 = float(np.mean(librosa.feature.spectral_rolloff(y=seg, sr=sr, roll_percent=0.25)))
    spec_flat = float(np.mean(librosa.feature.spectral_flatness(y=seg)))
    zcr_frames = librosa.feature.zero_crossing_rate(y=seg)
    zcr_val = float(np.mean(zcr_frames))
    zcr_std = float(np.std(zcr_frames))

    y_harm, _ = librosa.effects.hpss(seg, margin=(2.0, 1.0))
    harm_rms = float(np.mean(librosa.feature.rms(y=y_harm)))
    harm_ratio = float(harm_rms / (rms_val + 1e-9))

    S = np.abs(librosa.stft(seg))
    crest = float(np.mean(np.max(S, axis=0) / (np.mean(S, axis=0) + 1e-9)))
    flux = float(np.mean(librosa.onset.onset_strength(y=seg, sr=sr)))

    freqs = librosa.fft_frequencies(sr=sr)
    low_idx = np.where(freqs < 250)[0]
    mid_idx = np.where((freqs >= 250) & (freqs < 3000))[0]
    high_idx = np.where(freqs >= 3000)[0]

    low_e = float(np.sum(S[low_idx, :]**2))
    mid_e = float(np.sum(S[mid_idx, :]**2))
    high_e = float(np.sum(S[high_idx, :]**2))
    tot_e = low_e + mid_e + high_e + 1e-9

    row = {
        'rms': rms_val,
        'spectral_centroid': spec_cent,
        'spectral_bandwidth': spec_bw,
        'spectral_rolloff': spec_ro85,
        'spectral_rolloff_25': spec_ro25,
        'spectral_flatness': spec_flat,
        'zcr': zcr_val,
        'zcr_std': zcr_std,
        'harm_ratio': harm_ratio,
        'crest': crest,
        'flux': flux,
        'low_ratio': float(low_e / tot_e),
        'mid_ratio': float(mid_e / tot_e),
        'high_ratio': float(high_e / tot_e),
    }
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)
    contrast = librosa.feature.spectral_contrast(y=seg, sr=sr)
    for i in range(20):
        row[f'mfcc_mean_{i+1}'] = float(np.mean(mfcc[i]))
        row[f'mfcc_std_{i+1}'] = float(np.std(mfcc[i]))
        row[f'mfcc_delta_{i+1}'] = float(np.mean(mfcc_delta[i]))
    for i in range(contrast.shape[0]):
        row[f'contrast_mean_{i+1}'] = float(np.mean(contrast[i]))
    return row

def main():
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] File model belum ditemukan! Jalankan train.py terlebih dahulu.")
        return

    checkpoint   = joblib.load(MODEL_PATH)
    model        = checkpoint["model"]
    feature_cols = checkpoint["feature_cols"]

    print(BANNER)

    # Kalibrasi noise ruangan
    print("[*] Mengkalibrasi kebisingan ruangan selama 2 detik... Harap diam sejenak.")
    calib_audio  = sd.rec(int(2.0 * SR), samplerate=SR, channels=1, dtype='float32')
    sd.wait()
    ambient_rms     = float(np.sqrt(np.mean(calib_audio.flatten()**2)))
    noise_threshold = max(0.0015, ambient_rms * 1.3)
    print(f"[*] Kalibrasi selesai! Noise dasar: {ambient_rms:.5f} | Threshold: {noise_threshold:.5f}")
    print(f"[*] Model aktif: {os.path.basename(MODEL_PATH)}")
    print(f"[*] Log peringatan: {LOG_PATH}")
    print(f"\n[SISTEM AKTIF] Mendengarkan mikrofon...")
    print(f"[INFO] Kran KERAS  > {THRESHOLD_KERAS_SEC:.0f} detik -> peringatan kencang")
    print(f"[INFO] Kran PELAN  > {THRESHOLD_PELAN_SEC:.0f} detik -> peringatan boros")
    print("Tekan Ctrl+C untuk menghentikan.\n" + "─" * 70)

    BUFFER_SIZE = int(BUFFER_SEC * SR)
    audio_buffer          = np.zeros(BUFFER_SIZE, dtype=np.float32)
    durasi_keras_berjalan = 0.0
    durasi_pelan_berjalan = 0.0
    peringatan_ke         = 0

    chunk_samples = int(CHUNK_SEC * SR)
    stream = sd.InputStream(samplerate=SR, channels=1, blocksize=chunk_samples)
    stream.start()

    try:
        while True:
            new_chunk, _ = stream.read(chunk_samples)
            new_chunk = new_chunk.flatten()

            # Geser buffer dan isi bagian baru
            audio_buffer = np.roll(audio_buffer, -len(new_chunk))
            audio_buffer[-len(new_chunk):] = new_chunk

            seg     = audio_buffer.copy()
            raw_rms = float(np.sqrt(np.mean(seg**2)))
            meter   = create_volume_meter(raw_rms, max_scale=0.03, bar_length=12)
            waktu   = datetime.now().strftime("%H:%M:%S")

            # Jika hening (di bawah threshold noise) → reset pelan & keras
            if raw_rms < noise_threshold:
                durasi_keras_berjalan = max(0.0, durasi_keras_berjalan - 0.8)
                durasi_pelan_berjalan = max(0.0, durasi_pelan_berjalan - 0.8)
                print(f"[{waktu}] [Vol:{raw_rms:.4f}] [{meter}] -- HENING / KRAN MATI --")
                continue

            # Auto-gain normalisasi (mengatasi noise cancellation Intel mic)
            gain           = min(MAX_GAIN, TARGET_RMS / (raw_rms + 1e-6))
            normalized_seg = seg * gain

            # Ekstraksi fitur & prediksi
            row_dict = extract_features(normalized_seg, sr=SR)
            feats    = [row_dict.get(col, 0.0) for col in feature_cols]
            X_df     = pd.DataFrame([feats], columns=feature_cols)

            pred = model.predict(X_df.values)[0]
            prob = model.predict_proba(X_df.values)[0]

            # ── Kelas 0: Bukan air (ngobrol/kebisingan) ──
            if pred == 0:
                durasi_keras_berjalan = max(0.0, durasi_keras_berjalan - 0.8)
                durasi_pelan_berjalan = max(0.0, durasi_pelan_berjalan - 0.8)
                print(f"[{waktu}] [Vol:{raw_rms:.4f}] [{meter}] [BUKAN AIR {prob[0]*100:.0f}%] Ngobrol/Kebisingan")

            # ── Kelas 1: Kran pelan / hemat ──
            elif pred == 1:
                durasi_keras_berjalan = max(0.0, durasi_keras_berjalan - 0.4)
                durasi_pelan_berjalan += CHUNK_SEC
                print(f"[{waktu}] [Vol:{raw_rms:.4f}] [{meter}] [KRAN PELAN {prob[1]*100:.0f}%] Pelan:{durasi_pelan_berjalan:.0f}s")

            # ── Kelas 2: Kran keras / deras ──
            elif pred == 2:
                durasi_keras_berjalan += CHUNK_SEC
                durasi_pelan_berjalan = max(0.0, durasi_pelan_berjalan - 0.4)
                print(f"[{waktu}] [Vol:{raw_rms:.4f}] [{meter}] [KRAN KERAS {prob[2]*100:.0f}%] Keras:{durasi_keras_berjalan:.1f}s <<<")

            # ══════════════════════════════════════════════════════════
            #   PERINGATAN 1: KRAN KERAS > 10 DETIK
            # ══════════════════════════════════════════════════════════
            if durasi_keras_berjalan >= THRESHOLD_KERAS_SEC:
                peringatan_ke += 1
                print("\n" + "!" * 70)
                print(f"  !!! PERINGATAN #{peringatan_ke} - KRAN TERLALU KERAS/DERAS !!!")
                print(f"  Kran keras sudah {durasi_keras_berjalan:.1f} detik (batas: {THRESHOLD_KERAS_SEC:.0f} detik)")
                print(f"  Waktu  : {datetime.now().strftime('%H:%M:%S')}")
                print(f"  Pesan  : TOLONG KECILKAN KRAN AIR!")
                print(f"  Audio  : Menyuarakan peringatan...")
                print("!" * 70 + "\n")
                log_peringatan("KRAN KERAS", durasi_keras_berjalan, "Kran terlalu deras, tolong kecilkan")
                speak_warning("Peringatan, kran terlalu deras, tolong kecilkan kran")
                durasi_keras_berjalan = 0.0  # Reset setelah peringatan

            # ══════════════════════════════════════════════════════════
            #   PERINGATAN 2: KRAN PELAN > 20 DETIK
            # ══════════════════════════════════════════════════════════
            elif durasi_pelan_berjalan >= THRESHOLD_PELAN_SEC:
                peringatan_ke += 1
                print("\n" + "~" * 70)
                print(f"  ~~~ PERINGATAN #{peringatan_ke} - KRAN PELAN TERLALU LAMA ~~~")
                print(f"  Kran pelan sudah mengalir {durasi_pelan_berjalan:.0f} detik (batas: {THRESHOLD_PELAN_SEC:.0f} detik)")
                print(f"  Waktu  : {datetime.now().strftime('%H:%M:%S')}")
                print(f"  Pesan  : WUDHU SUDAH SELESAI, HARAP MATIKAN KRAN!")
                print(f"  Audio  : Menyuarakan peringatan...")
                print("~" * 70 + "\n")
                log_peringatan("KRAN PELAN LAMA", durasi_pelan_berjalan, "Kran pelan terlalu lama, harap matikan")
                speak_warning("Peringatan, kran sudah mengalir terlalu lama, harap matikan kran")
                durasi_pelan_berjalan = 0.0  # Reset setelah peringatan

    except KeyboardInterrupt:
        print("\n\n[*] Sistem monitoring dihentikan oleh pengguna.")
        print(f"[*] Total peringatan yang dipicu: {peringatan_ke}x")
        if os.path.exists(LOG_PATH):
            print(f"[*] Log peringatan tersimpan di: {LOG_PATH}")
    finally:
        stream.stop()
        stream.close()

if __name__ == "__main__":
    main()
