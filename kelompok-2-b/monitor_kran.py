import os
import sys
import time
import csv
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Pastikan output terminal mendukung UTF-8 di Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
import librosa
import win32com.client

# =====================================================================
#   KONFIGURASI UTAMA - UBAH NILAI INI SESUAI KEBUTUHAN
# =====================================================================
THRESHOLD_KERAS_SEC = 10.0   # Kran deras > 10 detik → peringatan kecilkan kran!
THRESHOLD_PELAN_SEC = 20.0   # Kran pelan > 20 detik → peringatan matikan kran!
WIN_SEC             = 2.0    # Panjang jendela analisis (detik)
HOP_SEC             = 1.0    # Lompatan antar jendela (detik)
SR                  = 22050  # Sample rate
# =====================================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "output", "model_aliran_air.pkl")
LOG_PATH   = os.path.join(BASE_DIR, "output", "log_peringatan.csv")

# Inisialisasi Text-to-Speech bawaan Windows
speaker = win32com.client.Dispatch("SAPI.SpVoice")

def speak_warning(message="Tolong hemat air wudhu"):
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
            writer.writerow(["Waktu", "Jenis Peringatan", "Durasi (detik)", "Pesan", "File Audio"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            jenis, f"{durasi:.1f}", pesan, ""
        ])

def extract_features_for_model(seg, sr=SR):
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

    row_dict = {
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
        row_dict[f'mfcc_mean_{i+1}'] = float(np.mean(mfcc[i]))
        row_dict[f'mfcc_std_{i+1}'] = float(np.std(mfcc[i]))
        row_dict[f'mfcc_delta_{i+1}'] = float(np.mean(mfcc_delta[i]))
    for i in range(contrast.shape[0]):
        row_dict[f'contrast_mean_{i+1}'] = float(np.mean(contrast[i]))
    return row_dict


def monitor_water_stream(audio_path, threshold_keras=THRESHOLD_KERAS_SEC, threshold_pelan=THRESHOLD_PELAN_SEC):
    """
    Simulasi monitoring kran air secara temporal (per detik):
    - Jika kran KERAS > threshold_keras (10s), memicu peringatan kecilkan kran.
    - Jika kran PELAN > threshold_pelan (20s), memicu peringatan matikan kran.
    """
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Model belum dilatih! Jalankan train.py terlebih dahulu.")
        return

    checkpoint   = joblib.load(MODEL_PATH)
    model        = checkpoint["model"]
    feature_cols = checkpoint["feature_cols"]

    print("=" * 70)
    print("  SISTEM CERDAS MONITORING PENGHEMATAN AIR WUDHU MASJID")
    print(f"  Target File     : {os.path.basename(audio_path)}")
    print(f"  Batas Kran Keras: {threshold_keras:.0f} detik (minta kecilkan)")
    print(f"  Batas Kran Pelan: {threshold_pelan:.0f} detik (minta matikan)")
    print(f"  Log Peringatan  : {LOG_PATH}")
    print("=" * 70)

    # Load audio
    y, sr       = librosa.load(audio_path, sr=SR)
    durasi_file = librosa.get_duration(y=y, sr=sr)
    print(f"[*] Durasi audio: {durasi_file:.1f} detik. Memulai simulasi streaming real-time...\n")

    win_samples = int(WIN_SEC * sr)
    hop_samples = int(HOP_SEC * sr)

    durasi_keras_berjalan = 0.0
    durasi_pelan_berjalan = 0.0
    durasi_total_kran     = 0.0
    total_air_ml          = 0.0
    warning_count         = 0
    current_sec           = 1

    for start in range(0, len(y) - win_samples + 1, hop_samples):
        seg = y[start : start + win_samples]

        row_dict = extract_features_for_model(seg, sr=sr)
        feats = [row_dict.get(col, 0.0) for col in feature_cols]
        X_df  = pd.DataFrame([feats], columns=feature_cols)

        pred = model.predict(X_df.values)[0]
        prob = model.predict_proba(X_df.values)[0]

        # Interpretasi hasil prediksi
        if pred == 0:  # Bukan air
            durasi_keras_berjalan = max(0.0, durasi_keras_berjalan - 0.5)
            durasi_pelan_berjalan = max(0.0, durasi_pelan_berjalan - 0.5)
            status = f"[BUKAN AIR {prob[0]*100:.1f}%] Suara ngobrol / kebisingan (diabaikan)"
        elif pred == 2:  # Kran keras
            durasi_keras_berjalan += HOP_SEC
            durasi_pelan_berjalan = max(0.0, durasi_pelan_berjalan - 0.5)
            durasi_total_kran     += HOP_SEC
            total_air_ml          += 120.0  # ~120 mL/detik untuk kran deras
            status = f"[KRAN KERAS {prob[2]*100:.1f}%] Keras: {durasi_keras_berjalan:.0f}s | Total: {durasi_total_kran:.0f}s"
        else:  # Kran pelan (pred == 1)
            durasi_pelan_berjalan += HOP_SEC
            durasi_keras_berjalan = max(0.0, durasi_keras_berjalan - 0.5)
            durasi_total_kran     += HOP_SEC
            total_air_ml          += 40.0  # ~40 mL/detik untuk kran pelan
            status = f"[KRAN PELAN {prob[1]*100:.1f}%] Pelan: {durasi_pelan_berjalan:.0f}s | Total: {durasi_total_kran:.0f}s"

        print(f"  Detik ke-{current_sec:03d} | {status} | Estimasi: {total_air_ml:.0f} mL")

        # Cek peringatan kran keras > threshold
        if durasi_keras_berjalan >= threshold_keras:
            warning_count += 1
            print("\n" + "!" * 70)
            print(f"  !!! PERINGATAN #{warning_count} - KRAN TERLALU KERAS/DERAS !!!")
            print(f"  Kran keras sudah berjalan {durasi_keras_berjalan:.0f} detik (batas: {threshold_keras:.0f} detik)!")
            print(f"  Estimasi air terbuang: {total_air_ml:.0f} mL")
            print(f"  Pesan Sistem: TOLONG KECILKAN KRAN AIR!")
            print(f"  Audio Aktif : Menyuarakan peringatan...")
            print("!" * 70 + "\n")
            log_peringatan("KRAN KERAS", durasi_keras_berjalan, "Kran terlalu deras, tolong kecilkan")
            speak_warning("Peringatan, kran terlalu deras, tolong kecilkan kran")
            durasi_keras_berjalan = 0.0  # Reset agar tidak spam

        # Cek peringatan kran pelan > threshold
        elif durasi_pelan_berjalan >= threshold_pelan:
            warning_count += 1
            print("\n" + "~" * 70)
            print(f"  ~~~ PERINGATAN #{warning_count} - KRAN PELAN TERLALU LAMA ~~~")
            print(f"  Kran pelan sudah berjalan {durasi_pelan_berjalan:.0f} detik (batas: {threshold_pelan:.0f} detik)!")
            print(f"  Estimasi air terpakai: {total_air_ml:.0f} mL")
            print(f"  Pesan Sistem: WUDHU SUDAH SELESAI, HARAP MATIKAN KRAN!")
            print(f"  Audio Aktif : Menyuarakan peringatan...")
            print("~" * 70 + "\n")
            log_peringatan("KRAN PELAN LAMA", durasi_pelan_berjalan, "Kran pelan terlalu lama, harap matikan")
            speak_warning("Peringatan, kran sudah mengalir terlalu lama, harap matikan kran")
            durasi_pelan_berjalan = 0.0

        # Simulasi delay real-time
        time.sleep(0.3)
        current_sec += 1

    # Rekap akhir
    print("\n" + "=" * 70)
    print("  REKAP AKHIR PENGGUNAAN AIR:")
    print(f"  Total Durasi Analisis: {durasi_file:.1f} detik")
    print(f"  Estimasi Air Terpakai: {total_air_ml / 1000:.2f} Liter")
    print(f"  Total Peringatan     : {warning_count}x")
    if warning_count > 0:
        print("  Status Konservasi    : TERDETEKSI PEMBOROSAN (Peringatan Diberikan)")
    else:
        print("  Status Konservasi    : BAIK / HEMAT AIR (Tidak Melampaui Batas)")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = input("Masukkan path file audio (.mp3/.wav): ").strip().strip('"')

    if os.path.exists(audio_file):
        monitor_water_stream(audio_file, threshold_keras=THRESHOLD_KERAS_SEC, threshold_pelan=THRESHOLD_PELAN_SEC)
    else:
        print(f"[ERROR] File '{audio_file}' tidak ditemukan!")
