import os
import sys
import time
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import soundfile as sf
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'dataset_audio_kaggle')
OUTPUT_DIR = os.path.join(BASE_DIR, 'hasil_ekstraksi')
CSV_EKSTRAKSI = os.path.join(OUTPUT_DIR, 'file_ekstraksi.csv')
CSV_SELEKSI = os.path.join(OUTPUT_DIR, 'file_seleksi_fitur.csv')

def ekstrak_fitur_segmen(y_seg, sr, f0_seg):
    """Mengekstrak 48 fitur akustik Librosa dari 1 segmen audio berdurasi 5 detik."""
    rms = librosa.feature.rms(y=y_seg)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y_seg)[0]
    silence_ratio = float(np.mean(rms < 0.01) * 100.0)
    skew_val = float(stats.skew(y_seg))
    kurt_val = float(stats.kurtosis(y_seg))

    S = np.abs(librosa.stft(y_seg))
    centroid  = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff   = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    flatness  = librosa.feature.spectral_flatness(S=S)[0]
    contrast  = librosa.feature.spectral_contrast(S=S, sr=sr)

    voiced_mask = rms[:len(f0_seg)] > 0.01
    f0_voiced = f0_seg[voiced_mask] if np.any(voiced_mask) else f0_seg
    f0_mean   = float(np.mean(f0_voiced))
    f0_std    = float(np.std(f0_voiced))
    f0_jitter = float(np.mean(np.abs(np.diff(f0_voiced)))) if len(f0_voiced) > 1 else 0.0

    mel_S = librosa.feature.melspectrogram(S=S**2, sr=sr)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_S), sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)

    fitur = {
        'rms_mean': float(np.mean(rms)),
        'rms_std': float(np.std(rms)),
        'zcr_mean': float(np.mean(zcr)),
        'zcr_std': float(np.std(zcr)),
        'silence_ratio': silence_ratio,
        'skewness': skew_val,
        'kurtosis': kurt_val,
        'centroid_mean': float(np.mean(centroid)),
        'centroid_std': float(np.std(centroid)),
        'bandwidth_mean': float(np.mean(bandwidth)),
        'bandwidth_std': float(np.std(bandwidth)),
        'rolloff_mean': float(np.mean(rolloff)),
        'rolloff_std': float(np.std(rolloff)),
        'flatness_mean': float(np.mean(flatness)),
        'flatness_std': float(np.std(flatness)),
        'contrast_mean': float(np.mean(contrast)),
        'contrast_std': float(np.std(contrast)),
        'f0_mean': f0_mean,
        'f0_std': f0_std,
        'f0_jitter': f0_jitter,
        'delta_mfcc_mean': float(np.mean(delta_mfcc)),
        'delta_mfcc_std': float(np.std(delta_mfcc)),
    }
    for i in range(13):
        fitur[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc[i]))
        fitur[f'mfcc_{i+1}_std']  = float(np.std(mfcc[i]))
    return fitur

def main():
    print('=== MEMULAI PENGOLAHAN DATASET & EKSTRAKSI FITUR ===')
    start_all = time.time()

    # 1. Baca data ekstraksi yang sudah ada jika ada
    if os.path.exists(CSV_EKSTRAKSI):
        df_existing = pd.read_csv(CSV_EKSTRAKSI)
        print(f'Memuat file ekstraksi yang sudah ada: {len(df_existing)} baris')
        existing_keys = set(zip(df_existing['kelas'], df_existing['file_induk']))
    else:
        df_existing = pd.DataFrame()
        existing_keys = set()

    # 2. Cari file yang belum diekstraksi
    kelas_config = [
        ('asli_human', 0),
        ('ai_manipulasi', 1)
    ]
    
    files_to_process = []
    for kelas_nama, label_id in kelas_config:
        folder_path = os.path.join(DATA_DIR, kelas_nama)
        if not os.path.exists(folder_path):
            continue
        all_wavs = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.wav')])
        for fname in all_wavs:
            if (kelas_nama, fname) not in existing_keys:
                files_to_process.append((kelas_nama, label_id, fname, os.path.join(folder_path, fname)))

    print(f'Total file baru yang perlu diekstraksi: {len(files_to_process)} file')

    new_rows = []
    TARGET_SR = 16000
    SAMPEL_5S = 5 * TARGET_SR

    # Warm up JIT for librosa.yin
    dummy_audio = np.random.randn(SAMPEL_5S).astype(np.float32)
    _ = librosa.yin(dummy_audio, fmin=60, fmax=360, sr=TARGET_SR, hop_length=1024)
    print('Numba JIT warm-up selesai.')

    count = 0
    t_start_files = time.time()
    for kelas_nama, label_id, fname, fpath in files_to_process:
        count += 1
        y_audio, sr = sf.read(fpath, dtype='float32')
        if y_audio.ndim > 1:
            y_audio = np.mean(y_audio, axis=1)

        # Hitung jumlah segmen 5 detik (maks 6 segmen / 30 detik)
        num_segs = max(1, min(6, int(len(y_audio) // SAMPEL_5S)))
        kode_file = fname.replace('.wav', '')

        for seg_idx in range(num_segs):
            y_seg = y_audio[seg_idx * SAMPEL_5S : (seg_idx + 1) * SAMPEL_5S]
            f0_seg = librosa.yin(y_seg, fmin=60, fmax=360, sr=TARGET_SR, hop_length=1024)
            fitur_seg = ekstrak_fitur_segmen(y_seg, TARGET_SR, f0_seg)

            row = {
                'id_sampel': f'{kelas_nama}_{kode_file}_seg{seg_idx+1}',
                'file_induk': fname,
                'segmen_ke': seg_idx + 1,
                'durasi_detik': 5,
                'kelas': kelas_nama,
                'label': label_id
            }
            row.update(fitur_seg)
            new_rows.append(row)

        if count % 20 == 0 or count == len(files_to_process):
            elapsed = round(time.time() - t_start_files, 1)
            print(f'Proses: {count}/{len(files_to_process)} file ({len(new_rows)} segmen terekstrak) [{elapsed}s]')

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        if not df_existing.empty:
            df_total = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_total = df_new
    else:
        df_total = df_existing

    # Simpan kembali file_ekstraksi.csv
    df_total.to_csv(CSV_EKSTRAKSI, index=False)
    print(f'\nDataset Ekstraksi berhasil disimpan ke: {CSV_EKSTRAKSI}')
    print(f'Total baris dataset ekstraksi: {len(df_total)} baris')
    print('Distribusi kelas:')
    print(df_total['kelas'].value_counts())

    # 3. SELEKSI FITUR (Top-20 Features menggunakan Random Forest Importance & Group Split)
    print('\n=== MELAKUKAN SELEKSI FITUR (TOP-20) ===')
    kolom_meta = ['id_sampel', 'file_induk', 'segmen_ke', 'durasi_detik', 'kelas', 'label']
    kolom_48 = [c for c in df_total.columns if c not in kolom_meta]

    # Group split berdasarkan file induk untuk evaluasi importance
    daftar_file_unik = sorted(df_total['file_induk'].unique())
    train_files, _ = train_test_split(daftar_file_unik, test_size=0.20, random_state=42)

    df_train_raw = df_total[df_total['file_induk'].isin(train_files)].copy()
    scaler_48 = StandardScaler()
    X_tr_48 = scaler_48.fit_transform(df_train_raw[kolom_48].values)
    y_tr_48 = df_train_raw['label'].values

    rf_ranker = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_ranker.fit(X_tr_48, y_tr_48)

    df_rank = pd.DataFrame({
        'Nama Fitur': kolom_48,
        'Importance (%)': rf_ranker.feature_importances_ * 100
    }).sort_values('Importance (%)', ascending=False).reset_index(drop=True)

    JUMLAH_FITUR_TERPILIH = 20
    fitur_terpilih = df_rank['Nama Fitur'].head(JUMLAH_FITUR_TERPILIH).tolist()

    print(f'Top-{JUMLAH_FITUR_TERPILIH} Fitur Terpilih (Total Bobot: {df_rank["Importance (%)"].head(JUMLAH_FITUR_TERPILIH).sum():.2f}%):')
    for idx, r in df_rank.head(JUMLAH_FITUR_TERPILIH).iterrows():
        print(f"  {idx+1:2d}. {r['Nama Fitur']:<20} : {r['Importance (%)']:.2f}%")

    # Simpan dataset hasil seleksi fitur
    df_seleksi = df_total[kolom_meta + fitur_terpilih].copy()
    df_seleksi.to_csv(CSV_SELEKSI, index=False)
    print(f'\nDataset hasil seleksi fitur berhasil disimpan ke: {CSV_SELEKSI}')
    print(f'Dimensi dataset seleksi: {df_seleksi.shape} (Berkurang dari {df_total.shape[1]} kolom menjadi {df_seleksi.shape[1]} kolom)')
    print(f'Selesai dalam {round(time.time() - start_all, 1)} detik.')

if __name__ == '__main__':
    main()
