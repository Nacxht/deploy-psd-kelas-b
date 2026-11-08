import os
import sys
import numpy as np
import soundfile as sf
import librosa
import scipy.stats as stats
import pandas as pd
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Tambahkan direktori saat ini ke sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)

from cnn_architecture import AudioCNN1D, CNNVoiceClassifier

csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'hasil_ekstraksi', 'file_seleksi_fitur.csv'))
df = pd.read_csv(csv_path)

kolom_meta = ['id_sampel', 'file_induk', 'segmen_ke', 'durasi_detik', 'kelas', 'label']
kolom_fitur = [
    'mfcc_8_mean', 'mfcc_11_mean', 'mfcc_7_mean', 'mfcc_8_std', 'mfcc_4_mean',
    'mfcc_5_mean', 'mfcc_6_std', 'skewness', 'mfcc_4_std', 'mfcc_11_std',
    'mfcc_1_mean', 'delta_mfcc_std', 'mfcc_12_mean', 'f0_std', 'mfcc_13_std',
    'mfcc_10_mean', 'mfcc_13_mean', 'mfcc_7_std', 'contrast_mean', 'mfcc_3_mean'
]

def ekstrak_fitur_segmen(y_seg, sr, f0_seg):
    """Ekstraksi fitur Librosa untuk 1 segmen 5 detik."""
    rms = librosa.feature.rms(y=y_seg)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y_seg)[0]
    silence_ratio = float(np.mean(rms < 0.01) * 100.0)
    skew_val = float(stats.skew(y_seg))
    kurt_val = float(stats.kurtosis(y_seg))

    S = np.abs(librosa.stft(y_seg))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

    voiced_mask = rms[:len(f0_seg)] > 0.01
    f0_voiced = f0_seg[voiced_mask] if np.any(voiced_mask) else f0_seg
    f0_mean = float(np.mean(f0_voiced))
    f0_std = float(np.std(f0_voiced))
    f0_jitter = float(np.mean(np.abs(np.diff(f0_voiced)))) if len(f0_voiced) > 1 else 0.0

    mel_S = librosa.feature.melspectrogram(S=S**2, sr=sr)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_S), sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)

    fitur = {
        'skewness': skew_val,
        'contrast_mean': float(np.mean(contrast)),
        'f0_std': f0_std,
        'delta_mfcc_std': float(np.std(delta_mfcc))
    }
    for i in range(13):
        fitur[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc[i]))
        fitur[f'mfcc_{i+1}_std'] = float(np.std(mfcc[i]))
    return fitur


def sinkronisasi_fitur_baru():
    """Memeriksa folder audio dan mengekstrak fitur HANYA untuk file baru."""
    if os.path.exists(CSV_SELEKSI):
        df_csv = pd.read_csv(CSV_SELEKSI)
        existing_pairs = set(zip(df_csv['kelas'], df_csv['file_induk']))
    else:
        df_csv = pd.DataFrame(columns=kolom_meta + kolom_fitur)
        existing_pairs = set()

    kelas_map = {
        'asli_human': 0,
        'ai_manipulasi': 1
    }

    new_rows = []
    print('=== MEMERIKSA DATASET AUDIO ===')
    for kelas_nama, label_id in kelas_map.items():
        folder = os.path.join(DATA_DIR, kelas_nama)
        if not os.path.exists(folder):
            continue
        daftar_wav = sorted([f for f in os.listdir(folder) if f.endswith('.wav')])
        file_baru = [f for f in daftar_wav if (kelas_nama, f) not in existing_pairs]
        print(f'Folder {kelas_nama}: Total {len(daftar_wav)} file | Ditemukan {len(file_baru)} file baru belum diekstrak.')

        for fname in file_baru:
            fpath = os.path.join(folder, fname)
            try:
                y, sr = sf.read(fpath, dtype='float32')
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                if sr != 16000:
                    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                    sr = 16000

                sampel_5s = 5 * 16000
                jml_seg = min(6, max(1, len(y) // sampel_5s))

                for s in range(jml_seg):
                    y_seg = y[s * sampel_5s : (s + 1) * sampel_5s]
                    f0_seg = librosa.yin(y_seg, fmin=50, fmax=400, sr=16000)
                    d = ekstrak_fitur_segmen(y_seg, 16000, f0_seg)

                    row = {
                        'id_sampel': f'{kelas_nama}_{os.path.splitext(fname)[0]}_seg{s+1}',
                        'file_induk': fname,
                        'segmen_ke': s + 1,
                        'durasi_detik': 5.0,
                        'kelas': kelas_nama,
                        'label': label_id
                    }
                    for k in kolom_fitur:
                        row[k] = d[k]
                    new_rows.append(row)
                print(f'  [+] Ekstraksi {fname} ({jml_seg} segmen @ 5s) -> Selesai.')
            except Exception as e:
                print(f'  [!] Gagal memproses {fname}: {e}')

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_combined = pd.concat([df_csv, df_new], ignore_index=True)
        os.makedirs(os.path.dirname(CSV_SELEKSI), exist_ok=True)
        df_combined.to_csv(CSV_SELEKSI, index=False)
        # Salin juga ke data/hasil_ekstraksi jika ada
        alt_csv = os.path.join(BASE_DIR, 'data', 'hasil_ekstraksi', 'file_seleksi_fitur.csv')
        if os.path.exists(os.path.dirname(alt_csv)):
            df_combined.to_csv(alt_csv, index=False)
        print(f'\nDataset berhasil diperbarui! Ditambahkan {len(new_rows)} baris segmen baru.')
        print(f'Total baris dataset sekarang: {len(df_combined)}')
        return df_combined
    else:
        print('\nSeluruh file audio sudah sinkron dengan CSV (tidak ada file baru).')
        return df_csv


def latih_model_cnn(df):
    """Melatih model 1D-CNN menggunakan dataset lengkap."""
    print('\n=== MEMULAI PELATIHAN MODEL DEEP LEARNING 1D-CNN ===')
    unique_files = sorted(df['file_induk'].unique())
    train_files, test_files = train_test_split(unique_files, test_size=0.20, random_state=42)

    df_train = df[df['file_induk'].isin(train_files)].copy()
    df_test = df[df['file_induk'].isin(test_files)].copy()

    print(f'Pembagian Data Bebas Kebocoran: {len(train_files)} file latih ({len(df_train)} segmen) | {len(test_files)} file uji ({len(df_test)} segmen)')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[kolom_fitur].values)
    y_train = df_train['label'].values
    X_test = scaler.transform(df_test[kolom_fitur].values)
    y_test = df_test['label'].values

    torch.manual_seed(42)
    np.random.seed(42)

    cnn = CNNVoiceClassifier(in_features=len(kolom_fitur), num_classes=2, lr=0.003, epochs=50)
    cnn.fit(X_train, y_train)

    preds = cnn.predict(X_test)
    probs = cnn.predict_proba(X_test)

    acc = accuracy_score(y_test, preds) * 100
    prec = precision_score(y_test, preds) * 100
    rec = recall_score(y_test, preds) * 100
    f1 = f1_score(y_test, preds) * 100

    metrics = {
        'accuracy': round(acc, 2),
        'precision': round(prec, 2),
        'recall': round(rec, 2),
        'f1_score': round(f1, 2)
    }

    print('\n=== HASIL EVALUASI MODEL BARU ===')
    print(f'Akurasi Data Uji : {acc:.2f}%')
    print(f'F1-Score         : {f1:.2f}%')
    print(f'Presisi          : {prec:.2f}%')
    print(f'Recall           : {rec:.2f}%')
    print('\nConfusion Matrix:\n', confusion_matrix(y_test, preds))

    mean_asli = df[df['label'] == 0][kolom_fitur].mean().to_dict()
    mean_ai = df[df['label'] == 1][kolom_fitur].mean().to_dict()

    paket_cnn = {
        'model_name': 'AudioCNN1D (Deep Learning Convolutional Neural Network)',
        'model': cnn,
        'scaler': scaler,
        'kolom_fitur': kolom_fitur,
        'metrics': metrics,
        'mean_asli': mean_asli,
        'mean_ai': mean_ai,
        'label_map': {0: 'Human Voice', 1: 'AI Generated Voice'},
        'sr': 16000,
        'durasi_segmen': 5
    }

    # Simpan ke backend dan web-statis
    pkl_targets = [
        os.path.join(SCRIPT_DIR, 'model_voice_cnn_5s.pkl'),
        os.path.join(BASE_DIR, 'web-statis-voice-detection', 'model_voice_cnn_5s.pkl')
    ]
    pt_targets = [
        os.path.join(SCRIPT_DIR, 'model_voice_cnn_5s.pt'),
        os.path.join(BASE_DIR, 'web-statis-voice-detection', 'model_voice_cnn_5s.pt')
    ]

    for p in pkl_targets:
        if os.path.exists(os.path.dirname(p)):
            joblib.dump(paket_cnn, p)
            print(f'Paket model (.pkl) berhasil disimpan: {p}')

    for pt in pt_targets:
        if os.path.exists(os.path.dirname(pt)):
            torch.save(cnn.model.state_dict(), pt)
            print(f'State dict PyTorch (.pt) berhasil disimpan: {pt}')

    print('\n[SUKSES] Model Deep Learning CNN telah diperbarui dan langsung aktif melayani prediksi!')


if __name__ == '__main__':
    df_lengkap = sinkronisasi_fitur_baru()
    latih_model_cnn(df_lengkap)
