"""
train.py
========
Pipeline Pelatihan Model Deteksi Aliran Air Wudhu Masjid (3 Kelas):
  [0] Bukan Air (Ngobrol, Lingkungan, Kipas Angin, Adzan, Angin/Blower)
  [1] Kran Pelan (Hemat Air)
  [2] Kran Keras (Boros Air)

Fitur Akustik Komprehensif (81 Fitur):
  - Spektral Dasar (6)       : RMS, Centroid, Bandwidth, Rolloff 85%, Flatness, ZCR
  - Akustik Khusus (8)       : Harm Ratio (Adzan/Vokal), Crest (Kipas vs Percikan),
                               Flux (Transien Cipratan), Low/Mid/High Energy Ratio,
                               ZCR Std (Fluktuasi Tetesan), Rolloff 25%
  - MFCC + Delta (60)        : 20 MFCC Mean, 20 MFCC Std, 20 MFCC Delta Mean
  - Spectral Contrast (7)    : 7 Sub-band Contrast Mean
"""

import os
import sys
import glob
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR   = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CATEGORIES = {
    "bukan_air": 0,
    "pelan":     1,
    "keras":     2
}
CLASS_NAMES = ["Bukan Air (Kipas/Adzan/Suasana)", "Kran Pelan (Hemat)", "Kran Keras (Boros)"]


def parse_args():
    parser = argparse.ArgumentParser(description='Train model deteksi aliran air wudhu.')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Batas jumlah file per kelas untuk debug / smoke test (opsional).')
    parser.add_argument('--limit', type=int, default=None,
                        help='Alias untuk --max-files.')
    return parser.parse_args()


def safe_load_audio(file_path, sr=22050):
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        if y is None or len(y) == 0 or np.isnan(y).any():
            raise ValueError('audio kosong atau nan')
        return y
    except Exception:
        return None


def collect_audio_files(max_files=None):
    file_list = []
    skipped = 0
    for cat_name, label_val in CATEGORIES.items():
        cat_folder = os.path.join(DATASET_DIR, cat_name)
        audio_files = (
            glob.glob(os.path.join(cat_folder, "*.mp3")) +
            glob.glob(os.path.join(cat_folder, "*.wav")) +
            glob.glob(os.path.join(cat_folder, "*.m4a"))
        )
        if max_files is not None:
            audio_files = audio_files[:max_files]
        for fpath in audio_files:
            y = safe_load_audio(fpath)
            if y is None:
                skipped += 1
                continue
            file_list.append((fpath, label_val))
    if skipped:
        print(f"[*] Melewati {skipped} file audio rusak / tidak dapat dibaca.")
    return file_list


def extract_rich_features(file_path, label, file_id,
                           window_sec=2.0, hop_sec=1.0, sr=22050):
    """
    Mengekstrak 81 fitur akustik per segmen jendela 2 detik
    """
    features = []
    try:
        y, _ = librosa.load(file_path, sr=sr)
        win_samples = int(window_sec * sr)
        hop_samples = int(hop_sec * sr)

        if len(y) < win_samples:
            y = np.pad(y, (0, win_samples - len(y)), mode='constant')

        for start in range(0, len(y) - win_samples + 1, hop_samples):
            seg = y[start : start + win_samples]

            # ── 1. Fitur Spektral Dasar ──
            rms_frames = librosa.feature.rms(y=seg)
            rms_val    = float(np.mean(rms_frames))
            spec_cent  = float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)))
            spec_bw    = float(np.mean(librosa.feature.spectral_bandwidth(y=seg, sr=sr)))
            spec_ro85  = float(np.mean(librosa.feature.spectral_rolloff(y=seg, sr=sr, roll_percent=0.85)))
            spec_ro25  = float(np.mean(librosa.feature.spectral_rolloff(y=seg, sr=sr, roll_percent=0.25)))
            spec_flat  = float(np.mean(librosa.feature.spectral_flatness(y=seg)))
            zcr_frames = librosa.feature.zero_crossing_rate(y=seg)
            zcr_val    = float(np.mean(zcr_frames))
            zcr_std    = float(np.std(zcr_frames))

            # ── 2. Pemisah Khusus: Adzan / Vokal (Harmonic HPSS) ──
            y_harm, y_perc = librosa.effects.hpss(seg, margin=(2.0, 1.0))
            harm_rms   = float(np.mean(librosa.feature.rms(y=y_harm)))
            harm_ratio = float(harm_rms / (rms_val + 1e-9))

            # ── 3. Pemisah Khusus: Kipas vs Cipratan Air (Crest & Flux) ──
            S = np.abs(librosa.stft(seg))
            # Crest factor: rasio puncak spektrum thdp rata-rata
            crest = float(np.mean(np.max(S, axis=0) / (np.mean(S, axis=0) + 1e-9)))
            # Spectral flux (onset strength)
            flux = float(np.mean(librosa.onset.onset_strength(y=seg, sr=sr)))

            # Sub-band Energy Distribution
            freqs = librosa.fft_frequencies(sr=sr)
            low_idx  = np.where(freqs < 250)[0]
            mid_idx  = np.where((freqs >= 250) & (freqs < 3000))[0]
            high_idx = np.where(freqs >= 3000)[0]

            low_e  = float(np.sum(S[low_idx, :]**2))
            mid_e  = float(np.sum(S[mid_idx, :]**2))
            high_e = float(np.sum(S[high_idx, :]**2))
            tot_e  = low_e + mid_e + high_e + 1e-9

            low_ratio  = float(low_e / tot_e)
            mid_ratio  = float(mid_e / tot_e)
            high_ratio = float(high_e / tot_e)

            # ── 4. MFCC + Delta ──
            mfcc       = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=20)
            mfcc_delta = librosa.feature.delta(mfcc)

            # ── 5. Spectral Contrast ──
            contrast = librosa.feature.spectral_contrast(y=seg, sr=sr)

            row = {
                'file_id':            file_id,
                'label':              label,
                'rms':                rms_val,
                'spectral_centroid':  spec_cent,
                'spectral_bandwidth': spec_bw,
                'spectral_rolloff':   spec_ro85,
                'spectral_rolloff_25':spec_ro25,
                'spectral_flatness':  spec_flat,
                'zcr':                zcr_val,
                'zcr_std':            zcr_std,
                'harm_ratio':         harm_ratio,
                'crest':              crest,
                'flux':               flux,
                'low_ratio':          low_ratio,
                'mid_ratio':          mid_ratio,
                'high_ratio':         high_ratio,
            }
            for i in range(20):
                row[f'mfcc_mean_{i+1}']  = float(np.mean(mfcc[i]))
                row[f'mfcc_std_{i+1}']   = float(np.std(mfcc[i]))
                row[f'mfcc_delta_{i+1}'] = float(np.mean(mfcc_delta[i]))
            for i in range(contrast.shape[0]):
                row[f'contrast_mean_{i+1}'] = float(np.mean(contrast[i]))

            features.append(row)

    except Exception as e:
        print(f"  [SKIP] {os.path.basename(file_path)}: {e}")

    return features


def main():
    args = parse_args()
    max_files = args.max_files if args.max_files is not None else args.limit

    print("=" * 70)
    print("  PELATIHAN MODEL AKUSTIK 3-KELAS: DETEKSI ALIRAN AIR WUDHU MASJID")
    print("  Fitur Pembeda Khusus Kipas Angin, Adzan, dan Cipratan Kran Air")
    print("=" * 70)

    file_list = collect_audio_files(max_files=max_files)
    if not file_list:
        raise RuntimeError("Tidak ada file audio valid yang bisa diproses. Cek dataset Anda.")

    for cat_name, label_val in CATEGORIES.items():
        cat_folder = os.path.join(DATASET_DIR, cat_name)
        audio_files = (
            glob.glob(os.path.join(cat_folder, "*.mp3")) +
            glob.glob(os.path.join(cat_folder, "*.wav")) +
            glob.glob(os.path.join(cat_folder, "*.m4a"))
        )
        if max_files is not None:
            audio_files = audio_files[:max_files]
        print(f"[*] Kategori '{cat_name.upper()}': {len(audio_files)} file audio yang dipertimbangkan")

    print(f"\n[*] Mengekstrak 81 fitur untuk {len(file_list)} file secara paralel (multi-core)...")
    results = joblib.Parallel(n_jobs=-1, batch_size=12)(
        joblib.delayed(extract_rich_features)(fpath, label_val, idx)
        for idx, (fpath, label_val) in enumerate(file_list)
    )

    all_data = []
    for feats in results:
        all_data.extend(feats)

    df = pd.DataFrame(all_data)
    print(f"\n[INFO] Total segmen (jendela 2 detik): {len(df)} baris")
    label_map = {0: '0 - Bukan Air (Kipas/Adzan)', 1: '1 - Kran Pelan', 2: '2 - Kran Keras'}
    print(f"[INFO] Distribusi per Kelas:\n{df['label'].value_counts().rename(label_map).to_string()}")

    csv_path = os.path.join(OUTPUT_DIR, "extracted_features.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Dataset fitur tersimpan: {csv_path}")

    feature_cols = [c for c in df.columns if c not in ['file_id', 'label']]
    X = df[feature_cols].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # ── Ensemble Model: ExtraTrees + Random Forest ──
    print("\n[*] Membangun dan melatih Ensemble Model (ExtraTrees + Tuned RF)...")
    et_model = ExtraTreesClassifier(
        n_estimators=600,
        max_features='sqrt',
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf_model = RandomForestClassifier(
        n_estimators=400,
        max_features='sqrt',
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # Voting ensemble soft probability
    ensemble_model = VotingClassifier(
        estimators=[('et', et_model), ('rf', rf_model)],
        voting='soft',
        n_jobs=-1
    )
    
    ensemble_model.fit(X_train, y_train)

    y_pred = ensemble_model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 70)
    print(f"  HASIL AKURASI 3 KELAS PADA DATA TEST: {acc * 100:.2f}%")
    print("=" * 70)
    print("\nLaporan Klasifikasi Detail:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

    # 5-Fold Stratified Cross Validation
    print("[*] Menjalankan 5-Fold Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Gunakan et_model untuk cross-val cepat dan konsisten
    cv_scores = cross_val_score(et_model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"    CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    print(f"    Per-fold   : {[f'{s*100:.2f}%' for s in cv_scores]}")

    # Simpan model
    model_path = os.path.join(OUTPUT_DIR, "model_aliran_air.pkl")
    joblib.dump({
        "model":        ensemble_model,
        "feature_cols": feature_cols,
        "class_names":  CLASS_NAMES
    }, model_path)
    print(f"\n[INFO] Model tersimpan: {model_path}")

    # ── Visualisasi Evaluasi ──
    plt.figure(figsize=(16, 5))

    # 1. Confusion Matrix
    plt.subplot(1, 3, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bukan Air', 'Pelan', 'Keras'],
                yticklabels=['Bukan Air', 'Pelan', 'Keras'],
                annot_kws={"size": 13})
    plt.title(f'Confusion Matrix\n(Akurasi Test: {acc*100:.2f}%)', fontsize=11, fontweight='bold')
    plt.xlabel('Prediksi Model')
    plt.ylabel('Kondisi Aktual')

    # 2. Feature Importance (dari ExtraTrees)
    plt.subplot(1, 3, 2)
    et_trained = ensemble_model.named_estimators_['et']
    importances = pd.Series(et_trained.feature_importances_, index=feature_cols)
    importances.sort_values(ascending=False).head(15).plot(kind='barh', color='#2b8cbe')
    plt.title('Top 15 Fitur Terpenting', fontsize=11, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # 3. Distribusi Dataset per Kelas
    plt.subplot(1, 3, 3)
    kelas_counts = pd.Series(y).value_counts().sort_index()
    kelas_labels = ['Bukan Air', 'Kran Pelan', 'Kran Keras']
    bars = plt.bar(kelas_labels, kelas_counts.values, color=['#e74c3c', '#2ecc71', '#3498db'])
    plt.title('Distribusi Dataset per Kelas', fontsize=11, fontweight='bold')
    plt.ylabel('Jumlah Segmen')
    for bar, val in zip(bars, kelas_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "evaluasi_model.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Grafik evaluasi diperbarui: {plot_path}")
    print("\n[SELESAI] Model sukses dilatih dan dievaluasi!")

if __name__ == "__main__":
    main()
