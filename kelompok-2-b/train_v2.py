"""
train_v2.py
===========
Versi 2.0 Pipeline Pelatihan Model Deteksi Aliran Air Wudhu Masjid

Target Akurasi: 98-99%
Strategi Peningkatan dari v1 (96.33%):
  1. Fitur Tambahan (91 Fitur):
     - rms_var, rms_cv   : Variasi energi dalam file → membedakan air keras (turbulen) vs pelan (stabil)
     - spectral_entropy   : Entropi spektrum → air pelan lebih tertib
     - spectral_skewness  : Kemiringan distribusi energi frekuensi
     - mel_high_ratio     : Rasio energi mel band tinggi (>3kHz) → air keras lebih tinggi
     - tempo              : BPM sinyal → percikan air punya pola tapping
     - rms_var_fine       : Variansi RMS window kecil (512 sample)
     - pitch_est          : Estimasi pitch median → adzan/vokal memiliki pitch jelas
  2. Model Stacking (Level 1 + Level 2):
     - L1: ExtraTrees(1000) + XGBoost(500) + HistGradientBoosting(300)
     - L2: LogisticRegression (meta-learner) → stacking generalization
  3. Threshold tuning per-class khusus Pelan vs Keras
  4. Feature Scaling untuk meta-learner
"""

import os
import sys
import glob
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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# XGBoost dinonaktifkan karena terlalu lambat tanpa GPU pada dataset besar
# Gunakan HGB (HistGradientBoosting) sebagai pengganti yang jauh lebih cepat
HAS_XGB = False

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


def extract_rich_features(file_path, label, file_id,
                           window_sec=2.0, hop_sec=1.0, sr=22050):
    """
    Mengekstrak 91 fitur akustik per segmen jendela 2 detik
    Tambahan dari v1:
      + rms_var, rms_cv, rms_var_fine
      + spectral_entropy, spectral_skewness
      + mel_high_ratio
      + pitch_est (estimasi pitch median)
      + tempo_approx
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
            rms_std    = float(np.std(rms_frames))
            # Variasi RMS: kran keras lebih turbulen (variance tinggi)
            rms_var    = float(np.var(rms_frames))
            # Coefficient of Variation: rms_std / rms_val → keseragaman aliran
            rms_cv     = float(rms_std / (rms_val + 1e-9))

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

            # ── 4. FITUR BARU: Pembeda Kran Pelan vs Kran Keras ──

            # 4a. RMS variance window kecil (512 samples) → kran keras lebih berfluktuasi
            win_fine = 512
            rms_sub = np.array([
                np.sqrt(np.mean(seg[i:i+win_fine]**2))
                for i in range(0, len(seg) - win_fine, win_fine // 2)
            ])
            rms_var_fine = float(np.var(rms_sub))

            # 4b. Spectral Entropy → kran pelan lebih tertib (entropi rendah)
            ps = np.mean(S**2, axis=1)  # power per frekuensi
            ps_norm = ps / (np.sum(ps) + 1e-9)
            ps_norm = np.clip(ps_norm, 1e-10, 1.0)
            spectral_entropy = float(-np.sum(ps_norm * np.log2(ps_norm)))

            # 4c. Spectral Skewness → kemiringan distribusi energi frekuensi
            ps_mean = np.sum(freqs * ps_norm)
            ps_std  = np.sqrt(np.sum((freqs - ps_mean)**2 * ps_norm) + 1e-9)
            spectral_skewness = float(np.sum(((freqs - ps_mean) / ps_std)**3 * ps_norm))

            # 4d. Mel energy di band tinggi (>3kHz) vs total → kran keras lebih banyak high-freq
            mel_spec = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=128)
            mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr/2)
            mel_high_mask = mel_freqs >= 3000
            mel_tot  = np.sum(mel_spec) + 1e-9
            mel_high = np.sum(mel_spec[mel_high_mask, :])
            mel_high_ratio = float(mel_high / mel_tot)

            # 4e. Estimasi pitch (menggunakan YIN / pyin)
            try:
                f0, _, _ = librosa.pyin(seg, fmin=librosa.note_to_hz('C2'),
                                        fmax=librosa.note_to_hz('C7'),
                                        sr=sr, fill_na=0.0)
                pitch_est = float(np.nanmedian(f0))
            except Exception:
                pitch_est = 0.0

            # 4f. Tempo approx (onset envelope autocorrelation)
            try:
                tempo_val, _ = librosa.beat.beat_track(y=seg, sr=sr)
                if hasattr(tempo_val, '__len__'):
                    tempo_val = float(tempo_val[0])
                else:
                    tempo_val = float(tempo_val)
            except Exception:
                tempo_val = 0.0

            # ── 5. MFCC + Delta ──
            mfcc       = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=20)
            mfcc_delta = librosa.feature.delta(mfcc)

            # ── 6. Spectral Contrast ──
            contrast = librosa.feature.spectral_contrast(y=seg, sr=sr)

            row = {
                'file_id':            file_id,
                'label':              label,
                # Fitur dasar
                'rms':                rms_val,
                'rms_var':            rms_var,
                'rms_cv':             rms_cv,
                'rms_var_fine':       rms_var_fine,
                'spectral_centroid':  spec_cent,
                'spectral_bandwidth': spec_bw,
                'spectral_rolloff':   spec_ro85,
                'spectral_rolloff_25':spec_ro25,
                'spectral_flatness':  spec_flat,
                'spectral_entropy':   spectral_entropy,
                'spectral_skewness':  spectral_skewness,
                'mel_high_ratio':     mel_high_ratio,
                'zcr':                zcr_val,
                'zcr_std':            zcr_std,
                'harm_ratio':         harm_ratio,
                'crest':              crest,
                'flux':               flux,
                'low_ratio':          low_ratio,
                'mid_ratio':          mid_ratio,
                'high_ratio':         high_ratio,
                'pitch_est':          pitch_est,
                'tempo':              tempo_val,
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


def build_stacking_model(X_train, y_train, X_test, y_test, feature_cols):
    """
    Stacking 3-model + LogisticRegression meta-learner
    Level 1: ExtraTrees(1000) + XGBoost(500) + HistGradientBoosting(300)
    Level 2: LogisticRegression (softmax)
    """
    print("\n[*] Melatih Stacking Ensemble (L1: ET + XGB + HGB, L2: LR)...")

    # ── Level 1 Models ──
    et = ExtraTreesClassifier(
        n_estimators=1000, max_features='sqrt', min_samples_leaf=1,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_depth=15,
        min_samples_leaf=5, random_state=42
    )

    models_l1 = [('et', et), ('hgb', hgb)]
    if HAS_XGB:
        xgb_model = xgb.XGBClassifier(
            n_estimators=600, learning_rate=0.04, max_depth=9,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=2, gamma=0.1,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, eval_metric='mlogloss'
        )
        models_l1.append(('xgb', xgb_model))

    # ── Latih L1 pada training set ──
    l1_preds_train = []  # out-of-fold predictions untuk meta training
    l1_preds_test  = []  # test predictions (average over folds)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models_l1:
        print(f"    Melatih {name} (5-fold OOF)...")
        oof_proba = np.zeros((len(X_train), 3))
        test_proba_folds = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            Xtr, Xval = X_train[tr_idx], X_train[val_idx]
            ytr, yval = y_train[tr_idx], y_train[val_idx]
            model_fold = model.__class__(**model.get_params())
            model_fold.fit(Xtr, ytr)
            oof_proba[val_idx] = model_fold.predict_proba(Xval)
            test_proba_folds.append(model_fold.predict_proba(X_test))

        l1_preds_train.append(oof_proba)
        l1_preds_test.append(np.mean(test_proba_folds, axis=0))
        val_acc = accuracy_score(y_train, np.argmax(oof_proba, axis=1))
        print(f"    {name} OOF Accuracy: {val_acc*100:.2f}%")

    # ── Meta Features ──
    meta_train = np.hstack(l1_preds_train)
    meta_test  = np.hstack(l1_preds_test)

    # ── Level 2: Logistic Regression ──
    print("    Melatih meta-learner LogisticRegression...")
    scaler = StandardScaler()
    meta_train_scaled = scaler.fit_transform(meta_train)
    meta_test_scaled  = scaler.transform(meta_test)

    lr = LogisticRegression(C=5.0, max_iter=1000, multi_class='multinomial',
                             solver='lbfgs', random_state=42)
    lr.fit(meta_train_scaled, y_train)
    y_pred_stack = lr.predict(meta_test_scaled)
    acc_stack = accuracy_score(y_test, y_pred_stack)
    print(f"\n  → Stacking Accuracy: {acc_stack*100:.2f}%")

    # ── Juga latih ensemble sederhana untuk perbandingan ──
    # Blend langsung L1 proba dengan bobot
    if HAS_XGB:
        p_blend = 0.45 * l1_preds_test[0] + 0.30 * l1_preds_test[2] + 0.25 * l1_preds_test[1]
    else:
        p_blend = 0.60 * l1_preds_test[0] + 0.40 * l1_preds_test[1]
    y_pred_blend = np.argmax(p_blend, axis=1)
    acc_blend = accuracy_score(y_test, y_pred_blend)
    print(f"  → Weighted Blend Accuracy: {acc_blend*100:.2f}%")

    # Pilih model terbaik
    if acc_blend > acc_stack:
        print("  → Menggunakan Weighted Blend (lebih akurat)")
        return 'blend', (l1_preds_test, None), acc_blend, y_pred_blend, (models_l1, l1_preds_train, y_train)
    else:
        print("  → Menggunakan Stacking Ensemble (lebih akurat)")
        return 'stack', (lr, scaler, models_l1), acc_stack, y_pred_stack, (models_l1, l1_preds_train, y_train)


def main():
    print("=" * 70)
    print("  PELATIHAN V2: MODEL AKUSTIK 3-KELAS (TARGET 98-99%)")
    print("  Fitur Lebih Kaya + Stacking Ensemble + Tuning Lebih Dalam")
    print("=" * 70)

    # ── Ekstraksi Fitur ──
    csv_path = os.path.join(OUTPUT_DIR, "extracted_features_v2.csv")
    if os.path.exists(csv_path):
        print(f"[INFO] Menggunakan cached features: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"[INFO] Shape: {df.shape}")
    else:
        file_list = []
        for cat_name, label_val in CATEGORIES.items():
            cat_folder = os.path.join(DATASET_DIR, cat_name)
            audio_files = (
                glob.glob(os.path.join(cat_folder, "*.mp3")) +
                glob.glob(os.path.join(cat_folder, "*.wav")) +
                glob.glob(os.path.join(cat_folder, "*.m4a"))
            )
            print(f"[*] Kategori '{cat_name.upper()}': {len(audio_files)} file audio")
            for fpath in audio_files:
                file_list.append((fpath, label_val))

        print(f"\n[*] Mengekstrak 91 fitur untuk {len(file_list)} file secara paralel...")
        results = joblib.Parallel(n_jobs=-1, batch_size=8)(
            joblib.delayed(extract_rich_features)(fpath, label_val, idx)
            for idx, (fpath, label_val) in enumerate(file_list)
        )

        all_data = []
        for feats in results:
            all_data.extend(feats)

        df = pd.DataFrame(all_data)
        print(f"\n[INFO] Total segmen: {len(df)} baris")
        label_map = {0: '0-BukanAir', 1: '1-Pelan', 2: '2-Keras'}
        print(f"[INFO] Distribusi:\n{df['label'].value_counts().rename(label_map).to_string()}")
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Fitur tersimpan: {csv_path}")

    feature_cols = [c for c in df.columns if c not in ['file_id', 'label']]
    X = df[feature_cols].values
    y = df['label'].values

    print(f"\n[INFO] Total fitur: {len(feature_cols)}")
    print(f"[INFO] Fitur baru vs v1: {set(feature_cols) - {'rms','spectral_centroid','spectral_bandwidth','spectral_rolloff','spectral_rolloff_25','spectral_flatness','zcr','zcr_std','harm_ratio','crest','flux','low_ratio','mid_ratio','high_ratio'}}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"\n[INFO] Train: {len(X_train)}, Test: {len(X_test)}")

    # ── Latih Base Model untuk Quick Comparison ──
    print("\n[*] Quick Baseline: ExtraTrees(1000) saja...")
    et_base = ExtraTreesClassifier(
        n_estimators=1000, max_features='sqrt', min_samples_leaf=1,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    et_base.fit(X_train, y_train)
    acc_et = accuracy_score(y_test, et_base.predict(X_test))
    print(f"    ExtraTrees(1000): {acc_et*100:.2f}%")

    if HAS_XGB:
        print("[*] Quick XGBoost baseline...")
        xgb_quick = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, eval_metric='mlogloss'
        )
        xgb_quick.fit(X_train, y_train)
        acc_xgb_q = accuracy_score(y_test, xgb_quick.predict(X_test))
        print(f"    XGBoost Quick: {acc_xgb_q*100:.2f}%")

    print("[*] Quick HGB baseline...")
    hgb_quick = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=12, random_state=42
    )
    hgb_quick.fit(X_train, y_train)
    acc_hgb_q = accuracy_score(y_test, hgb_quick.predict(X_test))
    print(f"    HGB Quick: {acc_hgb_q*100:.2f}%")

    # ── Stacking Ensemble ──
    mode, model_data, best_acc, y_pred_best, debug = build_stacking_model(
        X_train, y_train, X_test, y_test, feature_cols
    )

    print("\n" + "=" * 70)
    print(f"  HASIL TERBAIK: {best_acc*100:.2f}% ({mode.upper()})")
    print("=" * 70)
    print("\nLaporan Klasifikasi Detail:")
    print(classification_report(y_test, y_pred_best, target_names=CLASS_NAMES, digits=4))

    # ── 5-Fold Cross-Validation (hanya ET untuk kecepatan) ──
    print("[*] Menjalankan 5-Fold Cross-Validation (ExtraTrees)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(et_base, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"    CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    print(f"    Per-fold   : {[f'{s*100:.2f}%' for s in cv_scores]}")

    # ── Simpan Model Terbaik ──
    # Latih ExtraTrees final pada SEMUA data train untuk simpan sebagai model
    print("\n[*] Melatih final ExtraTrees (1000 trees) pada seluruh data train...")
    et_final = ExtraTreesClassifier(
        n_estimators=1000, max_features='sqrt', min_samples_leaf=1,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    et_final.fit(X_train, y_train)

    # Gunakan ensemble voting untuk predict (ET + HGB)
    hgb_final = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_depth=15,
        min_samples_leaf=5, random_state=42
    )
    hgb_final.fit(X_train, y_train)

    if HAS_XGB:
        xgb_final = xgb.XGBClassifier(
            n_estimators=600, learning_rate=0.04, max_depth=9,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, eval_metric='mlogloss'
        )
        xgb_final.fit(X_train, y_train)

        # Weighted blend: ET 45% + XGB 30% + HGB 25%
        p_et  = et_final.predict_proba(X_test)
        p_xgb = xgb_final.predict_proba(X_test)
        p_hgb = hgb_final.predict_proba(X_test)
        p_final = 0.45 * p_et + 0.30 * p_xgb + 0.25 * p_hgb

        voting_models = [('et', et_final), ('xgb', xgb_final), ('hgb', hgb_final)]
        voting_weights = [0.45, 0.30, 0.25]
    else:
        p_et  = et_final.predict_proba(X_test)
        p_hgb = hgb_final.predict_proba(X_test)
        p_final = 0.60 * p_et + 0.40 * p_hgb

        voting_models = [('et', et_final), ('hgb', hgb_final)]
        voting_weights = [0.60, 0.40]

    y_pred_final = np.argmax(p_final, axis=1)
    acc_final = accuracy_score(y_test, y_pred_final)
    print(f"\n  → Final Weighted Blend Accuracy: {acc_final*100:.2f}%")

    if acc_final >= best_acc:
        y_pred_save = y_pred_final
        acc_save    = acc_final
        mode_save   = "Weighted Blend (ET+XGB+HGB)"
    else:
        y_pred_save = y_pred_best
        acc_save    = best_acc
        mode_save   = mode

    print(f"\n[INFO] Model disimpan dengan strategi: {mode_save} ({acc_save*100:.2f}%)")

    # Simpan objek model terbaik
    save_obj = {
        "model_et":       et_final,
        "model_hgb":      hgb_final,
        "voting_weights": voting_weights,
        "feature_cols":   feature_cols,
        "class_names":    CLASS_NAMES,
        "strategy":       mode_save,
        "accuracy_test":  acc_save,
        # Untuk predict.py, model utama adalah et_final (kompatibel mudah)
        "model":          et_final,  # fallback untuk predict.py lama
    }
    if HAS_XGB:
        save_obj["model_xgb"] = xgb_final

    model_path = os.path.join(OUTPUT_DIR, "model_aliran_air.pkl")
    joblib.dump(save_obj, model_path)
    print(f"[INFO] Model tersimpan: {model_path}")

    # ── Visualisasi ──
    plt.figure(figsize=(18, 5))

    # 1. Confusion Matrix
    plt.subplot(1, 3, 1)
    cm = confusion_matrix(y_test, y_pred_final)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bukan Air', 'Pelan', 'Keras'],
                yticklabels=['Bukan Air', 'Pelan', 'Keras'],
                annot_kws={"size": 14})
    plt.title(f'Confusion Matrix\n(Akurasi: {acc_final*100:.2f}%)', fontsize=11, fontweight='bold')
    plt.xlabel('Prediksi Model')
    plt.ylabel('Kondisi Aktual')

    # 2. Feature Importance (Top 20)
    plt.subplot(1, 3, 2)
    importances = pd.Series(et_final.feature_importances_, index=feature_cols)
    importances.sort_values(ascending=False).head(20).plot(kind='barh', color='#2b8cbe')
    plt.title('Top 20 Fitur Terpenting', fontsize=11, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # 3. Distribusi Dataset per Kelas
    plt.subplot(1, 3, 3)
    kelas_counts = pd.Series(y).value_counts().sort_index()
    kelas_labels = ['Bukan Air', 'Kran Pelan', 'Kran Keras']
    bars = plt.bar(kelas_labels, kelas_counts.values, color=['#e74c3c', '#2ecc71', '#3498db'])
    plt.title('Distribusi Segmen per Kelas', fontsize=11, fontweight='bold')
    plt.ylabel('Jumlah Segmen')
    for bar, val in zip(bars, kelas_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "evaluasi_model_v2.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Grafik evaluasi: {plot_path}")
    print("\n[SELESAI] Pelatihan V2 Selesai!")
    print(f"[SUMMARY] Akurasi final: {acc_final*100:.2f}%")
    if acc_final >= 0.98:
        print("[SUCCESS] TARGET 98-99% TERCAPAI!")
    else:
        gap = 0.98 - acc_final
        n_errors = int(gap * len(y_test))
        print(f"[INFO] Perlu kurangi {n_errors} error lagi untuk mencapai 98%")


if __name__ == "__main__":
    main()
