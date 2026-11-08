import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import librosa

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "output", "model_aliran_air.pkl")


def extract_features_for_model(seg, sr=22050):
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

    y_harm, y_perc = librosa.effects.hpss(seg, margin=(2.0, 1.0))
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

    low_ratio = float(low_e / tot_e)
    mid_ratio = float(mid_e / tot_e)
    high_ratio = float(high_e / tot_e)

    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)
    contrast = librosa.feature.spectral_contrast(y=seg, sr=sr)

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
        'low_ratio': low_ratio,
        'mid_ratio': mid_ratio,
        'high_ratio': high_ratio,
    }
    for i in range(20):
        row_dict[f'mfcc_mean_{i+1}'] = float(np.mean(mfcc[i]))
        row_dict[f'mfcc_std_{i+1}'] = float(np.std(mfcc[i]))
        row_dict[f'mfcc_delta_{i+1}'] = float(np.mean(mfcc_delta[i]))
    for i in range(contrast.shape[0]):
        row_dict[f'contrast_mean_{i+1}'] = float(np.mean(contrast[i]))
    return row_dict


def predict_water_flow(audio_path):
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] File model belum ditemukan! Jalankan train.py terlebih dahulu.")
        return

    checkpoint = joblib.load(MODEL_PATH)
    model = checkpoint["model"]
    feature_cols = checkpoint["feature_cols"]
    class_names = checkpoint.get("class_names", ["Bukan Air", "Kran Pelan", "Kran Keras"])

    print(f"\n[*] Menganalisis file: {audio_path}")
    y, sr = librosa.load(audio_path, sr=22050)
    win_samples = int(2.0 * sr)
    hop_samples = int(1.0 * sr)

    if len(y) < win_samples:
        y = np.pad(y, (0, win_samples - len(y)), mode='constant')

    probabilities = []

    for start in range(0, len(y) - win_samples + 1, hop_samples):
        seg = y[start : start + win_samples]
        row_dict = extract_features_for_model(seg, sr=sr)
        feats = [row_dict.get(col, 0.0) for col in feature_cols]
        X_df = pd.DataFrame([feats], columns=feature_cols)
        prob = model.predict_proba(X_df.values)[0]
        probabilities.append(prob)

    avg_prob = np.mean(probabilities, axis=0)
    final_class = int(np.argmax(avg_prob))

    print("\n" + "=" * 60)
    print(f" HASIL DETEKSI : {class_names[final_class]}")
    for idx, name in enumerate(class_names):
        print(f" - Keyakinan {name} : {avg_prob[idx]*100:.2f}%")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = input("Masukkan path file audio (.mp3/.wav): ").strip().strip('"')
    
    if os.path.exists(audio_file):
        predict_water_flow(audio_file)
    else:
        print(f"[ERROR] File '{audio_file}' tidak ditemukan!")
