"""
predict_v2.py
=============
Prediksi Audio: Deteksi Aliran Air Wudhu Masjid (3 Kelas)
Mendukung model V2 (91 fitur) dengan Weighted Blend Ensemble

Gunakan: py -3.9 predict_v2.py <file_audio>
Contoh : py -3.9 predict_v2.py suara_test.wav
"""

import os
import sys
import numpy as np
import librosa
import joblib
import warnings
warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "output", "model_aliran_air.pkl")

CLASS_NAMES = ["Bukan Air (Kipas/Adzan/Suasana)", "Kran Pelan (Hemat)", "Kran Keras (Boros)"]
CLASS_EMOJIS = ["🔇", "💧", "🚿"]
CLASS_COLORS  = ["\033[91m", "\033[92m", "\033[94m"]  # merah, hijau, biru
RESET_COLOR   = "\033[0m"


def extract_features_v2(y, sr=22050):
    """
    Ekstraksi 91 fitur akustik dari sinyal audio
    Kompatibel dengan model V2 (train_v2.py)
    """
    # Pastikan minimal 2 detik
    win_samples = int(2.0 * sr)
    if len(y) < win_samples:
        y = np.pad(y, (0, win_samples - len(y)), mode='constant')
    seg = y[:win_samples]  # Ambil 2 detik pertama untuk prediksi cepat

    # ── Fitur Spektral Dasar ──
    rms_frames = librosa.feature.rms(y=seg)
    rms_val    = float(np.mean(rms_frames))
    rms_std    = float(np.std(rms_frames))
    rms_var    = float(np.var(rms_frames))
    rms_cv     = float(rms_std / (rms_val + 1e-9))

    spec_cent  = float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)))
    spec_bw    = float(np.mean(librosa.feature.spectral_bandwidth(y=seg, sr=sr)))
    spec_ro85  = float(np.mean(librosa.feature.spectral_rolloff(y=seg, sr=sr, roll_percent=0.85)))
    spec_ro25  = float(np.mean(librosa.feature.spectral_rolloff(y=seg, sr=sr, roll_percent=0.25)))
    spec_flat  = float(np.mean(librosa.feature.spectral_flatness(y=seg)))
    zcr_frames = librosa.feature.zero_crossing_rate(y=seg)
    zcr_val    = float(np.mean(zcr_frames))
    zcr_std    = float(np.std(zcr_frames))

    # ── Harmonic/HPSS ──
    y_harm, _ = librosa.effects.hpss(seg, margin=(2.0, 1.0))
    harm_rms   = float(np.mean(librosa.feature.rms(y=y_harm)))
    harm_ratio = float(harm_rms / (rms_val + 1e-9))

    # ── Crest & Flux ──
    S = np.abs(librosa.stft(seg))
    crest = float(np.mean(np.max(S, axis=0) / (np.mean(S, axis=0) + 1e-9)))
    flux  = float(np.mean(librosa.onset.onset_strength(y=seg, sr=sr)))

    # ── Sub-band Energy ──
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

    # ── Fitur Baru V2 ──
    # RMS variance window kecil
    win_fine = 512
    rms_sub = np.array([
        np.sqrt(np.mean(seg[i:i+win_fine]**2))
        for i in range(0, len(seg) - win_fine, win_fine // 2)
    ])
    rms_var_fine = float(np.var(rms_sub))

    # Spectral Entropy
    ps = np.mean(S**2, axis=1)
    ps_norm = ps / (np.sum(ps) + 1e-9)
    ps_norm = np.clip(ps_norm, 1e-10, 1.0)
    spectral_entropy = float(-np.sum(ps_norm * np.log2(ps_norm)))

    # Spectral Skewness
    ps_mean = np.sum(freqs * ps_norm)
    ps_std  = np.sqrt(np.sum((freqs - ps_mean)**2 * ps_norm) + 1e-9)
    spectral_skewness = float(np.sum(((freqs - ps_mean) / ps_std)**3 * ps_norm))

    # Mel High Ratio
    mel_spec = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=128)
    mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr/2)
    mel_high_mask = mel_freqs >= 3000
    mel_high_ratio = float(np.sum(mel_spec[mel_high_mask, :]) / (np.sum(mel_spec) + 1e-9))

    # Pitch estimation
    try:
        f0, _, _ = librosa.pyin(seg, fmin=librosa.note_to_hz('C2'),
                                fmax=librosa.note_to_hz('C7'), sr=sr, fill_na=0.0)
        pitch_est = float(np.nanmedian(f0))
    except Exception:
        pitch_est = 0.0

    # Tempo
    try:
        tempo_val, _ = librosa.beat.beat_track(y=seg, sr=sr)
        if hasattr(tempo_val, '__len__'):
            tempo_val = float(tempo_val[0])
        else:
            tempo_val = float(tempo_val)
    except Exception:
        tempo_val = 0.0

    # ── MFCC + Delta ──
    mfcc       = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)

    # ── Spectral Contrast ──
    contrast = librosa.feature.spectral_contrast(y=seg, sr=sr)

    # Susun vektor fitur
    feat = {
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
        feat[f'mfcc_mean_{i+1}']  = float(np.mean(mfcc[i]))
        feat[f'mfcc_std_{i+1}']   = float(np.std(mfcc[i]))
        feat[f'mfcc_delta_{i+1}'] = float(np.mean(mfcc_delta[i]))
    for i in range(contrast.shape[0]):
        feat[f'contrast_mean_{i+1}'] = float(np.mean(contrast[i]))

    return feat


def predict_file(file_path, model_data):
    """Prediksi satu file audio"""
    print(f"\n[*] Memuat audio: {os.path.basename(file_path)}")
    y, sr = librosa.load(file_path, sr=22050)
    print(f"    Durasi: {len(y)/sr:.1f} detik")

    feature_cols = model_data['feature_cols']
    feat = extract_features_v2(y, sr)

    # Susun vektor dalam urutan yang sama dengan saat training
    x = np.array([[feat.get(col, 0.0) for col in feature_cols]])

    # Prediksi dengan model terbaik yang tersedia
    results = {}

    if 'model_et' in model_data:
        et_m = model_data['model_et']
        p_et = et_m.predict_proba(x)[0]
        results['et'] = p_et

    if 'model_xgb' in model_data:
        xgb_m = model_data['model_xgb']
        p_xgb = xgb_m.predict_proba(x)[0]
        results['xgb'] = p_xgb

    if 'model_hgb' in model_data:
        hgb_m = model_data['model_hgb']
        p_hgb = hgb_m.predict_proba(x)[0]
        results['hgb'] = p_hgb

    # Weighted blend
    weights = model_data.get('voting_weights', [1.0])
    if 'et' in results and 'xgb' in results and 'hgb' in results:
        w = model_data.get('voting_weights', [0.45, 0.30, 0.25])
        p_final = w[0] * results['et'] + w[1] * results['xgb'] + w[2] * results['hgb']
    elif 'et' in results and 'hgb' in results:
        p_final = 0.6 * results['et'] + 0.4 * results['hgb']
    elif 'et' in results:
        p_final = results['et']
    else:
        # Fallback ke model utama
        p_final = model_data['model'].predict_proba(x)[0]

    pred_class = int(np.argmax(p_final))
    confidence = float(p_final[pred_class]) * 100

    return pred_class, confidence, p_final


def main():
    if len(sys.argv) < 2:
        print(f"Gunakan: py -3.9 predict_v2.py <file_audio>")
        print(f"Contoh : py -3.9 predict_v2.py suara_kran.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"[ERROR] File tidak ditemukan: {audio_path}")
        sys.exit(1)

    # Muat model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model tidak ditemukan: {MODEL_PATH}")
        print("Jalankan train_v2.py terlebih dahulu.")
        sys.exit(1)

    print("[*] Memuat model...")
    model_data = joblib.load(MODEL_PATH)
    strategy   = model_data.get('strategy', 'unknown')
    acc_test   = model_data.get('accuracy_test', 0)
    print(f"    Strategi  : {strategy}")
    if acc_test:
        print(f"    Akurasi Test : {acc_test*100:.2f}%")

    # Prediksi
    pred_class, confidence, probs = predict_file(audio_path, model_data)

    # Output
    print("\n" + "=" * 55)
    color = CLASS_COLORS[pred_class]
    emoji = CLASS_EMOJIS[pred_class]
    print(f"  {color}HASIL: {emoji} {CLASS_NAMES[pred_class]}{RESET_COLOR}")
    print(f"  Keyakinan: {confidence:.1f}%")
    print("=" * 55)
    print("\nDistribusi Probabilitas:")
    for i, (name, emoji, prob) in enumerate(zip(CLASS_NAMES, CLASS_EMOJIS, probs)):
        bar = "█" * int(prob * 40)
        print(f"  {emoji} {name[:30]:<30} {prob*100:5.1f}% {bar}")

    print()
    if pred_class == 0:
        print("  → Tidak ada aliran air terdeteksi")
    elif pred_class == 1:
        print("  → Kran menyala pelan - Mode hemat air ✓")
    else:
        print("  → Kran menyala keras - Pertimbangkan hemat air!")


if __name__ == "__main__":
    main()
