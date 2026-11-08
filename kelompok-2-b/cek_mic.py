import os
import sys
import time
import warnings
import sounddevice as sd
import numpy as np
import librosa
import joblib
import pandas as pd
import soundfile as sf

warnings.filterwarnings('ignore')

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "output", "model_aliran_air.pkl")

print("=" * 65)
print(" 🎙️ DIAGNOSTIK & TES MIKROFON LAPTOP (5 DETIK)")
print("=" * 65)
print("Petunjuk:")
print("1. Nyalakan kran air Anda sekarang!")
print("2. Perekaman akan dimulai dalam 3 detik...")
for i in range(3, 0, -1):
    print(f"   Mulai dalam {i}...")
    time.sleep(1)

print("\n🔴 [SEDANG MEREKAM 5 DETIK] Silakan alirkan kran...")
SR = 22050
duration = 5.0
audio = sd.rec(int(duration * SR), samplerate=SR, channels=1, dtype='float32')
sd.wait()
audio = audio.flatten()
print("⏹️ [SELESAI MEREKAM!]")

raw_rms = float(np.sqrt(np.mean(audio**2)))
peak = float(np.max(np.abs(audio)))
print("\n" + "-" * 50)
print(f"📊 DATA TEKNIS SUARA YANG TERTANGKAP:")
print(f"   Level Rata-rata (RMS) : {raw_rms:.6f}")
print(f"   Puncak Suara (Peak)   : {peak:.6f}")

# Simpan rekaman untuk diperiksa
out_wav = os.path.join(BASE_DIR, "rekaman_tes.wav")
sf.write(out_wav, audio, SR)
print(f"   File tersimpan di     : {out_wav}")

# Analisis dengan Model Machine Learning
if os.path.exists(MODEL_PATH):
    checkpoint = joblib.load(MODEL_PATH)
    model = checkpoint["model"]
    feature_cols = checkpoint["feature_cols"]

    # Normalisasi gain jika suara terlalu kecil
    target_rms = 0.05
    gain = min(20.0, target_rms / (raw_rms + 1e-6))
    seg = audio * gain

    rms = float(np.mean(librosa.feature.rms(y=seg)))
    spec_cent = float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=SR)))
    spec_bw = float(np.mean(librosa.feature.spectral_bandwidth(y=seg, sr=SR)))
    spec_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=seg, sr=SR)))
    spec_flat = float(np.mean(librosa.feature.spectral_flatness(y=seg)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=seg)))

    mfcc = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)
    contrast = librosa.feature.spectral_contrast(y=seg, sr=SR)

    row_dict = {
        'rms': rms,
        'spectral_centroid': spec_cent,
        'spectral_bandwidth': spec_bw,
        'spectral_rolloff': spec_rolloff,
        'spectral_flatness': spec_flat,
        'zcr': zcr
    }
    for i in range(20):
        row_dict[f'mfcc_mean_{i+1}'] = float(np.mean(mfcc[i]))
        row_dict[f'mfcc_std_{i+1}'] = float(np.std(mfcc[i]))
        row_dict[f'mfcc_delta_{i+1}'] = float(np.mean(mfcc_delta[i]))

    for i in range(contrast.shape[0]):
        row_dict[f'contrast_mean_{i+1}'] = float(np.mean(contrast[i]))

    feats = [row_dict[col] for col in feature_cols]
    X_df = pd.DataFrame([feats], columns=feature_cols)

    class_names = checkpoint.get("class_names", ["Bukan Air (Ngobrol/Suasana)", "Kran Pelan (Hemat)", "Kran Keras (Boros)"])
    pred = model.predict(X_df.values)[0]
    prob = model.predict_proba(X_df.values)[0]

    print("-" * 50)
    print(f"🤖 HASIL DETEKSI MODEL : {class_names[pred]}")
    for idx, name in enumerate(class_names):
        print(f"   Keyakinan {name:<30}: {prob[idx]*100:.2f}%")
    print("-" * 50)

print("\n🔊 Memutar kembali suara rekaman lewat speaker Anda...")
sd.play(audio * 3.0, SR)
sd.wait()
print("[*] Selesai! Jika suara rekaman tadi terdengar sangat kecil atau hening, berarti mikrofon laptop Anda teredam filter noise Windows.")
