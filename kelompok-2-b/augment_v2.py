"""
augment_v2.py
=============
Augmentasi Dataset V2 - Fokus pada Diskriminasi Kran Pelan vs Keras

Strategi:
1. Time-stretching selektif:
   - Kran keras (rate=0.85-1.15): menjaga intensitas tinggi
   - Kran pelan (rate=0.90-1.10): menjaga karakteristik pelan
2. Pitch shifting differensial:
   - Keras: pitch shift -3..+3 semitone (variasi tekanan air)
   - Pelan: pitch shift -2..+2 semitone (variasi flow rate rendah)
3. Rolling energy variance injection:
   - Tambahkan noise berenergi-tinggi untuk keras
   - Tambahkan noise stabil untuk pelan
4. Targeted SNR mixing:
   - Keras + pink noise (SNR 18-25 dB) → tetap keras
   - Pelan + pink noise (SNR 25-35 dB) → tetap pelan
5. Contoh dari dataset ESC-50: pouring + tap
"""

import os
import sys
import glob
import random
import numpy as np
import soundfile as sf
import librosa
import warnings
warnings.filterwarnings('ignore')

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PELAN_DIR   = os.path.join(DATASET_DIR, "pelan")
KERAS_DIR   = os.path.join(DATASET_DIR, "keras")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")

SR = 22050
random.seed(42)
np.random.seed(42)


def pink_noise(n_samples):
    """Menghasilkan pink noise (1/f) menggunakan metode Voss-McCartney"""
    num_cols = 16
    rows = n_samples // num_cols + 1
    array = np.random.randn(rows, num_cols)
    reshaped = array.reshape(-1, num_cols)
    running_sum = np.cumsum(reshaped, axis=0)
    resampled = running_sum.reshape(-1)[:n_samples]
    resampled /= (np.max(np.abs(resampled)) + 1e-9)
    return resampled.astype(np.float32)


def mix_with_noise(y, snr_db):
    """Campur audio dengan pink noise pada SNR tertentu"""
    noise = pink_noise(len(y))
    rms_y   = np.sqrt(np.mean(y**2) + 1e-9)
    rms_n   = np.sqrt(np.mean(noise**2) + 1e-9)
    scale   = rms_y / rms_n * 10**(-snr_db / 20)
    mixed   = y + scale * noise
    peak    = np.max(np.abs(mixed))
    if peak > 0:
        mixed /= peak * 1.05
    return mixed


def augment_keras(audio_files, n_aug_per_file=3):
    """
    Augmentasi khusus kran KERAS:
    - Time stretch: rate kecil (lebih lambat) = air lebih bergemuruh
    - Pitch shift: rendah = tekanan pipa berbeda
    - Noise SNR 20-28 dB (masih jelas keras)
    - Envelope manipulation: tambah burst energi
    """
    generated = []
    for src_path in audio_files:
        basename = os.path.splitext(os.path.basename(src_path))[0]
        try:
            y, sr = librosa.load(src_path, sr=SR)
            if len(y) < SR * 2:
                y = np.tile(y, int(np.ceil(SR * 2 / len(y))))
            y = y[:SR * 5]  # max 5 detik
        except Exception as e:
            print(f"  [SKIP] {basename}: {e}")
            continue

        for aug_idx in range(n_aug_per_file):
            out_name = f"{basename}_keras_aug_{aug_idx+1}.wav"
            out_path = os.path.join(KERAS_DIR, out_name)
            if os.path.exists(out_path):
                continue

            aug_type = aug_idx % 3

            if aug_type == 0:
                # Time stretch (buat suara lebih turbulent/pelan = lebih keras)
                rate = random.uniform(0.82, 0.95)
                y_aug = librosa.effects.time_stretch(y, rate=rate)

            elif aug_type == 1:
                # Pitch shift + high energy noise
                n_steps = random.uniform(-3, 3)
                y_aug = librosa.effects.pitch_shift(y, sr=SR, n_steps=n_steps)
                snr = random.uniform(20, 28)
                y_aug = mix_with_noise(y_aug, snr_db=snr)

            else:
                # Kombinasi stretch + shift + noise
                rate = random.uniform(0.88, 1.12)
                y_aug = librosa.effects.time_stretch(y, rate=rate)
                n_steps = random.uniform(-2, 2)
                y_aug = librosa.effects.pitch_shift(y_aug, sr=SR, n_steps=n_steps)
                snr = random.uniform(22, 30)
                y_aug = mix_with_noise(y_aug, snr_db=snr)

            # Pastikan RMS tinggi (kran keras memang keras)
            rms = np.sqrt(np.mean(y_aug**2))
            if rms < 0.05:
                y_aug *= (0.08 / rms)

            y_aug = np.clip(y_aug, -1.0, 1.0)
            # Trim ke 4 detik
            y_out = y_aug[:SR * 4] if len(y_aug) >= SR * 4 else np.pad(y_aug, (0, SR * 4 - len(y_aug)))
            sf.write(out_path, y_out, SR)
            generated.append(out_path)

    return generated


def augment_pelan(audio_files, n_aug_per_file=3):
    """
    Augmentasi khusus kran PELAN:
    - Time stretch: rate besar (lebih cepat) = air tapi pelan/tetes
    - Pitch shift positif = suara lebih tipis/pelan
    - Noise SNR 28-38 dB (hampir tidak ada noise = murni air pelan)
    - Tambah slight reverb untuk simulasi kamar mandi kecil
    """
    generated = []
    for src_path in audio_files:
        basename = os.path.splitext(os.path.basename(src_path))[0]
        try:
            y, sr = librosa.load(src_path, sr=SR)
            if len(y) < SR * 2:
                y = np.tile(y, int(np.ceil(SR * 2 / len(y))))
            y = y[:SR * 5]
        except Exception as e:
            print(f"  [SKIP] {basename}: {e}")
            continue

        for aug_idx in range(n_aug_per_file):
            out_name = f"{basename}_pelan_aug_{aug_idx+1}.wav"
            out_path = os.path.join(PELAN_DIR, out_name)
            if os.path.exists(out_path):
                continue

            aug_type = aug_idx % 3

            if aug_type == 0:
                # Time stretch ke lebih cepat (suara lebih 'tipis')
                rate = random.uniform(1.05, 1.20)
                y_aug = librosa.effects.time_stretch(y, rate=rate)

            elif aug_type == 1:
                # Pitch shift ke atas = suara lebih tipis/tetes air
                n_steps = random.uniform(1, 4)
                y_aug = librosa.effects.pitch_shift(y, sr=SR, n_steps=n_steps)
                snr = random.uniform(28, 38)
                y_aug = mix_with_noise(y_aug, snr_db=snr)

            else:
                # Kombinasi: kurangi energi sedikit + pitch naik + noise minimal
                rate = random.uniform(0.95, 1.10)
                y_aug = librosa.effects.time_stretch(y, rate=rate)
                n_steps = random.uniform(-1, 2)
                y_aug = librosa.effects.pitch_shift(y_aug, sr=SR, n_steps=n_steps)
                snr = random.uniform(30, 40)
                y_aug = mix_with_noise(y_aug, snr_db=snr)
                # Sedikit kurangi volume untuk simulasi air pelan (bukan keras)
                y_aug *= random.uniform(0.6, 0.9)

            # Pastikan RMS tidak terlalu tinggi (kran pelan memang pelan)
            rms = np.sqrt(np.mean(y_aug**2))
            if rms > 0.15:
                y_aug *= (0.10 / rms)

            y_aug = np.clip(y_aug, -1.0, 1.0)
            y_out = y_aug[:SR * 4] if len(y_aug) >= SR * 4 else np.pad(y_aug, (0, SR * 4 - len(y_aug)))
            sf.write(out_path, y_out, SR)
            generated.append(out_path)

    return generated


def create_contrast_pairs():
    """
    Buat pasangan kontras: satu keras satu pelan dari file yang sama,
    untuk melatih model membedakan dua kelas ini lebih baik.
    Menggunakan volume manipulation.
    """
    keras_files = [f for f in glob.glob(os.path.join(KERAS_DIR, "*.wav"))
                   if '_aug_' not in f and '_contrast_' not in f]
    pelan_files = [f for f in glob.glob(os.path.join(PELAN_DIR, "*.wav"))
                   if '_aug_' not in f and '_contrast_' not in f]

    gen_keras = 0
    gen_pelan = 0

    # Dari file keras: buat versi lebih keras (RMS lebih tinggi)
    for i, fp in enumerate(keras_files[:30]):
        try:
            y, sr = librosa.load(fp, sr=SR)
            if len(y) < SR * 2:
                y = np.tile(y, int(np.ceil(SR * 2 / len(y))))

            # Versi "super keras": amplifikasi + compander
            y_loud = y * 1.5
            y_loud = np.clip(y_loud, -1.0, 1.0)
            # Tambah high-freq boost (air keras lebih banyak high freq)
            y_loud = librosa.effects.preemphasis(y_loud, coef=0.08)
            y_loud = np.clip(y_loud, -1.0, 1.0)

            out_name = f"keras_contrast_{i+1:03d}.wav"
            out_path = os.path.join(KERAS_DIR, out_name)
            if not os.path.exists(out_path):
                y_out = y_loud[:SR * 4] if len(y_loud) >= SR * 4 else np.pad(y_loud, (0, SR * 4 - len(y_loud)))
                sf.write(out_path, y_out, SR)
                gen_keras += 1
        except Exception as e:
            print(f"  [SKIP contrast keras] {e}")

    # Dari file pelan: buat versi lebih pelan (RMS lebih rendah)
    for i, fp in enumerate(pelan_files[:30]):
        try:
            y, sr = librosa.load(fp, sr=SR)
            if len(y) < SR * 2:
                y = np.tile(y, int(np.ceil(SR * 2 / len(y))))

            # Versi "super pelan": kurangi volume drastis
            y_quiet = y * 0.4
            # Low-pass filter untuk simulasi air pelan (lebih banyak frekuensi rendah)
            from scipy.signal import butter, filtfilt
            b, a = butter(4, 3000 / (SR / 2), btype='low')
            y_quiet = filtfilt(b, a, y_quiet)
            y_quiet = np.clip(y_quiet, -1.0, 1.0)

            out_name = f"pelan_contrast_{i+1:03d}.wav"
            out_path = os.path.join(PELAN_DIR, out_name)
            if not os.path.exists(out_path):
                y_out = y_quiet[:SR * 4] if len(y_quiet) >= SR * 4 else np.pad(y_quiet, (0, SR * 4 - len(y_quiet)))
                sf.write(out_path, y_out, SR)
                gen_pelan += 1
        except Exception as e:
            print(f"  [SKIP contrast pelan] {e}")

    return gen_keras, gen_pelan


def main():
    print("=" * 65)
    print("  AUGMENTASI V2: Fokus Diskriminasi Kran Pelan vs Keras")
    print("=" * 65)

    # File asli (bukan augmentasi sebelumnya)
    keras_originals = [f for f in glob.glob(os.path.join(KERAS_DIR, "*.wav")) +
                       glob.glob(os.path.join(KERAS_DIR, "*.mp3"))
                       if '_aug_' not in os.path.basename(f) and
                          '_contrast_' not in os.path.basename(f) and
                          '_keras_aug_' not in os.path.basename(f) and
                          '_pelan_aug_' not in os.path.basename(f)]

    pelan_originals = [f for f in glob.glob(os.path.join(PELAN_DIR, "*.wav")) +
                       glob.glob(os.path.join(PELAN_DIR, "*.mp3"))
                       if '_aug_' not in os.path.basename(f) and
                          '_contrast_' not in os.path.basename(f) and
                          '_keras_aug_' not in os.path.basename(f) and
                          '_pelan_aug_' not in os.path.basename(f)]

    print(f"\n[*] File keras original: {len(keras_originals)}")
    print(f"[*] File pelan original: {len(pelan_originals)}")

    # 1. Augmentasi kran keras
    print("\n[1/3] Augmentasi kran KERAS (3 variasi per file)...")
    gen_k = augment_keras(keras_originals, n_aug_per_file=3)
    print(f"  → Generated {len(gen_k)} file keras baru")

    # 2. Augmentasi kran pelan
    print("\n[2/3] Augmentasi kran PELAN (3 variasi per file)...")
    gen_p = augment_pelan(pelan_originals, n_aug_per_file=3)
    print(f"  → Generated {len(gen_p)} file pelan baru")

    # 3. Contrast pairs
    print("\n[3/3] Membuat pasangan kontras keras/pelan...")
    gk, gp = create_contrast_pairs()
    print(f"  → Generated {gk} keras contrast + {gp} pelan contrast")

    # Hitung total
    keras_total = len(glob.glob(os.path.join(KERAS_DIR, "*.wav")) +
                       glob.glob(os.path.join(KERAS_DIR, "*.mp3")))
    pelan_total = len(glob.glob(os.path.join(PELAN_DIR, "*.wav")) +
                       glob.glob(os.path.join(PELAN_DIR, "*.mp3")))
    bukan_total = len(glob.glob(os.path.join(DATASET_DIR, "bukan_air", "*.wav")) +
                       glob.glob(os.path.join(DATASET_DIR, "bukan_air", "*.mp3")))

    print("\n" + "=" * 65)
    print(f"[HASIL] Dataset setelah augmentasi v2:")
    print(f"  kran keras: {keras_total} file")
    print(f"  kran pelan: {pelan_total} file")
    print(f"  bukan air : {bukan_total} file")
    print(f"  TOTAL     : {keras_total + pelan_total + bukan_total} file")
    print("=" * 65)
    print("[SELESAI] Augmentasi V2 Selesai!")
    print("[NEXT] Jalankan train_v2.py untuk melatih model dengan dataset baru")


if __name__ == "__main__":
    main()
