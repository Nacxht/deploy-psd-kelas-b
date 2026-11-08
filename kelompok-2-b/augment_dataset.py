"""
augment_dataset.py
==================
Melakukan augmentasi audio yang seimbang dan realistis:
1. Menghapus file augmentasi lama (_aug*.wav) agar bersih.
2. Untuk kelas air (PELAN & KERAS):
   - Augmentasi akustik: Gaussian noise (laptop mic), pitch shift, time stretch, volume scale.
   - Background Mix: Mencampur air dengan background desisan kipas atau adzan lemah (SNR 12-18 dB),
     agar model tetap mengenali aliran air meski ada kipas atau adzan di latar belakang.
3. Untuk BUKAN AIR:
   - Augmentasi suara kipas & adzan dengan variasi gain dan mic noise.
"""

import os
import sys
import glob
import numpy as np
import soundfile as sf
import librosa

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SR          = 22050

def clean_old_augmentations():
    print("[*] Membersihkan file augmentasi lama (*_aug*.wav)...")
    deleted = 0
    for f in glob.glob(os.path.join(DATASET_DIR, "*", "*_aug*.wav")):
        try:
            os.remove(f)
            deleted += 1
        except Exception:
            pass
    print(f"  -> Berhasil menghapus {deleted} file augmentasi lama.")

def add_gaussian_noise(y, noise_level=0.004):
    noise = np.random.normal(0, noise_level, len(y))
    return np.clip(y + noise, -1.0, 1.0)

def pitch_shift_aug(y, sr, n_steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def time_stretch_aug(y, rate):
    stretched = librosa.effects.time_stretch(y, rate=rate)
    if len(stretched) > len(y):
        stretched = stretched[:len(y)]
    else:
        stretched = np.pad(stretched, (0, len(y) - len(stretched)), mode='constant')
    return stretched

def volume_scale(y, factor):
    return np.clip(y * factor, -1.0, 1.0)

def mix_background(water_y, bg_y, snr_db=14.0):
    """Mencampur suara air dengan background noise (kipas/adzan) dengan SNR tertentu"""
    if len(bg_y) < len(water_y):
        repeats = int(np.ceil(len(water_y) / len(bg_y)))
        bg_y = np.tile(bg_y, repeats)[:len(water_y)]
    else:
        bg_y = bg_y[:len(water_y)]
        
    rms_w = np.sqrt(np.mean(water_y**2)) + 1e-9
    rms_b = np.sqrt(np.mean(bg_y**2)) + 1e-9
    
    desired_ratio = 10.0 ** (snr_db / 20.0)
    scaled_bg = bg_y * (rms_w / (desired_ratio * rms_b))
    
    mixed = water_y + scaled_bg
    return np.clip(mixed, -1.0, 1.0)

def main():
    print("=" * 70)
    print("  AUGMENTASI DATASET AUDIO - ROBUST & REALISTIS")
    print("=" * 70)

    clean_old_augmentations()

    # Siapkan sampel background untuk mix (kipas dan adzan)
    bukan_dir = os.path.join(DATASET_DIR, "bukan_air")
    kipas_samples = glob.glob(os.path.join(bukan_dir, "kipas_*.wav"))[:10]
    adzan_samples = (glob.glob(os.path.join(bukan_dir, "adzan_*.wav")) +
                     glob.glob(os.path.join(bukan_dir, "adzan_*.mp3")))[:10]

    bg_audios = []
    for f in (kipas_samples + adzan_samples):
        try:
            y_bg, _ = librosa.load(f, sr=SR)
            bg_audios.append(y_bg)
        except Exception:
            pass

    # Konfigurasi augmentasi per kelas
    # Pelan: 74 orig -> n_aug=3 -> ~300 file
    # Keras: 97 orig -> n_aug=2 -> ~300 file
    # Bukan_air: 669 orig (sudah sangat banyak & beragam) -> augmentasi selektif untuk kipas & adzan
    AUG_CONFIG = {
        "pelan": 3,
        "keras": 2,
    }

    for cat_name, n_aug in AUG_CONFIG.items():
        cat_dir = os.path.join(DATASET_DIR, cat_name)
        orig_files = [
            f for f in (
                glob.glob(os.path.join(cat_dir, "*.mp3")) +
                glob.glob(os.path.join(cat_dir, "*.wav")) +
                glob.glob(os.path.join(cat_dir, "*.m4a"))
            )
            if "_aug" not in os.path.basename(f)
        ]

        print(f"\n[*] Melakukan augmentasi untuk kelas '{cat_name.upper()}': {len(orig_files)} file asli...")
        created = 0

        for fpath in orig_files:
            bname = os.path.splitext(os.path.basename(fpath))[0]
            try:
                y, sr = librosa.load(fpath, sr=SR)
                if len(y) < SR:
                    continue

                for aug_idx in range(n_aug):
                    out_name = f"{bname}_aug{aug_idx+1}.wav"
                    out_path = os.path.join(cat_dir, out_name)

                    if aug_idx == 0:
                        # Noise mic laptop
                        y_aug = add_gaussian_noise(y, noise_level=0.005)
                    elif aug_idx == 1:
                        # Pitch shift ringan (+1.5 semitone)
                        y_aug = pitch_shift_aug(y, sr, n_steps=1.5)
                    elif aug_idx == 2 and bg_audios:
                        # Background mix dengan desisan kipas / adzan (SNR 14 dB)
                        bg_choice = bg_audios[np.random.randint(len(bg_audios))]
                        y_aug = mix_background(y, bg_choice, snr_db=np.random.uniform(12.0, 16.0))
                    else:
                        # Volume scaling (kran sedikit lebih jauh / dekat)
                        y_aug = volume_scale(y, factor=np.random.uniform(0.6, 1.2))

                    sf.write(out_path, y_aug.astype(np.float32), SR)
                    created += 1

            except Exception as e:
                pass

        total_cat = len(glob.glob(os.path.join(cat_dir, "*.*")))
        print(f"  -> Kelas {cat_name}: {created} file augmentasi baru dibuat (Total: {total_cat} file)")

    print("\n" + "=" * 70)
    print("  RINGKASAN TOTAL DATASET:")
    for c in ["bukan_air", "pelan", "keras"]:
        cnt = len(glob.glob(os.path.join(DATASET_DIR, c, "*.*")))
        print(f"  - {c:<12}: {cnt} file")
    print("=" * 70)

if __name__ == "__main__":
    main()
