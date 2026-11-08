"""
update_user_dataset.py
=======================
Memperbarui data suara user dari folder Downloads/all_data-20261102T110425Z-1-001/all_data
ke dalam dataset/keras dan dataset/pelan:
1. Menghapus HANYA data user lama (air_deras_* dan air_pelan_*) beserta augmentasinya.
2. Mempertahankan SEMUA data tambahan (esc50_*, air_*_cleaned_pure_*, *contrast_*, bukan_air/*).
3. Menyalin data baru dari Downloads:
   - air_deras_001.mp3 s/d air_deras_046.mp3
   - air_deras_047.m4a s/d air_deras_066.m4a (dikonversi ke .wav mono 22050Hz via ffmpeg)
   - air_pelan_001.mp3 s/d air_pelan_043.mp3
4. Menjalankan augmentasi realistis khusus untuk file user baru agar dataset seimbang dan kaya.
"""

import os
import sys
import glob
import shutil
import subprocess
import numpy as np
import soundfile as sf
import librosa
import imageio_ffmpeg

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
KERAS_DIR   = os.path.join(DATASET_DIR, "keras")
PELAN_DIR   = os.path.join(DATASET_DIR, "pelan")
BUKAN_DIR   = os.path.join(DATASET_DIR, "bukan_air")

DOWNLOADS_DIR = r"C:\Users\Raihan aryanova\Downloads\all_data-20261102T110425Z-1-001\all_data"
DL_DERAS = os.path.join(DOWNLOADS_DIR, "air_deras")
DL_PELAN = os.path.join(DOWNLOADS_DIR, "air_pelan")

SR = 22050

def step1_remove_old_user_data():
    print("[1/4] Menghapus data suara user lama (air_deras_* & air_pelan_*)...")
    
    deras_files = glob.glob(os.path.join(KERAS_DIR, "air_deras_*.*"))
    pelan_files = glob.glob(os.path.join(PELAN_DIR, "air_pelan_*.*"))
    
    print(f"  - Menemukan {len(deras_files)} file air_deras lama di keras")
    for f in deras_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"    Gagal hapus {f}: {e}")
            
    print(f"  - Menemukan {len(pelan_files)} file air_pelan lama di pelan")
    for f in pelan_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"    Gagal hapus {f}: {e}")

    print("  -> Data user lama berhasil dibersihkan. Data tambahan tetap utuh.")

def step2_copy_new_user_data():
    print("[2/4] Menyalin data suara user baru dari Downloads...")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    # 1. Keras / Deras
    deras_src = sorted(glob.glob(os.path.join(DL_DERAS, "*.*")))
    deras_copied = 0
    for f in deras_src:
        ext = os.path.splitext(f)[1].lower()
        bname = os.path.splitext(os.path.basename(f))[0]
        if ext == '.m4a':
            dst = os.path.join(KERAS_DIR, f"{bname}.wav")
            cmd = [ffmpeg_exe, '-y', '-i', f, '-ar', str(SR), '-ac', '1', dst]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            deras_copied += 1
        else:
            dst = os.path.join(KERAS_DIR, os.path.basename(f))
            shutil.copy2(f, dst)
            deras_copied += 1
    print(f"  -> Berhasil menyalin/mengonversi {deras_copied} file air_deras ke dataset/keras")

    # 2. Pelan
    pelan_src = sorted(glob.glob(os.path.join(DL_PELAN, "*.*")))
    pelan_copied = 0
    for f in pelan_src:
        dst = os.path.join(PELAN_DIR, os.path.basename(f))
        shutil.copy2(f, dst)
        pelan_copied += 1
    print(f"  -> Berhasil menyalin {pelan_copied} file air_pelan ke dataset/pelan")

def add_noise(y, noise_level=0.004):
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

def step3_augment_new_user_data():
    print("[3/4] Melakukan augmentasi akustik realistis untuk data user baru...")
    
    # Keras: 66 file asli -> buat 3 variasi per file -> ~264 file user
    # Pelan: 43 file asli -> buat 4 variasi per file -> ~215 file user
    aug_targets = [
        ("keras", "air_deras", KERAS_DIR, 3),
        ("pelan", "air_pelan", PELAN_DIR, 4)
    ]
    
    total_aug = 0
    for cat_name, prefix, cat_dir, n_aug in aug_targets:
        # Ambil hanya file asli user baru (bukan augmentasi)
        user_files = [
            f for f in glob.glob(os.path.join(cat_dir, f"{prefix}_*.*"))
            if "_aug" not in os.path.basename(f)
        ]
        print(f"  - Mengaugmentasi {len(user_files)} file {prefix} di {cat_name} ({n_aug} variasi/file)...")
        cat_aug_count = 0
        
        for fpath in user_files:
            bname = os.path.splitext(os.path.basename(fpath))[0]
            try:
                y, sr = librosa.load(fpath, sr=SR)
                if len(y) < SR:
                    continue
                
                for idx in range(n_aug):
                    out_path = os.path.join(cat_dir, f"{bname}_aug{idx+1}.wav")
                    
                    if idx == 0:
                        # Variasi noise mikrofon laptop
                        y_aug = add_noise(y, noise_level=np.random.uniform(0.003, 0.006))
                    elif idx == 1:
                        # Variasi pitch (+/- 1.5 semitone)
                        steps = 1.5 if (cat_name == "keras") else -1.5
                        y_aug = pitch_shift_aug(y, sr, n_steps=steps)
                    elif idx == 2:
                        # Variasi time stretch (sedikit lebih cepat / lambat)
                        rate = 1.08 if (cat_name == "keras") else 0.92
                        y_aug = time_stretch_aug(y, rate=rate)
                    else:
                        # Variasi volume (jarak mic sedikit lebih dekat / jauh)
                        y_aug = volume_scale(y, factor=np.random.uniform(0.7, 1.25))
                        
                    sf.write(out_path, y_aug.astype(np.float32), SR)
                    cat_aug_count += 1
            except Exception as e:
                pass
                
        print(f"    -> Dibuat {cat_aug_count} file augmentasi baru untuk {cat_name}.")
        total_aug += cat_aug_count

    print(f"  -> Total {total_aug} file augmentasi baru berhasil dibuat.")

def step4_summary():
    print("[4/4] Verifikasi dan Rekapitulasi Dataset Terkini:")
    print("=" * 65)
    for c in ["bukan_air", "pelan", "keras"]:
        cdir = os.path.join(DATASET_DIR, c)
        all_f = glob.glob(os.path.join(cdir, "*.*"))
        user_f = [f for f in all_f if os.path.basename(f).startswith("air_deras_") or os.path.basename(f).startswith("air_pelan_")]
        tambahan_f = [f for f in all_f if f not in user_f]
        print(f"  - Kategori {c:<10}: Total = {len(all_f):4d} | User (orig+aug) = {len(user_f):4d} | Tambahan = {len(tambahan_f):4d}")
    print("=" * 65)

if __name__ == "__main__":
    print("=" * 65)
    print("  UPDATE DATASET SUARA USER (TANPA MENGUBAH DATA TAMBAHAN)")
    print("=" * 65)
    step1_remove_old_user_data()
    step2_copy_new_user_data()
    step3_augment_new_user_data()
    step4_summary()
