"""
pisahkan_air_dan_adzan.py
=========================
Skrip untuk memisahkan dan mengkatalogkan file audio:
1. Mendeteksi apakah file di dataset/pelan dan dataset/keras adalah:
   - 'pure_air' : Aliran air kran murni tanpa suara adzan/vokal
   - 'campur_adzan' : Aliran air yang terekam saat ada kumandang adzan / suara vokal masjid
2. Mengekstrak suara Adzan murni (Harmonic track) menggunakan HPSS dan menyimpannya ke
   dataset/bukan_air/ sebagai contoh suara Adzan Masjid (Class 0: Bukan Air).
3. Mengekstrak suara Air murni yang bersih (Percussive/Noise residual) dan menyimpannya
   sebagai dataset variasi pure air tambahan.
4. Menyimpan katalog lengkap pemisahan ke output/pemisahan_air_adzan.csv
"""

import os
import sys
import glob
import librosa
import soundfile as sf
import numpy as np
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SR = 22050

def analyze_and_separate(cat_name):
    cat_dir = os.path.join(DATASET_DIR, cat_name)
    bukan_air_dir = os.path.join(DATASET_DIR, "bukan_air")
    os.makedirs(bukan_air_dir, exist_ok=True)
    
    all_files = sorted(
        glob.glob(os.path.join(cat_dir, "*.mp3")) +
        glob.glob(os.path.join(cat_dir, "*.wav")) +
        glob.glob(os.path.join(cat_dir, "*.m4a"))
    )
    # Hanya analisis file asli (bukan hasil augmentasi _aug)
    orig_files = [f for f in all_files if "_aug" not in os.path.basename(f) and "_cleaned_" not in os.path.basename(f)]
    
    print(f"\n[*] Menganalisis {len(orig_files)} file asli di kelas '{cat_name.upper()}'...")
    catalog = []
    
    for idx, fpath in enumerate(orig_files, 1):
        fname = os.path.basename(fpath)
        try:
            y, sr = librosa.load(fpath, sr=SR)
            dur = len(y) / sr
            
            # Harmonic-Percussive Source Separation
            # Margin=(2.5, 1.0) memisahkan vokal vokal bernada tajam (harm) dari noise air (perc)
            y_harm, y_perc = librosa.effects.hpss(y, margin=(2.5, 1.0))
            
            tot_rms  = float(np.mean(librosa.feature.rms(y=y)))
            harm_rms = float(np.mean(librosa.feature.rms(y=y_harm)))
            perc_rms = float(np.mean(librosa.feature.rms(y=y_perc)))
            flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
            
            # Hitung rasio energi di pita frekuensi vokal adzan (300 - 3000 Hz)
            S = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=sr)
            vocal_idx = np.where((freqs >= 250) & (freqs <= 3000))[0]
            vocal_ratio = float(np.sum(S[vocal_idx, :]**2) / (np.sum(S**2) + 1e-9))
            
            # Kriteria Adzan: Harmonic RMS tinggi, flatness rendah, energi vokal dominan
            is_campur_adzan = False
            if (harm_rms / (perc_rms + 1e-6) > 1.8 and flatness < 0.12) or (harm_rms > 0.05 and vocal_ratio > 0.60):
                is_campur_adzan = True
            
            tipe = "campur_adzan" if is_campur_adzan else "pure_air"
            
            # Jika campur adzan, kita lakukan pemisahan fisik audionya:
            adzan_file_saved = ""
            clean_air_file_saved = ""
            
            if is_campur_adzan:
                # 1. Simpan vokal adzan murni ke bukan_air (Class 0)
                adzan_name = f"adzan_dari_rekaman_{cat_name}_{idx:03d}.wav"
                adzan_path = os.path.join(bukan_air_dir, adzan_name)
                # Normalisasi suara adzan
                y_harm_norm = y_harm / (np.max(np.abs(y_harm)) + 1e-6) * 0.25
                sf.write(adzan_path, y_harm_norm.astype(np.float32), SR)
                adzan_file_saved = adzan_name
                
                # 2. Simpan residual suara air murni ke folder kelas air (Class 1 atau 2)
                clean_name = f"air_{cat_name}_cleaned_pure_{idx:03d}.wav"
                clean_path = os.path.join(cat_dir, clean_name)
                # Normalisasi suara air murni
                y_perc_norm = y_perc / (np.max(np.abs(y_perc)) + 1e-6) * 0.20
                sf.write(clean_path, y_perc_norm.astype(np.float32), SR)
                clean_air_file_saved = clean_name
            
            catalog.append({
                "kategori": cat_name,
                "nama_file": fname,
                "tipe": tipe,
                "durasi_detik": round(dur, 2),
                "rms_total": round(tot_rms, 4),
                "rms_harmonik": round(harm_rms, 4),
                "rms_perkusif": round(perc_rms, 4),
                "spectral_flatness": round(flatness, 4),
                "vocal_energy_ratio": round(vocal_ratio, 4),
                "ekstrak_adzan": adzan_file_saved,
                "ekstrak_air_bersih": clean_air_file_saved
            })
            
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")
            
    df_cat = pd.DataFrame(catalog)
    n_pure = len(df_cat[df_cat["tipe"] == "pure_air"])
    n_adzan = len(df_cat[df_cat["tipe"] == "campur_adzan"])
    print(f"  -> Hasil {cat_name}: {n_pure} PURE AIR, {n_adzan} CAMPUR ADZAN")
    return df_cat

def main():
    print("=" * 70)
    print("  PEMISAHAN DATASET: AIR PURE AIR VS AIR KENA ADZAN")
    print("  Menggunakan Harmonic-Percussive Source Separation (HPSS)")
    print("=" * 70)
    
    df_pelan = analyze_and_separate("pelan")
    df_keras = analyze_and_separate("keras")
    
    df_all = pd.concat([df_pelan, df_keras], ignore_index=True)
    out_csv = os.path.join(OUTPUT_DIR, "pemisahan_air_adzan.csv")
    df_all.to_csv(out_csv, index=False)
    
    print("\n" + "=" * 70)
    print(f"  RINGKASAN PEMISAHAN:")
    print(f"  Total File Asli Dianalisis : {len(df_all)}")
    print(f"  Air Murni (Pure Air)       : {len(df_all[df_all['tipe'] == 'pure_air'])} file")
    print(f"  Air Kena Adzan             : {len(df_all[df_all['tipe'] == 'campur_adzan'])} file")
    print(f"  -> File Vokal Adzan Diekstrak ke bukan_air : {len(df_all[df_all['tipe'] == 'campur_adzan'])} file")
    print(f"  -> File Air Bersih Ditambahkan ke kelas air: {len(df_all[df_all['tipe'] == 'campur_adzan'])} file")
    print(f"  Laporan lengkap tersimpan di: {out_csv}")
    print("=" * 70)

if __name__ == "__main__":
    main()
