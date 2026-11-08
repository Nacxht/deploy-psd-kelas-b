"""
tambah_bukan_air.py
===================
Menambah dataset 'bukan_air' secara komprehensif:
1. Suara KIPAS ANGIN berbagai variasi (mengatasi fan terdeteksi kran pelan):
   - Kipas berdiri (Speed 1, 2, 3)
   - Kipas dinding masjid (motor hum + blade pass)
   - Exhaust fan ventilasi tempat wudhu / toilet
   - Kipas osilasi (volume mengayun/swelling)
   - Kipas laptop / pendingin
2. Suara ADZAN MASJID & VOKAL:
   - Mengunduh rekaman Adzan asli (Wikimedia Commons) dan memotongnya jadi segmen 3 detik
   - Vokal adzan dari hasil pemisahan rekaman
3. Suara ALIRAN ANGIN / BLOWER nyata dari ESC-50 (kategori 'wind')
4. White, Pink, Brown noise, keyboard, ambient langkah kaki, obrolan
"""

import os
import sys
import glob
import urllib.request
import requests
import numpy as np
import soundfile as sf
import librosa
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(BASE_DIR, "dataset", "bukan_air")
os.makedirs(OUT_DIR, exist_ok=True)

SR       = 22050
DURATION = 3.0        # 3 detik per file
SAMPLES  = int(SR * DURATION)

def save_wav(filename, audio, sr=SR):
    path = os.path.join(OUT_DIR, filename)
    if not os.path.exists(path):
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        sf.write(path, audio, sr)
        return True
    return False

def white_noise(n, scale=0.15):
    return np.random.normal(0, scale, n)

def pink_noise(n, scale=0.15):
    """Filter 1/f pink noise"""
    white = np.random.randn(n)
    f = np.fft.rfftfreq(n)
    f[0] = 1e-6
    power = 1.0 / np.sqrt(f)
    pink = np.fft.irfft(np.fft.rfft(white) * power, n=n)
    pink = pink / (np.max(np.abs(pink)) + 1e-6) * scale
    return pink

def brown_noise(n, scale=0.15):
    """Brown noise (1/f^2) integrasi white noise"""
    w = np.random.randn(n)
    b = np.cumsum(w)
    b = b / (np.max(np.abs(b)) + 1e-6) * scale
    return b

def sine_tone(freq, n, amplitude=0.2, sr=SR):
    t = np.linspace(0, n/sr, n)
    return amplitude * np.sin(2 * np.pi * freq * t)

# =====================================================================
#   GENERATOR SUARA KIPAS ANGIN REALISTIS (BERBAGAI VARIASI)
# =====================================================================
def realistic_fan_sound(speed=1, fan_type="standing", sr=SR, duration=DURATION):
    """
    Membuat profil akustik kipas angin nyata:
    - Motor hum harmonik (50Hz PLN + 100, 150, 200, 250, 300Hz)
    - Blade-pass frequency (aerodynamic pulse 20-60 Hz)
    - Air turbulence friction (shaped bandpass pink noise)
    - Oscillating swell jika tipe 'oscillating'
    """
    n = int(sr * duration)
    t = np.linspace(0, duration, n)
    
    # 1. Komponen Aerodinamis (Aliran Udara)
    if speed == 1:      # Pelan
        air_scale = np.random.uniform(0.04, 0.08)
        base_air = pink_noise(n, scale=air_scale)
        bpf = np.random.uniform(25.0, 35.0)     # Blade pass freq ~30Hz
        mod_depth = 0.25
    elif speed == 2:    # Sedang
        air_scale = np.random.uniform(0.08, 0.14)
        base_air = pink_noise(n, scale=air_scale)
        bpf = np.random.uniform(40.0, 52.0)
        mod_depth = 0.35
    else:               # Kencang (Speed 3)
        air_scale = np.random.uniform(0.12, 0.22)
        base_air = 0.7 * pink_noise(n, scale=air_scale) + 0.3 * white_noise(n, scale=air_scale*0.5)
        bpf = np.random.uniform(55.0, 75.0)
        mod_depth = 0.45

    # Modulasi pulsa bilah kipas (blade flutter)
    blade_pulse = 1.0 + mod_depth * np.sin(2 * np.pi * bpf * t)
    audio = base_air * blade_pulse

    # 2. Komponen Motor Listrik (50Hz AC + harmonik)
    motor_vol = np.random.uniform(0.02, 0.06)
    audio += sine_tone(50.0,  n, amplitude=motor_vol, sr=sr)
    audio += sine_tone(100.0, n, amplitude=motor_vol * 0.5, sr=sr)
    audio += sine_tone(150.0, n, amplitude=motor_vol * 0.3, sr=sr)
    audio += sine_tone(200.0, n, amplitude=motor_vol * 0.2, sr=sr)

    # 3. Fitur Khusus berdasarkan Tipe Kipas
    if fan_type == "exhaust":
        # Exhaust fan: rongga saluran udara (resonansi duct)
        audio += sine_tone(280.0, n, amplitude=0.03, sr=sr)
        audio += brown_noise(n, scale=0.04)
    elif fan_type == "wall_masjid":
        # Kipas dinding masjid: getaran rangka besi/dinding
        audio += sine_tone(60.0, n, amplitude=0.03, sr=sr)
        audio += sine_tone(120.0, n, amplitude=0.02, sr=sr)
    elif fan_type == "oscillating":
        # Kipas geleng-geleng: volume membesar dan mengecil secara periodik
        osc_period = np.random.uniform(4.0, 7.0)
        osc_env = 0.6 + 0.4 * np.sin(2 * np.pi * (1.0 / osc_period) * t)
        audio = audio * osc_env
    elif fan_type == "laptop":
        # Kipas laptop: desisan nada tinggi (high-pitch rotor hiss)
        high_whine = np.random.uniform(1200.0, 2400.0)
        audio += sine_tone(high_whine, n, amplitude=0.015, sr=sr)

    return audio

# =====================================================================
#   UNDUH SAMPEL SUARA NYATA DARI INTERNET (ESC-50 & ADZAN)
# =====================================================================
def download_real_wind_and_air():
    """Mengunduh 40 sampel 'wind' (desisan hembusan angin/udara) nyata dari ESC-50"""
    print("\n[*] Mengunduh rekaman desisan angin & aliran udara nyata dari ESC-50...")
    esc_meta_url = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
    try:
        df_esc = pd.read_csv(esc_meta_url)
        wind_samples = df_esc[df_esc["category"] == "wind"]
        count = 0
        for idx, row in wind_samples.iterrows():
            filename = row["filename"]
            url = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{filename}"
            dest = os.path.join(OUT_DIR, f"wind_airflow_{count+1:03d}.wav")
            if not os.path.exists(dest):
                try:
                    urllib.request.urlretrieve(url, dest)
                    count += 1
                except Exception as e:
                    pass
        print(f"  -> {count} file desisan aliran angin nyata berhasil diunduh ke bukan_air!")
    except Exception as e:
        print(f"  [SKIP] Gagal unduh ESC-50 wind: {e}")

def download_and_slice_real_adhan():
    """Mengunduh audio Adzan asli dan memotong menjadi segmen 3 detik"""
    print("\n[*] Mengunduh rekaman Adzan Masjid asli (Wikimedia Commons)...")
    adhan_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/7/7d/The_Adhan_-_Muslim_Call_to_Prayer_-_Aaqib_Azeez.mp3"
    ]
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    saved_clips = 0
    for idx_u, url in enumerate(adhan_urls, 1):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                temp_mp3 = os.path.join(OUT_DIR, f"_temp_adhan_{idx_u}.mp3")
                with open(temp_mp3, "wb") as f:
                    f.write(r.content)
                
                # Load dan potong per 3 detik
                y, sr = librosa.load(temp_mp3, sr=SR)
                chunk_len = int(3.0 * sr)
                for start in range(0, len(y) - chunk_len, chunk_len):
                    clip = y[start : start + chunk_len]
                    clip_rms = float(np.mean(librosa.feature.rms(y=clip)))
                    if clip_rms > 0.02:   # Hanya simpan bagian yang ada suaranya
                        clip_name = f"adzan_masjid_asli_{idx_u}_{saved_clips+1:03d}.wav"
                        clip_path = os.path.join(OUT_DIR, clip_name)
                        if not os.path.exists(clip_path):
                            sf.write(clip_path, clip.astype(np.float32), sr)
                            saved_clips += 1
                
                if os.path.exists(temp_mp3):
                    os.remove(temp_mp3)
                    
        except Exception as e:
            print(f"  [SKIP] Gagal memproses adzan {idx_u}: {e}")
            
    print(f"  -> {saved_clips} segmen Adzan asli berhasil dibuat di bukan_air!")

# =====================================================================
#   MAIN
# =====================================================================
def main():
    print("=" * 70)
    print("  PENAMBAHAN DATASET BUKAN AIR: KIPAS ANGIN, ADZAN, ANGIN NYATA")
    print("=" * 70)

    dibuat = 0

    # 1. Kipas Angin Berdiri (Speed 1, 2, 3)
    print("\n[*] Membuat Variasi Kipas Angin Berdiri (Speed 1, 2, 3)...")
    for sp in [1, 2, 3]:
        for i in range(1, 26):
            audio = realistic_fan_sound(speed=sp, fan_type="standing")
            if save_wav(f"kipas_berdiri_speed{sp}_{i:03d}.wav", audio): dibuat += 1

    # 2. Kipas Dinding Masjid
    print("[*] Membuat Variasi Kipas Angin Dinding Masjid...")
    for i in range(1, 31):
        sp = np.random.choice([1, 2, 3])
        audio = realistic_fan_sound(speed=sp, fan_type="wall_masjid")
        if save_wav(f"kipas_dinding_masjid_{i:03d}.wav", audio): dibuat += 1

    # 3. Kipas Exhaust Fan Tempat Wudhu / Toilet
    print("[*] Membuat Variasi Exhaust Fan Tempat Wudhu / Toilet...")
    for i in range(1, 31):
        sp = np.random.choice([2, 3])
        audio = realistic_fan_sound(speed=sp, fan_type="exhaust")
        if save_wav(f"kipas_exhaust_wudhu_{i:03d}.wav", audio): dibuat += 1

    # 4. Kipas Osilasi (Volume Mengayun)
    print("[*] Membuat Variasi Kipas Angin Osilasi (Swelling)...")
    for i in range(1, 26):
        audio = realistic_fan_sound(speed=2, fan_type="oscillating")
        if save_wav(f"kipas_osilasi_{i:03d}.wav", audio): dibuat += 1

    # 5. Kipas Laptop / Pendingin
    print("[*] Membuat Variasi Kipas Laptop / Pendingin...")
    for i in range(1, 21):
        audio = realistic_fan_sound(speed=np.random.choice([1, 2, 3]), fan_type="laptop")
        if save_wav(f"kipas_laptop_{i:03d}.wav", audio): dibuat += 1

    # 6. Unduh Sampel Angin Nyata (ESC-50)
    download_real_wind_and_air()

    # 7. Unduh dan Potong Rekaman Adzan Asli
    download_and_slice_real_adhan()

    total_files = len(glob.glob(os.path.join(OUT_DIR, "*.wav")) + glob.glob(os.path.join(OUT_DIR, "*.mp3")))
    print("\n" + "=" * 70)
    print(f"  TOTAL FILE DI DATASET BUKAN_AIR SEKARANG: {total_files} file")
    print("=" * 70)

if __name__ == "__main__":
    main()
