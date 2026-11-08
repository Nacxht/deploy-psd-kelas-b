import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def buat_grafik_audio(file_audio, nama_gambar):
    try:
        print("Sedang memuat file audio...")
        # 1. Memuat file audio
        # y = data amplitudo audio, sr = sample rate (standar 22050 Hz)
        y, sr = librosa.load(file_audio)

        # Membuat kanvas/jendela grafik dengan ukuran 12x8 inci
        plt.figure(figsize=(12, 8))

        # --- GRAFIK 1: WAVEFORM (Gelombang Suara) ---
        plt.subplot(2, 1, 1) # Baris 1 dari 2
        librosa.display.waveshow(y, sr=sr, color='blue', alpha=0.7)
        plt.title(f'Waveform (Gelombang Amplitudo) - {file_audio}')
        plt.xlabel('Waktu (Detik)')
        plt.ylabel('Amplitudo')
        plt.grid(True, alpha=0.3)

        # --- GRAFIK 2: SPECTROGRAM (Frekuensi Suara) ---
        plt.subplot(2, 1, 2) # Baris 2 dari 2
        # Mengubah data audio ke skala Desibel (dB) menggunakan Short-Time Fourier Transform (STFT)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(img, format='%+2.0f dB')
        plt.title(f'Spectrogram (Frekuensi) - {file_audio}')
        plt.xlabel('Waktu (Detik)')
        plt.ylabel('Frekuensi (Hz)')

        # Merapikan tata letak agar teks tidak saling bertumpuk
        plt.tight_layout()

        # Menyimpan grafik menjadi file gambar PNG di dalam folder proyek
        plt.savefig(nama_gambar, dpi=300)
        print(f"Berhasil! Grafik telah disimpan sebagai gambar: {nama_gambar}")

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

# Jalankan fungsi dengan memanggil file MP3 hasil konversi sebelumnya
for i in range(110):
    buat_grafik_audio(f"after-knalpot_brong/brong_{i+1}.mp3", f"grafik-knalpot_brong/brong_{i+1}.png")
    buat_grafik_audio(f"after-knalpot_standart/standart_{i+1}.mp3", f"grafik-knalpot_standart/standart_{i+1}.png")
# buat_grafik_audio("Rekaman Baru.mp3", "grafik_output.png")