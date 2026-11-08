import os
import librosa
import numpy as np
from moviepy import AudioFileClip
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Fungsi membaca audio (Aman dari error MP3/M4A di Windows)
def muat_audio_aman(file_path):
    try:
        # librosa
        y, sr = librosa.load(file_path, duration=5.0)
        return y, sr
    except Exception:
        # Jika librosa gagal membaca mp3, gunakan moviepy untuk membuat file wav 
        temp_wav = "temp_baca.wav"
        klip = AudioFileClip(file_path)
        klip.write_audiofile(temp_wav, logger=None)
        klip.close()
        y, sr = librosa.load(temp_wav, duration=5.0)
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        return y, sr

# 2. Fungsi Ekstraksi Fitur Audio (Mengambil karakteristik suara: MFCC, frekuensi, dan energi)
def ekstrak_fitur(file_path):
    try:
        y, sr = muat_audio_aman(file_path)
        
        # a. MFCC (40 fitur karakteristik warna suara)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        
        # b. Spectral Centroid (rata-rata tinggi rendahnya frekuensi)
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        
        # c. RMS Energy (rata-rata keras/amplitudo suara)
        rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
        
        # Menggabungkan semua fitur menjadi 1 baris angka (42 fitur per audio)
        return np.hstack([mfcc, cent, rms])
    except Exception as e:
        print(f"Gagal memproses {file_path}: {e}")
        return None

# 3. Membaca kedua folder
daftar_folder = {
    "after-knalpot_standart": 0,  # Label 0 untuk Knalpot Standart
    "after-knalpot_brong": 1      # Label 1 untuk Knalpot Brong
}

fitur_list = []
label_list = []
file_list = [] # List untuk menyimpan path file

print("=== 1. MEMULAI EKSTRAKSI FITUR DARI DATA MENTAH ===")
for nama_folder, label in daftar_folder.items():
    if not os.path.exists(nama_folder):
        print(f"Folder {nama_folder} tidak ditemukan!")
        continue
        
    for nama_file in os.listdir(nama_folder):
        if nama_file.endswith(('.mp3', '.wav', '.m4a')):
            path_lengkap = os.path.join(nama_folder, nama_file)
            print(f"Mengekstrak: {path_lengkap}")
            
            fitur = ekstrak_fitur(path_lengkap)
            if fitur is not None:
                fitur_list.append(fitur)
                label_list.append(label)
                file_list.append(path_lengkap) # Simpan path file

X = np.array(fitur_list)
y = np.array(label_list)
files = np.array(file_list) # Jadikan array

print(f"\nTotal data berhasil diekstrak: {len(X)} file audio.")

# 4. Membagi Data Training (80%) dan Data Testing (20%)
# Tambahkan 'files' ke dalam proses split agar path file ikut terbagi
X_train, X_test, y_train, y_test, file_train, file_test = train_test_split(
    X, y, files, test_size=0.2, random_state=42, stratify=y
)

print(f"Jumlah Data Training : {len(X_train)} audio")
print(f"Jumlah Data Testing  : {len(X_test)} audio")

# 5. Melatih Model Machine Learning (Random Forest)
print("\n=== 2. MELATIH MODEL KLASIFIKASI ===")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Menguji Model dengan Data Testing
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

print(f"\n=== 3. HASIL EVALUASI MODEL ===")
print(f"Akurasi Model: {akurasi * 100:.2f}%\n")
print("Laporan Klasifikasi Detail:")
print(classification_report(y_test, y_pred, target_names=["Knalpot Standart (0)", "Knalpot Brong (1)"]))

# --- BAGIAN BARU: MENCARI DATA YANG SALAH PREDIKSI ---
print("\n=== 4. ANALISIS DATA YANG MENGURANGI AKURASI (SALAH TEBAK) ===")
jumlah_salah = 0

# Mengambil probabilitas keyakinan model untuk analisis tambahan
y_prob = model.predict_proba(X_test)

for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        jumlah_salah += 1
        label_asli = "Knalpot Brong" if y_test[i] == 1 else "Knalpot Standart"
        label_tebakan = "Knalpot Brong" if y_pred[i] == 1 else "Knalpot Standart"
        keyakinan = y_prob[i][y_pred[i]] * 100
        
        print(f"{jumlah_salah}. File: {file_test[i]}")
        print(f"   Seharusnya : {label_asli}")
        print(f"   Ditebak    : {label_tebakan} (Keyakinan: {keyakinan:.1f}%)")

if jumlah_salah == 0:
    print("Sempurna! Tidak ada data testing yang salah prediksi.")
else:
    print(f"\nTotal ada {jumlah_salah} file dari {len(y_test)} data testing yang salah.")
print("==============================================================\n")

# 7. Fungsi untuk mengetes 1 file audio secara langsung
def tes_prediksi_audio(file_uji):
    if not os.path.exists(file_uji):
        print(f"File '{file_uji}' tidak ditemukan untuk ditest.")
        return
        
    print(f"--- Mengetes File: {file_uji} ---")
    fitur_uji = ekstrak_fitur(file_uji)
    if fitur_uji is not None:
        prediksi = model.predict([fitur_uji])[0]
        probabilitas = model.predict_proba([fitur_uji])[0]
        
        hasil = "KNALPOT BRONG" if prediksi == 1 else "KNALPOT STANDART"
        keyakinan = probabilitas[prediksi] * 100
        print(f"Hasil Klasifikasi : {hasil}")
        print(f"Tingkat Keyakinan : {keyakinan:.2f}%\n")

# Contoh mengetes ulang salah satu file
# (Pastikan file ini benar-benar ada di direktori Anda)
# tes_prediksi_audio("Rekaman Baru.mp3")
# tes_prediksi_audio("123.mp3")