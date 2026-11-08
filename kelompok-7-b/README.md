# Sistem Deteksi Suara Kebocoran Gas/Pipa (LeakSound)
> **Tugas Besar Pengolahan Sinyal Digital (PSD)**  
> Implementasi Klasifikasi Suara Kebocoran menggunakan Algoritma Machine Learning Random Forest berbasis Ekstraksi Fitur Akustik.

---

## 📌 Deskripsi Proyek
Proyek ini mengidentifikasi suara kebocoran pipa/gas secara otomatis berdasarkan analisis karakteristik spektral dan domain waktu dari sinyal audio. Model dilatih menggunakan algoritma **Random Forest Classifier** (200 *decision trees*) dengan **50 fitur sinyal audio digital**:
* **40 Fitur MFCC**: Nilai *mean* (rata-rata) dan *standard deviation* (standar deviasi) dari 20 koefisien Mel-Frequency Cepstral Coefficients.
* **10 Fitur Spektral & Temporal**:
  * *Spectral Centroid* (pusat massa spektrum frekuensi)
  * *Spectral Flatness* (tingkat kerataan distribusi spektrum/derau)
  * *Spectral Rolloff* (frekuensi batas akumulasi energi spektral)
  * *Zero Crossing Rate (ZCR)* (laju perlintasan sinyal melewati titik nol)
  * *Root Mean Square (RMS) Energy* (kekuatan energi sinyal)

Inferensi dapat berjalan secara *real-time* maupun melalui unggahan file audio (*.wav / *.mp3) serta tombol uji cepat sampel audio.

---

## 🚀 Cara Menjalankan Aplikasi (Deployment Docker)

Sesuai ketentuan server laboratorium, aplikasi ini dirancang untuk dapat langsung dijalankan hanya dengan satu perintah:

```bash
docker compose up --build
```

Setelah proses build selesai, buka browser dan akses aplikasi melalui:
👉 **`http://localhost:3000`** *(atau sesuaikan IP server lab, misalnya: `http://<IP_SERVER_LAB>:3000`)*

### 🔌 Informasi Port
* **Port yang digunakan:** `3000` (Hanya membuka 1 port).

---

## 🛠️ Cara Menjalankan Tanpa Docker (Lokal)

Jika ingin menjalankan aplikasi secara langsung di mesin lokal tanpa Docker:

### Menggunakan Python:
1. Pastikan Python 3.10+ terinstal.
2. Pasang pustaka dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan server:
   ```bash
   python server.py
   ```
4. Buka browser di `http://localhost:3000`.

### Atau Menggunakan Node.js:
```bash
node server.js
```

---

## 📁 Struktur Berkas Repositori

```text
├── Dockerfile                   # Konfigurasi build container Docker
├── docker-compose.yml           # Konfigurasi orkestrasi Docker Compose
├── requirements.txt             # Daftar dependensi pustaka Python
├── README.md                    # Dokumentasi dan panduan proyek
├── server.py                    # Server web berbasis Python
├── server.js                    # Server web berbasis Node.js (opsional)
├── model/                       # Berkas bobot model ML & metadata ekstraksi
│   ├── random_forest_model.json # Model Random Forest dalam format JSON
│   ├── random_forest_model.joblib # Model Random Forest dalam format Joblib
│   ├── metadata_model.json      # Metadata konfigurasi audio (16 kHz, filterband)
│   └── data_testing.json        # Data pengujian evaluasi
├── web/                         # Antarmuka web sistem deteksi
│   ├── index.html               # Halaman dashboard monitoring & deteksi
│   ├── css/style.css            # Styling UI modern & responsif
│   ├── js/                      # Logika audio processing & predictor JS
│   └── samples/                 # Sampel audio uji cepat (Bocor & Normal)
├── dataset/                     # Data audio asli (bocor & normal)
└── dataset_bersih/              # Data audio yang telah dipreprocessing
```

---

## 📊 Dataset & Model
* Berkas model utama (`random_forest_model.json` dan `random_forest_model.joblib`) telah disertakan langsung di repositori sehingga tidak memerlukan unduhan eksternal tambahan saat deployment.
* Sampel audio uji siap pakai telah disertakan di folder `web/samples/` untuk kemudahan pengujian instan oleh dosen / asisten lab.

