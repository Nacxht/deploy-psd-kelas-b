# Proyek Sains Data: Klasifikasi & Deteksi Aliran Air Tempat Wudhu Masjid

Sistem cerdas berbasis Machine Learning (*Ensemble ExtraTrees + Tuned Random Forest*) untuk mendeteksi dan mengklasifikasikan aliran air kran di tempat wudhu masjid secara akustik real-time, guna mencegah pemborosan air dan mendukung konservasi air masjid.

Proyek ini telah dikontainerisasi menggunakan **Docker** dan siap di-deploy langsung pada **Server Lab Proxmox** hanya dengan satu perintah.

---

## 3 Kelas Klasifikasi

| Label | Kelas | Karakteristik Akustik | Tindakan Sistem |
|:---:|---|---|---|
| **0** | **Bukan Air** | Suara kipas angin, obrolan, ketikan keyboard, kumandang adzan, kebisingan lingkungan | Tidak memicu alarm sistem |
| **1** | **Kran Pelan (Hemat)** | Aliran air wajar, laminar, hemat air wudhu | Peringatan jika mengalir terus-menerus **> 20 detik** (indikasi lupa mematikan kran) |
| **2** | **Kran Keras (Boros)** | Aliran air bertekanan tinggi, turbulen, cipratan deras | Peringatan jika mengalir terus-menerus **> 10 detik** (indikasi pemborosan air) |

---

## Arsitektur & Performa Model

- **Jumlah Fitur Akustik**: 81 fitur (Spectral Centroid, Bandwidth, Rolloff, ZCR, HPSS Harmonic Ratio, Spectral Crest & Flux, Sub-band Energy Ratios, 20 MFCC + Delta, dan Spectral Contrast).
- **Algoritma**: Soft-Voting Ensemble (`ExtraTreesClassifier` + `RandomForestClassifier`).
- **Akurasi Data Uji (Test Accuracy)**: **97.31%**
- **5-Fold Stratified Cross-Validation**: **97.54% (±0.24%)**

---

## Cara Menjalankan Menggunakan Docker (Server Lab Proxmox)

> [!IMPORTANT]
> Sesuai ketentuan, aplikasi dapat langsung dijalankan hanya dengan satu perintah:

```bash
docker compose up --build
```

Setelah proses build selesai, buka browser dan akses aplikasi melalui:
```text
http://localhost:5000
```
atau jika diakses melalui IP Server Lab:
```text
http://<IP-SERVER-LAB>:5000
```

Port yang dibuka: **5000** (hanya 1 port).

---

## Fitur Aplikasi Web & REST API

### 1. Web Interface Interaktif
- **Upload File Audio**: Mendukung format `.wav`, `.mp3`, `.m4a`, `.ogg`, dan `.flac`.
- **Rekam Suara Langsung**: Merekam langsung melalui mikrofon browser client (durasi 5 detik).
- **Tombol Sampel Bawaan**: Uji coba instan dengan satu klik menggunakan sampel *Kran Keras*, *Kran Pelan*, dan *Bukan Air* yang telah disediakan.
- **Visualisasi Hasil**: Menampilkan probabilitas softmax tiap kelas, badge status peringatan, player audio, dan metrik akustik (RMS, frekuensi spektral centroid, ZCR).

### 2. REST API Endpoints
Aplikasi juga menyediakan API endpoint untuk integrasi sistem lain:

#### A. Health Check
```bash
curl http://localhost:5000/api/health
```
**Response JSON:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_features": 81,
  "classes": [
    "Bukan Air (Kipas/Adzan/Suasana)",
    "Kran Pelan (Hemat)",
    "Kran Keras (Boros)"
  ]
}
```

#### B. Prediksi Audio File
```bash
curl -X POST -F "audio=@samples/sample_kran_keras.mp3" http://localhost:5000/api/predict
```
**Response JSON:**
```json
{
  "status": "success",
  "filename": "sample_kran_keras.mp3",
  "predicted_class_idx": 2,
  "predicted_class_name": "Kran Keras (Boros)",
  "probabilities_pct": {
    "bukan_air": 0.04,
    "pelan": 0.26,
    "keras": 99.69
  },
  "info": {
    "label": "Kran Keras (Boros)",
    "alert": "Peringatan: Aliran Air Terlalu Deras!",
    "rekomendasi": "Boros air! Segera kecilkan debit kran untuk menghemat air wudhu."
  },
  "audio_meta": {
    "duration_sec": 10.79,
    "rms_volume": 0.0912,
    "spectral_centroid_hz": 4272.1,
    "zcr": 0.3095
  }
}
```

---

## Cara Menjalankan Lokal (Tanpa Docker)

Jika ingin menjalankan aplikasi langsung di environment Python lokal:

1. **Install dependensi**:
   ```powershell
   pip install -r requirements.txt
   ```
2. **Jalankan Web App**:
   ```powershell
   python app.py
   ```
   Buka `http://127.0.0.1:5000` di browser.

3. **Menjalankan Monitoring Mikrofon Laptop Real-Time (dengan Voice Warning)**:
   ```powershell
   python live_mic.py
   ```

4. **Melatih Ulang Model**:
   ```powershell
   python train.py
   ```

---

## Informasi Model & Dataset

- **Model Terlatih**: File model ensemble biner (`output/model_aliran_air.pkl`, ~47 MB) **sudah disertakan langsung di repository**, sehingga aplikasi dapat langsung digunakan tanpa perlu melakukan training ulang terlebih dahulu.
- **Sampel Audio Uji**: Disediakan 3 sampel representatif di folder `samples/` (`sample_kran_keras.mp3`, `sample_kran_pelan.mp3`, dan `sample_bukan_air.wav`).
- **Dataset Training**: Dataset audio mentah disimpan di folder `dataset/` (terdiri dari kategori `bukan_air`, `pelan`, dan `keras`). Untuk meregenerasi seluruh variasi data augmentasi, jalankan:
  ```powershell
  python augment_dataset.py
  ```

---

## Struktur Repositori

```text
proyek_hemat_air/
├── Dockerfile                  <-- Instruksi build image container Docker
├── docker-compose.yml          <-- Konfigurasi docker compose untuk Proxmox
├── requirements.txt            <-- Daftar dependensi Python
├── app.py                      <-- Web Application & REST API (Flask + Gunicorn)
├── samples/                    <-- File sampel audio bawaan untuk pengujian cepat
│   ├── sample_kran_keras.mp3
│   ├── sample_kran_pelan.mp3
│   └── sample_bukan_air.wav
├── output/
│   ├── model_aliran_air.pkl    <-- Model terlatih Ensemble ExtraTrees + RF (47 MB)
│   └── evaluasi_model.png      <-- Visualisasi Confusion Matrix & Top Features
├── dataset/                    <-- Audio dataset (bukan_air, pelan, keras)
├── train.py                    <-- Script pelatihan & ekstraksi 81 fitur audio
├── live_mic.py                 <-- Script monitoring real-time via mic laptop
├── predict.py                  <-- Script CLI klasifikasi satu file
├── augment_dataset.py          <-- Generator augmentasi akustik
├── README.md                   <-- Dokumentasi lengkap proyek
├── .dockerignore               <-- Filter build Docker
└── .gitignore                  <-- Filter commit Git
```

---

## Identitas Kelompok

- **Mata Kuliah**: Proyek Sains Data (PSD)
- **Dosen Pengampu**: Pak Mul
- **Topik**: Klasifikasi Audio / Akustik Aliran Air Tempat Wudhu Masjid
- **Anggota Kelompok**:
1. Raihan Aryanova Narendra - 240411100094
2. Muhammad Farhan - 240411100233
3. Joni Kumar Meghwar - 240411100234
