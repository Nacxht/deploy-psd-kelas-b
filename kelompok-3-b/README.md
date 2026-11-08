# Sistem Pemantauan Akustik Real-Time (Sawmill vs Non-Sawmill)

Aplikasi web pemantauan dan klasifikasi suara lingkungan secara kontinu dan real-time menggunakan model Machine Learning (**RandomForestClassifier**) untuk mendeteksi keberadaan aktivitas mesin gergaji kayu (*Sawmill*) versus kebisingan lingkungan normal (*Non-Sawmill*), dilengkapi dengan visualisasi osiloskop, analisis telemetri fitur audio Librosa, dan sistem peringatan otomatis ke WhatsApp via Webhook jika terdeteksi suara sawmill kontinu selama 10 detik atau lebih.

---

## 1. Persyaratan Sistem & Arsitektur

* **Platform Server Target:** Proxmox VE / Linux Server / Komputer Lokal
* **Container Engine:** Docker & Docker Compose
* **Port yang Dibuka:** **1 Port saja** (Port `5000`)
* **Spesifikasi Model:**
  * File Model: `model_terbaik_klasifikasi_audio.joblib`
  * Algoritma: `RandomForestClassifier` (Scikit-Learn)
  * Ukuran Model: **±360 KiB** (Model sudah disertakan langsung di dalam repositori ini, tidak memerlukan unduhan eksternal).
* **Ekstraksi Fitur:** 74 Fitur Audio Librosa (MFCC 20 koefisien, Chroma STFT 12 nada, Spectral Centroid, Spectral Bandwidth, Spectral Rolloff, Zero Crossing Rate, RMS) dengan normalisasi adaptif (*Automatic Gain Control*).
* **Sliding Window:** Durasi analisis 2.5 detik per window dengan pergeseran *hop* 0.5 detik.

---

## 2. Cara Menjalankan Aplikasi (Deployment Docker)

Docker Engine dan Docker Compose v2 harus sudah terpasang dan berjalan. Di Windows, instal Docker Desktop lalu tunggu sampai status Docker menunjukkan **Running**. Di server Proxmox/Linux, pastikan Docker Engine dan plugin Compose terpasang.

### Langkah 1: Kloning Repositori
```bash
git clone https://github.com/SolehMajid/Klasifikasi-Sawmil-.git
cd Klasifikasi-Sawmil-
```

### Langkah 2: Build dan jalankan
```bash
docker compose up --build
```
> **Catatan:** Docker akan otomatis mengunduh base image Python, memasang pustaka sistem Linux audio (`libsndfile1` & `ffmpeg`), memasang seluruh dependensi dari `requirements.txt`, dan menjalankan server Flask pada port 5000.

### Langkah 3: Akses Dashboard
Buka [http://localhost:5000](http://localhost:5000) di browser. Untuk server Proxmox, buka `http://<IP_SERVER_PROXMOX>:5000` jika port 5000 diizinkan oleh firewall.

Untuk menghentikan aplikasi, tekan `Ctrl+C`, lalu jalankan:
```bash
docker compose down
```

Jika port 5000 sudah digunakan, ubah hanya sisi kiri pemetaan port, misalnya `"5001:5000"`, lalu akses `http://localhost:5001`.

> **Akses mikrofon:** Browser mengizinkan akses mikrofon pada `localhost`. Jika membuka aplikasi dari komputer lain melalui alamat IP server, browser biasanya mewajibkan HTTPS. Untuk penggunaan jarak jauh, gunakan HTTPS melalui reverse proxy atau akses lokal saat pengujian.

---

## 3. Konfigurasi Lingkungan (Environment Variables)

Aplikasi telah dilengkapi proteksi kredensial menggunakan berkas `.env`. Template konfigurasi tersedia pada berkas `.env.example`:

| Variabel | Default | Keterangan |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | Host binding server Flask di dalam container |
| `PORT` | `5000` | Port aplikasi web |
| `WA_NOTIFICATION_ENABLED` | `false` | Mengaktifkan/menonaktifkan peringatan WhatsApp |
| `WA_METHOD` | `webhook` | Metode pengiriman notifikasi (`webhook` atau `fonnte`) |
| `WEBHOOK_URL` | - | URL endpoint bot WhatsApp |
| `WEBHOOK_SECRET` | - | Kunci otorisasi rahasia bot |
| `WA_TARGET_PHONE` | - | Nomor WhatsApp tujuan notifikasi |

> Tanpa berkas `.env`, dashboard dan fitur klasifikasi tetap berjalan. Notifikasi WhatsApp tidak aktif sampai kredensialnya dikonfigurasi. Untuk mengaktifkannya, salin `.env.example` menjadi `.env`, isi URL/token dan nomor tujuan yang valid, lalu jalankan ulang `docker compose up --build`. Jangan unggah `.env` ke GitHub.

---

## 4. Panduan Singkat Pengujian Fitur

1. **Pengujian Deteksi Akustik Kontinu:**
   * Buka dashboard di browser dan klik tombol **"Mulai Monitoring Kontinu"** (izinkan akses mikrofon browser).
   * Putar audio sampel gergaji mesin dekat mikrofon (jarak 10–30 cm).
   * Perhatikan indikator:
     * Gelombang osiloskop akan bereaksi secara real-time.
     * Bilah **Durasi Sawmill Kontinu** akan mulai menghitung (`0.5s -> 1.0s -> ... -> 10.0s`).
     * Jika suara sawmill bertahan terus selama **≥ 10 detik**, status notifikasi WhatsApp akan berubah menjadi `Terkirim` dan pesan peringatan otomatis dikirimkan.
2. **Pengujian Notifikasi Cepat:**
   * Klik tombol **"Tes Notif WA"** pada kontrol dashboard untuk memverifikasi jalur pengiriman pesan ke WhatsApp tanpa menunggu 10 detik.

---

## 5. Struktur Berkas Repositori

```text
├── Dockerfile              # Resep build container Linux & audio libraries
├── docker-compose.yml      # Konfigurasi container runner & port mapping 5000:5000
├── .dockerignore           # Mencegah file lokal dan rahasia masuk ke image
├── requirements.txt        # Daftar pustaka Python yang dibutuhkan
├── app.py                  # Aplikasi backend Flask, ring buffer & inferensi ML
├── model_terbaik_klasifikasi_audio.joblib # Model ML Random Forest (±360 KiB)
├── templates/
│   └── index.html          # Frontend dashboard responsif (Anti-Slop UI)
├── .env.example            # Template konfigurasi environment yang aman
├── .gitignore              # Daftar berkas yang diabaikan Git (.env, cache, dll)
└── README.md               # Dokumentasi dan petunjuk menjalankan proyek
```

---

## 6. Informasi Kelompok & Rekap Deploy

* **Mata Kuliah:** Proyek Sains Data (Semester 5)
* **Kelompok:** [Isi Nama / Nomor Kelompok Anda]
* **Anggota:**
  1. [Nama Mahasiswa 1] - [NIM]
  2. [Nama Mahasiswa 2] - [NIM]
* **Tautan Repositori GitHub:** [https://github.com/SolehMajid/Klasifikasi-Sawmil-](https://github.com/SolehMajid/Klasifikasi-Sawmil-)
