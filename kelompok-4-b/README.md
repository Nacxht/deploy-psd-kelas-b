# 🎙️ Voice Authenticity Detector — Human Voice vs AI Generated Voice

Sistem klasifikasi audio cerdas untuk mendeteksi keaslian suara (**Suara Manusia Asli** vs **Suara Manipulasi / AI Generated Voice**) menggunakan metodologi **CRISP-DM** dan model Deep Learning **1D-CNN (*One-Dimensional Convolutional Neural Network*)**.

---

## 👥 Anggota Kelompok (Kelompok 4)
1. **240411100009 — Muhammad Ainul Fuady**
2. **240411100010 — Irwan Dwi Mukhlisin**
3. **240411100105 — Fikri Mutawakkil**

---

## 📌 Link Repository GitHub
* **URL Repository Utama:** [https://github.com/muhammadainulfuady/AudioClasification](https://github.com/muhammadainulfuady/AudioClasification)
* **Akses:** Publik (*Public Repository*)

---

## 🚀 Panduan Menjalankan Aplikasi di Server Lab (Proxmox / Docker)

Aplikasi telah dikonfigurasi penuh agar dapat langsung dijalankan hanya dengan **1 perintah Docker Compose**:

### 1. Kloning Repository
```bash
git clone https://github.com/muhammadainulfuady/AudioClasification.git
cd AudioClasification
```

### 2. Build & Jalankan Kontainer Docker
```bash
docker compose up --build
```
> Kontainer akan otomatis mengunduh dependency sistem Linux (`libsndfile1`, `ffmpeg`), menginstal PyTorch CPU yang ringan, dan menyalakan server web Flask pada port `2006`.

### 3. Akses Antarmuka Web
Buka browser dan arahkan ke alamat:
* **Lokal:** `http://localhost:2006`
* **Server Lab (Proxmox):** `http://<IP_SERVER_LAB>:2006`

> **Informasi Port:** Port yang dibuka hanya **1 port saja**, yaitu port `2006` (`2006:2006`) agar tidak bentrok dengan kelompok lain di server lab.

Untuk menghentikan kontainer:
```bash
docker compose down
```

---

## Panduan Menjalankan Secara Lokal (Tanpa Docker)

Jika ingin menjalankan aplikasi secara langsung di Python lokal:

1. **Pastikan Python 3.10 atau 3.11 telah terinstal.**
2. **Install dependency:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Jalankan server Flask:**
   ```bash
   python frontend/app.py
   ```
4. Buka browser di `http://127.0.0.1:2006`.

---

## 📂 Struktur Direktori Proyek

```text
AudioClasification/
├── Dockerfile                  # Konfigurasi build image Docker
├── docker-compose.yml          # Konfigurasi compose untuk menjalankan layanan
├── requirements.txt            # Daftar pustaka Python yang dibutuhkan
├── cnn_architecture.py         # Definisi arsitektur PyTorch AudioCNN1D
├── README.md                   # Petunjuk penggunaan dan dokumentasi proyek
├── backend/
│   ├── main/
│   │   ├── model_voice_cnn_5s.pkl  # Paket model Deep Learning CNN siap pakai
│   │   ├── model_voice_cnn_5s.pt   # State dict PyTorch weights
│   │   ├── train_cnn_model.py      # Script pelatihan ulang model CNN
│   │   └── voices.ipynb            # Notebook riset lengkap 6 tahap CRISP-DM
├── frontend/
│   ├── app.py                  # Server web Flask backend
│   └── templates/
│       └── index.html          # Antarmuka frontend (TailwindCSS & Lucide Icons)
├── hasil_ekstraksi/
│   ├── file_ekstraksi.csv      # Dataset 48 fitur akustik lengkap (1.582 baris)
│   └── file_seleksi_fitur.csv  # Dataset Top-20 fitur terseleksi untuk modeling
└── data/
    └── dataset_audio_kaggle/
        ├── asli_human/         # 150 file WAV suara manusia (16.000 Hz 16-bit PCM Mono)
        └── ai_manipulasi/      # 151 file WAV suara AI (16.000 Hz 16-bit PCM Mono)
```

---

## 🧠 Informasi Model & Dataset

1. **Model Deep Learning 1D-CNN (`AudioCNN1D`)**:
   - Berkas model (`model_voice_cnn_5s.pkl` & `model_voice_cnn_5s.pt`) berukuran sangat ringkas (**~0.8 MB**) dan **sudah disertakan langsung di dalam repository**. Tidak memerlukan proses download tambahan dari drive eksternal.
   - Menggunakan arsitektur 3 blok konvolusi 1D (*Conv1d*, *BatchNorm1d*, *ReLU*, *MaxPool1d*, *Dropout*) dan *Fully Connected Layer*.

2. **Dataset Audio (301 Berkas Audio)**:
   - Terdiri atas **150 file WAV Human** (`audio_001.wav` s.d. `audio_150.wav`) dan **151 file WAV AI** (`audio_001.wav` s.d. `audio_151.wav`).
   - Format standar: **WAV 16-bit PCM, 16.000 Hz Mono**.
   - Seluruh dataset (**~281 MB**) sudah tersimpan langsung di dalam repositori `data/dataset_audio_kaggle/` (semua berkas $\le 28\text{ MB}$, aman dari batas GitHub 100 MB).

3. **Performa Evaluasi Model pada Data Uji (345 Sampel Segmen Uji)**:
   - **Akurasi (*Test Accuracy*):** **98.55%**
   - **F1-Score:** **98.59%**
   - **Presisi (*Precision*):** **99.43%**
   - **Sensitivitas (*Recall*):** **97.77%**
   - **Waktu Inferensi:** **< 0.9 detik** per pengujian suara.

