# Speech Emotion Recognition (SER) — Production Deployment

Layanan microservice produksi pengenalan emosi wicara berbasis **FastAPI + Uvicorn**, dikemas dalam kontainer **Docker & Docker Compose**, dan ditenagai oleh model **Master Hybrid Ensemble (845 Fitur Akustik Invarian CMVN)**.

---

## Alamat Akses dan Port Layanan

Microservice ini berjalan pada **Port `8015`**:

| Layanan | URL Akses Lokal | URL Akses Server VPS |
| :--- | :--- | :--- |
| **Web Dashboard Interaktif** | `http://localhost:8015/` | `http://<IP_SERVER>:8015/` |
| **Dokumentasi REST API (Swagger UI)** | `http://localhost:8015/docs` | `http://<IP_SERVER>:8015/docs` |
| **Healthcheck Status Server** | `http://localhost:8015/health` | `http://<IP_SERVER>:8015/health` |
| **Dokumentasi Redoc** | `http://localhost:8015/redoc` | `http://<IP_SERVER>:8015/redoc` |

---

## Cara Menjalankan Layanan

### Opsi 1: Menggunakan Docker (Rekomendasi Server Produksi)

Gunakan opsi ini saat aplikasi di-deploy ke server cloud atau VPS:

1. **Jalankan Kontainer di Latar Belakang (Background / Detached):**
   ```bash
   docker compose up --build -d
   ```
2. **Melihat Log Server:**
   ```bash
   docker compose logs -f
   ```
3. **Memeriksa Status Kesehatan Kontainer:**
   ```bash
   docker ps
   ```
4. **Menghentikan Layanan:**
   ```bash
   docker compose down
   ```

*Catatan: Konfigurasi Docker sudah dilengkapi `restart: unless-stopped` sehingga aplikasi otomatis menyala kembali jika server reboot atau restart.*

---

### Opsi 2: Menjalankan Langsung dengan Python (Lokal / Tanpa Docker)

Gunakan opsi ini jika ingin mencoba langsung di laptop tanpa memasang Docker Desktop:

1. **Instal Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Jalankan Aplikasi:**
   ```bash
   python app.py
   ```
   *Atau melalui runner Uvicorn:*
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8015
   ```

---

## Struktur Berkas Deployment

```text
6-deploy/
├── app.py                      # Server tunggal FastAPI, router REST API, & Web UI
├── Dockerfile                  # Cetak biru Linux container, FFmpeg, & libsndfile
├── docker-compose.yml          # Konfigurasi container service & port mapping 8015:8015
├── requirements.txt            # Dependensi runtime Python produksi
├── .dockerignore               # Filter file sampah (cache, venv, log)
├── README.md                   # Panduan pengoperasian deployment ini
├── models/
│   ├── master_hybrid_model.joblib            # Model produksi (Soft-Voting RBF-SVM + MLP)
│   ├── ravdess_prepared_features_and_scaler.npz # Parameter RobustScaler & manifest
│   └── ravdess_data_understanding_catalog.csv # Metadata label psikometri audio
└── test_web_model/             # Berkas sampel audio MP3 untuk tombol preset contoh
```

---

## Dokumentasi REST API

### 1. GET /health
Mengecek status kesehatan server dan kesiapan model.

**Contoh Respons:**
```json
{
  "status": "healthy",
  "service": "Speech Emotion Recognition API",
  "model": "Master Hybrid Ensemble (845 Invariant Acoustic Features)",
  "framework": "FastAPI + Uvicorn",
  "classes": ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]
}
```

### 2. POST /api/v1/predict
Mengklasifikasikan emosi dari berkas audio atau video secara real-time.

* **Parameter:** `file` (Multipart FormData: WAV, MP3, M4A, OGG, atau video MP4).
* **Contoh Pemanggilan via cURL:**
  ```bash
  curl -X POST "http://localhost:8015/api/v1/predict" \
       -H "accept: application/json" \
       -F "file=@contoh_suara.mp3"
  ```
* **Contoh Respons:**
  ```json
  {
    "status": "success",
    "endpoint": "POST /api/v1/predict",
    "input_file": "contoh_suara.mp3",
    "predicted_emotion": "Angry",
    "emotion_id": "Marah / Tegang",
    "emotion_code": "05",
    "status_label": "TERVERIFIKASI STABIL",
    "affective_profile": {
      "arousal": "Sangat Tinggi (+0.90)",
      "valence": "Sangat Negatif (-0.85)",
      "quadrant": "Kuadran II (Negatif - Berenergi Tinggi)",
      "pitch_mean_hz": 234.5
    },
    "inference_latency_ms": 78.4
  }
  ```

---

## Catatan Pengaturan Server

1. **Firewall / Security Group:**
   Buka izin akses masuk (Inbound Rule) untuk **Port `8015` TCP** pada panel cloud provider Anda.
2. **Izin Mikrofon Browser:**
   * Fitur **Unggah Berkas** dan **Contoh Audio** dapat berjalan di protokol HTTP maupun HTTPS.
   * Khusus untuk fitur **Rekam Mikrofon Langsung**, browser modern mewajibkan koneksi HTTPS jika diakses melalui IP publik server.
