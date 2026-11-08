# Base Image Python 3.9 Slim
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=5000

# Install dependencies sistem (ffmpeg & libsndfile1 untuk membaca audio MP3/WAV/M4A/OGG)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set direktori kerja
WORKDIR /app

# Salin dan install dependencies python terlebih dahulu untuk memanfaatkan Docker cache layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode proyek & model terlatih
COPY . .

# Ekspos port aplikasi (Port 5000)
EXPOSE 5000

# Healthcheck untuk memastikan container berjalan sehat di server Proxmox
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Jalankan server produksi menggunakan Gunicorn (WSGI)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
