# ==============================================================================
# Dockerfile: Voice Authenticity Detector (Human Voice vs AI Generated Voice)
# Base Image: Python 3.10-slim (Ringan, Cepat, dan Stabil untuk Audio ML)
# ==============================================================================

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=2006 \
    PYTHONPATH=/app \
    FLASK_APP=frontend/app.py \
    MPLCONFIGDIR=/tmp/matplotlib

# Install dependency sistem Linux (libsndfile1 & ffmpeg wajib untuk librosa & soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Direktori kerja di dalam kontainer
WORKDIR /app

# Salin requirements.txt terlebih dahulu untuk caching layer Docker
COPY requirements.txt .

# Install dependencies (gunakan PyTorch CPU agar build cepat ~180MB bukan CUDA ~4GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Salin seluruh isi proyek ke direktori kerja kontainer
COPY . .

# Buat direktori temporary untuk cache matplotlib
RUN mkdir -p /tmp/matplotlib && chmod -R 777 /tmp/matplotlib

# Buka hanya 1 port sesuai ketentuan server lab (Port 2006)
EXPOSE 2006

# Healthcheck untuk monitoring kontainer di Proxmox
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:2006/ || exit 1

# Perintah menjalankan aplikasi web Flask
CMD ["python", "-m", "flask", "--app", "frontend/app.py", "run", "--host=0.0.0.0", "--port=2006"]

