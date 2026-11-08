# Gunakan base image Python yang stabil dan ringan
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Direktori kerja di dalam container
WORKDIR /app

# Instal dependensi sistem Linux (libsndfile & ffmpeg wajib untuk library audio librosa/soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Salin requirements dan instal pustaka Python
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode proyek ke dalam container
COPY . /app/

# Expose port aplikasi (Port 5000)
EXPOSE 5000

# Perintah menjalankan aplikasi web
CMD ["python", "app.py"]
