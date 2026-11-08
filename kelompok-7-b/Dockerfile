# Menggunakan base image Python yang ringan dan stabil
FROM python:3.11-slim

# Set working directory di dalam container
WORKDIR /app

# Nonaktifkan buffering output agar log python langsung terlihat di console docker
ENV PYTHONUNBUFFERED=1
ENV PORT=3000

# Install dependensi sistem dasar untuk healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Salin file requirements terlebih dahulu untuk memanfaatkan caching layer Docker
COPY requirements.txt .

# Install dependensi pustaka Python
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi ke dalam container
COPY . .

# Port yang diekspos (maksimal 1-2 port sesuai ketentuan)
EXPOSE 3000

# Healthcheck untuk monitoring status aplikasi di container Proxmox/Docker
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/ || exit 1

# Perintah utama untuk menjalankan aplikasi
CMD ["python", "server.py"]

