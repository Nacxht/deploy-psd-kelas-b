FROM python:3.10-slim

# Menginstal dependensi sistem operasi (ffmpeg & libsndfile untuk memproses audio)
RUN apt-get update && apt-get install -y     libsndfile1     ffmpeg     && rm -rf /var/lib/apt/lists/*

# Menentukan direktori kerja di dalam container
WORKDIR /app

# Menyalin file requirements.txt dan menginstal library Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh file proyek ke dalam container
COPY . .

# Membuka port 5000 (hanya 1 port sesuai ketentuan)
EXPOSE 5000

# Menjalankan aplikasi menggunakan server production Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
