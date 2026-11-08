# syntax=docker/dockerfile:1
# speech emotion recognition (ser) unified fastapi microservice
FROM python:3.10-slim

# optimasi runtime python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# instal dependensi sistem operasi (ffmpeg & libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# instal dependensi python secara terisolasi dengan docker layer caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# salin seluruh source code, model, dan aset deploy
COPY . /app/

# port microservice
EXPOSE 8015

# uji kesehatan kontainer
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8015/health || exit 1

# jalankan server uvicorn berkinerja tinggi
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8015"]
