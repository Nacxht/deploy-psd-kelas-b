"""
app.py
======
Web Application & REST API - Sistem Cerdas Klasifikasi Aliran Air Wudhu Masjid
Berbasis Machine Learning (Ensemble ExtraTrees + Random Forest).
Dapat dijalankan langsung dengan:
    python app.py
atau melalui Docker:
    docker compose up --build
"""

import os
import sys
import tempfile
import warnings
import joblib
import numpy as np
import pandas as pd
import librosa
from flask import Flask, request, jsonify, render_template_string, send_from_directory

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # Max 32 MB file upload

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "output", "model_aliran_air.pkl")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

# Global variables for model
model_data = None
model = None
feature_cols = None
class_names = [
    "Bukan Air (Kipas/Adzan/Suasana)",
    "Kran Pelan (Hemat)",
    "Kran Keras (Boros)"
]

def load_ml_model():
    global model_data, model, feature_cols, class_names
    if os.path.exists(MODEL_PATH):
        try:
            model_data = joblib.load(MODEL_PATH)
            model = model_data.get("model", model_data)
            feature_cols = model_data.get("feature_cols", None)
            if "class_names" in model_data:
                class_names = model_data["class_names"]
            print(f"[*] Model berhasil dimuat dari {MODEL_PATH}")
            if feature_cols:
                print(f"    Jumlah fitur: {len(feature_cols)}")
        except Exception as e:
            print(f"[!] Gagal memuat model: {e}")
    else:
        print(f"[!] Model tidak ditemukan di {MODEL_PATH}")

load_ml_model()

def extract_features(y, sr=22050):
    """Mengekstrak 81 fitur akustik yang identik dengan pipeline training."""
    if len(y) < sr * 2:
        y = np.pad(y, (0, int(sr * 2 - len(y))), mode='constant')

    win_samples = int(2.0 * sr)
    hop_samples = int(1.0 * sr)

    all_segment_features = []

    for start in range(0, len(y) - win_samples + 1, hop_samples):
        seg = y[start : start + win_samples]

        rms_frames = librosa.feature.rms(y=seg)
        rms_val = float(np.mean(rms_frames))
        spec_cent = float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)))
        spec_bw = float(np.mean(librosa.feature.spectral_bandwidth(y=seg, sr=sr)))
        spec_ro85 = float(np.mean(librosa.feature.spectral_rolloff(y=seg, sr=sr, roll_percent=0.85)))
        spec_ro25 = float(np.mean(librosa.feature.spectral_rolloff(y=seg, sr=sr, roll_percent=0.25)))
        spec_flat = float(np.mean(librosa.feature.spectral_flatness(y=seg)))
        zcr_frames = librosa.feature.zero_crossing_rate(y=seg)
        zcr_val = float(np.mean(zcr_frames))
        zcr_std = float(np.std(zcr_frames))

        y_harm, _ = librosa.effects.hpss(seg, margin=(2.0, 1.0))
        harm_rms = float(np.mean(librosa.feature.rms(y=y_harm)))
        harm_ratio = float(harm_rms / (rms_val + 1e-9))

        S = np.abs(librosa.stft(seg))
        crest = float(np.mean(np.max(S, axis=0) / (np.mean(S, axis=0) + 1e-9)))
        flux = float(np.mean(librosa.onset.onset_strength(y=seg, sr=sr)))

        freqs = librosa.fft_frequencies(sr=sr)
        low_idx = np.where(freqs < 250)[0]
        mid_idx = np.where((freqs >= 250) & (freqs < 3000))[0]
        high_idx = np.where(freqs >= 3000)[0]

        low_e = float(np.sum(S[low_idx, :]**2))
        mid_e = float(np.sum(S[mid_idx, :]**2))
        high_e = float(np.sum(S[high_idx, :]**2))
        tot_e = low_e + mid_e + high_e + 1e-9

        low_ratio = float(low_e / tot_e)
        mid_ratio = float(mid_e / tot_e)
        high_ratio = float(high_e / tot_e)

        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        contrast = librosa.feature.spectral_contrast(y=seg, sr=sr)

        row = {
            'rms': rms_val,
            'spectral_centroid': spec_cent,
            'spectral_bandwidth': spec_bw,
            'spectral_rolloff': spec_ro85,
            'spectral_rolloff_25': spec_ro25,
            'spectral_flatness': spec_flat,
            'zcr': zcr_val,
            'zcr_std': zcr_std,
            'harm_ratio': harm_ratio,
            'crest': crest,
            'flux': flux,
            'low_ratio': low_ratio,
            'mid_ratio': mid_ratio,
            'high_ratio': high_ratio,
        }
        for i in range(20):
            row[f'mfcc_mean_{i+1}'] = float(np.mean(mfcc[i]))
            row[f'mfcc_std_{i+1}'] = float(np.std(mfcc[i]))
            row[f'mfcc_delta_{i+1}'] = float(np.mean(mfcc_delta[i]))
        for i in range(contrast.shape[0]):
            row[f'contrast_mean_{i+1}'] = float(np.mean(contrast[i]))

        all_segment_features.append(row)

    if not all_segment_features:
        raise ValueError("Audio terlalu pendek atau tidak memiliki segmen valid.")

    df_feats = pd.DataFrame(all_segment_features)
    if feature_cols:
        return df_feats[feature_cols].values, df_feats
    return df_feats.values, df_feats

def predict_audio_array(y, sr=22050):
    """Memprediksi array audio menggunakan agregasi probabilitas per segmen."""
    if model is None:
        raise RuntimeError("Model Machine Learning belum dimuat.")

    X, df_feats = extract_features(y, sr=sr)
    probs_per_segment = model.predict_proba(X)
    avg_probs = np.mean(probs_per_segment, axis=0)
    pred_class_idx = int(np.argmax(avg_probs))

    # Ringkasan akustik
    audio_duration = float(len(y) / sr)
    avg_rms = float(df_feats['rms'].mean())
    avg_centroid = float(df_feats['spectral_centroid'].mean())
    avg_zcr = float(df_feats['zcr'].mean())

    status_deskripsi = {
        0: {"label": "Bukan Air", "badge": "bg-secondary", "alert": "Aktivitas Non-Air (Adzan / Suasana / Obrolan / Noise)", "rekomendasi": "Tidak ada peringatan sistem."},
        1: {"label": "Kran Pelan (Hemat)", "badge": "bg-success", "alert": "Aliran Air Wajar & Hemat", "rekomendasi": "Debit air ideal untuk wudhu. Beri peringatan hanya jika lupa mematikan > 20 detik."},
        2: {"label": "Kran Keras (Boros)", "badge": "bg-danger", "alert": "Peringatan: Aliran Air Terlalu Deras!", "rekomendasi": "Boros air! Segera kecilkan debit kran untuk menghemat air wudhu."}
    }

    return {
        "status": "success",
        "predicted_class_idx": pred_class_idx,
        "predicted_class_name": class_names[pred_class_idx],
        "info": status_deskripsi[pred_class_idx],
        "probabilities": {
            "bukan_air": float(avg_probs[0]),
            "pelan": float(avg_probs[1]),
            "keras": float(avg_probs[2])
        },
        "probabilities_pct": {
            "bukan_air": round(float(avg_probs[0]) * 100, 2),
            "pelan": round(float(avg_probs[1]) * 100, 2),
            "keras": round(float(avg_probs[2]) * 100, 2)
        },
        "audio_meta": {
            "duration_sec": round(audio_duration, 2),
            "segments_analyzed": len(X),
            "rms_volume": round(avg_rms, 4),
            "spectral_centroid_hz": round(avg_centroid, 1),
            "zcr": round(avg_zcr, 4)
        }
    }

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sistem Deteksi Aliran Air Wudhu - Proyek Sains Data</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <style>
        body { background-color: #0f172a; color: #f8fafc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .card-custom { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; }
        .header-box { background: linear-gradient(135deg, #0ea5e9 0%, #1e40af 100%); border-radius: 14px; padding: 28px 24px; box-shadow: 0 10px 25px -5px rgba(14, 165, 233, 0.3); }
        .btn-sample { background-color: #334155; border: 1px solid #475569; color: #e2e8f0; transition: all 0.2s; }
        .btn-sample:hover { background-color: #0ea5e9; color: white; border-color: #0284c7; }
        .progress-custom { height: 18px; border-radius: 8px; background-color: #334155; }
        .drop-zone { border: 2px dashed #475569; border-radius: 12px; padding: 30px; text-align: center; cursor: pointer; transition: border-color 0.2s; }
        .drop-zone:hover { border-color: #38bdf8; background-color: rgba(56, 189, 248, 0.05); }
    </style>
</head>
<body class="py-4">
<div class="container" style="max-width: 900px;">
    <!-- HEADER -->
    <div class="header-box mb-4 text-white">
        <div class="d-flex align-items-center justify-content-between flex-wrap gap-2">
            <div>
                <span class="badge bg-light text-primary fw-bold mb-2">TUGAS PROYEK SAINS DATA (PSD)</span>
                <h2 class="fw-bold mb-1"><i class="fa-solid fa-faucet-drip me-2"></i>Klasifikasi Aliran Air Tempat Wudhu</h2>
                <p class="mb-0 text-white-50">Deteksi Akustik Konservasi Air Masjid Berbasis Ensemble Machine Learning</p>
            </div>
            <div class="text-end">
                <span class="badge bg-success px-3 py-2"><i class="fa-solid fa-server me-1"></i>Server Lab Proxmox / Docker</span>
            </div>
        </div>
    </div>

    <!-- MAIN CARD -->
    <div class="card card-custom p-4 mb-4">
        <h5 class="fw-bold text-info mb-3"><i class="fa-solid fa-upload me-2"></i>Input Suara untuk Klasifikasi</h5>
        
        <!-- Opsi 1: Upload File -->
        <div class="drop-zone mb-3" id="dropZone" onclick="document.getElementById('audioInput').click()">
            <i class="fa-solid fa-cloud-arrow-up fa-3x text-info mb-2"></i>
            <h6 class="fw-bold mb-1">Klik atau Tarik File Audio ke Sini</h6>
            <small class="text-secondary">Mendukung format WAV, MP3, M4A, OGG, FLAC (Max 32MB)</small>
            <input type="file" id="audioInput" accept="audio/*" style="display: none;" onchange="handleFileSelect(this.files[0])">
        </div>

        <div id="fileInfoBadge" class="alert alert-info py-2 d-none d-flex justify-content-between align-items-center">
            <span><i class="fa-solid fa-music me-2"></i><b id="fileNameTxt">file.wav</b></span>
            <audio id="audioPreview" controls style="height: 32px;"></audio>
        </div>

        <!-- Opsi 2: Rekam dari Browser Mic & Opsi 3: Sampel -->
        <div class="row g-2 mb-3">
            <div class="col-md-5">
                <button id="btnRecord" class="btn btn-outline-warning w-100 py-2" onclick="toggleRecord()">
                    <i class="fa-solid fa-microphone me-2"></i><span id="recordTxt">Rekam Mikrofon (5 Detik)</span>
                </button>
            </div>
            <div class="col-md-7">
                <div class="btn-group w-100" role="group">
                    <button class="btn btn-sample btn-sm" onclick="testSample('sample_kran_keras.mp3', 'Kran Keras (Boros)')"><i class="fa-solid fa-water me-1"></i>Sampel Keras</button>
                    <button class="btn btn-sample btn-sm" onclick="testSample('sample_kran_pelan.mp3', 'Kran Pelan (Hemat)')"><i class="fa-solid fa-droplet me-1"></i>Sampel Pelan</button>
                    <button class="btn btn-sample btn-sm" onclick="testSample('sample_bukan_air.wav', 'Bukan Air (Adzan/Noise)')"><i class="fa-solid fa-volume-xmark me-1"></i>Sampel Bukan Air</button>
                </div>
            </div>
        </div>

        <button id="btnPredict" class="btn btn-primary w-100 py-2 fw-bold" onclick="submitAudio()" disabled>
            <i class="fa-solid fa-wand-magic-sparkles me-2"></i>Jalankan Analisis & Klasifikasi
        </button>
    </div>

    <!-- HASIL ANALISIS -->
    <div id="resultCard" class="card card-custom p-4 d-none mb-4">
        <h5 class="fw-bold text-info mb-3"><i class="fa-solid fa-chart-pie me-2"></i>Hasil Deteksi & Prediksi</h5>
        
        <div id="alertBanner" class="alert p-3 mb-3 d-flex align-items-center gap-3">
            <i id="statusIcon" class="fa-solid fa-circle-check fa-2x"></i>
            <div>
                <h4 id="statusTitle" class="fw-bold mb-0">Kran Pelan (Hemat)</h4>
                <p id="statusDesc" class="mb-0 text-white-50">Aliran air wajar dan hemat</p>
            </div>
        </div>

        <!-- Probability Bars -->
        <h6 class="fw-bold mb-2">Tingkat Keyakinan Model (Probabilitas Softmax):</h6>
        <div class="mb-2">
            <div class="d-flex justify-content-between small mb-1">
                <span><i class="fa-solid fa-droplet text-success me-1"></i>Kran Pelan (Hemat)</span>
                <b id="pctPelan">0%</b>
            </div>
            <div class="progress progress-custom">
                <div id="barPelan" class="progress-bar bg-success" style="width: 0%"></div>
            </div>
        </div>
        <div class="mb-2">
            <div class="d-flex justify-content-between small mb-1">
                <span><i class="fa-solid fa-water text-danger me-1"></i>Kran Keras (Boros)</span>
                <b id="pctKeras">0%</b>
            </div>
            <div class="progress progress-custom">
                <div id="barKeras" class="progress-bar bg-danger" style="width: 0%"></div>
            </div>
        </div>
        <div class="mb-3">
            <div class="d-flex justify-content-between small mb-1">
                <span><i class="fa-solid fa-volume-xmark text-secondary me-1"></i>Bukan Air (Kipas / Adzan / Noise)</span>
                <b id="pctBukan">0%</b>
            </div>
            <div class="progress progress-custom">
                <div id="barBukan" class="progress-bar bg-secondary" style="width: 0%"></div>
            </div>
        </div>

        <!-- Audio Metadata Metrics -->
        <div class="row g-2 text-center pt-2 border-top border-secondary">
            <div class="col-3">
                <small class="text-secondary d-block">Durasi</small>
                <b id="metaDuration" class="text-info">-</b>
            </div>
            <div class="col-3">
                <small class="text-secondary d-block">RMS Volume</small>
                <b id="metaRms" class="text-info">-</b>
            </div>
            <div class="col-3">
                <small class="text-secondary d-block">Spektral Centroid</small>
                <b id="metaCentroid" class="text-info">-</b>
            </div>
            <div class="col-3">
                <small class="text-secondary d-block">Zero Crossing Rate</small>
                <b id="metaZcr" class="text-info">-</b>
            </div>
        </div>
    </div>

    <!-- FOOTER & API INFO -->
    <div class="text-center text-secondary small">
        <p class="mb-1">Proyek Sains Data &copy; 2026 - Server Deployment via Docker on Proxmox</p>
        <span>REST API: <code>POST /api/predict</code> (Form-data: <code>audio</code>) | Health check: <code>GET /api/health</code></span>
    </div>
</div>

<script>
let currentFile = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

function handleFileSelect(file) {
    if (!file) return;
    currentFile = file;
    document.getElementById('fileNameTxt').innerText = file.name;
    document.getElementById('fileInfoBadge').classList.remove('d-none');
    document.getElementById('audioPreview').src = URL.createObjectURL(file);
    document.getElementById('btnPredict').disabled = false;
}

function testSample(filename, label) {
    fetch('/api/sample/' + filename)
        .then(res => res.blob())
        .then(blob => {
            const file = new File([blob], filename, { type: blob.type || 'audio/wav' });
            handleFileSelect(file);
            document.getElementById('fileNameTxt').innerText = label + " (" + filename + ")";
            submitAudio();
        })
        .catch(err => alert("Gagal mengambil sampel: " + err));
}

async function toggleRecord() {
    const btn = document.getElementById('btnRecord');
    const txt = document.getElementById('recordTxt');
    
    if (!isRecording) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            
            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const recFile = new File([audioBlob], "rekaman_mic.wav", { type: 'audio/wav' });
                handleFileSelect(recFile);
            };
            
            mediaRecorder.start();
            isRecording = true;
            btn.classList.replace('btn-outline-warning', 'btn-danger');
            txt.innerText = "Merekam... Klik untuk Berhenti";
            
            // Auto stop 5 detik
            setTimeout(() => { if (isRecording) toggleRecord(); }, 5000);
        } catch(e) {
            alert("Izin mikrofon ditolak atau tidak didukung: " + e.message);
        }
    } else {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        isRecording = false;
        btn.classList.replace('btn-danger', 'btn-outline-warning');
        txt.innerText = "Rekam Mikrofon (5 Detik)";
    }
}

function submitAudio() {
    if (!currentFile) return;
    
    const btn = document.getElementById('btnPredict');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Mengekstrak Fitur & Mengklasifikasi...';

    const formData = new FormData();
    formData.append('audio', currentFile);

    fetch('/api/predict', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles me-2"></i>Jalankan Analisis & Klasifikasi';
        
        if (data.status === 'success') {
            displayResults(data);
        } else {
            alert('Error: ' + (data.message || 'Gagal menganalisis audio.'));
        }
    })
    .catch(err => {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles me-2"></i>Jalankan Analisis & Klasifikasi';
        alert('Terjadi kesalahan jaringan: ' + err);
    });
}

function displayResults(data) {
    const card = document.getElementById('resultCard');
    card.classList.remove('d-none');
    
    const banner = document.getElementById('alertBanner');
    const icon = document.getElementById('statusIcon');
    const title = document.getElementById('statusTitle');
    const desc = document.getElementById('statusDesc');

    title.innerText = data.predicted_class_name;
    desc.innerText = data.info.alert + " — " + data.info.rekomendasi;

    banner.className = 'alert p-3 mb-3 d-flex align-items-center gap-3';
    if (data.predicted_class_idx === 1) {
        banner.classList.add('alert-success');
        icon.className = 'fa-solid fa-circle-check fa-2x text-success';
    } else if (data.predicted_class_idx === 2) {
        banner.classList.add('alert-danger');
        icon.className = 'fa-solid fa-triangle-exclamation fa-2x text-danger';
    } else {
        banner.classList.add('alert-secondary');
        icon.className = 'fa-solid fa-volume-xmark fa-2x text-secondary';
    }

    // Probabilities
    document.getElementById('pctPelan').innerText = data.probabilities_pct.pelan + '%';
    document.getElementById('barPelan').style.width = data.probabilities_pct.pelan + '%';

    document.getElementById('pctKeras').innerText = data.probabilities_pct.keras + '%';
    document.getElementById('barKeras').style.width = data.probabilities_pct.keras + '%';

    document.getElementById('pctBukan').innerText = data.probabilities_pct.bukan_air + '%';
    document.getElementById('barBukan').style.width = data.probabilities_pct.bukan_air + '%';

    // Meta
    document.getElementById('metaDuration').innerText = data.audio_meta.duration_sec + ' detik';
    document.getElementById('metaRms').innerText = data.audio_meta.rms_volume;
    document.getElementById('metaCentroid').innerText = data.audio_meta.spectral_centroid_hz + ' Hz';
    document.getElementById('metaZcr').innerText = data.audio_meta.zcr;

    card.scrollIntoView({ behavior: 'smooth' });
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": class_names,
        "num_features": len(feature_cols) if feature_cols else 0
    })

@app.route('/api/sample/<filename>', methods=['GET'])
def get_sample(filename):
    return send_from_directory(SAMPLES_DIR, filename)

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"status": "error", "message": "Field 'audio' tidak ditemukan pada request."}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"status": "error", "message": "Tidak ada file yang dipilih."}), 400

    ext = os.path.splitext(audio_file.filename)[1].lower()
    suffix = ext if ext in ['.wav', '.mp3', '.m4a', '.ogg', '.flac'] else '.wav'

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            audio_file.save(tmp_path)

        y, sr = librosa.load(tmp_path, sr=22050, mono=True)
        if len(y) == 0:
            return jsonify({"status": "error", "message": "File audio kosong atau tidak dapat dibaca."}), 400

        result = predict_audio_array(y, sr=sr)
        result["filename"] = audio_file.filename
        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Menjalankan server aplikasi di port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
