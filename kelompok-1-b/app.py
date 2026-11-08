import os
import joblib
import librosa
import numpy as np
from flask import Flask, render_template, request, jsonify
from moviepy import AudioFileClip
from sklearn.ensemble import RandomForestClassifier
from werkzeug.utils import secure_filename

app = Flask(__name__)
MODEL_PATH = "model_knalpot.pkl"

# ==========================================
# BAGIAN 1: FUNGSI EKSTRAKSI & MODEL ML
# ==========================================
def muat_audio_aman(file_path):
    try:
        y, sr = librosa.load(file_path, duration=5.0)
        return y, sr
    except Exception:
        temp_wav = "temp_baca.wav"
        klip = AudioFileClip(file_path)
        klip.write_audiofile(temp_wav, logger=None)
        klip.close()
        y, sr = librosa.load(temp_wav, duration=5.0)
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        return y, sr

def ekstrak_fitur(file_path):
    try:
        y, sr = muat_audio_aman(file_path)
        if len(y) == 0:
            return None, 0, 0
        
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
        
        fitur = np.hstack([mfcc, cent, rms])
        volume_max = float(np.max(np.abs(y)))
        rms_val = float(rms[0]) if isinstance(rms, np.ndarray) else float(rms)
        
        return fitur, volume_max, rms_val
    except Exception as e:
        print(f"Gagal memproses {file_path}: {e}")
        return None, 0, 0

def siapkan_model():
    # Jika file model sudah ada, langsung muat (tidak perlu training ulang yang lama)
    if os.path.exists(MODEL_PATH):
        print(f"Memuat model instan dari {MODEL_PATH}...")
        return joblib.load(MODEL_PATH)
    
    print("File model belum ada. Mengekstrak dataset dan melatih model baru...")
    daftar_folder = {"after-knalpot_standart": 0, "after-knalpot_brong": 1}
    fitur_list, label_list = [], []

    for nama_folder, label in daftar_folder.items():
        if os.path.exists(nama_folder):
            for nama_file in os.listdir(nama_folder):
                if nama_file.endswith(('.mp3', '.wav', '.m4a')):
                    path_lengkap = os.path.join(nama_folder, nama_file)
                    fitur, _, _ = ekstrak_fitur(path_lengkap)
                    if fitur is not None:
                        fitur_list.append(fitur)
                        label_list.append(label)

    if len(fitur_list) > 0:
        X = np.array(fitur_list)
        y = np.array(label_list)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        joblib.dump(clf, MODEL_PATH) # Simpan model ke file .pkl
        print(f"Model berhasil dilatih dan disimpan ke {MODEL_PATH}!")
        return clf
    else:
        print("PERINGATAN: Dataset tidak ditemukan dan model_knalpot.pkl belum ada!")
        return None

model = siapkan_model()

# ==========================================
# BAGIAN 2: FLASK WEB ROUTING
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model belum siap (Dataset/Model .pkl tidak ditemukan di server)"}), 500

    if 'audio' not in request.files:
        return jsonify({"error": "Tidak ada file audio"}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "File kosong"}), 400

    ext = os.path.splitext(file.filename)[1]
    if not ext:
        ext = ".wav"
    temp_path = secure_filename(f"temp_uji{ext}")
    file.save(temp_path)

    fitur_uji, volume, rms_val = ekstrak_fitur(temp_path)
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if fitur_uji is None:
        return jsonify({"error": "Gagal mengekstrak audio. Pastikan format file didukung."}), 500

    # Cek ambang batas keheningan (Hanya berlaku jika dari rekaman mic realtime)
    mode_input = request.form.get('mode', 'mic')
    if mode_input == 'mic' and volume < 0.015:
        return jsonify({
            "hasil": "SEPI",
            "confidence": 100.0,
            "rms": round(rms_val, 4),
            "volume": round(volume, 4)
        })

    # Prediksi dan ambil persentase keyakinan (Confidence Score)
    prediksi = int(model.predict([fitur_uji])[0])
    probabilitas = model.predict_proba([fitur_uji])[0]
    keyakinan = float(probabilitas[prediksi] * 100)

    hasil = "BRONG" if prediksi == 1 else "STANDART"
    return jsonify({
        "hasil": hasil,
        "confidence": round(keyakinan, 2),
        "rms": round(rms_val, 4),
        "volume": round(volume, 4)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)