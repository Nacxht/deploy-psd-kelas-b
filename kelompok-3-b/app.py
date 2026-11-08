import os
import sys
import io
import time
import math
import warnings
import threading
import queue
from collections import deque
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import joblib
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

# Muat variabel environment dari .env jika ada
load_dotenv()

# Pastikan console stdout UTF-8 pada Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ==============================================================================
# 1. KONFIGURASI AUDIO & MODEL
# ==============================================================================
SAMPLE_RATE = 22050          # Sample rate model training
WINDOW_DURATION = 2.5        # Durasi window analisis (detik)
HOP_DURATION = 0.5           # Interval hop geser window (detik)
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION)  # 55,125 sampel
HOP_SAMPLES = int(SAMPLE_RATE * HOP_DURATION)        # 11,025 sampel

SILENCE_THRESHOLD = 0.003    # Ambang batas hening (RMS)
GAIN_BOOST = 1.0             # Default software gain multiplier

MODEL_FILE = 'model_terbaik_klasifikasi_audio.joblib'
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILE)

# ==============================================================================
# KONFIGURASI NOTIFIKASI WHATSAPP (Aman: Dimuat dari .env)
# ==============================================================================
WA_NOTIFICATION_ENABLED = os.environ.get("WA_NOTIFICATION_ENABLED", "True").lower() in ("true", "1", "t")
WA_METHOD = os.environ.get("WA_METHOD", "webhook")  # "webhook" atau "fonnte"
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
FONNTE_TOKEN = os.environ.get("FONNTE_TOKEN", "")
WA_TARGET_PHONE = os.environ.get("WA_TARGET_PHONE", "")   # Dimuat dari file .env (aman dari GitHub)

SAWMILL_TRIGGER_DURATION = 10.0   # Butuh minimal 10 detik terdeteksi Sawmill kontinu
NOTIFICATION_COOLDOWN = 120.0     # Jeda 2 menit antar notifikasi agar tidak spam

wa_state = {
    'sawmill_start_time': None,
    'consecutive_seconds': 0.0,
    'alert_sent_for_session': False,
    'last_alert_time': 0.0,
    'last_status_message': 'Siaga',
}
wa_lock = threading.Lock()

def send_whatsapp_alert(duration, confidence, prob_sawmil, rms):
    """Mengirim notifikasi WhatsApp ke webhook/gateway secara asinkron (background thread)."""
    def _worker():
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        pesan = (
            f"🚨 *PERINGATAN AKUSTIK: SUARA SAWMILL TERDETEKSI* 🚨\n\n"
            f"Sistem mendeteksi suara mesin gergaji (Sawmill) secara kontinu selama lebih dari *{int(duration)} detik*.\n\n"
            f"📊 *Detail Deteksi:*\n"
            f"• Waktu: {now_str}\n"
            f"• Durasi Terdeteksi: {duration:.1f} detik\n"
            f"• Keyakinan Model: {confidence}%\n"
            f"• Probabilitas Sawmill: {prob_sawmil}%\n"
            f"• Intensitas Audio (RMS): {rms}\n\n"
            f"⚠️ _Pesan otomatis dikirim oleh Sistem Pemantauan Akustik Real-Time._"
        )

        try:
            print(f"[WA NOTIF] Mengirim notifikasi WhatsApp via {WA_METHOD}...", flush=True)
            if WA_METHOD == "webhook":
                payload = {
                    "secret_key": WEBHOOK_SECRET,
                    "target": WA_TARGET_PHONE,
                    "pesan": pesan,
                    "secret": WEBHOOK_SECRET,
                    "message": pesan
                }
                if WA_TARGET_PHONE:
                    payload["phone"] = WA_TARGET_PHONE
                    payload["to"] = WA_TARGET_PHONE

                headers = {
                    "Content-Type": "application/json",
                    "X-Secret-Key": WEBHOOK_SECRET,
                    "Authorization": f"Bearer {WEBHOOK_SECRET}"
                }
                resp = requests.post(WEBHOOK_URL, json=payload, headers=headers, timeout=10)
                print(f"[WA NOTIF] Webhook response ({resp.status_code}): {resp.text[:200]}", flush=True)
                with wa_lock:
                    wa_state['last_status_message'] = f"Terkirim ({time.strftime('%H:%M:%S')})"

            elif WA_METHOD == "fonnte":
                headers = {"Authorization": FONNTE_TOKEN}
                payload = {
                    "target": WA_TARGET_PHONE,
                    "message": pesan
                }
                resp = requests.post("https://api.fonnte.com/send", data=payload, headers=headers, timeout=10)
                print(f"[WA NOTIF] Fonnte response ({resp.status_code}): {resp.text[:200]}", flush=True)
                with wa_lock:
                    wa_state['last_status_message'] = f"Terkirim ({time.strftime('%H:%M:%S')})"

        except Exception as err:
            print(f"[WA NOTIF ERROR] Gagal mengirim WhatsApp: {err}", flush=True)
            with wa_lock:
                wa_state['last_status_message'] = f"Gagal: {str(err)[:25]}"

    threading.Thread(target=_worker, daemon=True).start()

# 74 Kolom fitur persis sesuai model training
FEATURE_COLUMNS = [
    'mfcc_1_mean', 'mfcc_1_std', 'mfcc_2_mean', 'mfcc_2_std', 'mfcc_3_mean', 'mfcc_3_std',
    'mfcc_4_mean', 'mfcc_4_std', 'mfcc_5_mean', 'mfcc_5_std', 'mfcc_6_mean', 'mfcc_6_std',
    'mfcc_7_mean', 'mfcc_7_std', 'mfcc_8_mean', 'mfcc_8_std', 'mfcc_9_mean', 'mfcc_9_std',
    'mfcc_10_mean', 'mfcc_10_std', 'mfcc_11_mean', 'mfcc_11_std', 'mfcc_12_mean', 'mfcc_12_std',
    'mfcc_13_mean', 'mfcc_13_std', 'mfcc_14_mean', 'mfcc_14_std', 'mfcc_15_mean', 'mfcc_15_std',
    'mfcc_16_mean', 'mfcc_16_std', 'mfcc_17_mean', 'mfcc_17_std', 'mfcc_18_mean', 'mfcc_18_std',
    'mfcc_19_mean', 'mfcc_19_std', 'mfcc_20_mean', 'mfcc_20_std',
    'chroma_1_mean', 'chroma_1_std', 'chroma_2_mean', 'chroma_2_std', 'chroma_3_mean', 'chroma_3_std',
    'chroma_4_mean', 'chroma_4_std', 'chroma_5_mean', 'chroma_5_std', 'chroma_6_mean', 'chroma_6_std',
    'chroma_7_mean', 'chroma_7_std', 'chroma_8_mean', 'chroma_8_std', 'chroma_9_mean', 'chroma_9_std',
    'chroma_10_mean', 'chroma_10_std', 'chroma_11_mean', 'chroma_11_std', 'chroma_12_mean', 'chroma_12_std',
    'spectral_centroid_mean', 'spectral_centroid_std',
    'spectral_bandwidth_mean', 'spectral_bandwidth_std',
    'spectral_rolloff_mean', 'spectral_rolloff_std',
    'zcr_mean', 'zcr_std',
    'rms_mean', 'rms_std'
]

# Load model ke memori
model = None
try:
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
    elif os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    print(f"[INFO] Model ML berhasil dimuat: {type(model).__name__}", flush=True)
except Exception as e:
    print(f"[ERROR] Gagal memuat model: {e}", flush=True)

# ==============================================================================
# 2. AUDIO RING BUFFER & STABILIZER STATE
# ==============================================================================
class AudioRingBuffer:
    """Ring buffer thread-safe untuk sliding window audio stream."""
    def __init__(self, max_samples):
        self.max_samples = max_samples
        self.buffer = np.zeros(max_samples, dtype=np.float32)
        self.lock = threading.Lock()
        self.samples_written = 0

    def append(self, samples):
        with self.lock:
            n = len(samples)
            if n >= self.max_samples:
                self.buffer[:] = samples[-self.max_samples:]
            else:
                self.buffer[:-n] = self.buffer[n:]
                self.buffer[-n:] = samples
            self.samples_written += n

    def get_window(self):
        with self.lock:
            return self.buffer.copy(), self.samples_written

    def reset(self):
        with self.lock:
            self.buffer.fill(0)
            self.samples_written = 0

ring_buffer = AudioRingBuffer(WINDOW_SAMPLES)

class DetectionStabilizer:
    """
    Stabilisasi prediksi menggunakan Moving Average dan Hysteresis Threshold.
    Menghilangkan pergantian label cepat (flicker).
    """
    def __init__(self, history_size=6, sawmil_thresh=0.55, nonsawmil_thresh=0.45):
        self.history_size = history_size
        self.sawmil_thresh = sawmil_thresh
        self.nonsawmil_thresh = nonsawmil_thresh
        self.prob_history = deque(maxlen=history_size)
        self.stable_label = "Non-Sawmill"
        self.stable_counter = 0
        self.lock = threading.Lock()

    def update(self, raw_prob_sawmil, is_silent=False):
        with self.lock:
            if is_silent:
                # Suara hening otomatis Non-Sawmill
                self.prob_history.append(0.0)
                self.stable_label = "Non-Sawmill"
                return "Non-Sawmill", "Non-Sawmill", 0.0, 0.0

            self.prob_history.append(raw_prob_sawmil)
            smoothed_prob = float(np.mean(self.prob_history))

            # Instant Prediction
            instant_label = "Sawmill" if raw_prob_sawmil >= 0.50 else "Non-Sawmill"

            # Stable Prediction dengan Hysteresis
            if smoothed_prob >= self.sawmil_thresh:
                if self.stable_label != "Sawmill":
                    self.stable_counter += 1
                    if self.stable_counter >= 2:  # Butuh minimal 2 window konsisten
                        self.stable_label = "Sawmill"
                        self.stable_counter = 0
                else:
                    self.stable_counter = 0
            elif smoothed_prob <= self.nonsawmil_thresh:
                if self.stable_label != "Non-Sawmill":
                    self.stable_counter += 1
                    if self.stable_counter >= 2:
                        self.stable_label = "Non-Sawmill"
                        self.stable_counter = 0
                else:
                    self.stable_counter = 0
            # Jika di antara threshold (0.45 - 0.55), pertahankan stable_label sebelumnya

            return instant_label, self.stable_label, raw_prob_sawmil, smoothed_prob

    def reset(self):
        with self.lock:
            self.prob_history.clear()
            self.stable_label = "Non-Sawmill"
            self.stable_counter = 0

stabilizer = DetectionStabilizer()

# State Terkini Sistem untuk Monitoring Dashboard
system_state = {
    'status': 'Siaga',
    'mic_active': False,
    'source': 'Browser Mic',
    'instant_prediction': 'Menunggu...',
    'stable_prediction': 'Menunggu...',
    'confidence': 0.0,
    'prob_sawmil': 0.0,
    'prob_nonsawmil': 100.0,
    'sawmill_duration': 0.0,
    'wa_status': 'Siaga',
    'rms': 0.0,
    'is_silent': True,
    'stats': {
        'centroid': 0.0,
        'rolloff': 0.0,
        'zcr': 0.0,
        'bandwidth': 0.0
    },
    'history_timeline': deque(maxlen=30),  # Data poin grafik probabilitas
    'last_updated': time.strftime("%H:%M:%S")
}
state_lock = threading.Lock()

# ==============================================================================
# 3. EKSTRAKSI FITUR DENGAN NORMALISASI ADAPTIF (AGC)
# ==============================================================================
def extract_features_sliding(y, sr=SAMPLE_RATE, n_mfcc=20):
    """
    Ekstraksi 74 fitur Librosa dengan normalisasi adaptif.
    Memastikan suara HP / mic jauh tetap terdeteksi konsisten sesuai skala training.
    """
    # 1. Hitung RMS murni sebelum normalisasi
    raw_rms = float(np.sqrt(np.mean(y**2)))
    is_silent = (raw_rms < SILENCE_THRESHOLD)

    # 2. Normalisasi Puncak (AGC) agar perbedaan jarak HP / volume speaker tidak merusak MFCC
    max_amp = float(np.max(np.abs(y)))
    if max_amp > 1e-4:
        # Penskalaan ke nominal level ~0.7 (sesuai amplitudo rata-rata dataset MP3)
        y_proc = (y / max_amp) * 0.70
    else:
        y_proc = y + np.random.normal(0, 1e-6, size=len(y)).astype(np.float32)

    features = {}

    # 1. MFCC (20 Koefisien)
    mfcc = librosa.feature.mfcc(y=y_proc, sr=sr, n_mfcc=n_mfcc)
    for i in range(n_mfcc):
        features[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc[i]))
        features[f'mfcc_{i+1}_std'] = float(np.std(mfcc[i]))

    # 2. Chroma STFT (12 Nada)
    chroma = librosa.feature.chroma_stft(y=y_proc, sr=sr)
    for i in range(chroma.shape[0]):
        features[f'chroma_{i+1}_mean'] = float(np.mean(chroma[i]))
        features[f'chroma_{i+1}_std'] = float(np.std(chroma[i]))

    # 3. Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=y_proc, sr=sr)
    features['spectral_centroid_mean'] = float(np.mean(spec_cent))
    features['spectral_centroid_std'] = float(np.std(spec_cent))

    # 4. Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y_proc, sr=sr)
    features['spectral_bandwidth_mean'] = float(np.mean(spec_bw))
    features['spectral_bandwidth_std'] = float(np.std(spec_bw))

    # 5. Spectral Rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(y=y_proc, sr=sr)
    features['spectral_rolloff_mean'] = float(np.mean(spec_rolloff))
    features['spectral_rolloff_std'] = float(np.std(spec_rolloff))

    # 6. Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=y_proc)
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std'] = float(np.std(zcr))

    # 7. RMS Energy (gunakan RMS dari audio ternormalisasi agar konsisten dengan range dataset)
    rms_calc = librosa.feature.rms(y=y_proc)
    features['rms_mean'] = float(np.mean(rms_calc))
    features['rms_std'] = float(np.std(rms_calc))

    return features, raw_rms, is_silent

# ==============================================================================
# 4. INFERENCE PIPELINE TERPISAH (NON-BLOCKING)
# ==============================================================================
inference_queue = queue.Queue(maxsize=3)

def run_inference_on_window(audio_window):
    """Menjalankan ekstraksi fitur dan prediksi pada 1 window audio 2.5 detik."""
    global model
    if model is None:
        return

    try:
        t_start = time.time()
        features_dict, raw_rms, is_silent = extract_features_sliding(audio_window, sr=SAMPLE_RATE)

        if is_silent:
            raw_sawmil_prob = 0.0
        else:
            df_in = pd.DataFrame([features_dict])[FEATURE_COLUMNS]
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(df_in)[0]
                raw_sawmil_prob = float(proba[1])
            else:
                pred = int(model.predict(df_in)[0])
                raw_sawmil_prob = 1.0 if pred == 1 else 0.0

        # Stabilisasi
        inst_label, stbl_label, raw_p, smooth_p = stabilizer.update(raw_sawmil_prob, is_silent=is_silent)

        prob_saw_pct = round(smooth_p * 100, 1)
        prob_non_pct = round((1.0 - smooth_p) * 100, 1)
        conf_val = prob_saw_pct if stbl_label == "Sawmill" else prob_non_pct

        # Evaluasi Durasi Deteksi Sawmill Kontinu untuk Notifikasi WhatsApp (>= 10 detik)
        now_time = time.time()
        if WA_NOTIFICATION_ENABLED and stbl_label == "Sawmill" and not is_silent:
            with wa_lock:
                if wa_state['sawmill_start_time'] is None:
                    wa_state['sawmill_start_time'] = now_time

                sawmill_duration = round(now_time - wa_state['sawmill_start_time'], 1)
                wa_state['consecutive_seconds'] = sawmill_duration

                cooldown_passed = (now_time - wa_state['last_alert_time']) >= NOTIFICATION_COOLDOWN
                if sawmill_duration >= SAWMILL_TRIGGER_DURATION and (not wa_state['alert_sent_for_session'] or cooldown_passed):
                    wa_state['alert_sent_for_session'] = True
                    wa_state['last_alert_time'] = now_time
                    wa_state['last_status_message'] = f"Mengirim notif ({sawmill_duration}s)..."
                    print(f"[ALERT] Sawmill terdeteksi kontinu {sawmill_duration}s! Mengirim notifikasi WhatsApp...", flush=True)
                    send_whatsapp_alert(sawmill_duration, conf_val, prob_saw_pct, round(raw_rms, 4))
        else:
            with wa_lock:
                wa_state['sawmill_start_time'] = None
                wa_state['consecutive_seconds'] = 0.0
                wa_state['alert_sent_for_session'] = False

        # Perbarui Global System State untuk Web Dashboard
        with state_lock:
            system_state['status'] = 'Aktif Memantau'
            system_state['mic_active'] = True
            system_state['instant_prediction'] = inst_label
            system_state['stable_prediction'] = stbl_label
            system_state['confidence'] = conf_val
            system_state['prob_sawmil'] = prob_saw_pct
            system_state['prob_nonsawmil'] = prob_non_pct
            with wa_lock:
                system_state['sawmill_duration'] = wa_state['consecutive_seconds']
                system_state['wa_status'] = wa_state['last_status_message']
            system_state['rms'] = round(raw_rms, 4)
            system_state['is_silent'] = is_silent
            system_state['stats'] = {
                'centroid': round(features_dict['spectral_centroid_mean'], 1),
                'rolloff': round(features_dict['spectral_rolloff_mean'], 1),
                'zcr': round(features_dict['zcr_mean'], 4),
                'bandwidth': round(features_dict['spectral_bandwidth_mean'], 1)
            }
            system_state['last_updated'] = time.strftime("%H:%M:%S")
            system_state['history_timeline'].append({
                'time': system_state['last_updated'],
                'prob_sawmil': prob_saw_pct,
                'stable': stbl_label,
                'rms': round(raw_rms, 4)
            })

    except Exception as e:
        print(f"[Inference Error]: {e}", flush=True)

def inference_worker_thread():
    """Worker background agar proses inferensi tidak memblokir streaming audio."""
    while True:
        try:
            window = inference_queue.get()
            run_inference_on_window(window)
            inference_queue.task_done()
        except Exception as e:
            time.sleep(0.05)

# Mulai worker thread inferensi saat startup
worker = threading.Thread(target=inference_worker_thread, daemon=True)
worker.start()

# ==============================================================================
# 5. HARDWARE MICROPHONE STREAM (PYTHON SOUNDDEVICE) - OPSIONAL JIKA TERSEDIA
# ==============================================================================
hw_stream = None
hw_running = False

def hw_audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[HW Audio Status]: {status}", flush=True)
    samples = indata[:, 0]  # Mono
    ring_buffer.append(samples)

def start_hardware_microphone():
    global hw_stream, hw_running
    try:
        import sounddevice as sd
        if hw_running:
            return True, "Hardware mic sudah berjalan"

        hw_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=HOP_SAMPLES,
            callback=hw_audio_callback
        )
        hw_stream.start()
        hw_running = True
        print(f"[INFO] Hardware microphone SoundDevice aktif pada {SAMPLE_RATE} Hz.", flush=True)
        return True, "Hardware mic aktif"
    except Exception as e:
        return False, f"SoundDevice tidak tersedia: {e}"

def stop_hardware_microphone():
    global hw_stream, hw_running
    if hw_stream:
        try:
            hw_stream.stop()
            hw_stream.close()
        except:
            pass
    hw_stream = None
    hw_running = False

# Periodic trigger loop untuk sliding window inferensi (tiap 0.5 detik)
def sliding_window_timer_thread():
    last_samples_processed = 0
    while True:
        time.sleep(HOP_DURATION)
        window, current_samples = ring_buffer.get_window()
        # Jika ada audio baru masuk (minimal hop_samples)
        if current_samples - last_samples_processed >= (HOP_SAMPLES // 2) and current_samples >= (SAMPLE_RATE):
            last_samples_processed = current_samples
            if not inference_queue.full():
                inference_queue.put(window)

timer_thread = threading.Thread(target=sliding_window_timer_thread, daemon=True)
timer_thread.start()

# ==============================================================================
# 6. FLASK ROUTE API
# ==============================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stream_chunk', methods=['POST'])
def stream_chunk():
    """
    Menerima potongan kecil audio kontinu (Float32 PCM / WAV) dari browser microphone.
    Langsung dimasukkan ke RingBuffer tanpa blocking.
    """
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file'}), 400

    audio_file = request.files['audio']
    raw_bytes = audio_file.read()

    if len(raw_bytes) == 0:
        return jsonify({'success': False, 'error': 'Empty chunk'}), 400

    try:
        # Decode WAV langsung di memori
        bio = io.BytesIO(raw_bytes)
        y, sr = sf.read(bio, dtype='float32')

        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Resample jika browser mengirim pada sample rate berbeda
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

        # Masukkan ke RingBuffer
        ring_buffer.append(y)

        # Ambil state terkini
        with state_lock:
            resp_state = {
                'success': True,
                'instant_prediction': system_state['instant_prediction'],
                'stable_prediction': system_state['stable_prediction'],
                'confidence': system_state['confidence'],
                'prob_sawmil': system_state['prob_sawmil'],
                'prob_nonsawmil': system_state['prob_nonsawmil'],
                'sawmill_duration': system_state.get('sawmill_duration', 0.0),
                'wa_status': system_state.get('wa_status', 'Siaga'),
                'rms': system_state['rms'],
                'is_silent': system_state['is_silent'],
                'stats': system_state['stats'],
                'last_updated': system_state['last_updated']
            }

        return jsonify(resp_state)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/live_state', methods=['GET'])
def get_live_state():
    """Endpoint polling data real-time untuk dashboard visual."""
    with state_lock:
        data = {
            'status': system_state['status'],
            'mic_active': system_state['mic_active'],
            'instant_prediction': system_state['instant_prediction'],
            'stable_prediction': system_state['stable_prediction'],
            'confidence': system_state['confidence'],
            'prob_sawmil': system_state['prob_sawmil'],
            'prob_nonsawmil': system_state['prob_nonsawmil'],
            'sawmill_duration': system_state.get('sawmill_duration', 0.0),
            'wa_status': system_state.get('wa_status', 'Siaga'),
            'rms': system_state['rms'],
            'is_silent': system_state['is_silent'],
            'stats': system_state['stats'],
            'history': list(system_state['history_timeline']),
            'last_updated': system_state['last_updated']
        }
    return jsonify(data)

@app.route('/api/test_wa', methods=['POST'])
def test_wa():
    """Endpoint manual untuk menguji Webhook WhatsApp dengan satu klik."""
    try:
        send_whatsapp_alert(
            duration=10.0,
            confidence=95.0,
            prob_sawmil=98.5,
            rms=0.045
        )
        return jsonify({'success': True, 'message': 'Pesan tes sedang dikirim ke Webhook WhatsApp'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_state():
    ring_buffer.reset()
    stabilizer.reset()
    with wa_lock:
        wa_state['sawmill_start_time'] = None
        wa_state['consecutive_seconds'] = 0.0
        wa_state['alert_sent_for_session'] = False
        wa_state['last_status_message'] = 'Siaga'
    with state_lock:
        system_state['instant_prediction'] = 'Menunggu...'
        system_state['stable_prediction'] = 'Menunggu...'
        system_state['confidence'] = 0.0
        system_state['prob_sawmil'] = 0.0
        system_state['prob_nonsawmil'] = 100.0
        system_state['sawmill_duration'] = 0.0
        system_state['wa_status'] = 'Siaga'
        system_state['rms'] = 0.0
        system_state['history_timeline'].clear()
    return jsonify({'success': True, 'message': 'State berhasil direset'})

if __name__ == '__main__':
    # Mode uji cepat WhatsApp via terminal: python app.py --test-wa
    if '--test-wa' in sys.argv:
        print("=================================================================")
        print("  PENGUJIAN LANGSUNG NOTIFIKASI WHATSAPP")
        print(f"  Metode: {WA_METHOD}")
        print(f"  Target: {WA_TARGET_PHONE}")
        print(f"  URL   : {WEBHOOK_URL}")
        print("=================================================================")
        send_whatsapp_alert(duration=10.0, confidence=95.0, prob_sawmil=98.5, rms=0.045)
        print("[INFO] Menunggu respons server webhook (3 detik)...")
        time.sleep(3)
        print("[INFO] Selesai! Silakan cek WhatsApp Anda.")
        sys.exit(0)

    print("=================================================================")
    print("  AUDIO MONITORING & DETECTION (SAWMIL VS NON-SAWMIL)")
    print("  Sliding Window: 2.5s | Hop: 0.5s | Sample Rate: 22050 Hz")
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    print(f"  Akses Dashboard pada: http://{host}:{port}")
    print("=================================================================")
    app.run(host=host, port=port, debug=False, threaded=True)
