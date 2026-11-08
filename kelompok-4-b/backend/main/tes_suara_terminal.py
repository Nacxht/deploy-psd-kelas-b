import os
import sys
import time
import queue
import warnings
import joblib
import librosa
import sounddevice as sd
import numpy as np
import scipy.stats as stats

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

try:
    from cnn_architecture import AudioCNN1D, CNNVoiceClassifier
except ImportError:
    pass

MODEL_PATH = os.path.join(BASE_DIR, 'model_voice_cnn_5s.pkl')


def ekstrak_fitur_segmen(y_seg, sr, f0_seg):
    """Mengekstrak 48 fitur audio Librosa dari 1 segmen berdurasi 5 detik utuh."""
    rms = librosa.feature.rms(y=y_seg)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y_seg)[0]
    silence_ratio = float(np.mean(rms < 0.01) * 100.0)
    skew_val = float(stats.skew(y_seg))
    kurt_val = float(stats.kurtosis(y_seg))

    S = np.abs(librosa.stft(y_seg))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

    # Gunakan ambang suara aktif yang bersih agar noise ruangan tidak merusak f0_std
    ambang_voiced = max(0.01, float(np.percentile(rms, 35)))
    voiced_mask = rms[:len(f0_seg)] > ambang_voiced
    f0_voiced = f0_seg[voiced_mask] if np.any(voiced_mask) else f0_seg
    f0_mean = float(np.mean(f0_voiced))
    f0_std = float(np.std(f0_voiced))
    f0_jitter = float(np.mean(np.abs(np.diff(f0_voiced)))) if len(f0_voiced) > 1 else 0.0

    mel_S = librosa.feature.melspectrogram(S=S**2, sr=sr)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_S), sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)

    fitur = {
        'rms_mean': float(np.mean(rms)),
        'rms_std': float(np.std(rms)),
        'zcr_mean': float(np.mean(zcr)),
        'zcr_std': float(np.std(zcr)),
        'silence_ratio': silence_ratio,
        'skewness': skew_val,
        'kurtosis': kurt_val,
        'centroid_mean': float(np.mean(centroid)),
        'centroid_std': float(np.std(centroid)),
        'bandwidth_mean': float(np.mean(bandwidth)),
        'bandwidth_std': float(np.std(bandwidth)),
        'rolloff_mean': float(np.mean(rolloff)),
        'rolloff_std': float(np.std(rolloff)),
        'flatness_mean': float(np.mean(flatness)),
        'flatness_std': float(np.std(flatness)),
        'contrast_mean': float(np.mean(contrast)),
        'contrast_std': float(np.std(contrast)),
        'f0_mean': f0_mean,
        'f0_std': f0_std,
        'f0_jitter': f0_jitter,
        'delta_mfcc_mean': float(np.mean(delta_mfcc)),
        'delta_mfcc_std': float(np.std(delta_mfcc)),
    }
    for i in range(13):   
        fitur[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc[i]))
        fitur[f'mfcc_{i+1}_std'] = float(np.std(mfcc[i]))
    return fitur


def olah_10_detik_utuh(y_10s, sr, bundle):
    """
    Mengolah rekaman 10 detik UTUH (tanpa dipotong-potong saat jeda) tepat setelah 10 detik selesai.
    """
    y_10s = y_10s.astype(np.float32)
    y_10s = y_10s - np.mean(y_10s)

    # Normalisasi RMS pada bagian bersuara agar setara dengan dataset (~0.022)
    rms_frames = librosa.feature.rms(y=y_10s)[0]
    aktif_mask = rms_frames > np.percentile(rms_frames, 30)
    rms_aktif = float(np.mean(rms_frames[aktif_mask])) if np.any(aktif_mask) else float(np.sqrt(np.mean(y_10s**2)))
    if rms_aktif > 1e-4:
        y_10s = np.clip(y_10s * (0.022 / rms_aktif), -1.0, 1.0)

    sampel_5s = 5 * sr
    f0_10s = librosa.yin(y_10s, fmin=60, fmax=360, sr=sr)
    mid_f0 = len(f0_10s) // 2

    # Bagi 10 detik utuh menjadi 2 blok 5 detik (0-5s dan 5-10s)
    fit_seg1 = ekstrak_fitur_segmen(y_10s[:sampel_5s], sr, f0_10s[:mid_f0])
    fit_seg2 = ekstrak_fitur_segmen(y_10s[sampel_5s:2*sampel_5s], sr, f0_10s[mid_f0:])

    kolom_fitur = bundle['kolom_fitur']
    X_raw = np.array([
        [fit_seg1[k] for k in kolom_fitur],
        [fit_seg2[k] for k in kolom_fitur]
    ])
    X_scaled = bundle['scaler'].transform(X_raw)

    # Gabungkan prediksi Random Forest & SVM (jika tersedia) pada 10 detik tersebut
    proba_rf = bundle['model'].predict_proba(X_scaled)
    if 'svm_model' in bundle:
        proba_svm = bundle['svm_model'].predict_proba(X_scaled)
        proba_gabungan = 0.5 * proba_rf + 0.5 * proba_svm
    else:
        proba_gabungan = proba_rf

    # Beri bobot sedikit lebih besar pada 5 detik kedua (karena detik ke-1 biasanya baru mulai bunyi)
    mean_proba = 0.45 * proba_gabungan[0] + 0.55 * proba_gabungan[1]

    # Rata-rata fitur 10 detik untuk diagnosis
    fit_rata = {k: (fit_seg1[k] + fit_seg2[k]) / 2.0 for k in kolom_fitur}
    return mean_proba, proba_gabungan, fit_rata


def main():
    if not os.path.exists(MODEL_PATH):
        print(f'[ERROR] File model tidak ditemukan: {MODEL_PATH}')
        sys.exit(1)

    bundle = joblib.load(MODEL_PATH)
    sr = bundle.get('sr', 16000)

    CHUNK_DETIK = 0.25                 # Baca stream tiap 0.25 detik
    CHUNK_SAMPEL = int(CHUNK_DETIK * sr)
    TARGET_10S_SAMPEL = 10 * sr        # 10 detik penuh (160.000 sampel)
    AMBANG_MULAI_BICARA = 0.006        # Trigger mulai merekam 10 detik ketika ada suara masuk

    q_audio = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        q_audio.put(indata[:, 0].copy())

    dev_info = sd.query_devices(kind='input')
    print('================================================================================')
    print('   DETEKSI SUARA REAL-TIME PER 10 DETIK UTUH (LANGSUNG KELUAR DI DETIK KE-10)   ')
    print('================================================================================')
    print(f' Mikrofon Aktif : {dev_info["name"]}')
    print(' Cara Kerja     : 1. Saat hening, program menunggu suara masuk.')
    print('                  2. Begitu terdengar suara, program merekam 10 DETIK UTUH tanpa')
    print('                     nyicil tebakan di tengah jalan.')
    print('                  3. Tepat di detik ke-10, langsung keluar HASIL DETEKSI!')
    print(' Tekan [Ctrl + C] untuk berhenti.')
    print('================================================================================\n')

    sedang_merekam_10s = False
    buffer_10s = np.zeros(0, dtype=np.float32)
    sesi_ke = 1

    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype='float32',
                            blocksize=CHUNK_SAMPEL, callback=audio_callback):
            while True:
                chunk = q_audio.get()
                rms_chunk = float(np.sqrt(np.mean(chunk**2)))
                level_bar = min(20, int((rms_chunk / 0.08) * 20))
                meter_str = '#' * level_bar + '-' * (20 - level_bar)

                if not sedang_merekam_10s:
                    # Masih menunggu suara pertama masuk
                    if rms_chunk >= AMBANG_MULAI_BICARA:
                        sedang_merekam_10s = True
                        buffer_10s = chunk.copy()
                        print(f'\n [SESI #{sesi_ke}] Suara terdeteksi! Mengumpulkan 10 detik utuh...')
                    else:
                        print(f' [MENUNGGU SUARA] [{meter_str}] Silakan bicara / putar suara...       ', end='\r', flush=True)
                else:
                    # Sedang merekam 10 detik secara UTUH & KONTINU (tanpa membuang jeda di tengah)
                    buffer_10s = np.concatenate([buffer_10s, chunk])
                    durasi_jalan = min(10.0, len(buffer_10s) / sr)
                    prog_fill = int((durasi_jalan / 10.0) * 20)
                    prog_bar = '=' * prog_fill + '.' * (20 - prog_fill)

                    print(f' [MEREKAM 10 DETIK] [{prog_bar}] {durasi_jalan:4.1f} / 10.0 detik (Vol: [{meter_str[:10]}])', end='\r', flush=True)

                    # Tepat saat mencapai 10.0 detik -> Langsung olah & keluarkan hasil!
                    if len(buffer_10s) >= TARGET_10S_SAMPEL:
                        y_10s_utuh = buffer_10s[:TARGET_10S_SAMPEL]
                        print(f'\n [MENGOLAH] 10.0 detik selesai! Mengekstrak fitur & mendeteksi...')

                        mean_proba, proba_seg, fit_rata = olah_10_detik_utuh(y_10s_utuh, sr, bundle)
                        idx_final = int(np.argmax(mean_proba))
                        vonis = 'SUARA MANUSIA ASLI (HUMAN VOICE)' if idx_final == 0 else 'SUARA MANIPULASI AI (AI VOICE)'

                        print('--------------------------------------------------------------------------------')
                        print(f' [HASIL DETEKSI 10 DETIK - SESI #{sesi_ke}]')
                        print(f' >>> KEPUTUSAN  : {vonis}')
                        print(f' >>> KEYAKINAN  : {mean_proba[idx_final]*100:.2f}%  (Manusia Asli: {mean_proba[0]*100:.2f}% | AI: {mean_proba[1]*100:.2f}%)')
                        print(f' >>> Detail 10s : Bagian Awal (0-5s) -> Asli {proba_seg[0,0]*100:.1f}% / AI {proba_seg[0,1]*100:.1f}%')
                        print(f'                  Bagian Akhir(5-10s)-> Asli {proba_seg[1,0]*100:.1f}% / AI {proba_seg[1,1]*100:.1f}%')
                        print(f' >>> Fitur Utama: mfcc_8={fit_rata["mfcc_8_mean"]:.2f} | mfcc_11={fit_rata["mfcc_11_mean"]:.2f} | f0_std={fit_rata["f0_std"]:.2f} Hz')
                        print('--------------------------------------------------------------------------------\n')

                        # Kosongkan antrean audio yang menumpuk saat proses ekstraksi & siap untuk sesi berikutnya
                        while not q_audio.empty():
                            try:
                                q_audio.get_nowait()
                            except queue.Empty:
                                break

                        sedang_merekam_10s = False
                        buffer_10s = np.zeros(0, dtype=np.float32)
                        sesi_ke += 1

    except KeyboardInterrupt:
        print('\n\n[SELESAI] Program dihentikan.')


if __name__ == '__main__':
    main()
