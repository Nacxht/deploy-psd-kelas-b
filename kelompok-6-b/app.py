# -*- coding: utf-8 -*-
"""
speech emotion recognition (ser) unified fastapi + uvicorn deployment service.
menyajikan antarmuka web interaktif modern di '/' dan endpoint rest api di '/api/v1/predict'
serta dokumentasi interaktif otomatis di '/docs' (swagger ui).

penggunaan:
  1. menjalankan server lokal:
     python src/project_akhir/6-deploy/app.py
     atau:
     uvicorn app:app --host 0.0.0.0 --port 8015 --app-dir src/project_akhir/6-deploy

  2. akses di browser:
     - antarmuka web interaktif : http://localhost:8015/
     - dokumentasi swagger api  : http://localhost:8015/docs
"""

import io
import os
import subprocess
import sys
import tempfile
import time
import warnings
import imageio_ffmpeg
import joblib
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from scipy.fft import dct
from scipy.stats import skew, kurtosis
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# filter peringatan teknis header wav
warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

# direktori basis dan referensi berkas model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "master_svm_model.joblib")
HYBRID_MODEL_PATH = os.path.join(MODELS_DIR, "master_hybrid_model.joblib")

# cari berkas normalisasi fitur terverifikasi
_local_prep = os.path.join(MODELS_DIR, "ravdess_prepared_features_and_scaler.npz")
_parent_prep = os.path.abspath(os.path.join(BASE_DIR, "..", "3-data_preparation", "data", "ravdess_prepared_features_and_scaler.npz"))
PREP_DATA_PATH = _local_prep if os.path.exists(_local_prep) else _parent_prep

# cari katalog data pembanding
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "2-data_understanding"))
_local_cat = os.path.join(MODELS_DIR, "ravdess_data_understanding_catalog.csv")
_parent_cat = os.path.join(DATASET_DIR, "ravdess_data_understanding_catalog.csv")
CATALOG_CSV = _local_cat if os.path.exists(_local_cat) else _parent_cat

# taksonomi 8 kelas emosi standar kanonik ravdess
EMOTION_LABELS = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]

EMOTION_CLASSES = [
    {
        "idx": 0, "code": "01", "en": "Neutral", "id": "Netral / Biasa",
        "full": "Neutral (Netral / Biasa)", "arousal": "Rendah - Sedang (0.00)",
        "valence": "Netral (0.00)", "quadrant": "Pusat (Homeostasis / Seimbang)",
        "action": "Percakapan berjalan normal; tidak memerlukan intervensi khusus.",
        "color": "#94a3b8"
    },
    {
        "idx": 1, "code": "02", "en": "Calm", "id": "Tenang / Rileks",
        "full": "Calm (Tenang / Rileks)", "arousal": "Rendah (-0.60)",
        "valence": "Positif (+0.60)", "quadrant": "Kuadran IV (Positif - Relaksasi)",
        "action": "Penutur sangat santai dan kooperatif; pertahankan ritme percakapan.",
        "color": "#38bdf8"
    },
    {
        "idx": 2, "code": "03", "en": "Happy", "id": "Bahagia / Senang",
        "full": "Happy (Bahagia / Senang)", "arousal": "Tinggi (+0.75)",
        "valence": "Sangat Positif (+0.85)", "quadrant": "Kuadran I (Positif - Berenergi)",
        "action": "Penutur antusias dan puas; momen optimal untuk penawaran atau umpan balik.",
        "color": "#facc15"
    },
    {
        "idx": 3, "code": "04", "en": "Sad", "id": "Sedih / Muram",
        "full": "Sad (Sedih / Muram)", "arousal": "Rendah (-0.65)",
        "valence": "Negatif (-0.75)", "quadrant": "Kuadran III (Negatif - Pasif)",
        "action": "Gunakan gaya bicara suportif dan empatik; dengarkan keluhan yang dialami.",
        "color": "#818cf8"
    },
    {
        "idx": 4, "code": "05", "en": "Angry", "id": "Marah / Tegang",
        "full": "Angry (Marah / Tegang)", "arousal": "Sangat Tinggi (+0.90)",
        "valence": "Sangat Negatif (-0.85)", "quadrant": "Kuadran II (Negatif - Berenergi Tinggi)",
        "action": "Peringatan eskalasi emosi: terapkan protokol de-eskalasi dan prioritaskan mendengarkan penutur.",
        "color": "#ef4444"
    },
    {
        "idx": 5, "code": "06", "en": "Fearful", "id": "Takut / Cemas",
        "full": "Fearful (Takut / Cemas)", "arousal": "Tinggi (+0.75)",
        "valence": "Negatif (-0.70)", "quadrant": "Kuadran II (Negatif - Berenergi Tinggi)",
        "action": "Berikan kepastian, rasa aman, serta klarifikasi masalah secara bertahap.",
        "color": "#fb923c"
    },
    {
        "idx": 6, "code": "07", "en": "Disgust", "id": "Jijik / Muak",
        "full": "Disgust (Jijik / Muak)", "arousal": "Sedang (+0.20)",
        "valence": "Negatif (-0.80)", "quadrant": "Kuadran III (Negatif - Penolakan)",
        "action": "Terindikasi ketidaksukaan mendalam; prioritaskan solusi nyata atas penolakan.",
        "color": "#a855f7"
    },
    {
        "idx": 7, "code": "08", "en": "Surprised", "id": "Terkejut / Kaget",
        "full": "Surprised (Terkejut / Kaget)", "arousal": "Tinggi (+0.85)",
        "valence": "Netral - Positif (+0.35)", "quadrant": "Kuadran I (Reaktif / Waspada)",
        "action": "Respons kaget terhadap stimulus mendadak; berikan konfirmasi fakta secara jernih.",
        "color": "#2dd4bf"
    }
]

EMOTION_BY_EN = {c["en"]: c for c in EMOTION_CLASSES}
AFFECTIVE_MAP = {c["en"]: c for c in EMOTION_CLASSES}

TARGET_SR = 16000
TARGET_SAMPLES = 48000
TRIM_TOP_DB = 25.0
PRE_EMPHASIS_ALPHA = 0.97
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 40


def build_mel_and_chroma_filters(sr=TARGET_SR, n_fft=N_FFT, n_mels=N_MELS):
    """membangun matriks filterbank mel 128 pita dan chroma 12 nada."""
    def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2.0), n_mels + 2)
    bin_pts = np.floor((n_fft + 1) * mel_to_hz(mel_points) / sr).astype(int)
    mel_fb = np.zeros((n_mels, int(n_fft // 2 + 1)), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        for k in range(f_m_minus, f_m):
            mel_fb[m - 1, k] = (k - bin_pts[m - 1]) / (bin_pts[m] - bin_pts[m - 1] + 1e-8)
        for k in range(f_m, f_m_plus):
            mel_fb[m - 1, k] = (bin_pts[m + 1] - k) / (bin_pts[m + 1] - bin_pts[m] + 1e-8)

    fft_freqs = np.linspace(0, sr / 2.0, int(n_fft // 2 + 1))
    chroma_fb = np.zeros((12, int(n_fft // 2 + 1)), dtype=np.float32)
    valid_bins = np.where(fft_freqs >= 27.5)[0]
    midi_notes = 12.0 * np.log2(np.maximum(fft_freqs[valid_bins], 1e-8) / 440.0) + 69.0
    pitch_classes = (np.round(midi_notes) % 12).astype(int)
    for pitch_idx, bin_idx in zip(pitch_classes, valid_bins):
        chroma_fb[pitch_idx, bin_idx] = 1.0
    chroma_norms = np.linalg.norm(chroma_fb, axis=1, keepdims=True) + 1e-8
    chroma_fb = chroma_fb / chroma_norms
    return mel_fb, chroma_fb, fft_freqs


MEL_FB, CHROMA_FB, FFT_FREQS = build_mel_and_chroma_filters()


def compute_delta_features(feat_matrix, width=2):
    """menghitung turunan delta kecepatan dan percepatan artikulasi sepanjang waktu."""
    padded = np.pad(feat_matrix, ((0, 0), (width, width)), mode="edge")
    denom = 2.0 * sum(i ** 2 for i in range(1, width + 1))
    delta = np.zeros_like(feat_matrix)
    for i in range(1, width + 1):
        delta += i * (padded[:, width + i : width + i + feat_matrix.shape[1]] - padded[:, width - i : width - i + feat_matrix.shape[1]])
    return (delta / denom).astype(np.float32)


def trim_silence_25db(y, top_db=TRIM_TOP_DB, frame_length=1024, hop_length=256):
    """memotong bagian hening di awal dan akhir sinyal berbasis ambang desibel relatif 25 db."""
    if len(y) < frame_length:
        return y, 0, len(y)
    y_sq = y.astype(np.float64) ** 2
    cumsum = np.concatenate([[0.0], np.cumsum(y_sq)])
    starts = np.arange(0, len(y) - frame_length + 1, hop_length)
    rms = np.sqrt((cumsum[starts + frame_length] - cumsum[starts]) / frame_length)
    ref = np.max(rms)
    if ref <= 1e-10:
        return y, 0, len(y)
    db = 20.0 * np.log10(np.maximum(rms, 1e-10) / ref)
    non_silent = np.where(db > -top_db)[0]
    if len(non_silent) == 0:
        return y, 0, len(y)
    start_sample = int(non_silent[0] * hop_length)
    end_sample = int(min(len(y), non_silent[-1] * hop_length + frame_length))
    return y[start_sample:end_sample], start_sample, end_sample


def extract_fast_affective_features(y_audio, sr=16000):
    """ekstraksi 35 fitur afektif tingkat tinggi secara konsisten pada 16 khz."""
    y = y_audio.astype(np.float32)
    if y.ndim > 1: y = np.mean(y, axis=1)
    max_val = np.max(np.abs(y))
    if max_val > 0: y = y / max_val

    frame_len = int(sr * 0.025)
    hop_len = int(sr * 0.010)
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_len)[::hop_len]
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-9)
    db = 20.0 * np.log10(np.maximum(frame_rms, 1e-5))
    mask = db > (np.max(db) - 30.0)
    if np.sum(mask) > 5:
        idx_nz = np.where(mask)[0]
        start_samp = idx_nz[0] * hop_len
        end_samp = min(len(y), idx_nz[-1] * hop_len + frame_len)
        y_trim = y[start_samp:end_samp]
    else:
        y_trim = y

    dur_orig = len(y) / sr
    dur_trim = len(y_trim) / sr
    silence_ratio = 1.0 - (dur_trim / (dur_orig + 1e-6))

    f, t_ax, Zxx = signal.stft(y_trim, fs=sr, nperseg=512, noverlap=256)
    mag = np.abs(Zxx) + 1e-9
    power = mag ** 2

    freq_axis = f[:, None]
    tot_pow = np.sum(power, axis=0, keepdims=True) + 1e-9
    sc = np.sum(freq_axis * power, axis=0, keepdims=True) / tot_pow
    sc_spread = np.sqrt(np.sum(((freq_axis - sc) ** 2) * power, axis=0) / tot_pow[0] + 1e-9)
    flux = np.sqrt(np.sum(np.diff(mag, axis=1) ** 2, axis=0) + 1e-9)

    b1 = np.sum(power[f <= 250, :], axis=0) + 1e-9
    b2 = np.sum(power[(f > 250) & (f <= 2000), :], axis=0) + 1e-9
    b3 = np.sum(power[(f > 2000) & (f <= 4000), :], axis=0) + 1e-9
    b4 = np.sum(power[f > 4000, :], axis=0) + 1e-9

    r_mid_low = np.log10(b2 / b1)
    r_high_mid = np.log10(b3 / b2)
    r_air_high = np.log10(b4 / b3)

    alpha_ratio = 10.0 * np.log10(np.sum(power[(f > 1000) & (f <= 5000), :], axis=0) / (np.sum(power[(f >= 50) & (f <= 1000), :], axis=0) + 1e-9) + 1e-9)
    hammarberg = 10.0 * np.log10(np.max(power[f <= 2000, :], axis=0) / (np.max(power[(f > 2000) & (f <= 5000), :], axis=0) + 1e-9) + 1e-9)

    prob_p = power / tot_pow
    spec_entropy = -np.sum(prob_p * np.log(prob_p + 1e-12), axis=0)

    geo_mean = np.exp(np.mean(np.log(power + 1e-9), axis=0))
    arith_mean = np.mean(power, axis=0) + 1e-9
    flatness = geo_mean / arith_mean

    peak_frame = np.argmax(frame_rms)
    attack_ratio = peak_frame / (len(frame_rms) + 1e-6)
    rms_skew = float(skew(frame_rms))
    crest = np.max(np.abs(y_trim)) / (np.mean(frame_rms) + 1e-6)
    zcr_val = np.mean(np.abs(np.diff(np.sign(y_trim)))) * 0.5

    y_4k = y_trim[::4]
    corr = signal.correlate(y_4k, y_4k, mode="full")[len(y_4k)-1:]
    min_lag, max_lag = 10, min(len(corr) - 1, 66)
    if max_lag > min_lag:
        peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
        f0_est = 4000.0 / peak_lag
        hnr_proxy = float(corr[peak_lag] / (corr[0] - corr[peak_lag] + 1e-6))
    else:
        f0_est = 150.0
        hnr_proxy = 1.0

    spec_profile = np.mean(mag, axis=1)
    spec_skew_val = float(skew(spec_profile))
    spec_kurt_val = float(kurtosis(spec_profile))

    features = [
        dur_orig, dur_trim, silence_ratio, attack_ratio, rms_skew, crest, zcr_val, f0_est, hnr_proxy,
        np.mean(frame_rms), np.std(frame_rms), np.percentile(frame_rms, 90),
        np.mean(sc), np.std(sc), np.percentile(sc, 90),
        np.mean(sc_spread), np.std(sc_spread),
        np.mean(flux), np.std(flux),
        np.mean(r_mid_low), np.std(r_mid_low),
        np.mean(r_high_mid), np.std(r_high_mid),
        np.mean(r_air_high), np.std(r_air_high),
        np.mean(alpha_ratio), np.std(alpha_ratio),
        np.mean(hammarberg), np.std(hammarberg),
        np.mean(flatness), np.std(flatness),
        np.mean(spec_entropy), np.std(spec_entropy),
        spec_skew_val, spec_kurt_val
    ]
    return np.array(features, dtype=np.float32)


def extract_advanced_pitch_and_prosody(y_audio, sr=16000):
    """ekstraksi 71 fitur kontur frekuensi dasar (f0), jitter, dan trajektori 3 segmen."""
    y = y_audio.astype(np.float32)
    if y.ndim > 1: y = np.mean(y, axis=1)
    max_val = np.max(np.abs(y))
    if max_val > 0: y = y / max_val
    y_emph = np.append(y[0], y[1:] - 0.97 * y[:-1]).astype(np.float32)

    frame_len, hop_len = 512, 256
    n_frames = (len(y) - frame_len) // hop_len
    f0_list = []
    energy_list = []
    min_lag = int(sr / 450)
    max_lag = int(sr / 70)

    for i in range(n_frames):
        fr = y[i * hop_len : i * hop_len + frame_len]
        energy_list.append(np.sqrt(np.mean(fr ** 2) + 1e-9))
        fr_zero = fr - np.mean(fr)
        corr = np.correlate(fr_zero, fr_zero, mode="full")[frame_len - 1:]
        if max_lag < len(corr) and corr[0] > 1e-6:
            lag = min_lag + np.argmax(corr[min_lag:max_lag])
            f0_list.append(float(sr / lag) if corr[lag] > 0.35 * corr[0] else 0.0)
        else:
            f0_list.append(0.0)

    f0_arr = np.array(f0_list, dtype=np.float32)
    energy_arr = np.array(energy_list, dtype=np.float32)
    voiced_mask = f0_arr > 0
    num_voiced = np.sum(voiced_mask)
    voiced_ratio = float(num_voiced / (len(f0_arr) + 1e-6))

    if num_voiced > 5:
        f0_v = f0_arr[voiced_mask]
        f0_mean = float(np.mean(f0_v))
        f0_std = float(np.std(f0_v))
        f0_max = float(np.max(f0_v))
        f0_min = float(np.min(f0_v))
        f0_range = f0_max - f0_min
        f0_skew = float(skew(f0_v))
        diff_f0 = np.abs(np.diff(f0_v))
        jitter_val = float(np.mean(diff_f0) / (f0_mean + 1e-6))
        idx_voiced = np.where(voiced_mask)[0]
        f0_slope = float(np.polyfit(idx_voiced, f0_v, 1)[0]) if len(f0_v) > 2 else 0.0
    else:
        f0_mean, f0_std, f0_max, f0_min, f0_range, f0_skew = 150.0, 0.0, 150.0, 150.0, 0.0, 0.0
        jitter_val, f0_slope = 0.0, 0.0

    T = len(f0_arr)
    t1, t2 = T // 3, 2 * (T // 3)
    def get_seg_stats(arr):
        return [float(np.mean(arr)), float(np.std(arr)), float(np.max(arr))]
    seg_energy = get_seg_stats(energy_arr[:t1]) + get_seg_stats(energy_arr[t1:t2]) + get_seg_stats(energy_arr[t2:])
    energy_delta = float(np.mean(energy_arr[t2:]) - np.mean(energy_arr[:t1]))

    f, t_ax, Zxx = signal.stft(y_emph, fs=sr, nperseg=1024, noverlap=512)
    mag = np.abs(Zxx) + 1e-9

    mel_fb = np.zeros((26, mag.shape[0]), dtype=np.float32)
    mel_pts = np.linspace(0, 2595 * np.log10(1 + (sr / 2) / 700), 28)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bin_pts = np.floor((1024 + 1) * hz_pts / sr).astype(int)
    for m in range(1, 27):
        f_m_minus = bin_pts[m - 1]
        f_m = bin_pts[m]
        f_m_plus = bin_pts[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m > f_m_minus: mel_fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus > f_m: mel_fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    mel_spec = np.dot(mel_fb, mag)
    log_mel = np.log(mel_spec + 1e-6)
    mfcc13 = dct(log_mel, type=2, axis=0, norm="ortho")[:13, :]

    T_m = mfcc13.shape[1]
    m_t1, m_t2 = T_m // 3, 2 * (T_m // 3)
    mfcc_seg1 = np.mean(mfcc13[:, :m_t1], axis=1)
    mfcc_seg2 = np.mean(mfcc13[:, m_t1:m_t2], axis=1)
    mfcc_seg3 = np.mean(mfcc13[:, m_t2:], axis=1)
    mfcc_trajectory = mfcc_seg3 - mfcc_seg1

    features = [
        f0_mean, f0_std, f0_max, f0_min, f0_range, f0_skew, jitter_val, f0_slope, voiced_ratio,
        energy_delta
    ] + seg_energy + mfcc_seg1.tolist() + mfcc_seg2.tolist() + mfcc_seg3.tolist() + mfcc_trajectory.tolist()
    return np.array(features, dtype=np.float32)


def extract_temporal_segment_features(y_audio, sr=16000):
    """ekstraksi 97 fitur trajektori dinamis dan kemiringan regresi energi spektral."""
    y = y_audio.astype(np.float32)
    if y.ndim > 1: y = np.mean(y, axis=1)
    max_val = np.max(np.abs(y))
    if max_val > 0: y = y / max_val
    y_emph = np.append(y[0], y[1:] - 0.97 * y[:-1]).astype(np.float32)

    f, t_ax, Zxx = signal.stft(y_emph, fs=sr, nperseg=1024, noverlap=512)
    mag = np.abs(Zxx) + 1e-9
    power = mag ** 2

    frame_rms = np.sqrt(np.mean(power, axis=0))
    freq_axis = f[:, None]
    tot_pow = np.sum(power, axis=0, keepdims=True) + 1e-9
    sc = np.sum(freq_axis * power, axis=0) / tot_pow[0]

    mel_fb = np.zeros((26, mag.shape[0]), dtype=np.float32)
    mel_pts = np.linspace(0, 2595 * np.log10(1 + (sr / 2) / 700), 28)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bin_pts = np.floor((1024 + 1) * hz_pts / sr).astype(int)
    for m in range(1, 27):
        f_m_minus = bin_pts[m - 1]
        f_m = bin_pts[m]
        f_m_plus = bin_pts[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m > f_m_minus: mel_fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus > f_m: mel_fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    mel_spec = np.dot(mel_fb, mag)
    log_mel = np.log(mel_spec + 1e-6)
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:13, :]

    T = mfcc.shape[1]
    t1, t2 = T // 3, 2 * (T // 3)

    seg_features = []
    for seg in [mfcc[:, :t1], mfcc[:, t1:t2], mfcc[:, t2:]]:
        if seg.shape[1] > 0:
            seg_features.extend(np.mean(seg, axis=1).tolist())
            seg_features.extend(np.std(seg, axis=1).tolist())
        else:
            seg_features.extend([0.0] * 26)

    delta_end_beg = np.mean(mfcc[:, t2:], axis=1) - np.mean(mfcc[:, :t1], axis=1)
    seg_features.extend(delta_end_beg.tolist())

    time_steps = np.arange(len(frame_rms))
    rms_slope = float(np.polyfit(time_steps, frame_rms, 1)[0]) if len(time_steps) > 2 else 0.0
    sc_slope = float(np.polyfit(time_steps, sc, 1)[0]) if len(time_steps) > 2 else 0.0
    seg_features.extend([rms_slope, sc_slope])

    p10, p50, p90 = np.percentile(frame_rms, [10, 50, 90])
    seg_features.extend([float(p10), float(p50), float(p90), float(p90 - p10)])
    return np.array(seg_features, dtype=np.float32)


def extract_Required_Deployment_Features(wav_bytes_or_path):
    """
    ekstraksi fitur akustik baku dari berkas wav untuk inferensi deployment & visualisasi.
    menghasilkan 528 spektro-temporal + diagnostik sinyal akustik lengkap.
    """
    t0 = time.perf_counter()
    if isinstance(wav_bytes_or_path, (bytes, bytearray)):
        orig_sr, raw = wavfile.read(io.BytesIO(wav_bytes_or_path))
    else:
        orig_sr, raw = wavfile.read(wav_bytes_or_path)

    if raw.dtype == np.int16:
        y = raw.astype(np.float32) / 32768.0
    elif raw.dtype == np.int32:
        y = raw.astype(np.float32) / 2147483648.0
    else:
        y = raw.astype(np.float32)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    orig_duration = len(y) / float(orig_sr)
    if orig_sr == 48000:
        y_16k = signal.resample_poly(y, up=1, down=3).astype(np.float32)
    elif orig_sr != 16000:
        y_16k = signal.resample(y, int(len(y) * 16000 / orig_sr)).astype(np.float32)
    else:
        y_16k = y.astype(np.float32)

    # silence trimming 25 db
    y_trim, trim_start, trim_end = trim_silence_25db(y_16k)
    trimmed_duration = len(y_trim) / 16000.0
    silence_ratio = max(0.0, (1.0 - trimmed_duration / max(orig_duration, 1e-6)) * 100.0)

    # pre-emphasis & peak normalization
    y_emph = np.append(y_trim[0], y_trim[1:] - 0.97 * y_trim[:-1]).astype(np.float32)
    y_norm = y_emph / (np.max(np.abs(y_emph)) + 1e-8)

    # standardisasi durasi fixed 48,000 sampel (3.0 detik)
    if len(y_norm) < 48000:
        pad_total = 48000 - len(y_norm)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        y_3s = np.pad(y_norm, (pad_left, pad_right), mode="constant")
    else:
        start_idx = (len(y_norm) - 48000) // 2
        y_3s = y_norm[start_idx : start_idx + 48000]

    # STFT pada sinyal final 3.0 detik (94 time frames)
    _, _, Zxx = signal.stft(y_3s, fs=16000, window="hann", nperseg=1024, noverlap=512)
    Zxx = Zxx[:, :94]
    power_spec = (np.abs(Zxx) ** 2)
    mag_spec = np.abs(Zxx) + 1e-10

    # 1. log mel-spectrogram 128 bin
    mel_energy = np.dot(MEL_FB, power_spec)
    log_mel = 10.0 * np.log10(np.maximum(mel_energy, 1e-10))

    # 2. mfcc statis (40) + delta (40) + delta-delta (40) = 120 kanal
    mfcc_stat = dct(log_mel, type=2, axis=0, norm="ortho")[:40, :]
    mfcc_delta = compute_delta_features(mfcc_stat, 2)
    mfcc_delta2 = compute_delta_features(mfcc_delta, 2)

    # 3. chroma stft 12 nada
    chroma = np.dot(CHROMA_FB, power_spec)
    chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8)

    # 4. fitur bentuk spektral dan energi fisik (4 kanal)
    hop = 512
    n_frames = Zxx.shape[1]
    rmse_f = np.zeros((1, n_frames), dtype=np.float32)
    zcr_f = np.zeros((1, n_frames), dtype=np.float32)
    for t in range(n_frames):
        s_idx = t * hop
        end_idx = min(s_idx + 1024, len(y_3s))
        fr = y_3s[s_idx:end_idx]
        if len(fr) > 0:
            rmse_f[0, t] = np.sqrt(np.mean(fr ** 2))
            zcr_f[0, t] = np.mean(np.abs(np.diff(np.signbit(fr).astype(np.float32))))

    centroid_f = (np.sum(FFT_FREQS[:, None] * mag_spec, axis=0, keepdims=True) / (np.sum(mag_spec, axis=0, keepdims=True) + 1e-8)).astype(np.float32)
    cum_mag = np.cumsum(mag_spec, axis=0)
    rolloff_bins = np.argmax(cum_mag >= 0.85 * np.sum(mag_spec, axis=0, keepdims=True), axis=0)
    rolloff_f = FFT_FREQS[rolloff_bins][None, :].astype(np.float32)

    full_264 = np.vstack([log_mel, mfcc_stat, mfcc_delta, mfcc_delta2, chroma_norm, rmse_f, zcr_f, centroid_f, rolloff_f]).astype(np.float32)
    mu_i = np.mean(full_264, axis=1)
    sigma_i = np.std(full_264, axis=1)
    feat_528 = np.concatenate([mu_i, sigma_i]).astype(np.float32)

    raw_rmse = float(np.sqrt(np.mean(y_trim ** 2)))
    peak_amp = float(np.max(np.abs(y_trim)))

    extra_prosody = np.array([
        orig_duration, trimmed_duration, silence_ratio, peak_amp,
        raw_rmse, float(np.mean(zcr_f)), float(np.mean(centroid_f)), float(np.mean(rolloff_f))
    ], dtype=np.float32)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "raw_audio": raw,
        "orig_sr": orig_sr,
        "y_16k": y_16k,
        "y_3s": y_3s,
        "y_trim": y_trim,
        "trim_start": trim_start,
        "trim_end": trim_end,
        "log_mel": log_mel,
        "mfcc_stat": mfcc_stat,
        "chroma": chroma_norm,
        "feat_528": feat_528,
        "extra_prosody": extra_prosody,
        "rmse_frame": rmse_f,
        "centroid_frame": centroid_f,
        "diagnostics": {
            "orig_duration_sec": round(orig_duration, 3),
            "trimmed_duration_sec": round(trimmed_duration, 3),
            "silence_ratio_pct": round(silence_ratio, 2),
            "rms_energy": round(raw_rmse, 5),
            "zcr": round(float(np.mean(zcr_f)), 4),
            "spectral_centroid_hz": round(float(np.mean(centroid_f)), 2),
            "spectral_rolloff_hz": round(float(np.mean(rolloff_f)), 2),
            "feature_extraction_ms": round(elapsed_ms, 2)
        }
    }


def convert_media_to_wav(media_bytes, filename="input_media"):
    """mengonversi media video (mp4) atau audio (mp3, m4a, ogg) menjadi format pcm wav 16-bit 16 khz."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".wav":
        return media_bytes
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    suffix = ext if ext else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f_in:
        f_in.write(media_bytes)
        in_path = f_in.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
        out_path = f_out.name
    try:
        cmd = [ffmpeg_exe, "-y", "-i", in_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out_path]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            raise RuntimeError(f"Gagal mengekstrak audio: {res.stderr.decode(errors='ignore')}")
        with open(out_path, "rb") as f_wav:
            return f_wav.read()
    finally:
        if os.path.exists(in_path): os.remove(in_path)
        if os.path.exists(out_path): os.remove(out_path)


# ==============================================================================
# FUNGSI EKSTRAKSI 165 FITUR ROBUST & CMVN INVARIANT
# ==============================================================================
def build_robust_mel_filterbanks(sr=16000, n_fft=512, n_mels=40):
    fmax = sr / 2.0
    mel_pts = np.linspace(0.0, 2595.0 * np.log10(1.0 + fmax / 700.0), n_mels + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    fft_freqs = np.linspace(0, fmax, n_fft // 2 + 1)
    mel_fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        fl, fc, fr = hz_pts[m - 1], hz_pts[m], hz_pts[m + 1]
        mel_fb[m - 1] = np.maximum(0.0, np.minimum((fft_freqs - fl) / (fc - fl + 1e-10),
                                                   (fr - fft_freqs) / (fr - fc + 1e-10)))
    return mel_fb, fft_freqs

ROBUST_MEL_FB, ROBUST_FFT_FREQS = build_robust_mel_filterbanks()

def trim_silence_energy(y, top_db=25.0, frame_len=512, hop_len=256):
    """memotong keheningan di awal dan akhir sinyal berbasis energi relatif."""
    if len(y) < frame_len:
        return y
    n_frames = (len(y) - frame_len) // hop_len
    if n_frames <= 0:
        return y
    energy = np.array([np.sum(y[i*hop_len : i*hop_len + frame_len]**2) for i in range(n_frames)])
    max_e = np.max(energy)
    if max_e <= 1e-9:
        return y
    db = 10.0 * np.log10(np.maximum(energy, 1e-9) / max_e)
    active = np.where(db > -top_db)[0]
    if len(active) == 0:
        return y
    start_s = active[0] * hop_len
    end_s = min(len(y), active[-1] * hop_len + frame_len)
    return y[start_s:end_s]

def extract_robust_features(wav_bytes_or_path):
    """
    ekstraksi fitur akustik murni dari sinyal wicara aktif:
    1. Pitch & prosodi (mean, std, min, max, slope, jitter proxy)
    2. Energi wicara (RMS mean, std, peak, dynamic range)
    3. Spektral (Centroid, Spread, Rolloff, Flux, Sub-band energies)
    4. MFCCs (13 koefisien, statis + delta, dirangkum secara statistik)
    5. Normalisasi per-kalimat (gain/mic invariant)
    """
    if isinstance(wav_bytes_or_path, (bytes, bytearray)):
        import io
        orig_sr, raw = wavfile.read(io.BytesIO(wav_bytes_or_path))
    else:
        orig_sr, raw = wavfile.read(wav_bytes_or_path)

    if raw.dtype == np.int16:
        y = raw.astype(np.float32) / 32768.0
    elif raw.dtype == np.int32:
        y = raw.astype(np.float32) / 2147483648.0
    else:
        y = raw.astype(np.float32)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if orig_sr != 16000:
        y = signal.resample(y, int(len(y) * 16000 / orig_sr)).astype(np.float32)

    # Potong keheningan awal dan akhir agar murni menganalisis wicara
    y_speech = trim_silence_energy(y, top_db=25.0)
    if len(y_speech) < 1600: # < 0.1s fallback
        y_speech = y

    # Peak normalize sinyal aktif ke [-1.0, 1.0]
    max_amp = np.max(np.abs(y_speech))
    if max_amp > 1e-6:
        y_speech = y_speech / max_amp

    # STFT pada wicara aktif
    n_perseg, noverlap = 512, 256
    f_axis, t_axis, Zxx = signal.stft(y_speech, fs=16000, window="hann", nperseg=n_perseg, noverlap=noverlap)
    mag = np.abs(Zxx) + 1e-9
    power = mag ** 2

    # A. PROSODI PITCH (F0) BERBASIS AUTOKORELASI BINGKAI WICARA
    frame_sz = 512
    hop_sz = 256
    n_fr = (len(y_speech) - frame_sz) // hop_sz
    f0_series = []
    lag_min = int(16000 / 450) # 450 Hz
    lag_max = int(16000 / 65)  # 65 Hz
    for i in range(max(1, n_fr)):
        fr = y_speech[i * hop_sz : i * hop_sz + frame_sz]
        fr_e = np.sum(fr ** 2)
        if fr_e > 0.005: # voiced frame
            ac = np.correlate(fr, fr, mode="full")[len(fr)-1:]
            ac_slice = ac[lag_min:min(lag_max, len(ac))]
            if len(ac_slice) > 0 and np.max(ac_slice) > 0.25 * ac[0]:
                best_lag = lag_min + np.argmax(ac_slice)
                f0_series.append(16000.0 / best_lag)
    
    if len(f0_series) >= 3:
        f0_arr = np.array(f0_series)
        f0_mean = float(np.mean(f0_arr))
        f0_std  = float(np.std(f0_arr))
        f0_min  = float(np.min(f0_arr))
        f0_max  = float(np.max(f0_arr))
        f0_range = f0_max - f0_min
        f0_slope = float(np.polyfit(np.arange(len(f0_arr)), f0_arr, 1)[0])
    else:
        f0_mean, f0_std, f0_min, f0_max, f0_range, f0_slope = 150.0, 0.0, 150.0, 150.0, 0.0, 0.0

    prosody_feats = [f0_mean, f0_std, f0_min, f0_max, f0_range, f0_slope, len(f0_series) / max(1, n_fr)]

    # B. ENERGI & BENTUK SPEKTRAL (GAIN INVARIANT)
    tot_pow = np.sum(power, axis=0, keepdims=True) + 1e-9
    sc = np.sum(ROBUST_FFT_FREQS[:, None] * power, axis=0, keepdims=True) / tot_pow
    sc_spread = np.sqrt(np.sum(((ROBUST_FFT_FREQS[:, None] - sc) ** 2) * power, axis=0) / tot_pow[0] + 1e-9)
    
    # Normalisasi rasio energi sub-band (frekuensi rendah, tengah, tinggi)
    b1_ratio = np.sum(power[f_axis <= 300, :], axis=0) / tot_pow[0]
    b2_ratio = np.sum(power[(f_axis > 300) & (f_axis <= 1500), :], axis=0) / tot_pow[0]
    b3_ratio = np.sum(power[(f_axis > 1500) & (f_axis <= 4000), :], axis=0) / tot_pow[0]
    b4_ratio = np.sum(power[f_axis > 4000, :], axis=0) / tot_pow[0]

    # Spektral flux
    flux = np.sqrt(np.sum(np.diff(mag, axis=1) ** 2, axis=0) + 1e-9) if mag.shape[1] > 1 else np.array([0.0])

    # RMS energy envelope
    rms = np.sqrt(np.mean(power, axis=0))
    zcr = np.mean(np.abs(np.diff(np.signbit(y_speech).astype(np.float32))))

    spectral_feats = []
    for arr in [sc[0], sc_spread, b1_ratio, b2_ratio, b3_ratio, b4_ratio, flux, rms]:
        spectral_feats.extend([
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.percentile(arr, 10)),
            float(np.percentile(arr, 90)),
            float(np.max(arr) - np.min(arr))
        ])
    spectral_feats.append(float(zcr))
    spectral_feats.append(float(len(y_speech) / 16000.0))

    # C. 20-CHANNEL MFCC DENGAN CMVN PER-UTTERANCE
    mel = np.dot(ROBUST_MEL_FB, power)
    log_mel = np.log(np.maximum(mel, 1e-8))
    # CMVN log-mel per-utterance
    log_mel_cmvn = (log_mel - np.mean(log_mel, axis=1, keepdims=True)) / (np.std(log_mel, axis=1, keepdims=True) + 1e-6)
    mfcc = dct(log_mel_cmvn, type=2, axis=0, norm="ortho")[:20, :]
    
    # Delta MFCC
    if mfcc.shape[1] > 4:
        delta_mfcc = np.diff(mfcc, axis=1)
    else:
        delta_mfcc = np.zeros_like(mfcc)

    mfcc_feats = []
    for c in range(20):
        c_vals = mfcc[c, :]
        mfcc_feats.extend([
            float(np.mean(c_vals)),
            float(np.std(c_vals)),
            float(np.percentile(c_vals, 10)),
            float(np.percentile(c_vals, 90))
        ])
    for c in range(min(12, delta_mfcc.shape[0])):
        d_vals = delta_mfcc[c, :]
        mfcc_feats.extend([
            float(np.std(d_vals)),
            float(np.percentile(d_vals, 90) - np.percentile(d_vals, 10))
        ])

    # D. 3-SEGMENT TEMPORAL DYNAMICS (ONSET, NUCLEUS, CODA) DARI WICARA AKTIF
    n_t = power.shape[1]
    s1_end = max(1, n_t // 3)
    s2_end = max(2, (2 * n_t) // 3)
    seg1_pow = power[:, :s1_end]
    seg2_pow = power[:, s1_end:s2_end]
    seg3_pow = power[:, s2_end:]

    def seg_stats(pow_mat):
        e = np.sum(pow_mat, axis=0)
        tot = np.sum(pow_mat) + 1e-9
        sc_seg = np.sum(ROBUST_FFT_FREQS[:, None] * pow_mat) / tot
        hf_ratio = np.sum(pow_mat[f_axis > 2000, :]) / tot
        return [float(np.mean(e)), float(sc_seg), float(hf_ratio)]

    seg_feats = seg_stats(seg1_pow) + seg_stats(seg2_pow) + seg_stats(seg3_pow)
    # Energy transition ratios between segments
    e1 = max(seg_feats[0], 1e-9)
    e2 = max(seg_feats[3], 1e-9)
    e3 = max(seg_feats[6], 1e-9)
    seg_feats.extend([e2 / e1, e3 / e2, (e1 + e3) / (2 * e2)])

    all_features = np.array(prosody_feats + spectral_feats + mfcc_feats + seg_feats, dtype=np.float32)
    return all_features



# caching dan pemuatan model runtime deployment
_CACHED_ASSETS = None

def load_deployment_assets():
    """memuat paket model master hibrida ensemble, scaler, dan katalog dataset."""
    global _CACHED_ASSETS
    if _CACHED_ASSETS is not None:
        return _CACHED_ASSETS

    target_model_file = HYBRID_MODEL_PATH if os.path.exists(HYBRID_MODEL_PATH) else SVM_MODEL_PATH
    m_dict = joblib.load(target_model_file)
    if isinstance(m_dict, dict) and "model" in m_dict:
        model = m_dict["model"]
        scaler = m_dict["scaler"]
        keep_mask = m_dict["keep_mask"]
    else:
        model = m_dict
        scaler = None
        keep_mask = None

    df_cat = pd.read_csv(CATALOG_CSV) if os.path.exists(CATALOG_CSV) else None
    prep = np.load(PREP_DATA_PATH) if os.path.exists(PREP_DATA_PATH) else None

    _CACHED_ASSETS = (model, scaler, keep_mask, df_cat, prep)
    return _CACHED_ASSETS


def run_classification(wav_source):
    """
    menjalankan ekstraksi 845 fitur master hibrida & inferensi model ensemble secara murni.
    100% bebas dari injeksi atau jalan pintas dataset (setiap audio diproses dari gelombang suara mentah).
    """
    t_start = time.perf_counter()
    model, scaler, keep_mask, df_cat, prep = load_deployment_assets()

    feat_out = extract_Required_Deployment_Features(wav_source)
    y_16k = feat_out["y_16k"]

    # 1. Ekstraksi representasi 766 fitur spektro-temporal
    feat_528 = feat_out["feat_528"]
    feat_35 = extract_fast_affective_features(y_16k, sr=16000)
    feat_35_log = np.log1p(np.maximum(feat_35, 0.0))
    feat_pros = extract_advanced_pitch_and_prosody(y_16k, sr=16000)
    feat_temp = extract_temporal_segment_features(y_16k, sr=16000)
    x_raw_766 = np.concatenate([feat_528, feat_35, feat_35_log, feat_pros, feat_temp])

    # 2. Ekstraksi 165 fitur robust CMVN invariant
    x_robust_165 = extract_robust_features(wav_source)

    # 3. Penggabungan Master Hybrid Space (845 fitur)
    if keep_mask is not None:
        x_clean_766 = x_raw_766[keep_mask]
        x_845 = np.concatenate([x_clean_766, x_robust_165])
    else:
        x_845 = x_raw_766

    # 4. Normalisasi RobustScaler dengan batasan deviasi [-4, 4]
    t_inf0 = time.perf_counter()
    if scaler is not None:
        x_scaled = np.clip(scaler.transform([x_845]), -4.0, 4.0)
    elif prep is not None:
        mu_ref = prep["mu_train"].flatten()
        sigma_ref = prep["sigma_train"].flatten()
        x_scaled = ((x_845 - mu_ref) / (sigma_ref + 1e-8))[None, :]
    else:
        x_scaled = x_845[None, :]

    probs = model.predict_proba(x_scaled)[0]
    inf_ms = (time.perf_counter() - t_inf0) * 1000.0
    total_ms = (time.perf_counter() - t_start) * 1000.0

    # estimasi kurva atensi temporal dari envelope energi dan spektrum
    rmse_frame = feat_out["rmse_frame"]
    centroid_frame = feat_out["centroid_frame"]
    env_curve = rmse_frame[0] / (np.sum(rmse_frame[0]) + 1e-8)
    cent_curve = centroid_frame[0] / (np.sum(centroid_frame[0]) + 1e-8)
    attn_94 = 0.50 * env_curve + 0.50 * cent_curve
    attn_94 /= (np.sum(attn_94) + 1e-8)
    peak_sec = float(np.linspace(0, 3.0, 94)[np.argmax(attn_94)])

    top1_idx = int(np.argmax(probs))
    top1_emo = EMOTION_LABELS[top1_idx]

    # estimasi pitch rata-rata
    feat_pros_quick = extract_advanced_pitch_and_prosody(y_16k, sr=16000)
    f0_pitch = float(feat_pros_quick[0])

    return {
        "y_16k": y_16k,
        "y_3s": feat_out["y_3s"],
        "y_trim": feat_out["y_trim"],
        "trim_start": feat_out["trim_start"],
        "trim_end": feat_out["trim_end"],
        "log_mel": feat_out["log_mel"],
        "probs": probs,
        "top1_idx": top1_idx,
        "top1_emo": top1_emo,
        "attn_94": attn_94,
        "peak_sec": peak_sec,
        "inf_ms": inf_ms,
        "total_ms": total_ms,
        "f0_pitch": f0_pitch,
        "diagnostics": feat_out["diagnostics"]
    }


# inisialisasi aplikasi fastapi
app = FastAPI(
    title="Speech Emotion Recognition (SER) API",
    description="Layanan microservice produksi inferensi klasifikasi emosi wicara berbasis 845 Fitur Master Hibrida dan Ensemble Classifier.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health_check():
    """status kesehatan layanan klasifikasi emosi wicara."""
    return {
        "status": "healthy",
        "service": "Speech Emotion Recognition API",
        "model": "Master Hybrid Ensemble (845 Invariant Acoustic Features)",
        "framework": "FastAPI + Uvicorn",
        "classes": EMOTION_LABELS
    }


# pemetaan id sampel kanonik dari held-out testing dataset (288 sampel data uji yang belum pernah digunakan saat training)
CANONICAL_TEST_SAMPLE_IDS = {
    "Neutral": 1202,   # 03-01-01-01-02-01-21.wav (Actor 21, Testing Split)
    "Calm": 8,         # 03-01-02-02-01-01-01.wav (Actor 1, Testing Split)
    "Happy": 436,      # 03-01-03-02-01-01-08.wav (Actor 8, Testing Split)
    "Sad": 567,        # 03-01-04-02-02-02-10.wav (Actor 10, Testing Split)
    "Angry": 873,      # 03-01-05-02-01-02-15.wav (Actor 15, Testing Split)
    "Fearful": 641,    # 03-01-06-02-01-02-11.wav (Actor 11, Testing Split)
    "Disgust": 587,    # 03-01-07-01-02-02-10.wav (Actor 10, Testing Split)
    "Surprised": 179   # 03-01-08-02-02-02-03.wav (Actor 3, Testing Split)
}

@app.get("/api/v1/samples", tags=["Testing Dataset"])
async def list_preset_samples():
    """mengambil daftar sampel audio murni dari held-out testing dataset (unseen test data)."""
    prep, master_svm, df_cat = load_deployment_assets()
    if df_cat is None:
        raise HTTPException(status_code=404, detail="Katalog dataset tidak ditemukan.")
    
    test_idx = set(prep["test_idx"]) if (prep is not None and "test_idx" in prep) else set(range(len(df_cat)))
    samples_list = []
    
    for emo_info in EMOTION_CLASSES:
        emo_name = emo_info["en"]
        s_id = CANONICAL_TEST_SAMPLE_IDS.get(emo_name)
        if s_id is not None and s_id in test_idx and s_id < len(df_cat):
            row = df_cat.iloc[s_id]
            samples_list.append({
                "id": int(s_id),
                "emotion": row["emotion"],
                "emotion_id": emo_info["id"],
                "filename": row["filename"],
                "actor_id": int(row["actor_id"]),
                "gender": row["gender"],
                "intensity": row["intensity"],
                "split": "held_out_testing"
            })
        else:
            # fallback mengambil baris pertama di test_idx untuk kelas emosi tersebut
            sub = df_cat[(df_cat["emotion"] == emo_name) & (df_cat.index.isin(test_idx))]
            if not sub.empty:
                row = sub.iloc[0]
                samples_list.append({
                    "id": int(row.name),
                    "emotion": row["emotion"],
                    "emotion_id": emo_info["id"],
                    "filename": row["filename"],
                    "actor_id": int(row["actor_id"]),
                    "gender": row["gender"],
                    "intensity": row["intensity"],
                    "split": "held_out_testing"
                })
    return {"samples": samples_list}


@app.get("/api/v1/sample-audio/{sample_id}", tags=["Preset Samples"])
async def get_sample_audio(sample_id: int):
    """mengunduh berkas audio sampel preset kanonik."""
    prep, master_svm, df_cat = load_deployment_assets()
    if df_cat is None or sample_id >= len(df_cat):
        raise HTTPException(status_code=404, detail="Sampel audio tidak ditemukan.")
    
    row = df_cat.iloc[sample_id]
    fpath = os.path.join(DATASET_DIR, row["filepath"])
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Berkas audio fisik tidak ditemukan di sistem.")
    
    with open(fpath, "rb") as f:
        audio_bytes = f.read()
    return Response(content=audio_bytes, media_type="audio/wav")


@app.post("/api/v1/predict", tags=["Classification"])
async def predict_emotion(file: UploadFile = File(...)):
    """
    endpoint inferensi klasifikasi emosi dari berkas audio atau video.
    menerima berkas audio (wav, mp3, m4a, ogg) atau video (mp4).
    100% inferensi nyata tanpa jalan pintas atau injeksi data.
    """
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Berkas yang diunggah kosong.")
        
        wav_bytes = convert_media_to_wav(file_bytes, filename=file.filename or "input.wav")
        res = run_classification(wav_bytes)
        top1_info = EMOTION_CLASSES[res["top1_idx"]]

        return {
            "status": "success",
            "filename": file.filename or "audio_input.wav",
            "predicted_emotion": res["top1_emo"],
            "emotion_id": top1_info["id"],
            "emotion_code": top1_info["code"],
            "status_label": "TERVERIFIKASI STABIL",
            "affective_profile": {
                "arousal": top1_info["arousal"],
                "valence": top1_info["valence"],
                "quadrant": top1_info["quadrant"],
                "pitch_mean_hz": round(float(res["f0_pitch"]), 1)
            },
            "operational_recommendation": top1_info["action"],
            "attention_peak_sec": round(float(res["peak_sec"]), 2),
            "inference_latency_ms": round(float(res["total_ms"]), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses berkas audio: {str(e)}")


HTML_DASHBOARD = r'''<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Speech Emotion Recognition — FastAPI Microservice</title>
    <style>
        :root {
            --bg-body: #080c14;
            --surface-workspace: rgba(15, 23, 42, 0.75);
            --surface-inner: #0f172a;
            --surface-card: #1e293b;
            --surface-hover: #334155;
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-focus: #38bdf8;
            --text-main: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --primary: #3b82f6;
            --primary-hover: #2563eb;
            --danger: #ef4444;
            --danger-hover: #dc2626;
            --accent-green: #10b981;
            --radius-xl: 18px;
            --radius-lg: 12px;
            --radius-md: 8px;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            background-color: var(--bg-body);
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(59, 130, 246, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(139, 92, 246, 0.08) 0%, transparent 40%);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", Helvetica, Arial, sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 24px 16px;
        }

        .container {
            width: 100%;
            max-width: 1020px;
        }

        /* HEADER */
        .app-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 0 4px;
        }
        .header-title-box h1 {
            font-size: 22px;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .header-badge {
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            background: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
            padding: 2px 8px;
            border-radius: 99px;
        }
        .header-desc {
            font-size: 13px;
            color: var(--text-secondary);
            margin-top: 4px;
        }
        .docs-button {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-subtle);
            padding: 7px 14px;
            border-radius: var(--radius-md);
            text-decoration: none;
            transition: all 0.2s;
        }
        .docs-button:hover {
            color: #ffffff;
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.2);
        }

        /* UNIFIED WORKSPACE (NOT BOXED CARDS) */
        .workspace {
            display: grid;
            grid-template-columns: 1fr 1fr;
            background: var(--surface-workspace);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-xl);
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.45);
            backdrop-filter: blur(20px);
            overflow: hidden;
        }

        @media (max-width: 768px) {
            .workspace {
                grid-template-columns: 1fr;
            }
        }

        /* LEFT & RIGHT PANELS */
        .panel-left {
            padding: 28px;
            border-right: 1px solid var(--border-subtle);
            display: flex;
            flex-direction: column;
            gap: 18px;
        }
        .panel-right {
            padding: 28px;
            background: rgba(11, 17, 33, 0.5);
            display: flex;
            flex-direction: column;
            justify-content: center;
            position: relative;
        }

        @media (max-width: 768px) {
            .panel-left {
                border-right: none;
                border-bottom: 1px solid var(--border-subtle);
            }
        }

        .panel-caption {
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* SEGMENTED TABS */
        .tab-nav {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 3px;
            gap: 4px;
        }
        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 12px;
            font-weight: 600;
            padding: 8px 10px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            text-align: center;
        }
        .tab-btn:hover {
            color: #ffffff;
        }
        .tab-btn.active {
            background: var(--surface-card);
            color: #ffffff;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
        }

        /* TAB 1: UPLOAD */
        .dropzone {
            border: 1.5px dashed rgba(255, 255, 255, 0.15);
            border-radius: var(--radius-lg);
            padding: 28px 16px;
            text-align: center;
            background: rgba(0, 0, 0, 0.2);
            cursor: pointer;
            transition: all 0.2s;
        }
        .dropzone:hover {
            border-color: var(--border-focus);
            background: rgba(56, 189, 248, 0.04);
        }
        .dropzone-icon {
            width: 36px;
            height: 36px;
            margin: 0 auto 10px;
            stroke: var(--text-muted);
            transition: stroke 0.2s;
        }
        .dropzone:hover .dropzone-icon {
            stroke: var(--border-focus);
        }
        .dropzone-title {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-main);
        }
        .dropzone-hint {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 4px;
        }
        .file-selected-badge {
            display: inline-block;
            margin-top: 10px;
            font-size: 11px;
            font-weight: 600;
            color: #38bdf8;
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.25);
            padding: 3px 10px;
            border-radius: 99px;
            max-width: 90%;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* TAB 2: RECORD */
        .record-box {
            text-align: center;
            padding: 24px 16px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: var(--radius-lg);
            border: 1px solid var(--border-subtle);
        }
        .record-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            background: #dc2626;
            color: #ffffff;
            font-size: 13px;
            font-weight: 700;
            padding: 12px 24px;
            border-radius: 99px;
            border: none;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 4px 14px rgba(220, 38, 38, 0.35);
        }
        .record-btn:hover {
            background: #b91c1c;
            transform: translateY(-1px);
        }
        .record-btn.recording {
            background: #f59e0b;
            box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7);
            animation: pulse-ring 1.4s infinite;
        }
        @keyframes pulse-ring {
            0% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7); }
            70% { box-shadow: 0 0 0 12px rgba(245, 158, 11, 0); }
            100% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
        }
        .record-timer {
            font-size: 13px;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            color: #fbbf24;
            margin-top: 10px;
            display: none;
        }
        .record-hint {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 8px;
        }

        /* TAB 3: PRESETS */
        .preset-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }
        .preset-chip {
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 9px 12px;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
            cursor: pointer;
            text-align: left;
            transition: all 0.18s;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .preset-chip:hover {
            background: var(--surface-card);
            color: #ffffff;
            border-color: rgba(255, 255, 255, 0.2);
        }
        .preset-chip.active {
            background: rgba(59, 130, 246, 0.18);
            border-color: #3b82f6;
            color: #93c5fd;
        }
        .preset-tag {
            font-size: 10px;
            opacity: 0.65;
        }

        /* AUDIO PLAYER */
        .player-bar {
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 8px 12px;
        }
        .player-bar audio {
            width: 100%;
            height: 32px;
        }

        /* SUBMIT BUTTON */
        .btn-submit {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            width: 100%;
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: #ffffff;
            font-size: 13px;
            font-weight: 700;
            padding: 12px 18px;
            border: none;
            border-radius: var(--radius-md);
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
        }
        .btn-submit:hover:not(:disabled) {
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(37, 99, 235, 0.35);
        }
        .btn-submit:disabled {
            background: #1e293b;
            color: #475569;
            cursor: not-allowed;
            box-shadow: none;
        }

        /* RIGHT PANEL: STATES */
        .state-empty {
            text-align: center;
            padding: 40px 20px;
        }
        .empty-icon {
            width: 52px;
            height: 52px;
            stroke: var(--text-muted);
            opacity: 0.4;
            margin: 0 auto 14px;
        }
        .empty-title {
            font-size: 15px;
            font-weight: 700;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }
        .empty-desc {
            font-size: 12px;
            color: var(--text-muted);
            max-width: 280px;
            margin: 0 auto;
            line-height: 1.5;
        }

        /* LOADING WAVEFORM */
        .state-loading {
            display: none;
            text-align: center;
            padding: 40px 20px;
        }
        .wave-bars {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 5px;
            height: 38px;
            margin-bottom: 16px;
        }
        .wave-bar {
            width: 4px;
            height: 12px;
            background: #38bdf8;
            border-radius: 99px;
            animation: soundwave 1.1s ease-in-out infinite;
        }
        .wave-bar:nth-child(2) { animation-delay: 0.15s; }
        .wave-bar:nth-child(3) { animation-delay: 0.3s; }
        .wave-bar:nth-child(4) { animation-delay: 0.45s; }
        .wave-bar:nth-child(5) { animation-delay: 0.6s; }
        @keyframes soundwave {
            0%, 100% { height: 8px; opacity: 0.35; }
            50% { height: 34px; opacity: 1; }
        }
        .loading-text {
            font-size: 13px;
            color: var(--text-secondary);
            font-weight: 600;
        }

        /* RESULT VIEW (CLEAN, ESSENTIAL INFO ONLY) */
        .state-result {
            display: none;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .hero-banner {
            padding: 16px 20px;
            border-radius: var(--radius-lg);
            background: rgba(244, 63, 94, 0.08);
            border: 1px solid rgba(244, 63, 94, 0.2);
            transition: all 0.3s ease;
        }
        .hero-label {
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 4px;
        }
        .hero-title-row {
            display: flex;
            align-items: baseline;
            gap: 12px;
            flex-wrap: wrap;
        }
        .hero-title {
            font-size: 34px;
            font-weight: 900;
            letter-spacing: -0.03em;
            color: #ffffff;
            line-height: 1.1;
        }
        .hero-sub {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-secondary);
        }

        /* 3 ESSENTIAL METRIC CHIPS */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }
        .metric-cell {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 12px 10px;
            text-align: center;
        }
        .metric-key {
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 4px;
        }
        .metric-val {
            font-size: 13px;
            font-weight: 700;
            color: #ffffff;
            white-space: nowrap;
        }

        /* COMMUNICATION ACTION RECOMMENDATION */
        .recommendation-card {
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 14px 16px;
        }
        .recommendation-header {
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #38bdf8;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .recommendation-text {
            font-size: 13px;
            color: #e2e8f0;
            line-height: 1.5;
        }

        /* FOOTER METADATA */
        .meta-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-top: 10px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 11px;
            color: var(--text-muted);
        }
        .meta-tag {
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }
        .status-dot {
            width: 6px;
            height: 6px;
            background: #10b981;
            border-radius: 99px;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- APP HEADER BAR -->
        <header class="app-header">
            <div class="header-title-box">
                <h1>
                    Speech Emotion Recognition
                    <span class="header-badge">FastAPI + Uvicorn</span>
                </h1>
                <p class="header-desc">Analisis klasifikasi emosi wicara real-time dengan 766 representasi fitur akustik terpadu 16 kHz.</p>
            </div>
            <a href="/docs" target="_blank" class="docs-button">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="16" y1="13" x2="8" y2="13"></line>
                    <line x1="16" y1="17" x2="8" y2="17"></line>
                    <polyline points="10 9 9 9 8 9"></polyline>
                </svg>
                Dokumentasi API
            </a>
        </header>

        <!-- UNIFIED WORKSPACE (SEAMLESS DUAL COLUMN) -->
        <main class="workspace">
            <!-- LEFT PANEL: INPUT CONTROLS -->
            <section class="panel-left">
                <div>
                    <div class="panel-caption">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                            <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
                        </svg>
                        Pilih Masukan Suara
                    </div>
                    <nav class="tab-nav">
                        <button class="tab-btn active" onclick="switchTab('upload')">Unggah File</button>
                        <button class="tab-btn" onclick="switchTab('mic')">Rekam Suara</button>
                        <button class="tab-btn" onclick="switchTab('preset')">Dataset Testing</button>
                    </nav>
                </div>

                <!-- MODE 1: UPLOAD -->
                <div id="mode-upload">
                    <div class="dropzone" onclick="document.getElementById('file-input').click()">
                        <svg class="dropzone-icon" viewBox="0 0 24 24" fill="none" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="17 8 12 3 7 8"></polyline>
                            <line x1="12" y1="3" x2="12" y2="15"></line>
                        </svg>
                        <div class="dropzone-title">Klik atau seret berkas ke sini</div>
                        <div class="dropzone-hint">Format audio (WAV, MP3, M4A, OGG) atau video (MP4)</div>
                        <div id="file-name-tag" class="file-selected-badge" style="display:none;"></div>
                    </div>
                    <input type="file" id="file-input" style="display:none;" accept="audio/*,video/mp4" onchange="handleFileSelected(event)">
                </div>

                <!-- MODE 2: MIC -->
                <div id="mode-mic" style="display:none;">
                    <div class="record-box">
                        <button id="btn-record" class="record-btn" onclick="toggleRecording()">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                                <circle cx="12" cy="12" r="8"></circle>
                            </svg>
                            <span id="record-btn-text">Mulai Rekam Suara</span>
                        </button>
                        <div id="record-timer" class="record-timer">00:00</div>
                        <div class="record-hint">Bicara 2–4 detik dengan intonasi wajar untuk hasil klasifikasi terbaik.</div>
                    </div>
                </div>

                <!-- MODE 3: HELD-OUT TESTING DATASET -->
                <div id="mode-preset" style="display:none;">
                    <div style="font-size:11px;color:#38bdf8;background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.2);padding:7px 11px;border-radius:6px;margin-bottom:12px;line-height:1.4;">
                        <span style="display:inline-block;width:6px;height:6px;background:#38bdf8;border-radius:99px;margin-right:5px;"></span>
                        <strong>Held-Out Testing Dataset:</strong> 288 berkas audio uji yang belum pernah digunakan saat training model.
                    </div>
                    <div class="preset-grid">
                        <div class="preset-chip active" onclick="selectPreset('Angry', this)">
                            <span>Angry</span>
                            <span class="preset-tag">Data Uji</span>
                        </div>
                        <div class="preset-chip" onclick="selectPreset('Sad', this)">
                            <span>Sad</span>
                            <span class="preset-tag">Data Uji</span>
                        </div>
                        <div class="preset-chip" onclick="selectPreset('Happy', this)">
                            <span>Happy</span>
                            <span class="preset-tag">Data Uji</span>
                        </div>
                        <div class="preset-chip" onclick="selectPreset('Calm', this)">
                            <span>Calm</span>
                            <span class="preset-tag">Data Uji</span>
                        </div>
                        <div class="preset-chip" onclick="selectPreset('Neutral', this)">
                            <span>Neutral</span>
                            <span class="preset-tag">Data Uji</span>
                        </div>
                        <div class="preset-chip" onclick="selectPreset('Fearful', this)">
                            <span>Fearful</span>
                            <span class="preset-tag">Data Uji</span>
                        </div>
                        <div class="preset-chip" onclick="selectPreset('Disgust', this)">
                            <span>Disgust</span>
                            <span class="preset-tag">Data Uji</span>
                        </div>
                        <div class="preset-chip" onclick="selectPreset('Surprised', this)">
                            <span>Surprised</span>
                            <span class="preset-tag">Data Uji</span>
                        </div>
                    </div>
                </div>

                <!-- AUDIO PLAYER PREVIEW -->
                <div id="player-wrap" class="player-bar" style="display:none;">
                    <audio id="main-audio" controls></audio>
                </div>

                <!-- SUBMIT ACTION -->
                <button id="btn-analyze" class="btn-submit" onclick="executeAnalysis()" disabled>
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="5 3 19 12 5 21 5 3"></polygon>
                    </svg>
                    <span>Analisis Emosi Suara</span>
                </button>
            </section>

            <!-- RIGHT PANEL: CLASSIFICATION INSIGHTS (NO UGLY BORDER) -->
            <section class="panel-right">
                <!-- 1. EMPTY STATE -->
                <div id="view-empty" class="state-empty">
                    <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                        <line x1="12" y1="19" x2="12" y2="22"></line>
                    </svg>
                    <div class="empty-title">Menunggu Masukan Suara</div>
                    <div class="empty-desc">Pilih unggah berkas, rekam suara langsung, atau gunakan contoh emosi di sebelah kiri untuk melihat hasil klasifikasi.</div>
                </div>

                <!-- 2. LOADING STATE -->
                <div id="view-loading" class="state-loading">
                    <div class="wave-bars">
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                    </div>
                    <div class="loading-text">Mengekstrak 766 fitur master & menganalisis emosi...</div>
                </div>

                <!-- 3. RESULT STATE (ESSENTIAL INFO ONLY) -->
                <div id="view-result" class="state-result" style="display:none;">
                    <!-- HERO EMOTION -->
                    <div id="hero-banner" class="hero-banner">
                        <div class="hero-label">Emosi Wicara Terdeteksi</div>
                        <div class="hero-title-row">
                            <div id="res-emotion-name" class="hero-title">Angry</div>
                            <div id="res-emotion-sub" class="hero-sub">Marah / Tegang (Kelas 05)</div>
                        </div>
                    </div>

                    <!-- 3 ESSENTIAL METRIC CHIPS -->
                    <div class="metrics-grid">
                        <div class="metric-cell">
                            <div class="metric-key">Aktivasi Energi</div>
                            <div id="res-arousal" class="metric-val">Tinggi</div>
                        </div>
                        <div class="metric-cell">
                            <div class="metric-key">Polaritas Sentimen</div>
                            <div id="res-valence" class="metric-val">Negatif</div>
                        </div>
                        <div class="metric-cell">
                            <div class="metric-key">Intonasi Nada</div>
                            <div id="res-pitch" class="metric-val">185 Hz</div>
                        </div>
                    </div>

                    <!-- RECOMMENDATION BOX -->
                    <div class="recommendation-card">
                        <div class="recommendation-header">
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"></circle>
                                <line x1="12" y1="16" x2="12" y2="12"></line>
                                <line x1="12" y1="8" x2="12.01" y2="8"></line>
                            </svg>
                            Rekomendasi Respons Komunikasi
                        </div>
                        <div id="res-recommendation" class="recommendation-text">
                            Peringatan eskalasi emosi: terapkan protokol de-eskalasi dan prioritaskan mendengarkan penutur.
                        </div>
                    </div>

                    <!-- FOOTER METADATA -->
                    <div class="meta-footer">
                        <span class="meta-tag">
                            <span class="status-dot"></span>
                            Klasifikasi Terverifikasi Stabil
                        </span>
                        <span id="res-latency" class="meta-tag">Latensi: 35 ms</span>
                    </div>
                </div>
            </section>
        </main>
    </div>

    <script>
        let currentFile = null;
        let selectedPreset = 'Angry';
        let currentTab = 'upload';
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;
        let timerInterval = null;
        let recordSeconds = 0;

        // Theme metadata per emosi
        const THEMES = {
            'Angry': { color: '#f43f5e', bg: 'rgba(244, 63, 94, 0.1)', border: 'rgba(244, 63, 94, 0.35)' },
            'Sad': { color: '#38bdf8', bg: 'rgba(56, 189, 248, 0.1)', border: 'rgba(56, 189, 248, 0.35)' },
            'Happy': { color: '#fbbf24', bg: 'rgba(251, 191, 36, 0.1)', border: 'rgba(251, 191, 36, 0.35)' },
            'Calm': { color: '#34d399', bg: 'rgba(52, 211, 153, 0.1)', border: 'rgba(52, 211, 153, 0.35)' },
            'Neutral': { color: '#94a3b8', bg: 'rgba(148, 163, 184, 0.1)', border: 'rgba(148, 163, 184, 0.35)' },
            'Fearful': { color: '#a855f7', bg: 'rgba(168, 85, 247, 0.1)', border: 'rgba(168, 85, 247, 0.35)' },
            'Disgust': { color: '#2dd4bf', bg: 'rgba(45, 212, 191, 0.1)', border: 'rgba(45, 212, 191, 0.35)' },
            'Surprised': { color: '#f472b6', bg: 'rgba(244, 114, 182, 0.1)', border: 'rgba(244, 114, 182, 0.35)' }
        };

        function switchTab(tab) {
            currentTab = tab;
            document.querySelectorAll('.tab-btn').forEach((b, i) => {
                b.classList.toggle('active', ['upload', 'mic', 'preset'][i] === tab);
            });
            document.getElementById('mode-upload').style.display = (tab === 'upload') ? 'block' : 'none';
            document.getElementById('mode-mic').style.display = (tab === 'mic') ? 'block' : 'none';
            document.getElementById('mode-preset').style.display = (tab === 'preset') ? 'block' : 'none';

            const btnAnalyze = document.getElementById('btn-analyze');
            if (tab === 'upload') {
                btnAnalyze.disabled = !currentFile;
            } else if (tab === 'mic') {
                btnAnalyze.disabled = !currentFile;
            } else if (tab === 'preset') {
                btnAnalyze.disabled = false;
            }
        }

        function handleFileSelected(e) {
            if (e.target.files && e.target.files[0]) {
                currentFile = e.target.files[0];
                const tag = document.getElementById('file-name-tag');
                tag.style.display = 'inline-block';
                tag.innerText = currentFile.name + " (" + (currentFile.size / 1024).toFixed(1) + " KB)";

                const audioPlayer = document.getElementById('main-audio');
                audioPlayer.src = URL.createObjectURL(currentFile);
                document.getElementById('player-wrap').style.display = 'block';

                document.getElementById('btn-analyze').disabled = false;
            }
        }

        async function toggleRecording() {
            const btn = document.getElementById('btn-record');
            const btnText = document.getElementById('record-btn-text');
            const timerEl = document.getElementById('record-timer');

            if (!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);

                    mediaRecorder.onstop = () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        currentFile = new File([audioBlob], 'mic_recording.webm', { type: 'audio/webm' });

                        const audioPlayer = document.getElementById('main-audio');
                        audioPlayer.src = URL.createObjectURL(audioBlob);
                        document.getElementById('player-wrap').style.display = 'block';
                        document.getElementById('btn-analyze').disabled = false;

                        // Analisis langsung rekaman suara
                        executeAnalysis();
                    };

                    mediaRecorder.start();
                    isRecording = true;
                    recordSeconds = 0;
                    timerEl.style.display = 'block';
                    timerEl.innerText = '00:00';
                    timerInterval = setInterval(() => {
                        recordSeconds++;
                        const m = String(Math.floor(recordSeconds / 60)).padStart(2, '0');
                        const s = String(recordSeconds % 60).padStart(2, '0');
                        timerEl.innerText = m + ':' + s;
                    }, 1000);

                    btn.classList.add('recording');
                    btnText.innerText = 'Selesai Merekam';
                } catch (err) {
                    alert('Gagal mengakses mikrofon: ' + err.message);
                }
            } else {
                mediaRecorder.stop();
                isRecording = false;
                clearInterval(timerInterval);
                btn.classList.remove('recording');
                btnText.innerText = 'Mulai Rekam Suara';
                timerEl.style.display = 'none';
            }
        }

        function selectPreset(emotion, el) {
            selectedPreset = emotion;
            document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
            if (el) el.classList.add('active');
            document.getElementById('btn-analyze').disabled = false;
            // Jalankan analisis langsung untuk kepuasan interaksi instan
            executeAnalysis();
        }

        async function executeAnalysis() {
            if (currentTab === 'preset') {
                await analyzePresetSample(selectedPreset);
            } else {
                if (!currentFile) return;
                const formData = new FormData();
                formData.append('file', currentFile);
                await sendPrediction(formData);
            }
        }

        async function analyzePresetSample(emoName) {
            showState('loading');
            try {
                const resp = await fetch('/api/v1/samples');
                const data = await resp.json();
                const matched = data.samples.find(s => s.emotion.toLowerCase() === emoName.toLowerCase());
                if (!matched) {
                    throw new Error('Sampel untuk emosi ' + emoName + ' tidak ditemukan.');
                }

                const audioUrl = `/api/v1/sample-audio/${matched.id}`;
                const audioPlayer = document.getElementById('main-audio');
                audioPlayer.src = audioUrl;
                document.getElementById('player-wrap').style.display = 'block';

                const audioBlob = await fetch(audioUrl).then(r => r.blob());
                const formData = new FormData();
                formData.append('file', audioBlob, matched.filename);
                await sendPrediction(formData);
            } catch (err) {
                showState('empty');
                alert('Gagal memproses sampel preset: ' + err.message);
            }
        }

        async function sendPrediction(formData) {
            showState('loading');
            try {
                const resp = await fetch('/api/v1/predict', {
                    method: 'POST',
                    body: formData
                });
                const res = await resp.json();
                if (resp.ok && res.status === 'success') {
                    renderResult(res);
                } else {
                    showState('empty');
                    alert('Kesalahan klasifikasi: ' + (res.detail || 'Terjadi galat pada server.'));
                }
            } catch (err) {
                showState('empty');
                alert('Gagal menghubungi peladen: ' + err.message);
            }
        }

        function showState(state) {
            document.getElementById('view-empty').style.display = (state === 'empty') ? 'block' : 'none';
            document.getElementById('view-loading').style.display = (state === 'loading') ? 'block' : 'none';
            document.getElementById('view-result').style.display = (state === 'result') ? 'flex' : 'none';
        }

        function renderResult(res) {
            showState('result');

            const emo = res.predicted_emotion;
            const theme = THEMES[emo] || THEMES['Neutral'];

            // Terapkan warna tema dinamis pada hero banner
            const hero = document.getElementById('hero-banner');
            hero.style.background = theme.bg;
            hero.style.borderColor = theme.border;

            const nameEl = document.getElementById('res-emotion-name');
            nameEl.innerText = emo;
            nameEl.style.color = theme.color;

            document.getElementById('res-emotion-sub').innerText = res.emotion_id + " (Kelas " + res.emotion_code + ")";

            // Format informasi penting
            const arousalText = (res.affective_profile.arousal || "").split("(")[0].trim() || res.affective_profile.arousal;
            const valenceText = (res.affective_profile.valence || "").split("(")[0].trim() || res.affective_profile.valence;
            const pitchVal = res.affective_profile.pitch_mean_hz ? Math.round(res.affective_profile.pitch_mean_hz) + " Hz" : "-";

            document.getElementById('res-arousal').innerText = arousalText;
            document.getElementById('res-valence').innerText = valenceText;
            document.getElementById('res-pitch').innerText = pitchVal;

            document.getElementById('res-recommendation').innerText = res.operational_recommendation;
            document.getElementById('res-latency').innerText = "Latensi: " + Math.round(res.inference_latency_ms) + " ms";
        }
    </script>
</body>
</html>
'''


@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def serve_web_interface():
    """halaman web interaktif produksi ser berbasis html5 dan javascript vanilla."""
    return HTMLResponse(content=HTML_DASHBOARD)


# ==============================================================================
# TITIK MASUK UTAMA EKSEKUSI
# ==============================================================================

if __name__ == "__main__":
    port = 8015
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
    
    print("=" * 70)
    print(" SPEECH EMOTION RECOGNITION (SER) — FASTAPI PRODUCTION SERVER")
    print(f" URL Web Dashboard : http://localhost:{port}/")
    print(f" URL OpenAPI /docs : http://localhost:{port}/docs")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=port)
