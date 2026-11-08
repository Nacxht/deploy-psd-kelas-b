
const fs = require("fs");
const data = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));

const audio = data.audio;
const window = data.window;
const expected = data.expected_features;

const N = data.n_fft;
const hop = data.hop_length;
const frameLength = data.frame_length;
const sr = data.sampling_rate;

function fft(re, im) {
    const n = re.length;

    for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;

        if (i < j) {
            [re[i], re[j]] = [re[j], re[i]];
            [im[i], im[j]] = [im[j], im[i]];
        }
    }

    for (let len = 2; len <= n; len <<= 1) {
        const angle = -2 * Math.PI / len;
        const wr0 = Math.cos(angle);
        const wi0 = Math.sin(angle);

        for (let i = 0; i < n; i += len) {
            let wr = 1;
            let wi = 0;

            for (let j = 0; j < len / 2; j++) {
                const u = i + j;
                const v = u + len / 2;

                const vr = re[v] * wr - im[v] * wi;
                const vi = re[v] * wi + im[v] * wr;

                re[v] = re[u] - vr;
                im[v] = im[u] - vi;
                re[u] += vr;
                im[u] += vi;

                const nextWr = wr * wr0 - wi * wi0;
                wi = wr * wi0 + wi * wr0;
                wr = nextWr;
            }
        }
    }
}

function meanStd(values) {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;

    const variance = values.reduce(
        (a, b) => a + (b - mean) ** 2, 0
    ) / values.length;

    return [mean, Math.sqrt(variance)];
}

// STFT dan magnitude spectrum.
const jumlahFrame = Math.floor((audio.length - N) / hop) + 1;
const spectra = [];

for (let f = 0; f < jumlahFrame; f++) {
    const awal = f * hop;
    const re = new Array(N);
    const im = new Array(N).fill(0);

    for (let i = 0; i < N; i++) {
        re[i] = audio[awal + i] * window[i];
    }

    fft(re, im);

    const magnitude = [];

    for (let k = 0; k <= N / 2; k++) {
        magnitude.push(Math.hypot(re[k], im[k]));
    }

    spectra.push(magnitude);
}

const hasil = {};

// 1. Spectral centroid.
const centroid = [];

for (const spectrum of spectra) {
    const total = spectrum.reduce((a, b) => a + b, 0);

    let weighted = 0;

    for (let k = 0; k < spectrum.length; k++) {
        weighted += (k * sr / N) * spectrum[k];
    }

    centroid.push(total === 0 ? 0 : weighted / total);
}

[hasil.centroid_mean, hasil.centroid_std] = meanStd(centroid);

// 2. Spectral flatness.
const flatness = [];

for (const spectrum of spectra) {
    const power = spectrum.map(v => Math.max(1e-10, v ** 2));

    const logMean = power.reduce(
        (a, b) => a + Math.log(b), 0
    ) / power.length;

    const arithmeticMean = power.reduce(
        (a, b) => a + b, 0
    ) / power.length;

    flatness.push(Math.exp(logMean) / arithmeticMean);
}

[hasil.flatness_mean, hasil.flatness_std] = meanStd(flatness);

// 3. Spectral rolloff 85%.
const rolloff = [];

for (const spectrum of spectra) {
    const total = spectrum.reduce((a, b) => a + b, 0);
    const batas = 0.85 * total;

    let akumulasi = 0;
    let indeks = 0;

    for (let k = 0; k < spectrum.length; k++) {
        akumulasi += spectrum[k];

        if (akumulasi >= batas) {
            indeks = k;
            break;
        }
    }

    rolloff.push(indeks * sr / N);
}

[hasil.rolloff_mean, hasil.rolloff_std] = meanStd(rolloff);

// 4. Zero Crossing Rate.
const zcr = [];
const jumlahFrameWaktu = Math.floor(
    (audio.length - frameLength) / hop
) + 1;

for (let f = 0; f < jumlahFrameWaktu; f++) {
    const awal = f * hop;
    let crossings = 0;

    for (let i = 1; i < frameLength; i++) {
        const sebelumnya = audio[awal + i - 1];
        const sekarang = audio[awal + i];

        // zero_pos=True: nilai nol dianggap positif.
        if (
            (sebelumnya < 0 && sekarang >= 0) ||
            (sebelumnya >= 0 && sekarang < 0)
        ) {
            crossings++;
        }
    }

    zcr.push(crossings / frameLength);
}

[hasil.zcr_mean, hasil.zcr_std] = meanStd(zcr);

// 5. RMS tanpa window.
const rms = [];

for (let f = 0; f < jumlahFrameWaktu; f++) {
    const awal = f * hop;
    let total = 0;

    for (let i = 0; i < frameLength; i++) {
        const nilai = audio[awal + i];
        total += nilai * nilai;
    }

    rms.push(Math.sqrt(total / frameLength));
}

[hasil.rms_mean, hasil.rms_std] = meanStd(rms);

// Perbandingan dengan Python.
let maxError = 0;
let totalError = 0;
let berbeda = 0;

console.log("=== PERBANDINGAN 10 FITUR ===");

for (const nama of Object.keys(hasil)) {
    const error = Math.abs(hasil[nama] - expected[nama]);

    maxError = Math.max(maxError, error);
    totalError += error;

    if (error > 1e-4) berbeda++;

    console.log(
        nama,
        "| Python:", expected[nama],
        "| JavaScript:", hasil[nama],
        "| Selisih:", error
    );
}

console.log("\n=== RINGKASAN UJI FITUR ===");
console.log("Jumlah fitur:", Object.keys(hasil).length);
console.log("Selisih maksimum:", maxError);
console.log("Rata-rata selisih:", totalError / Object.keys(hasil).length);
console.log("Fitur berbeda (> 1e-4):", berbeda);

console.log(
    "Status:",
    maxError <= 1e-4 ? "LOLOS" : "PERLU DIPERIKSA"
);
