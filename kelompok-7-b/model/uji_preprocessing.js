
const fs = require("fs");

const data = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const sos = data.sos_coefficients;
const input = data.audio_input;
const expected = new Float32Array(data.expected_output);

function hitungZi(sos) {
    let scale = 1.0;
    const zi = [];

    for (const section of sos) {
        const [b0, b1, b2, a0, a1, a2] = section;

        const B0 = b0 / a0;
        const B1 = b1 / a0;
        const B2 = b2 / a0;
        const A1 = a1 / a0;
        const A2 = a2 / a0;

        const gain = (B0 + B1 + B2) / (1 + A1 + A2);
        const z0 = gain - B0;
        const z1 = B2 - A2 * gain;

        zi.push([scale * z0, scale * z1]);
        scale *= gain;
    }

    return zi;
}

function sosfilt(sos, x, zi) {
    let hasil = Array.from(x);

    for (let s = 0; s < sos.length; s++) {
        const [b0, b1, b2, a0, a1, a2] = sos[s];

        const B0 = b0 / a0;
        const B1 = b1 / a0;
        const B2 = b2 / a0;
        const A1 = a1 / a0;
        const A2 = a2 / a0;

        let z1 = zi[s][0];
        let z2 = zi[s][1];

        const keluaran = new Array(hasil.length);

        for (let i = 0; i < hasil.length; i++) {
            const x0 = hasil[i];
            const y = B0 * x0 + z1;

            z1 = B1 * x0 - A1 * y + z2;
            z2 = B2 * x0 - A2 * y;

            keluaran[i] = y;
        }

        hasil = keluaran;
    }

    return hasil;
}

function sosfiltfilt(sos, x) {
    const jumlahSection = sos.length;

    const zerosB = sos.filter(s => s[2] === 0).length;
    const zerosA = sos.filter(s => s[5] === 0).length;

    const padlen = 3 * (
        2 * jumlahSection + 1 - Math.min(zerosB, zerosA)
    );

    if (x.length <= padlen) {
        throw new Error("Audio terlalu pendek untuk padding filter.");
    }

    const ziDasar = hitungZi(sos);

    // Odd extension di bagian awal.
    const kiri = [];
    for (let i = padlen; i >= 1; i--) {
        kiri.push(2 * x[0] - x[i]);
    }

    // Odd extension di bagian akhir.
    const kanan = [];
    for (let i = 1; i <= padlen; i++) {
        kanan.push(2 * x[x.length - 1] - x[x.length - 1 - i]);
    }

    const diperluas = [...kiri, ...x, ...kanan];

    // Filter maju.
    const ziMaju = ziDasar.map(z => [
        z[0] * diperluas[0],
        z[1] * diperluas[0]
    ]);

    const maju = sosfilt(sos, diperluas, ziMaju);

    // Balik, filter lagi, kemudian balik kembali.
    const terbalik = maju.reverse();

    const ziMundur = ziDasar.map(z => [
        z[0] * terbalik[0],
        z[1] * terbalik[0]
    ]);

    const mundur = sosfilt(sos, terbalik, ziMundur).reverse();

    return mundur.slice(padlen, padlen + x.length);
}

// Preprocessing dan normalisasi.
let hasil = sosfiltfilt(sos, input);

const maksimum = Math.max(...hasil.map(v => Math.abs(v)));

hasil = new Float32Array(
    hasil.map(v => v / (maksimum + data.epsilon))
);

// Bandingkan dengan hasil Python.
let maxError = 0;
let totalError = 0;
let berbeda = 0;

for (let i = 0; i < hasil.length; i++) {
    const error = Math.abs(hasil[i] - expected[i]);

    maxError = Math.max(maxError, error);
    totalError += error;

    if (error > 1e-6) berbeda++;
}

console.log("=== HASIL UJI PREPROCESSING JAVASCRIPT ===");
console.log("Jumlah sampel:", hasil.length);
console.log("Maksimum selisih:", maxError);
console.log("Rata-rata selisih:", totalError / hasil.length);
console.log("Sampel berbeda (> 1e-6):", berbeda);

console.log(
    "Status:",
    maxError <= 1e-6 ? "LOLOS" : "PERLU DIPERIKSA"
);
