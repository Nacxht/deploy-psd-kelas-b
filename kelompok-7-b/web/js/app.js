
// ============================================================
// APLIKASI UTAMA DETEKSI SUARA KEBOCORAN
// ============================================================

// Elemen HTML
const micStatus = document.getElementById("micStatus");
const waveCanvas = document.getElementById("waveCanvas");
const wavePlaceholder = document.getElementById("wavePlaceholder");
const recordStatus = document.getElementById("recordStatus");
const recordTimer = document.getElementById("recordTimer");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const uploadBtn = document.getElementById("uploadBtn");
const audioFileInput = document.getElementById("audioFileInput");
const sampleBocorBtn = document.getElementById("sampleBocorBtn");
const sampleNormalBtn = document.getElementById("sampleNormalBtn");

const processStatus = document.getElementById("processStatus");
const resultDisplay = document.getElementById("resultDisplay");

const bocorValue = document.getElementById("bocorValue");
const bocorBar = document.getElementById("bocorBar");
const normalValue = document.getElementById("normalValue");
const normalBar = document.getElementById("normalBar");

const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const historyBody = document.getElementById("historyBody");

let nomorRiwayat = 0;

// ============================================================
// KONFIGURASI APLIKASI
// ============================================================

const TARGET_SAMPLE_RATE = 16000;
const DURASI_SEGMENT = 1;
const JUMLAH_SAMPEL = TARGET_SAMPLE_RATE * DURASI_SEGMENT;

// ============================================================
// VARIABEL APLIKASI
// ============================================================

let audioContext = null;
let mediaStream = null;
let sumberAudio = null;
let audioProcessor = null;
let gainNode = null;

let sedangMerekam = false;
let modelSiap = false;

let bufferMasuk = [];
let jumlahBuffer = 0;

let antrianProses = Promise.resolve();

let waktuMulai = 0;
let animationId = null;

let audioTerakhir = null;
let samplingRateRekaman = null;

// ============================================================
// FUNGSI STATUS
// ============================================================

function ubahStatusMic(teks) {
    micStatus.textContent = teks;
}

function ubahStatusRekaman(teks) {
    recordStatus.textContent = teks;
}

function ubahStatusProses(teks) {
    processStatus.textContent = teks;
}

// ============================================================
// MEMUAT KONFIGURASI DAN MODEL
// ============================================================

async function inisialisasiAplikasi() {
    try {
        ubahStatusMic("Memuat sistem...");
        ubahStatusProses("Memuat konfigurasi audio dan model...");

        await Promise.all([
            muatKonfigurasiAudio(),
            muatModelRandomForest()
        ]);

        modelSiap = true;

        ubahStatusMic("Sistem siap");
        ubahStatusProses("Model Random Forest berhasil dimuat.");

        startBtn.disabled = false;
        stopBtn.disabled = true;

        console.log("Konfigurasi audio berhasil dimuat.");
        console.log("Model Random Forest berhasil dimuat.");

    } catch (error) {
        modelSiap = false;

        ubahStatusMic("Sistem bermasalah");
        ubahStatusProses("Gagal memuat sistem: " + error.message);

        startBtn.disabled = true;

        console.error("Kesalahan inisialisasi:", error);
    }
}

// ============================================================
// RESAMPLING AUDIO KE 16000 Hz
// ============================================================

async function resamplingAudio(audio, samplingRateAsal) {

    if (samplingRateAsal === TARGET_SAMPLE_RATE) {
        return Float32Array.from(audio);
    }

    const OfflineContext =
        window.OfflineAudioContext ||
        window.webkitOfflineAudioContext;

    if (!OfflineContext) {
        throw new Error(
            "Browser tidak mendukung OfflineAudioContext."
        );
    }

    const panjangOutput = Math.round(
        audio.length * TARGET_SAMPLE_RATE / samplingRateAsal
    );

    const offlineContext = new OfflineContext(
        1,
        panjangOutput,
        TARGET_SAMPLE_RATE
    );

    const audioBuffer = offlineContext.createBuffer(
        1,
        audio.length,
        samplingRateAsal
    );

    audioBuffer.copyToChannel(audio, 0);

    const sumber = offlineContext.createBufferSource();
    sumber.buffer = audioBuffer;
    sumber.connect(offlineContext.destination);
    sumber.start(0);

    const hasilRendering = await offlineContext.startRendering();

    return Float32Array.from(
        hasilRendering.getChannelData(0)
    );
}

// ============================================================
// MEMPROSES SATU SEGMENT AUDIO
// ============================================================

async function prosesSegmentAudio(audioAsal) {

    try {
        ubahStatusProses("Melakukan resampling audio...");

        const audio16k = await resamplingAudio(
            audioAsal,
            samplingRateRekaman
        );

        let audioFinal = audio16k;
        if (audioFinal.length > JUMLAH_SAMPEL) {
            audioFinal = audioFinal.subarray(0, JUMLAH_SAMPEL);
        } else if (audioFinal.length < JUMLAH_SAMPEL) {
            const padded = new Float32Array(JUMLAH_SAMPEL);
            padded.set(audioFinal);
            audioFinal = padded;
        }

        ubahStatusProses("Melakukan preprocessing audio...");

        const audioBersih = preprocessingAudio(audioFinal);

        ubahStatusProses("Mengekstraksi 50 fitur audio...");

        const fitur = ekstraksiSemuaFitur(audioBersih);

        if (fitur.length !== 50) {
            throw new Error(
                "Jumlah fitur tidak sesuai: " + fitur.length
            );
        }

        ubahStatusProses("Menjalankan prediksi Random Forest...");

        const hasilPrediksi = await prediksiRandomForest(fitur);

        tampilkanHasilPrediksi(hasilPrediksi);

    } catch (error) {
        console.error("Kesalahan pemrosesan audio:", error);

        ubahStatusProses(
            "Gagal memproses audio: " + error.message
        );
    }
}

// ============================================================
// MENAMPILKAN HASIL PREDIKSI
// ============================================================

function tampilkanHasilPrediksi(hasil) {

    const prediksi = (hasil.prediksi || "").toLowerCase();

    const probabilitasBocor = Number(
        hasil.probabilitasPerKelas?.bocor ??
        hasil.probabilitas?.[0] ??
        0
    );

    const probabilitasNormal = Number(
        hasil.probabilitasPerKelas?.normal ??
        hasil.probabilitas?.[1] ??
        0
    );

    if (
        !Number.isFinite(probabilitasBocor) ||
        !Number.isFinite(probabilitasNormal)
    ) {
        throw new Error("Probabilitas prediksi tidak valid.");
    }

    const persenBocor = probabilitasBocor * 100;
    const persenNormal = probabilitasNormal * 100;

    if (prediksi === "bocor") {
        resultDisplay.className = "result-display is-leak";
        resultDisplay.innerHTML = `
            <div class="result-symbol" style="background: #fee2e2; color: #dc2626;">⚠</div>
            <h2 style="color: #dc2626;">TERDETEKSI KEBOCORAN</h2>
            <p>Keyakinan: ${persenBocor.toFixed(1)}%</p>
        `;
    } else {
        resultDisplay.className = "result-display is-normal";
        resultDisplay.innerHTML = `
            <div class="result-symbol" style="background: #dcfce7; color: #16a34a;">✓</div>
            <h2 style="color: #16a34a;">KONDISI NORMAL</h2>
            <p>Keyakinan: ${persenNormal.toFixed(1)}%</p>
        `;
    }

    bocorValue.textContent =
        persenBocor.toFixed(2) + "%";

    normalValue.textContent =
        persenNormal.toFixed(2) + "%";

    bocorBar.style.width =
        persenBocor + "%";

    normalBar.style.width =
        persenNormal + "%";

    ubahStatusProses(
        "Prediksi selesai menggunakan " +
        hasil.jumlahPohon + " pohon Random Forest."
    );

    tambahRiwayat(
        prediksi,
        persenBocor,
        persenNormal
    );

    console.log("HASIL PREDIKSI:", hasil);
}

// ============================================================
// MENAMBAHKAN RIWAYAT DETEKSI
// ============================================================

function tambahRiwayat(prediksi, bocor, normal) {

    const emptyPlaceholder = historyBody.querySelector(".empty-history");
    if (emptyPlaceholder) {
        emptyPlaceholder.closest("tr")?.remove();
    }

    nomorRiwayat++;
    const baris = document.createElement("tr");
    const waktu = new Date().toLocaleTimeString("id-ID");

    const badgeStyle = (prediksi || "").toLowerCase() === "bocor"
        ? "background: #fee2e2; color: #dc2626; padding: 3px 8px; border-radius: 6px; font-weight: bold; font-size: 11px;"
        : "background: #dcfce7; color: #16a34a; padding: 3px 8px; border-radius: 6px; font-weight: bold; font-size: 11px;";

    baris.innerHTML = `
        <td>${nomorRiwayat}</td>
        <td>${waktu}</td>
        <td><span style="${badgeStyle}">${prediksi.toUpperCase()}</span></td>
        <td>Bocor: ${bocor.toFixed(1)}% | Normal: ${normal.toFixed(1)}%</td>
    `;

    historyBody.prepend(baris);
}

// ============================================================
// MENGAMBIL BUFFER AUDIO PER SATU DETIK
// ============================================================

function masukkanBufferAudio(data) {

    bufferMasuk.push(Float32Array.from(data));
    jumlahBuffer += data.length;

    while (
        jumlahBuffer >=
        Math.round(audioContext.sampleRate * DURASI_SEGMENT)
    ) {
        const ukuranSegment = Math.round(
            audioContext.sampleRate * DURASI_SEGMENT
        );

        const segment = new Float32Array(ukuranSegment);

        let posisi = 0;

        while (posisi < ukuranSegment) {

            const bagianPertama = bufferMasuk[0];

            const sisa = ukuranSegment - posisi;

            if (bagianPertama.length <= sisa) {

                segment.set(bagianPertama, posisi);

                posisi += bagianPertama.length;

                bufferMasuk.shift();

            } else {

                segment.set(
                    bagianPertama.subarray(0, sisa),
                    posisi
                );

                bufferMasuk[0] =
                    bagianPertama.subarray(sisa);

                posisi += sisa;
            }
        }

        jumlahBuffer -= ukuranSegment;

        antrianProses = antrianProses.then(
            () => prosesSegmentAudio(segment)
        );
    }
}

// ============================================================
// MENAMPILKAN GELOMBANG AUDIO
// ============================================================

function gambarGelombang(audio) {

    const canvas = waveCanvas;
    const ctx = canvas.getContext("2d");

    const lebar = canvas.clientWidth;
    const tinggi = canvas.clientHeight;

    if (!lebar || !tinggi) {
        return;
    }

    const rasio = window.devicePixelRatio || 1;

    if (
        canvas.width !== Math.round(lebar * rasio) ||
        canvas.height !== Math.round(tinggi * rasio)
    ) {
        canvas.width = Math.round(lebar * rasio);
        canvas.height = Math.round(tinggi * rasio);
    }

    ctx.setTransform(rasio, 0, 0, rasio, 0, 0);
    ctx.clearRect(0, 0, lebar, tinggi);

    ctx.beginPath();

    const tengah = tinggi / 2;
    const langkah = Math.max(
        1,
        Math.floor(audio.length / lebar)
    );

    for (let x = 0; x < lebar; x++) {

        const indeks = Math.min(
            x * langkah,
            audio.length - 1
        );

        const y = tengah - audio[indeks] * tengah * 0.85;

        if (x === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }

    ctx.strokeStyle = "#22c55e";
    ctx.lineWidth = 2;
    ctx.stroke();
}

// ============================================================
// TIMER REKAMAN
// ============================================================

function perbaruiTimer() {

    if (!sedangMerekam) {
        return;
    }

    const detik = Math.floor(
        (performance.now() - waktuMulai) / 1000
    );

    const menit = Math.floor(detik / 60);
    const sisaDetik = detik % 60;

    recordTimer.textContent =
        String(menit).padStart(2, "0") +
        ":" +
        String(sisaDetik).padStart(2, "0");

    animationId = requestAnimationFrame(perbaruiTimer);
}

// ============================================================
// MEMULAI REKAMAN MIKROFON
// ============================================================

async function mulaiRekaman() {

    if (!modelSiap) {
        ubahStatusProses("Model belum siap.");
        return;
    }

    if (sedangMerekam) {
        return;
    }

    try {

        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error(
                "Akses mikrofon tidak tersedia. Gunakan localhost atau HTTPS."
            );
        }

        ubahStatusMic("Meminta izin mikrofon...");
        ubahStatusProses("Menghubungkan mikrofon...");

        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            }
        });

        const AudioContextClass =
            window.AudioContext ||
            window.webkitAudioContext;

        audioContext = new AudioContextClass();

        samplingRateRekaman = audioContext.sampleRate;

        await audioContext.resume();

        sumberAudio = audioContext.createMediaStreamSource(
            mediaStream
        );

        audioProcessor = audioContext.createScriptProcessor(
            4096,
            1,
            1
        );

        gainNode = audioContext.createGain();
        gainNode.gain.value = 0;

        bufferMasuk = [];
        jumlahBuffer = 0;

        antrianProses = Promise.resolve();

        audioProcessor.onaudioprocess = function(event) {

            if (!sedangMerekam) {
                return;
            }

            const data = event.inputBuffer.getChannelData(0);

            const salinan = Float32Array.from(data);

            audioTerakhir = salinan;

            if (wavePlaceholder) {
                wavePlaceholder.style.display = "none";
            }

            gambarGelombang(salinan);

            masukkanBufferAudio(salinan);
        };

        sumberAudio.connect(audioProcessor);
        audioProcessor.connect(gainNode);
        gainNode.connect(audioContext.destination);

        sedangMerekam = true;
        waktuMulai = performance.now();

        ubahStatusMic("Mikrofon aktif");
        ubahStatusRekaman("Sedang merekam");
        ubahStatusProses(
            "Audio sedang dikumpulkan untuk proses deteksi."
        );

        startBtn.disabled = true;
        stopBtn.disabled = false;

        perbaruiTimer();

        console.log(
            "Sampling rate mikrofon:",
            samplingRateRekaman
        );

    } catch (error) {

        console.error("Kesalahan mikrofon:", error);

        ubahStatusMic("Mikrofon tidak tersedia");
        ubahStatusProses(error.message);

        await hentikanPerangkatAudio();

        sedangMerekam = false;

        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
}

// ============================================================
// MENGHENTIKAN PERANGKAT AUDIO
// ============================================================

async function hentikanPerangkatAudio() {

    if (audioProcessor) {
        audioProcessor.onaudioprocess = null;
        audioProcessor.disconnect();
        audioProcessor = null;
    }

    if (sumberAudio) {
        sumberAudio.disconnect();
        sumberAudio = null;
    }

    if (gainNode) {
        gainNode.disconnect();
        gainNode = null;
    }

    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

    if (audioContext) {
        await audioContext.close();
        audioContext = null;
    }
}

// ============================================================
// MENGHENTIKAN REKAMAN
// ============================================================

async function hentikanRekaman() {

    if (!sedangMerekam) {
        return;
    }

    sedangMerekam = false;

    if (animationId !== null) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }

    ubahStatusMic("Mikrofon tidak aktif");
    ubahStatusRekaman("Rekaman dihentikan");

    startBtn.disabled = false;
    stopBtn.disabled = true;

    await hentikanPerangkatAudio();

    // Tunggu seluruh segment yang sudah masuk antrean selesai.
    await antrianProses;

    bufferMasuk = [];
    jumlahBuffer = 0;

    ubahStatusProses(
        "Rekaman selesai. Segment audio yang lengkap telah diproses."
    );

    console.log("Rekaman dihentikan.");
}

// ============================================================
// MEMBERSIHKAN RIWAYAT
// ============================================================

function bersihkanRiwayat() {

    nomorRiwayat = 0;

    // Mengembalikan riwayat ke pesan awal
    historyBody.innerHTML = `
        <tr>
            <td colspan="4" class="empty-history">
                Belum ada riwayat deteksi.
            </td>
        </tr>
    `;

    // Mengembalikan hasil klasifikasi ke kondisi awal
    resultDisplay.className = "result-display";
    resultDisplay.innerHTML = `
        <div class="result-symbol">?</div>
        <h2>Belum Ada Hasil</h2>
        <p>Silakan lakukan perekaman audio atau uji sampel.</p>
    `;

    // Mengembalikan probabilitas ke 0%
    bocorValue.textContent = "0.00%";
    normalValue.textContent = "0.00%";

    // Mengosongkan progress bar
    bocorBar.style.width = "0%";
    normalBar.style.width = "0%";

    ubahStatusProses("Riwayat dan hasil deteksi berhasil dibersihkan.");
    console.log("Riwayat dan hasil deteksi berhasil dibersihkan.");
}

// ============================================================
// PENGUJIAN AUDIO DARI FILE / SAMPEL
// ============================================================

async function prosesAudioBufferLangsung(audioBuffer, labelNama) {
    if (!modelSiap) {
        ubahStatusProses("Model belum siap.");
        return;
    }

    try {
        ubahStatusMic("Menganalisis audio...");
        ubahStatusProses(`Menganalisis audio "${labelNama}"...`);

        const channelData = audioBuffer.getChannelData(0);
        const srAsal = audioBuffer.sampleRate;

        // Tampilkan gelombang di canvas
        if (wavePlaceholder) {
            wavePlaceholder.style.display = "none";
        }
        gambarGelombang(channelData.subarray(0, Math.min(channelData.length, srAsal)));

        // Ambil 1 detik pertama atau lakukan padding jika durasi kurang
        const durasiSampel = Math.min(channelData.length, Math.round(srAsal * DURASI_SEGMENT));
        const segmenAsal = channelData.subarray(0, durasiSampel);

        ubahStatusProses("Resampling ke 16.000 Hz...");
        const audio16k = await resamplingAudio(segmenAsal, srAsal);

        let audioFinal = audio16k;
        if (audioFinal.length > JUMLAH_SAMPEL) {
            audioFinal = audioFinal.subarray(0, JUMLAH_SAMPEL);
        } else if (audioFinal.length < JUMLAH_SAMPEL) {
            const padded = new Float32Array(JUMLAH_SAMPEL);
            padded.set(audioFinal);
            audioFinal = padded;
        }

        ubahStatusProses("Preprocessing & ekstraksi 50 fitur...");
        const audioBersih = preprocessingAudio(audioFinal);
        const fitur = ekstraksiSemuaFitur(audioBersih);

        ubahStatusProses("Menjalankan klasifikasi Random Forest...");
        const hasilPrediksi = await prediksiRandomForest(fitur);

        tampilkanHasilPrediksi(hasilPrediksi);
        ubahStatusMic("Selesai");
        ubahStatusProses(`Selesai menguji "${labelNama}".`);

    } catch (error) {
        console.error("Gagal memproses audio:", error);
        ubahStatusMic("Gagal");
        ubahStatusProses("Gagal memproses: " + error.message);
    }
}

async function tanganiUnggahFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
        ubahStatusProses(`Membaca file "${file.name}"...`);
        const arrayBuffer = await file.arrayBuffer();
        const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
        const tempCtx = new AudioCtxClass();
        const decoded = await tempCtx.decodeAudioData(arrayBuffer);
        await tempCtx.close();

        await prosesAudioBufferLangsung(decoded, file.name);
    } catch (err) {
        console.error("Gagal membaca file audio:", err);
        ubahStatusProses("Gagal membaca file: " + err.message);
    } finally {
        event.target.value = "";
    }
}

async function ujiSampelAudio(url, label) {
    try {
        ubahStatusProses(`Mengunduh sampel ${label}...`);
        const res = await fetch(url);
        if (!res.ok) throw new Error("Gagal mengunduh sampel: " + res.status);
        const arrayBuffer = await res.arrayBuffer();
        const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
        const tempCtx = new AudioCtxClass();
        const decoded = await tempCtx.decodeAudioData(arrayBuffer);
        await tempCtx.close();

        await prosesAudioBufferLangsung(decoded, label);
    } catch (err) {
        console.error("Gagal uji sampel:", err);
        ubahStatusProses("Gagal uji sampel: " + err.message);
    }
}

// ============================================================
// EVENT LISTENER
// ============================================================

startBtn.addEventListener("click", mulaiRekaman);

stopBtn.addEventListener("click", hentikanRekaman);

clearHistoryBtn.addEventListener(
    "click",
    bersihkanRiwayat
);

if (uploadBtn && audioFileInput) {
    uploadBtn.addEventListener("click", () => audioFileInput.click());
    audioFileInput.addEventListener("change", tanganiUnggahFile);
}

if (sampleBocorBtn) {
    sampleBocorBtn.addEventListener("click", () => {
        ujiSampelAudio("./samples/sampel_bocor.wav", "Sampel Kebocoran Gas");
    });
}

if (sampleNormalBtn) {
    sampleNormalBtn.addEventListener("click", () => {
        ujiSampelAudio("./samples/sampel_normal.wav", "Sampel Normal");
    });
}

// ============================================================
// INISIALISASI
// ============================================================

startBtn.disabled = true;
stopBtn.disabled = true;

inisialisasiAplikasi();
