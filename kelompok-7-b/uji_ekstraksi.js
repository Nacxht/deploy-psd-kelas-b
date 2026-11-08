
const fs = require("fs");
const vm = require("vm");
const path = require("path");

const folderModel = path.join(__dirname, "web", "model");

const konteks = {
    console: console,

    fetch: async function(url) {
        const namaFile = url.startsWith("model/")
            ? url.substring("model/".length)
            : url;

        const lokasi = path.join(folderModel, namaFile);

        return {
            ok: fs.existsSync(lokasi),
            status: fs.existsSync(lokasi) ? 200 : 404,

            json: async function() {
                return JSON.parse(
                    fs.readFileSync(lokasi, "utf8")
                );
            }
        };
    }
};

vm.createContext(konteks);

const kode = fs.readFileSync(
    path.join(__dirname, "web", "js", "audio-processing.js"),
    "utf8"
);

vm.runInContext(kode, konteks);

(async () => {
    try {
        // Muat konfigurasi terlebih dahulu
        await konteks.muatKonfigurasiAudio();

        // Buat audio sintetis dan jalankan ekstraksi
        const hasil = vm.runInContext(`
            (() => {
                const samplingRate = 16000;
                const audio = new Float64Array(samplingRate);

                for (let i = 0; i < audio.length; i++) {
                    const waktu = i / samplingRate;

                    audio[i] =
                        0.5 * Math.sin(2 * Math.PI * 440 * waktu) +
                        0.2 * Math.sin(2 * Math.PI * 1000 * waktu);
                }

                const hasilEkstraksi = ekstraksiSemuaFitur(audio);

                return {
                    jumlahSampel: audio.length,
                    tipeHasil: typeof hasilEkstraksi,
                    kunciHasil: Object.keys(hasilEkstraksi || {}),
                    hasil: hasilEkstraksi
                };
            })()
        `, konteks);

        console.log("\\n=== HASIL EKSTRAKSI AUDIO ===");
        console.log("Jumlah sampel:", hasil.jumlahSampel);
        console.log("Tipe hasil:", hasil.tipeHasil);
        console.log("Kunci hasil:", hasil.kunciHasil);

        console.log("\\n=== DATA HASIL EKSTRAKSI ===");
        console.log(hasil.hasil);

        console.log("\\nSTATUS: EKSTRAKSI BERHASIL");

    } catch (error) {
        console.error("\\nSTATUS: EKSTRAKSI GAGAL");
        console.error(error.stack);
        process.exitCode = 1;
    }
})();
