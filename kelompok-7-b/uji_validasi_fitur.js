
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
        await konteks.muatKonfigurasiAudio();

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

                const fitur = ekstraksiSemuaFitur(audio);

                return {
                    jumlahFitur: fitur.length,

                    jumlahNumerik: fitur.filter(
                        nilai => typeof nilai === "number"
                    ).length,

                    jumlahTidakValid: fitur.filter(
                        nilai => !Number.isFinite(nilai)
                    ).length,

                    sesuaiJumlah:
                        fitur.length === daftarFiturAudio.length,

                    pasangan: daftarFiturAudio.map(
                        (nama, index) => ({
                            nama: nama,
                            nilai: fitur[index]
                        })
                    )
                };
            })()
        `, konteks);

        console.log("\\n=== VALIDASI JUMLAH FITUR ===");
        console.log("Jumlah fitur:", hasil.jumlahFitur);
        console.log("Jumlah fitur numerik:", hasil.jumlahNumerik);
        console.log("Jumlah nilai tidak valid:", hasil.jumlahTidakValid);
        console.log("Sesuai jumlah fitur model:", hasil.sesuaiJumlah);

        console.log("\\n=== 5 FITUR PERTAMA ===");
        console.log(hasil.pasangan.slice(0, 5));

        console.log("\\n=== 10 FITUR TERAKHIR ===");
        console.log(hasil.pasangan.slice(-10));

        console.log("\\n=== STATUS VALIDASI ===");

        if (
            hasil.jumlahFitur === 50 &&
            hasil.jumlahNumerik === 50 &&
            hasil.jumlahTidakValid === 0 &&
            hasil.sesuaiJumlah
        ) {
            console.log("BERHASIL: 50 fitur valid.");
        } else {
            throw new Error("Validasi 50 fitur belum terpenuhi.");
        }

    } catch (error) {
        console.error("GAGAL:", error.stack);
        process.exitCode = 1;
    }
})();
