
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
        const hasil = await konteks.muatKonfigurasiAudio();

        console.log("\n=== HASIL PEMUATAN KONFIGURASI ===");
        console.log(hasil);

        console.log("\n=== PEMERIKSAAN VARIABEL GLOBAL ===");

        const pemeriksaan = vm.runInContext(`
            ({
                parameterAudio: parameterAudio !== null,
                jumlahFitur: daftarFiturAudio.length,
                referensiEkstraksi: referensiEkstraksi !== null,
                referensiPreprocessing: referensiPreprocessing !== null
            })
        `, konteks);

        console.log(pemeriksaan);

        console.log("\n=== UKURAN MATRIKS ===");

        const ukuran = vm.runInContext(`
            ({
                melFilterbank: [
                    referensiEkstraksi.mel_filterbank.length,
                    referensiEkstraksi.mel_filterbank[0].length
                ],
                dct: [
                    referensiEkstraksi.dct_matrix.length,
                    referensiEkstraksi.dct_matrix[0].length
                ]
            })
        `, konteks);

        console.log("Mel Filterbank:", ukuran.melFilterbank);
        console.log("DCT:", ukuran.dct);

        console.log("\nSTATUS: PEMUATAN BERHASIL");

    } catch (error) {
        console.error("\nSTATUS: GAGAL");
        console.error(error.stack);
        process.exitCode = 1;
    }
})();
