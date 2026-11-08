
// ============================================
// RANDOM FOREST PREDICTOR
// Versi Browser - JavaScript
// ============================================

let modelRandomForest = null;

// ============================================
// MEMUAT MODEL RANDOM FOREST
// ============================================

async function muatModelRandomForest() {

    try {

        const response = await fetch(
            "./model/random_forest_model.json"
        );

        if (!response.ok) {
            throw new Error(
                "Gagal memuat model Random Forest: " +
                response.status
            );
        }

        const model = await response.json();

        // Validasi struktur dasar model
        if (model.model_type !== "RandomForestClassifier") {
            throw new Error("Jenis model tidak sesuai.");
        }

        if (model.n_estimators !== 200) {
            throw new Error("Jumlah pohon tidak sesuai.");
        }

        if (model.n_features !== 50) {
            throw new Error("Jumlah fitur model tidak sesuai.");
        }

        if (
            !Array.isArray(model.classes) ||
            model.classes.length !== 2
        ) {
            throw new Error("Struktur kelas tidak valid.");
        }

        if (
            !Array.isArray(model.feature_names) ||
            model.feature_names.length !== 50
        ) {
            throw new Error("Daftar fitur model tidak valid.");
        }

        if (
            !Array.isArray(model.trees) ||
            model.trees.length !== model.n_estimators
        ) {
            throw new Error("Struktur pohon tidak valid.");
        }

        modelRandomForest = model;

        console.log("Model Random Forest berhasil dimuat.");
        console.log("Jumlah pohon:", model.trees.length);
        console.log("Jumlah fitur:", model.n_features);
        console.log("Kelas:", model.classes);

        return true;

    } catch (error) {

        modelRandomForest = null;

        console.error(
            "Kesalahan pemuatan model:",
            error
        );

        throw error;
    }
}


// ============================================
// PREDIKSI SATU POHON
// ============================================

function prediksiPohon(pohon, fitur) {

    const nodes = pohon.nodes;
    let indeks = 0;

    while (true) {

        const node = nodes[indeks];

        if (!node) {
            throw new Error(
                "Node tidak ditemukan pada pohon " +
                pohon.tree_id
            );
        }

        if (node.is_leaf) {

            const nilai = node.value;

            const total = nilai.reduce(
                (a, b) => a + b,
                0
            );

            if (total <= 0) {
                throw new Error(
                    "Nilai probabilitas leaf tidak valid."
                );
            }

            return nilai.map(
                v => v / total
            );
        }

        const nilaiFitur = fitur[node.feature];

        if (nilaiFitur <= node.threshold) {
            indeks = node.left;
        } else {
            indeks = node.right;
        }
    }
}


// ============================================
// PREDIKSI RANDOM FOREST
// ============================================

function prediksiRandomForest(fitur) {

    if (!modelRandomForest) {
        throw new Error(
            "Model belum dimuat. Jalankan " +
            "muatModelRandomForest() terlebih dahulu."
        );
    }

    if (
        !Array.isArray(fitur) &&
        !ArrayBuffer.isView(fitur)
    ) {
        throw new Error(
            "Fitur harus berupa array numerik."
        );
    }

    if (fitur.length !== modelRandomForest.n_features) {
        throw new Error(
            "Jumlah fitur tidak sesuai dengan model."
        );
    }

    for (let i = 0; i < fitur.length; i++) {

        if (
            typeof fitur[i] !== "number" ||
            !Number.isFinite(fitur[i])
        ) {
            throw new Error(
                "Nilai fitur ke-" + i + " tidak valid."
            );
        }
    }

    const totalProbabilitas = new Array(
        modelRandomForest.classes.length
    ).fill(0);

    for (const pohon of modelRandomForest.trees) {

        const probabilitas = prediksiPohon(
            pohon,
            fitur
        );

        for (
            let i = 0;
            i < probabilitas.length;
            i++
        ) {
            totalProbabilitas[i] += probabilitas[i];
        }
    }

    const rataRata = totalProbabilitas.map(
        nilai => nilai / modelRandomForest.trees.length
    );

    let indeksTerpilih = 0;

    for (let i = 1; i < rataRata.length; i++) {

        if (
            rataRata[i] >
            rataRata[indeksTerpilih]
        ) {
            indeksTerpilih = i;
        }
    }

    return {

        prediksi:
            modelRandomForest.classes[indeksTerpilih],

        probabilitas: rataRata,

        probabilitasPerKelas: Object.fromEntries(
            modelRandomForest.classes.map(
                (kelas, i) => [kelas, rataRata[i]]
            )
        ),

        jumlahPohon:
            modelRandomForest.trees.length
    };
}


// ============================================
// INFORMASI MODEL
// ============================================

function getInformasiModel() {

    if (!modelRandomForest) {
        throw new Error("Model belum dimuat.");
    }

    return {
        algoritma: modelRandomForest.model_type,
        jumlahPohon: modelRandomForest.n_estimators,
        jumlahFitur: modelRandomForest.n_features,
        kelas: [...modelRandomForest.classes],
        featureNames: [...modelRandomForest.feature_names]
    };
}


const fs = require("fs");
const path = require("path");

async function jalankanPengujian() {

    const model = JSON.parse(
        fs.readFileSync(path.join(__dirname, "web", "model", "random_forest_model.json"), "utf8")
    );

    const testing = JSON.parse(
        fs.readFileSync(path.join(__dirname, "model", "data_testing.json"), "utf8")
    );

    const expected = JSON.parse(
        fs.readFileSync(path.join(__dirname, "model", "hasil_testing_javascript.json"), "utf8")
    );

    // Simulasi fetch untuk memuat model JSON
    global.fetch = async function() {
        return {
            ok: true,
            json: async () => model
        };
    };

    await muatModelRandomForest();

    let jumlahPrediksiSama = 0;
    let selisihMaksimum = 0;

    for (let i = 0; i < testing.X_test.length; i++) {

        const hasil = prediksiRandomForest(
            testing.X_test[i]
        );

        // Bandingkan kelas prediksi
        if (hasil.prediksi === expected.prediksi[i]) {
            jumlahPrediksiSama++;
        }

        // Bandingkan probabilitas
        for (let j = 0; j < hasil.probabilitas.length; j++) {

            const selisih = Math.abs(
                hasil.probabilitas[j] -
                expected.probabilitas[i][j]
            );

            selisihMaksimum = Math.max(
                selisihMaksimum,
                selisih
            );
        }
    }

    console.log("\n======================================");
    console.log("HASIL PENGUJIAN PREDICTOR BROWSER");
    console.log("======================================");

    console.log("Jumlah data:", testing.X_test.length);

    console.log(
        "Prediksi sama:",
        jumlahPrediksiSama,
        "/",
        testing.X_test.length
    );

    console.log(
        "Selisih probabilitas maksimum:",
        selisihMaksimum
    );

    console.log(
        "Kelas model:",
        model.classes
    );

    console.log(
        "Jumlah pohon:",
        model.trees.length
    );

    console.log(
        "Jumlah fitur:",
        model.n_features
    );

    console.log(
        "STATUS:",
        jumlahPrediksiSama === testing.X_test.length
            ? "BERHASIL"
            : "PERLU PEMERIKSAAN"
    );
}

jalankanPengujian().catch(error => {
    console.error("PENGUJIAN GAGAL:", error);
    process.exit(1);
});
