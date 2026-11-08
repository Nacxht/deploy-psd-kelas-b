
const fs = require("fs");
const path = require("path");

const lokasiModel = path.join(__dirname, "random_forest_model.json");
const model = JSON.parse(fs.readFileSync(lokasiModel, "utf8"));

function prediksiPohon(pohon, fitur) {
    const nodes = pohon.nodes;
    let indeks = 0;

    while (true) {
        const node = nodes[indeks];

        if (node.is_leaf) {
            const nilai = node.value;
            const total = nilai.reduce((a, b) => a + b, 0);

            return nilai.map(v => v / total);
        }

        const nilaiFitur = fitur[node.feature];

        if (nilaiFitur <= node.threshold) {
            indeks = node.left;
        } else {
            indeks = node.right;
        }
    }
}

function prediksiRandomForest(fitur) {
    if (fitur.length !== model.n_features) {
        throw new Error("Jumlah fitur tidak sesuai dengan model.");
    }

    const totalProbabilitas = new Array(model.classes.length).fill(0);

    for (const pohon of model.trees) {
        const probabilitas = prediksiPohon(pohon, fitur);

        for (let i = 0; i < probabilitas.length; i++) {
            totalProbabilitas[i] += probabilitas[i];
        }
    }

    const rataRata = totalProbabilitas.map(
        nilai => nilai / model.trees.length
    );

    let indeksTerpilih = 0;

    for (let i = 1; i < rataRata.length; i++) {
        if (rataRata[i] > rataRata[indeksTerpilih]) {
            indeksTerpilih = i;
        }
    }

    return {
        prediksi: model.classes[indeksTerpilih],
        probabilitas: rataRata
    };
}

module.exports = {
    model,
    prediksiRandomForest
};
