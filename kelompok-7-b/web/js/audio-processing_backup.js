
// ============================================
// AUDIO PROCESSING
// Preprocessing dan ekstraksi 50 fitur audio
// ============================================

let parameterAudio = null;
let daftarFiturAudio = null;
let referensiEkstraksi = null;
let referensiPreprocessing = null;

// Memuat seluruh konfigurasi audio
async function muatKonfigurasiAudio() {
    const folderModel = "model/";

    const [
        metadataResponse,
        fiturResponse,
        ekstraksiResponse,
        preprocessingResponse
    ] = await Promise.all([
        fetch(folderModel + "metadata_model.json"),
        fetch(folderModel + "daftar_fitur.json"),
        fetch(folderModel + "referensi_ekstraksi_js.json"),
        fetch(folderModel + "referensi_preprocessing_js.json")
    ]);

    if (
        !metadataResponse.ok ||
        !fiturResponse.ok ||
        !ekstraksiResponse.ok ||
        !preprocessingResponse.ok
    ) {
        throw new Error("Gagal memuat konfigurasi audio.");
    }

    parameterAudio = await metadataResponse.json();
    daftarFiturAudio = await fiturResponse.json();
    referensiEkstraksi = await ekstraksiResponse.json();
    referensiPreprocessing = await preprocessingResponse.json();

    if (daftarFiturAudio.length !== 50) {
        throw new Error("Jumlah fitur tidak sesuai.");
    }

    if (
        referensiEkstraksi.feature_names.length !== 50 ||
        referensiEkstraksi.feature_names.some(
            (nama, indeks) => nama !== daftarFiturAudio[indeks]
        )
    ) {
        throw new Error("Urutan fitur referensi tidak sesuai.");
    }

    return {
        status: "berhasil",
        jumlahFitur: daftarFiturAudio.length,
        samplingRate: parameterAudio.sampling_rate
    };
}


// Menghitung koefisien dan kondisi awal filter SOS
function hitungZi(sos, x) {
    const [b0, b1, b2, a0, a1, a2] = sos;

    return {
        b0: b0 / a0,
        b1: b1 / a0,
        b2: b2 / a0,
        a1: a1 / a0,
        a2: a2 / a0,
        x1: x,
        x2: x,
        y1: 0,
        y2: 0
    };
}


// Filter Second-Order Sections (SOS)
function sosfilt(sos, audio) {
    let hasil = Float64Array.from(audio);

    for (const bagian of sos) {
        const [b0, b1, b2, a0, a1, a2] = bagian;

        let x1 = 0;
        let x2 = 0;
        let y1 = 0;
        let y2 = 0;

        for (let i = 0; i < hasil.length; i++) {
            const x = hasil[i];

            const y = (
                (b0 * x) +
                (b1 * x1) +
                (b2 * x2) -
                (a1 * y1) -
                (a2 * y2)
            ) / a0;

            hasil[i] = y;

            x2 = x1;
            x1 = x;
            y2 = y1;
            y1 = y;
        }
    }

    return hasil;
}


// Filtering maju dan mundur tanpa pergeseran fase
function sosfiltfilt(sos, audio) {
    const hasilMaju = sosfilt(sos, audio);

    const terbalik = Float64Array.from(hasilMaju).reverse();

    const hasilMundur = sosfilt(sos, terbalik);

    return Float64Array.from(hasilMundur).reverse();
}


// Preprocessing dan normalisasi audio
function preprocessingAudio(audio) {
    if (!referensiPreprocessing) {
        throw new Error("Konfigurasi preprocessing belum dimuat.");
    }

    const parameter = referensiPreprocessing;

    if (audio.length === 0) {
        throw new Error("Audio kosong.");
    }

    const hasilFilter = sosfiltfilt(
        parameter.sos_coefficients,
        audio
    );

    let maksimum = 0;

    for (const nilai of hasilFilter) {
        maksimum = Math.max(maksimum, Math.abs(nilai));
    }

    const hasil = new Float32Array(hasilFilter.length);

    for (let i = 0; i < hasil.length; i++) {
        hasil[i] = hasilFilter[i] / (maksimum + parameter.epsilon);
    }

    return hasil;
}


// Fast Fourier Transform (FFT)
function hitungFFT(input) {
    const N = input.length;

    if (N !== 512) {
        throw new Error("FFT harus menggunakan 512 titik.");
    }

    const real = Float64Array.from(input);
    const imag = new Float64Array(N);

    let j = 0;

    for (let i = 1; i < N; i++) {
        let bit = N >> 1;

        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }

        j ^= bit;

        if (i < j) {
            [real[i], real[j]] = [real[j], real[i]];
        }
    }

    for (let size = 2; size <= N; size *= 2) {
        const angle = -2 * Math.PI / size;
        const wr = Math.cos(angle);
        const wi = Math.sin(angle);

        for (let start = 0; start < N; start += size) {
            let ur = 1;
            let ui = 0;

            for (let k = 0; k < size / 2; k++) {
                const even = start + k;
                const odd = even + size / 2;

                const tr = ur * real[odd] - ui * imag[odd];
                const ti = ur * imag[odd] + ui * real[odd];

                real[odd] = real[even] - tr;
                imag[odd] = imag[even] - ti;

                real[even] += tr;
                imag[even] += ti;

                const nextUr = ur * wr - ui * wi;
                ui = ur * wi + ui * wr;
                ur = nextUr;
            }
        }
    }

    return { real, imag };
}


// Windowing Hann
function terapkanWindowHann(frame) {
    if (frame.length !== 512) {
        throw new Error("Frame harus berukuran 512 titik.");
    }

    const hasil = new Float64Array(512);

    for (let i = 0; i < 512; i++) {
        hasil[i] = frame[i] * windowHann[i];
    }

    return hasil;
}

const windowHann = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.168375916970614e-05, 0.0002467198171342, 0.0005550625190150482, 0.0009866357858642205, 0.001541333133436018, 0.002219017698460002, 0.0030195222724101467, 0.0039426493427611176, 0.0049881711417212315, 0.00615582970243117, 0.007445336922613066, 0.00885637463565564, 0.010388594689117125, 0.012041619030626338, 0.013815039801161721, 0.015708419435684462, 0.017721290771101017, 0.019853157161528523, 0.02210349260083494, 0.024471741852423234, 0.02695732058622735, 0.029559615522887273, 0.032277984585066366, 0.035111757055874326, 0.03806023374435663, 0.04112268715800943, 0.044298361682277465, 0.04758647376699032, 0.05098621211969223, 0.054496737905816106, 0.05811718495565327, 0.06184665997806821, 0.06568424278090446, 0.06962898649802823, 0.07367991782295391, 0.07783603724899246, 0.08209631931586497, 0.08645971286271914, 0.09092514128748835, 0.09549150281252633, 0.10015767075645471, 0.1049224938121548, 0.10978479633083521, 0.11474337861210543, 0.11979701719998453, 0.1249444651847702, 0.13018445251069527, 0.13551568628929433, 0.14093685111840565, 0.14644660940672627, 0.15204360170384285, 0.15772644703565564, 0.16349374324511345, 0.16934406733817414, 0.17527597583490823, 0.1812880051256553, 0.18737867183214757, 0.19354647317351187, 0.19978988733705805, 0.2061073738537635, 0.21249737397836072, 0.21895831107393465, 0.2254885910009342, 0.23208660251050178, 0.2387507176420256, 0.24547929212481456, 0.2522706657837964, 0.2591231629491425, 0.2660350928697134, 0.2730047501302266, 0.2800304150720424, 0.28711035421746367, 0.29424282069744573, 0.30142605468260975, 0.30865828381745514, 0.31593772365766115, 0.3232625781103715, 0.3306310398773544, 0.3380412909009254, 0.34549150281252633, 0.3529798373838481, 0.3605044469803854, 0.3680634750173137, 0.3756550564175727, 0.38327731807204746, 0.39092837930172886, 0.3986063523217438, 0.4063093427071378, 0.4140354498602953, 0.4217827674798847, 0.4295493840312088, 0.4373333832178479, 0.4451328444544775, 0.45294584334074295, 0.46077045213607765, 0.46860474023534343, 0.4764467746451787, 0.48429462046093597, 0.49214634134408974, 0.5000000000000001, 0.5078536586559105, 0.5157053795390641, 0.5235532253548214, 0.5313952597646567, 0.5392295478639226, 0.5470541566592573, 0.5548671555455227, 0.5626666167821522, 0.5704506159687914, 0.5782172325201156, 0.5859645501397048, 0.5936906572928624, 0.6013936476782563, 0.6090716206982714, 0.6167226819279529, 0.6243449435824275, 0.6319365249826866, 0.6394955530196147, 0.647020162616152, 0.6545084971874738, 0.6619587090990748, 0.6693689601226458, 0.6767374218896287, 0.6840622763423391, 0.691341716182545, 0.6985739453173904, 0.7057571793025545, 0.7128896457825364, 0.7199695849279576, 0.7269952498697734, 0.7339649071302868, 0.7408768370508578, 0.7477293342162039, 0.7545207078751857, 0.7612492823579744, 0.7679133974894985, 0.774511408999066, 0.7810416889260654, 0.7875026260216393, 0.7938926261462366, 0.8002101126629422, 0.8064535268264883, 0.8126213281678527, 0.8187119948743449, 0.8247240241650918, 0.8306559326618261, 0.8365062567548868, 0.8422735529643444, 0.8479563982961572, 0.8535533905932737, 0.8590631488815945, 0.8644843137107059, 0.8698155474893049, 0.8750555348152298, 0.8802029828000155, 0.8852566213878947, 0.890215203669165, 0.8950775061878453, 0.8998423292435453, 0.9045084971874737, 0.9090748587125118, 0.913540287137281, 0.9179036806841352, 0.9221639627510076, 0.9263200821770463, 0.9303710135019719, 0.9343157572190957, 0.9381533400219318, 0.9418828150443468, 0.9455032620941841, 0.9490137878803079, 0.9524135262330098, 0.9557016383177226, 0.9588773128419905, 0.9619397662556435, 0.9648882429441258, 0.9677220154149337, 0.9704403844771128, 0.9730426794137726, 0.9755282581475768, 0.9778965073991651, 0.9801468428384716, 0.9822787092288991, 0.9842915805643155, 0.9861849601988384, 0.9879583809693737, 0.9896114053108829, 0.9911436253643444, 0.9925546630773869, 0.9938441702975689, 0.9950118288582788, 0.996057350657239, 0.9969804777275899, 0.99778098230154, 0.998458666866564, 0.9990133642141358, 0.9994449374809851, 0.9997532801828658, 0.9999383162408303, 1.0, 0.9999383162408303, 0.9997532801828658, 0.9994449374809851, 0.9990133642141358, 0.998458666866564, 0.99778098230154, 0.9969804777275899, 0.9960573506572389, 0.9950118288582788, 0.9938441702975689, 0.9925546630773869, 0.9911436253643443, 0.9896114053108829, 0.9879583809693737, 0.9861849601988382, 0.9842915805643155, 0.982278709228899, 0.9801468428384715, 0.977896507399165, 0.9755282581475767, 0.9730426794137725, 0.9704403844771127, 0.9677220154149336, 0.9648882429441257, 0.9619397662556433, 0.9588773128419905, 0.9557016383177226, 0.9524135262330098, 0.9490137878803078, 0.9455032620941839, 0.9418828150443466, 0.9381533400219317, 0.9343157572190957, 0.9303710135019717, 0.926320082177046, 0.9221639627510074, 0.9179036806841351, 0.9135402871372809, 0.9090748587125116, 0.9045084971874736, 0.8998423292435451, 0.8950775061878451, 0.8902152036691648, 0.8852566213878945, 0.8802029828000153, 0.8750555348152297, 0.8698155474893048, 0.8644843137107057, 0.8590631488815943, 0.8535533905932736, 0.8479563982961571, 0.8422735529643443, 0.8365062567548867, 0.8306559326618257, 0.8247240241650917, 0.8187119948743447, 0.8126213281678525, 0.8064535268264882, 0.800210112662942, 0.7938926261462366, 0.7875026260216389, 0.781041688926065, 0.7745114089990657, 0.767913397489498, 0.7612492823579742, 0.7545207078751854, 0.7477293342162037, 0.7408768370508576, 0.7339649071302867, 0.7269952498697734, 0.7199695849279573, 0.712889645782536, 0.7057571793025541, 0.69857394531739, 0.6913417161825447, 0.6840622763423388, 0.6767374218896285, 0.6693689601226456, 0.6619587090990747, 0.6545084971874737, 0.6470201626161516, 0.6394955530196142, 0.6319365249826862, 0.6243449435824272, 0.6167226819279525, 0.6090716206982711, 0.6013936476782561, 0.5936906572928623, 0.5859645501397047, 0.5782172325201155, 0.570450615968791, 0.5626666167821518, 0.5548671555455224, 0.5470541566592569, 0.5392295478639223, 0.5313952597646565, 0.5235532253548212, 0.515705379539064, 0.5078536586559104, 0.5, 0.4921463413440893, 0.4842946204609355, 0.4764467746451784, 0.4686047402353431, 0.46077045213607737, 0.4529458433407427, 0.44513284445447726, 0.4373333832178478, 0.4295493840312087, 0.4217827674798846, 0.41403544986029484, 0.40630934270713737, 0.39860635232174346, 0.3909283793017285, 0.3832773180720471, 0.37565505641757246, 0.36806347501731346, 0.3605044469803853, 0.35297983738384797, 0.34549150281252633, 0.33804129090092494, 0.330631039877354, 0.3232625781103712, 0.3159377236576608, 0.30865828381745497, 0.3014260546826096, 0.29424282069744556, 0.28711035421746367, 0.2800304150720424, 0.27300475013022624, 0.266035092869713, 0.25912316294914206, 0.252270665783796, 0.24547929212481417, 0.23875071764202543, 0.23208660251050156, 0.22548859100093405, 0.21895831107393465, 0.21249737397836072, 0.2061073738537631, 0.19978988733705771, 0.19354647317351148, 0.18737867183214718, 0.18128800512565496, 0.17527597583490806, 0.16934406733817398, 0.1634937432451133, 0.15772644703565564, 0.15204360170384285, 0.14644660940672594, 0.14093685111840532, 0.135515686289294, 0.130184452510695, 0.12494446518477009, 0.11979701719998442, 0.11474337861210532, 0.1097847963308351, 0.1049224938121548, 0.10015767075645471, 0.09549150281252605, 0.09092514128748808, 0.08645971286271892, 0.08209631931586475, 0.07783603724899235, 0.0736799178229538, 0.06962898649802812, 0.06568424278090434, 0.06184665997806821, 0.05811718495565327, 0.054496737905815884, 0.05098621211969201, 0.0475864737669901, 0.04429836168227724, 0.04112268715800932, 0.038060233744356575, 0.03511175705587427, 0.03227798458506631, 0.029559615522887273, 0.02695732058622735, 0.024471741852423068, 0.02210349260083483, 0.019853157161528356, 0.017721290771100906, 0.015708419435684406, 0.013815039801161666, 0.012041619030626283, 0.01038859468911707, 0.00885637463565564, 0.007445336922612955, 0.006155829702431059, 0.004988171141721176, 0.003942649342761062, 0.0030195222724101467, 0.0022190176984599463, 0.001541333133436018, 0.0009866357858642205, 0.0005550625190149927, 0.0002467198171342, 6.168375916970614e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];


// ============================================
// SHORT-TIME FOURIER TRANSFORM (STFT)
// ============================================

function hitungSTFT(audio) {
    const frameLength = 400;
    const hopLength = 160;
    const nFFT = 512;
    const paddingAwal = Math.floor((nFFT - frameLength) / 2);

    if (!(audio instanceof Float32Array ||
          audio instanceof Float64Array)) {
        throw new Error("Input audio harus Float32Array atau Float64Array.");
    }

    if (audio.length < frameLength) {
        throw new Error("Audio terlalu pendek untuk STFT.");
    }

    const jumlahFrame =
        Math.floor((audio.length - frameLength) / hopLength) + 1;

    const hasil = [];

    for (let frameIndex = 0; frameIndex < jumlahFrame; frameIndex++) {
        const awal = frameIndex * hopLength;
        const frame = new Float64Array(nFFT);

        for (let i = 0; i < frameLength; i++) {
            frame[i + paddingAwal] = audio[awal + i];
        }

        const frameWindow = terapkanWindowHann(frame);
        const spektrum = hitungFFT(frameWindow);

        hasil.push({
            real: spektrum.real,
            imag: spektrum.imag
        });
    }

    return hasil;
}


// ============================================
// EKSTRAKSI MEL SPECTROGRAM DAN MFCC
// ============================================

// Menghitung power spectrum dari STFT
function hitungSpektrumDaya(stft) {
    return stft.map(frame => {
        const hasil = new Float64Array(257);

        for (let i = 0; i < 257; i++) {
            hasil[i] =
                frame.real[i] * frame.real[i] +
                frame.imag[i] * frame.imag[i];
        }

        return hasil;
    });
}

// Menghitung Mel Spectrogram
function hitungMelSpektrogram(spektrumDaya) {
    if (!referensiEkstraksi) {
        throw new Error("Referensi ekstraksi belum dimuat.");
    }

    const filterbank = referensiEkstraksi.mel_filterbank;
    const jumlahMel = 40;

    return spektrumDaya.map(frame => {
        const hasil = new Float64Array(jumlahMel);

        for (let m = 0; m < jumlahMel; m++) {
            let total = 0;

            for (let k = 0; k < 257; k++) {
                total += filterbank[m][k] * frame[k];
            }

            hasil[m] = total;
        }

        return hasil;
    });
}

// Konversi power spectrogram ke decibel
function konversiPowerKeDb(melSpektrogram) {
    const amin = 1e-10;
    const topDb = 80;

    let nilaiMaksimum = -Infinity;

    const hasil = melSpektrogram.map(frame =>
        Float64Array.from(frame, nilai => {
            const db = 10 * Math.log10(Math.max(amin, nilai));
            nilaiMaksimum = Math.max(nilaiMaksimum, db);
            return db;
        })
    );

    const batasBawah = nilaiMaksimum - topDb;

    for (const frame of hasil) {
        for (let i = 0; i < frame.length; i++) {
            frame[i] = Math.max(frame[i], batasBawah);
        }
    }

    return hasil;
}

// Menghitung MFCC menggunakan DCT-II
function hitungMFCC(melDb) {
    if (!referensiEkstraksi) {
        throw new Error("Referensi ekstraksi belum dimuat.");
    }

    const matriksDCT = referensiEkstraksi.dct_matrix;
    const jumlahMFCC = 20;

    return melDb.map(frame => {
        const hasil = new Float64Array(jumlahMFCC);

        for (let k = 0; k < jumlahMFCC; k++) {
            let total = 0;

            for (let m = 0; m < 40; m++) {
                total += matriksDCT[k][m] * frame[m];
            }

            hasil[k] = total;
        }

        return hasil;
    });
}

// Menghitung mean dan standar deviasi MFCC
function ekstraksiStatistikMFCC(mfcc) {
    if (mfcc.length === 0) {
        throw new Error("Data MFCC kosong.");
    }

    const hasil = {};

    for (let k = 0; k < 20; k++) {
        let total = 0;

        for (const frame of mfcc) {
            total += frame[k];
        }

        const mean = total / mfcc.length;

        let jumlahKuadrat = 0;

        for (const frame of mfcc) {
            jumlahKuadrat += Math.pow(frame[k] - mean, 2);
        }

        const std = Math.sqrt(jumlahKuadrat / mfcc.length);

        hasil[`mfcc_${k + 1}_mean`] = mean;
        hasil[`mfcc_${k + 1}_std`] = std;
    }

    return hasil;
}

// Fungsi utama ekstraksi 40 fitur MFCC
function ekstraksiFiturMFCC(audio) {
    const stft = hitungSTFT(audio);
    const power = hitungSpektrumDaya(stft);
    const mel = hitungMelSpektrogram(power);
    const melDb = konversiPowerKeDb(mel);
    const mfcc = hitungMFCC(melDb);

    return {
        jumlahFrame: mfcc.length,
        jumlahMFCC: 20,
        fitur: ekstraksiStatistikMFCC(mfcc)
    };
}
