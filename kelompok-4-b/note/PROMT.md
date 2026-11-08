# Catatan Evaluasi Proyek Sains Data - Pertemuan ke-10

## 1. Ekstraksi dan Seleksi Fitur

Pada proyek klasifikasi audio untuk deteksi suara manusia asli dan suara hasil manipulasi AI, terdapat beberapa hal yang perlu diperhatikan dalam proses ekstraksi dan seleksi fitur.

### Ketentuan:

1. Ekstraksi seluruh fitur  
   Lakukan ekstraksi terhadap seluruh fitur audio yang telah ditentukan. Misalnya, pada tahap awal terdapat 48 fitur yang berhasil diekstraksi.

2. Seleksi fitur (Feature Selection)  
   Setelah seluruh fitur berhasil diekstraksi, lakukan proses seleksi fitur untuk menentukan fitur-fitur terbaik yang akan digunakan dalam proses klasifikasi.

3. Pembuatan dataset hasil seleksi  
   Setelah seleksi fitur, jumlah kolom fitur pada dataset akan berkurang sesuai dengan fitur yang terpilih. Contohnya, dari 48 fitur awal menjadi 20, 15, atau sejumlah fitur terbaik lainnya.

4. Dataset yang digunakan untuk pemodelan  
   Dataset yang digunakan pada tahap pemodelan dan klasifikasi adalah file CSV terbaru yang telah melalui proses seleksi fitur, bukan dataset hasil ekstraksi awal.

5. Output hasil ekstraksi dan seleksi fitur  
   Hasil proses ini terdiri dari dua output:
   - File CSV hasil ekstraksi seluruh fitur.
   - File CSV hasil seleksi fitur yang akan digunakan untuk pemodelan.

6. Pengelolaan folder output  
   File CSV yang sebelumnya masih berada di luar folder hasil perlu dirapikan ke dalam folder khusus agar struktur proyek lebih terorganisasi.

   Contoh struktur folder:

   ```text
   hasil_ekstraksi/
   ├── file_ekstraksi.csv
   └── file_seleksi_fitur.csv
   ```

## 2. Pembuatan Wireframe Proyek

Buatlah wireframe yang menggambarkan bagaimana sistem klasifikasi audio bekerja, mulai dari pengguna memberikan input audio hingga sistem menghasilkan hasil klasifikasi.

Catatan penting:

- Wireframe harus sesuai dengan konsep proyek yang dikembangkan.
- Proyek ini bukan sistem streaming atau klasifikasi audio secara langsung (real-time).
- Oleh karena itu, wireframe tidak perlu menggambarkan proses seseorang sedang menelepon dan suaranya langsung diklasifikasikan oleh sistem.
- Gambarkan alur penggunaan sistem yang sebenarnya, yaitu pengguna mengunggah atau memasukkan file audio, kemudian sistem memproses audio tersebut dan menampilkan hasil klasifikasi.

Contoh alur wireframe:

1. Pengguna membuka aplikasi.
2. Pengguna mengunggah file audio.
3. Sistem melakukan preprocessing dan ekstraksi fitur audio.
4. Sistem melakukan klasifikasi menggunakan model yang telah dilatih.
5. Sistem menampilkan hasil klasifikasi, misalnya:
   - Suara Manusia Asli.
   - Suara Manipulasi AI.

## 3. Pendekatan CRISP-DM

Dalam pengembangan proyek, gunakan pendekatan CRISP-DM (Cross-Industry Standard Process for Data Mining) sebagai kerangka kerja penelitian dan pengembangan sistem.

Tahapan CRISP-DM yang perlu diperhatikan ini di taroh pada voices.ipynb:

1. Business Understanding – Memahami tujuan dan permasalahan klasifikasi audio.
2. Data Understanding – Memahami dataset audio yang digunakan.
3. Data Preparation – Melakukan preprocessing, ekstraksi fitur, dan seleksi fitur.
4. Modeling – Melakukan proses pelatihan model klasifikasi.
5. Evaluation – Mengevaluasi performa model menggunakan metrik evaluasi yang sesuai.
6. Deployment – Mengimplementasikan model ke dalam aplikasi klasifikasi audio.

## 4. Website (Frontend dan Integrasi Model)

Pada pengembangan website, tim menggunakan Flask (Python) sebagai framework untuk membangun frontend dan mengintegrasikan model klasifikasi audio ke dalam aplikasi.

### Ketentuan dan Harapan:

1. Pengujian audio secara langsung melalui website  
   Pengguna dapat mengunggah file audio melalui website untuk melakukan pengujian klasifikasi.

2. Integrasi model klasifikasi  
   File audio yang diunggah harus langsung diproses oleh sistem menggunakan model klasifikasi yang telah dilatih sebelumnya. Pengguna tidak perlu melakukan preprocessing, ekstraksi fitur, maupun seleksi fitur secara manual. Jadi pas upload langsung keluar hasilnay gitu loh.

3. Hasil klasifikasi secara otomatis  
   Setelah proses pengujian selesai, website langsung menampilkan hasil klasifikasi audio, misalnya:
   - AI Generated Voice: Audio terindikasi sebagai suara hasil manipulasi AI.
   - Human Voice: Audio terindikasi sebagai suara manusia asli.

4. Tidak perlu menampilkan akurasi model pada halaman pengujian  
   Website tidak perlu menampilkan nilai akurasi, precision, recall, F1-score, maupun metrik evaluasi model lainnya pada halaman pengujian pengguna. Hal ini karena performa model telah dievaluasi pada tahap penelitian sebelumnya dan hasil evaluasi tersebut menjadi dasar kepercayaan terhadap model yang digunakan.

5. Desain antarmuka yang profesional  
   Hindari penggunaan emoji atau ikon dekoratif yang tidak relevan, seperti emoji hati, pesawat, dan sejenisnya. Gunakan ikon yang profesional, konsisten, serta sesuai dengan konteks aplikasi.

6. Penggunaan ikon open-source  
   Apabila website membutuhkan ikon, gunakan library ikon open-source yang tersedia, seperti:
   - [Lucide](https://lucide.dev/)
   - [Font Awesome Free](https://fontawesome.com/)
   - [Bootstrap Icons](https://icons.getbootstrap.com/)

   Ikon harus digunakan secara konsisten dan memiliki fungsi yang jelas dalam antarmuka aplikasi.

### Gambaran alur penggunaan website:

1. Pengguna membuka website klasifikasi audio.
2. Pengguna mengunggah file audio.
3. Pengguna menekan tombol untuk melakukan klasifikasi.
4. Sistem memproses audio secara otomatis menggunakan pipeline preprocessing dan ekstraksi fitur yang sesuai dengan model.
5. Sistem menjalankan prediksi menggunakan model klasifikasi yang telah dilatih.
6. Website menampilkan hasil klasifikasi tanpa perlu menampilkan metrik evaluasi model.

Catatan penting: Proses preprocessing dan ekstraksi fitur tetap dilakukan oleh sistem di belakang layar. Pengguna hanya perlu mengunggah audio dan menerima hasil klasifikasi.
