# Brief Rekaman Loom — Freight Rate ML Assessment

Dokumen kerja untuk 1 kali rekaman. Total durasi target **2 menit 30 detik**. Kamera wajib menyala (wajah terlihat, pojok kanan bawah). Layar utama: slide ringkas atau langsung kode — jangan dua-duanya.

## 1. Setup sebelum menekan Record

- Tutup tab Chrome yang tidak perlu. Sisakan tiga tab: `github.com/fajarsetiawan17/freight-rate-ml`, file `report/Freight_Rate_ML_Report.pdf` halaman 2, dan editor `pipeline.py`.
- Editor: font 16 pt, zoom 150 persen, panel terminal di bawah. Kode harus terbaca di video 720p.
- Terminal siap di folder repo. Jalankan sekali sebagai percobaan supaya output sudah hangat dan tidak menunggu build.
- Mikrofon: uji 5 detik. Level bicara normal, jangan mendesis. Ruangan tanpa kipas.
- Pencahayaan di depan wajah, bukan di belakang. Jendela jangan di belakang kepala.
- Loom: set resolusi 1080p, kamera on, mikrofon on, **jangan** rekam tab saja.
- Notifikasi Windows: aktifkan Focus Assist selama rekaman.

## 2. Peta waktu

| Segmen | Durasi | Yang tampil |
|---|---|---|
| Pembuka | 0:00–0:12 | Kamera + judul repo |
| Temuan eksplorasi data | 0:12–0:45 | Halaman 2 report (scatter + histogram) |
| Masalah kualitas data | 0:45–1:15 | Tabel data-quality di README |
| Alasan model | 1:15–1:45 | Halaman 4 report (ablation) |
| Split dan validasi | 1:45–2:10 | Terminal: `python pipeline.py --cv` |
| Walkthrough kode | 2:10–2:30 | `pipeline.py` + `score.py` hijau |

Kalau kelebihan waktu, potong di segmen walkthrough kode — bagian yang paling mudah diringkas. Jangan pernah memotong segmen kualitas data atau split.

## 3. Naskah per segmen

### 0:00–0:12 — Pembuka
Tampil: kamera, lalu share screen ke repo.

> "Halo, saya Fajar Setiawan. Ini submission assessment Machine Learning Engineer untuk prediksi freight rate. Saya akan lewatkan empat hal: temuan data, masalah kualitas data, alasan pemilihan model, dan cara saya split serta memvalidasi."

Jangan baca CV. Jangan minta maaf. Dua belas detik saja.

### 0:12–0:45 — Temuan eksplorasi data
Tampil: `Freight_Rate_ML_Report.pdf` halaman 2.

> "Temuan terbesar dari eksplorasi: rate hampir merupakan fungsi murni jarak. Korelasi log rate terhadap log distance 0,99. Artinya model tidak perlu menemukan ulang hubungan jarak–harga, cukup belajar koreksi level di atasnya. Yang tersisa hanyalah efek peralatan, lane, berat, dan hari pengiriman.
>
> Sebaran rate sangat right-skewed, jadi saya memodelkan log dari rate, bukan rate mentah."

Sorot grafik scatter kiri dengan kursor saat menyebut 0,99.

### 0:45–1:15 — Masalah kualitas data
Tampil: tabel "Data-quality fixes" di README repo.

> "Ada enam masalah yang saya temukan dan tangani.
>
> Pertama, 599 baris punya rate mustahil — di bawah satu dolar atau di atas enam dolar per mil. Itu label error, bukan harga. Saya buang dari training, tapi load-nya tetap saya beri prediksi.
>
> Kedua, 292 baris punya berat negatif, minimum minus 47.500 pon. Ini tanda yang terbalik, jadi saya ambil nilai absolutnya.
>
> Ketiga, berat hilang di 300 baris training dan 165 baris validasi. Saya isi dengan median training, dan nilai median yang sama dipakai ulang ke validasi dan Desember supaya tidak ada leakage.
>
> Keempat, `market_index` hilang di 374 baris dan saya perlakukan sama.
>
> Kelima, 495 baris punya rute, tanggal, dan peralatan identik tapi rate berbeda. Itu bukan duplikat — itu load terpisah. Saya biarkan.
>
> Keenam, file Desember tidak membawa koordinat kota sama sekali. Saya bangun lookup 64 kota dari data training."

Kalau waktu mepet, lewati poin kelima. Poin pertama dan kedua wajib.

### 1:15–1:45 — Alasan model
Tampil: tabel ablation halaman 4.

> "Saya pakai LightGBM dengan objektif L1 pada log rate. Alasannya tiga: L1 tahan outlier dan mendekati optimasi median, yang sesuai karena rate punya ekor panjang; gradient boosting menangani interaksi jarak–peralatan–lane tanpa saya tulis manual; dan datasetnya tabular, jadi tidak ada keuntungan dari model yang lebih berat.
>
> Sepuluh fitur: log jarak, log berat, peralatan, asal, tujuan, titik tengah lat, titik tengah lon, delta utara-selatan, delta timur-barat, dan hari dalam minggu.
>
> Dua kolom sengaja saya buang: `market_index` dan `quote_signal`. Keduanya bergeser level antara jendela pengembangan dan jendela prediksi. Kalau dimasukkan, model mengekstrapolasi level yang pasarnya tidak bayar. Buktinya di ablation: MAE naik dari 53,96 dolar menjadi 95,48 dolar hanya karena menambah `market_index`. Jadi membuangnya bukan kelalaian, itu keputusan yang diukur."

Angka 53,96 dan 95,48 harus disebut persis. Ini bukti paling kuat di seluruh rekaman.

### 1:45–2:10 — Split dan validasi
Tampil: terminal, jalankan `python pipeline.py --cv`.

> "Validasi memakai rolling-origin, tiga fold expanding window. Setiap fold melatih sampai akhir satu bulan lalu memprediksi bulan berikutnya yang belum pernah dilihat. Saya tidak pakai random K-fold karena submission sebenarnya adalah forecast dua bulan ke depan — random split akan bocor antar waktu dan skornya jadi terlalu optimistis.
>
> Hasilnya MAE 53,56, 56,38, dan 51,93 dolar, MAPE sekitar 2,1 persen. Bias antar fold berubah tanda, dari plus 23 ke minus 27 dolar, yang menandakan tidak ada drift sistematis yang tertangkap model.
>
> Angka-angka ini yang saya laporkan, bukan skor pada data latih."

Saat terminal selesai, tunjuk tiga baris fold dan kolom MAE.

### 2:10–2:30 — Walkthrough kode
Tampil: `pipeline.py`.

> Kode masuk lewat satu entry point, `pipeline.py`, 171 baris. Alur utamanya: `clean` menangani masalah kualitas data yang tadi, `features` membangun sepuluh fitur, `drop_bad_labels` membuang label rusak, `fit` melatih pada log rate, `predict` mengembalikan rate asli, `rolling_cv` menghasilkan tabel tadi, dan `metrics` menghitung MAE, RMSE, MAPE, dan bias.
>
> Yang penting: fungsi `clean` mengembalikan nilai imputasi, dan nilai itu dipakai ulang ke validasi dan Desember. Jadi tidak ada statistik dari data uji yang bocor ke training.
>
> Terakhir, `score.py` bawaan assessment saya jalankan apa adanya dan hasilnya pass. Output-nya di folder `outputs`, chart Desember ada di `scorer_results`."

Tunjuk `def clean` dan baris `fills`, lalu tutup dengan `score.py` yang hijau.

## 4. Kalimat yang harus dihindari

- "Saya coba-coba" atau "kebetulan" — ganti dengan keputusan yang diukur.
- Menyebut angka yang tidak ada di report. Semua angka di naskah ini sudah ada di dokumen.
- "Model ini akurat" tanpa angka. Selalu MAE atau MAPE.
- Permintaan maaf di awal. Mulai langsung.
- Membaca tabel baris per baris. Sebut nomor penting saja.

## 5. Jika ditanya lanjutan

**Kenapa tidak XGBoost atau CatBoost?** Hasil setara pada data tabular seukuran ini; LightGBM paling cepat dilatih dan sudah punya penanganan kategori native, jadi tidak perlu one-hot.

**Kenapa L1 dan bukan RMSE?** Rate punya ekor panjang. RMSE akan didominasi beberapa load mahal, dan metrik scorer berbasis persentase — L1 pada log paling dekat ke situ.

**Kenapa tidak ada kalibrasi level?** Sudah diuji. Kalibrasi mean residual 30, 45, 60, dan 90 hari rata-rata memperburuk MAE dari 56,71 ke 74,80 dolar. Ditolak karena hasil ukur, bukan karena tidak dicoba.

**Berapa skor baseline?** Baseline dolar per mil konstan menghasilkan MAPE 8,19 persen. Model 1,99 persen.

**Kenapa 600 estimator?** Itu hasil early stopping di tiga fold CV, 486, 553, dan 691, diskalakan ke ukuran training penuh.

## 6. Checklist setelah rekaman

- Durasi 2:00–3:00. Di bawah 2 menit terasa terburu-buru, di atas 3 menit dipotong reviewer.
- Wajah terlihat minimal 80 persen durasi.
- Tidak ada notifikasi, tidak ada tab sensitif, tidak ada token atau password di layar.
- Suara jelas tanpa clipping. Putar ulang 30 detik pertama untuk cek.
- Link Loom diset **anyone with the link can view**, bukan private.
- Link ditempel di README repo dan di body pengiriman.

## 7. Angka rujukan

| Metrik | Nilai |
|---|---|
| Baris training mentah | 48.000 |
| Baris dipakai training | 47.401 |
| Baris dibuang (label rusak) | 599 |
| Load validasi | 12.000 |
| Holdout MAE | 51,93 dolar |
| Holdout RMSE | 277,61 dolar |
| Holdout MAPE | 1,99 persen |
| CV MAE tiga fold | 53,56 / 56,38 / 51,93 |
| CV MAPE | 2,23 / 2,18 / 1,99 persen |
| Baseline dolar per mil MAPE | 8,19 persen |
| MAE dengan `market_index` | 95,48 dolar |
| MAE dengan `quote_signal` | 63,88 dolar |
| Prediksi Desember | 811–831 dolar, rata-rata 822 |
