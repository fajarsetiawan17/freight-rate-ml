# Brief Rekaman Loom — Freight Rate ML Assessment

Dokumen kerja untuk 1 kali rekaman. Target durasi **2 menit 45 detik**. Kamera wajib menyala (wajah terlihat, pojok kanan bawah).

Demo dilakukan **di dalam notebook Colab**, bukan di editor lokal. Semua bagian assessment bisa ditunjukkan dari satu tab: eksplorasi, kualitas data, CV, scorer, dan shape file submission.

Notebook: `report/Freight_Rate_Walkthrough.ipynb` — 22 cell, 11 seksi, ~3 menit untuk `Run all` di CPU gratis.

## 1. Setup sebelum menekan Record

**Wajib: jalankan `Runtime → Run all` sampai selesai SEBELUM merekam.** Notebook harus sudah punya seluruh output tercetak dan chart ter-render. Kalau tidak, rekaman akan mati menunggu 3 menit.

- Buka notebook lewat link Colab dari README repo. Jangan buka file `.ipynb` mentah dari GitHub.
- Tunggu semua cell berstatus selesai, chart Desember terlihat, dan TOC kiri terisi 11 seksi.
- Lipat panel kiri (TOC) kalau layar sempit — tapi biarkan TOC terbuka kalau ingin terlihat terstruktur. Pilih satu, jangan berubah di tengah rekaman.
- Zoom browser 110–125 persen. Font cell default 14 px sudah terbaca di 1080p, jangan diperkecil.
- Sembunyikan sidebar kiri Colab yang lain (Files, Secrets) — jangan sampai folder Drive pribadi terlihat.
- Tutup tab lain. Sisakan dua: notebook Colab, dan `github.com/fajarsetiawan17/freight-rate-ml`.
- Mikrofon: uji 5 detik. Level bicara normal, tanpa desis. Ruangan tanpa kipas.
- Pencahayaan di depan wajah. Jendela jangan di belakang kepala.
- Loom: 1080p, kamera on, mikrofon on, **jangan** rekam tab saja.
- Aktifkan Focus Assist Windows selama rekaman.

### Cell mana yang dijalankan live, mana yang cukup ditunjuk

| Cell | Perlakuan | Alasan |
|---|---|---|
| §1 statistik + korelasi | **live** | Instan, dan angkanya jadi bukti |
| Rate per mile (chart) | **live** | Instan, chart muncul di tempat |
| §2 kualitas data | **live** | Instan, ini bukti terkuat |
| §3 `pipeline.py --cv` | **cached** | 1–2 menit di Colab — jangan pernah live |
| Ablation (baca CSV) | **live** | Instan |
| §4 `score.py` | **cached** | 10–20 detik, terlalu lama untuk segmen 10 detik |
| Chart Desember | **cached** | Sudah ter-render dari run sebelumnya |
| §5 shape file | **live** | Instan |

Saat menunjuk output cached, tekan cell-nya sekali supaya ter-highlight, lalu bicara. Jangan pura-pura menjalankan.

## 2. Peta waktu

| Segmen | Durasi | Yang tampil |
|---|---|---|
| Pembuka | 0:00–0:12 | Kamera, lalu tab Colab |
| Setup dan peta repo | 0:12–0:25 | §0 Setup |
| Temuan eksplorasi | 0:25–0:55 | §1 + chart rate per mile |
| Kualitas data | 0:55–1:25 | §2 Data-quality issues |
| Alasan model + ablation | 1:25–1:50 | §3 judul + tabel ablation |
| Split dan CV | 1:50–2:15 | §3 output `--cv` |
| Scorer dan chart Desember | 2:15–2:40 | §4 + chart |
| Penutup + file submission | 2:40–2:45 | §5 |

Kalau kelebihan waktu, potong segmen penutup dan segmen setup. Jangan pernah memotong kualitas data atau split.

## 3. Naskah per segmen

### 0:00–0:12 — Pembuka
Tampil: kamera, lalu share screen ke tab Colab.

> "Halo, saya Fajar Setiawan. Ini submission assessment Machine Learning Engineer untuk prediksi freight rate. Semua yang saya tunjukkan di sini berjalan dari satu notebook Colab yang memanggil kode submission asli."

Dua belas detik. Jangan baca CV, jangan minta maaf.

### 0:12–0:25 — Setup dan peta repo
Tampil: §0 Setup, sudah selesai dijalankan.

> "Notebook ini clone repo-nya sendiri lalu install LightGBM, scikit-learn, dan matplotlib. Total 22 cell, dan seluruh pipeline jalan di CPU gratis dalam tiga menit. Data yang dipakai adalah data yang disuplai assessment, ikut di dalam repo."

Sorot output `data: ['december_chart_inputs.csv', 'train_test.csv', ...]`.

### 0:25–0:55 — Temuan eksplorasi
Tampil: §1 lalu chart rate per mile.

> "Temuan terbesar: rate hampir merupakan fungsi murni jarak. Korelasi log rate terhadap log distance 0,99 setelah baris harga mustahil dibuang — dan tetap 0,967 kalau semua baris ikut dihitung. Artinya model tidak perlu menemukan ulang hubungan jarak–harga, cukup belajar koreksi level di atasnya.
>
> Sebaran rate right-skewed dengan skewness 1,9, jadi saya memodelkan log dari rate, bukan rate mentah.
>
> Grafik kanan memperlihatkan rate per mil: median 2,15 dolar, tapi ada ekor sampai 14 dolar. Itu bukan harga, itu kesalahan label — dan itu masuk ke bagian berikutnya."

Tunjuk angka 0.9919 di output, lalu grafik histogram.

### 0:55–1:25 — Kualitas data
Tampil: §2 Data-quality issues, jalankan live.

> "Enam masalah, dan semuanya terhitung di cell ini.
>
> Pertama, 599 baris punya rate mustahil — di bawah satu dolar atau di atas enam dolar per mil. Itu label error, bukan harga. Saya buang dari training, tapi load-nya tetap dapat prediksi.
>
> Kedua, 292 baris punya berat negatif, minimum minus 47.500 pon. Tanda terbalik, jadi saya ambil nilai absolutnya.
>
> Ketiga, berat hilang di 300 baris training dan 165 baris validasi. Saya isi median training, dan median yang sama dipakai ulang ke validasi dan Desember supaya tidak ada leakage.
>
> Keempat, `market_index` hilang di 374 baris, diperlakukan sama.
>
> Kelima, 495 baris punya rute, tanggal, dan peralatan identik tapi rate berbeda. Itu load terpisah, bukan duplikat. Saya biarkan.
>
> Keenam, file Desember tidak membawa koordinat kota sama sekali — kolomnya cuma tujuh. Saya bangun lookup 64 kota dari data training."

Kalau waktu mepet, lewati poin kelima. Poin satu dan dua wajib.

### 1:25–1:50 — Alasan model dan ablation
Tampil: judul §3, lalu tabel ablation.

> "Saya pakai LightGBM dengan objektif L1 pada log rate. Tiga alasan: L1 tahan outlier dan mendekati optimasi median, cocok untuk target berekor panjang; gradient boosting menangani interaksi jarak–peralatan–lane tanpa saya tulis manual; dan datanya tabular, jadi model lebih berat tidak membeli apa pun.
>
> Sepuluh fitur: log jarak, log berat, peralatan, asal, tujuan, titik tengah lintang, titik tengah bujur, delta utara-selatan, delta timur-barat, dan hari dalam minggu.
>
> Dua kolom sengaja dibuang: `market_index` dan `quote_signal`. Keduanya bergeser level antara jendela pengembangan dan jendela prediksi. Kalau dimasukkan, model mengekstrapolasi level yang pasarnya tidak bayar. Ablation mengukurnya: MAE naik dari 53,96 ke 95,48 dolar hanya karena menambah `market_index`."

Angka 53,96 dan 95,48 harus disebut persis. Ini bukti terkuat di seluruh rekaman.

### 1:50–2:15 — Split dan CV
Tampil: output `!python pipeline.py --cv` (cached).

> "Validasi memakai rolling-origin, tiga fold expanding window. Setiap fold melatih sampai akhir satu bulan lalu memprediksi bulan berikutnya yang belum pernah dilihat. Saya tidak pakai random K-fold, karena submission sebenarnya adalah forecast dua bulan ke depan — random split akan bocor antar waktu dan skornya jadi terlalu optimistis.
>
> Hasilnya MAE 53,56, 56,38, dan 51,93 dolar, MAPE sekitar 2,1 persen. Bias berubah tanda antar fold, dari plus 23 ke minus 27 dolar — tidak ada drift sistematis yang lolos.
>
> Cell ini saya jalankan sebelum rekaman: butuh satu sampai dua menit di Colab. Angka yang saya laporkan ini, bukan skor di data latih."

Saat menunjuk, sorot kolom MAE dan `best_iter` 486, 553, 691.

### 2:15–2:40 — Scorer dan chart Desember
Tampil: §4 output lalu chart.

> "Terakhir, `score.py` bawaan assessment saya jalankan apa adanya, tanpa modifikasi. Hasilnya: 12.000 prediksi final tervalidasi, 31 prediksi Desember tervalidasi, dan chart-nya dibuat.
>
> Chart inilah artefak yang diminta brief. Prediksi Desember bergerak 811 sampai 831 dolar, rata-rata 822. Puncak di pertengahan minggu, dasar hari Minggu — pola mingguan yang masuk akal untuk angkutan barang."

Tunjuk baris "Validated 12,000 final predictions", lalu chart.

### 2:40–2:45 — Penutup
Tampil: §5 Submission files.

> "Dua file keluar: `validation_predictions.csv` dengan tepat dua kolom, dan file Desember dengan tujuh kolom asli di urutan asli plus `predicted_rate`. Kode, data, report, dan notebook ini semuanya ada di repo publik. Terima kasih."

Tunjuk `(12000, 2)` dan `(31, 7)`.

## 4. Kalimat yang harus dihindari

- "Saya coba-coba" atau "kebetulan" — ganti dengan keputusan yang diukur.
- Menyebut angka yang tidak ada di notebook atau report. Semua angka di naskah ini sudah ada.
- "Model ini akurat" tanpa angka. Selalu MAE atau MAPE.
- Permintaan maaf di awal. Mulai langsung.
- Membaca tabel baris per baris. Sebut nomor penting saja.
- "Colab-nya lambat" atau komentar soal koneksi. Kalau ada loading, diam dan tunjuk cell berikutnya.

## 5. Jika ditanya lanjutan

**Kenapa Colab, bukan repo saja?** Repo sudah cukup untuk menjalankan; notebook hanya membuat seluruh alur bisa diverifikasi dalam satu klik tanpa install lokal.

**Kenapa tidak XGBoost atau CatBoost?** Hasil setara pada data tabular seukuran ini; LightGBM paling cepat dilatih dan punya penanganan kategori native, jadi tidak perlu one-hot.

**Kenapa L1 dan bukan RMSE?** Rate punya ekor panjang. RMSE akan didominasi beberapa load mahal, dan metrik scorer berbasis persentase — L1 pada log paling dekat ke situ.

**Kenapa tidak ada kalibrasi level?** Sudah diuji. Kalibrasi mean residual 30, 45, 60, dan 90 hari memperburuk MAE dari 56,71 ke 74,80 dolar. Ditolak karena hasil ukur.

**Berapa skor baseline?** Baseline dolar per mil konstan menghasilkan MAPE 8,19 persen. Model 1,99 persen.

**Kenapa 600 estimator?** Hasil early stopping tiga fold, 486, 553, dan 691, diskalakan ke ukuran training penuh.

**Kenapa notebook tidak membuat report PDF?** Report butuh langkah Word/LibreOffice. Karena itu PDF-nya ikut dikirim di repo, bukan di-generate ulang di Colab.

## 6. Checklist setelah rekaman

- Durasi 2:30–3:00. Di bawah 2 menit terasa terburu-buru, di atas 3 menit dipotong reviewer.
- Wajah terlihat minimal 80 persen durasi.
- Notebook sudah `Run all` sebelum rekaman; tidak ada cell yang masih berputar.
- Tidak ada notifikasi, tidak ada tab Drive pribadi, tidak ada token atau password di layar.
- Suara jelas tanpa clipping. Putar ulang 30 detik pertama untuk cek.
- Link Loom diset **anyone with the link can view**, bukan private.
- Link ditempel di README repo dan di body pengiriman.

## 7. Angka rujukan

| Metrik | Nilai |
|---|---|
| Cell notebook | 22, dalam 11 seksi |
| Baris training mentah | 48.000 |
| Baris dipakai training | 47.401 |
| Baris dibuang (label rusak) | 599 |
| Load validasi | 12.000 |
| Korelasi log jarak–log rate (baris valid) | 0,9919 |
| Korelasi log jarak–log rate (semua baris) | 0,9674 |
| Skewness rate | 1,90 |
| Median dolar per mil | 2,15 |
| Holdout MAE | 51,93 dolar |
| Holdout RMSE | 277,61 dolar |
| Holdout MAPE | 1,99 persen |
| CV MAE tiga fold | 53,56 / 56,38 / 51,93 |
| CV MAPE | 2,23 / 2,18 / 1,99 persen |
| Baseline dolar per mil MAPE | 8,19 persen |
| MAE dengan `market_index` | 95,48 dolar |
| MAE dengan `quote_signal` | 63,88 dolar |
| Prediksi Desember | 811–831 dolar, rata-rata 822 |
| Bentuk file submission | (12000, 2) dan (31, 7) |
