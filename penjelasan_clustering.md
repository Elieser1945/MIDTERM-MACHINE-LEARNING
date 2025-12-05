# 🧩 Customer Clustering – Credit Card Usage

---

## 🎯 Deskripsi Singkat & Tujuan

Proyek ini membangun **pipeline unsupervised learning end-to-end** untuk melakukan **customer clustering** berdasarkan pola penggunaan kartu kredit dan perilaku pembayaran pelanggan.  
Tujuan utamanya adalah mengelompokkan pelanggan menjadi segmen-segmen yang memiliki karakteristik serupa, sehingga bisa dimanfaatkan untuk strategi bisnis seperti penawaran produk, promosi, atau manajemen risiko.[web:31][web:32]

Secara garis besar, notebook mencakup:
- Exploratory Data Analysis (EDA)
- Pembersihan data & penanganan missing values
- Penanganan outlier
- Feature engineering & feature scaling
- Clustering dengan **K-Means**, **Hierarchical Clustering**, dan **DBSCAN**
- Penentuan jumlah klaster optimal (Elbow & Silhouette)
- Evaluasi & visualisasi klaster (PCA plot, pairplot, boxplot, dendrogram, silhouette plot)
- Interpretasi karakteristik tiap klaster dan kesimpulan akhir[web:31][web:36][web:39]

---

## 📊 Dataset

Dataset: `clusteringmidterm.csv`  
Dataset ini berisi informasi penggunaan kartu kredit dan perilaku pembayaran pelanggan. Setiap baris mewakili satu pelanggan unik, dengan fitur-fitur utama sebagai berikut:

- `CUST_ID` — ID unik pelanggan  
- `BALANCE` — saldo rata-rata / saldo belum dibayar  
- `BALANCE_FREQUENCY` — seberapa sering saldo diperbarui  
- `PURCHASES`, `ONEOFF_PURCHASES`, `INSTALLMENTS_PURCHASES` — total dan tipe pengeluaran  
- `CASH_ADVANCE`, `CASH_ADVANCE_FREQUENCY`, `CASH_ADVANCE_TRX` — aktivitas penarikan tunai  
- `PURCHASES_FREQUENCY`, `ONEOFF_PURCHASES_FREQUENCY`, `PURCHASES_INSTALLMENTS_FREQUENCY`, `PURCHASES_TRX` — intensitas transaksi pembelian  
- `CREDIT_LIMIT` — batas maksimal kredit  
- `PAYMENTS`, `MINIMUM_PAYMENTS` — perilaku pembayaran  
- `PRC_FULL_PAYMENT` — proporsi pembayaran penuh  
- `TENURE` — lama kepemilikan akun (bulan)[web:31][web:34]

Dalam pipeline:
- `CUST_ID` hanya digunakan sebagai identitas, **tidak** dipakai dalam perhitungan clustering.
- Seluruh fitur numerik dibersihkan, diimputasi, dan dinormalisasi sebelum dipakai untuk model clustering.[web:31][web:36]

---

## 🧱 Alur Pipeline & Preprocessing

### 1. EDA (Exploratory Data Analysis)

Langkah awal EDA mencakup:
- Menampilkan statistik deskriptif (`describe`) untuk memahami skala, sebaran, dan kemungkinan skewness tiap fitur.
- Menghitung persentase **missing values** per kolom.
- Menggambarkan **histogram** beberapa fitur penting seperti `BALANCE`, `PURCHASES`, `CASH_ADVANCE`, `CREDIT_LIMIT`, dan `PAYMENTS`.
- Membuat **pairplot** pada subset data & subset fitur untuk melihat korelasi visual dan pola cluster kasar secara 2D.[web:31][web:32]

### 2. Pembersihan Data & Missing Values

Tahap pembersihan mencakup:
- Menghapus `CUST_ID` dari matriks fitur (disimpan terpisah jika ingin dipakai untuk pelabelan hasil).
- Memilih hanya kolom numerik untuk proses clustering.
- Mengisi missing values menggunakan **median** (`SimpleImputer(strategy="median")`) untuk mengurangi pengaruh outlier sekaligus menjaga informasi distribusi.[web:31][web:36]

### 3. Penanganan Outlier

Outlier ditangani dengan **clipping** nilai ke dalam rentang persentil bawah–atas (misalnya 1–99) pada setiap fitur numerik:
- Nilai di bawah persentil bawah digeser ke batas bawah.
- Nilai di atas persentil atas digeser ke batas atas.

Pendekatan ini menjaga ukuran sample tetap sama, sekaligus mengurangi dominasi nilai ekstrem pada jarak Euclidean yang digunakan sebagian besar algoritma clustering.[web:35][web:36]

### 4. Feature Engineering & Scaling

Beberapa fitur turunan yang relevan ditambahkan, misalnya (jika kolom tersedia):

- Rasio pembelian terhadap limit:
  \[
  \text{PURCHASES\_TO\_LIMIT} = \frac{\text{PURCHASES}}{\text{CREDIT\_LIMIT} + \epsilon}
  \]
- Rasio cash advance terhadap limit:
  \[
  \text{CASHADV\_TO\_LIMIT} = \frac{\text{CASH\_ADVANCE}}{\text{CREDIT\_LIMIT} + \epsilon}
  \]
- Rasio pembayaran terhadap saldo:
  \[
  \text{PAYMENTS\_TO\_BALANCE} = \frac{\text{PAYMENTS}}{\text{BALANCE} + \epsilon}
  \]

dengan \(\epsilon\) adalah konstanta kecil (misalnya \(10^{-6}\)) untuk menghindari pembagian dengan nol.[web:31][web:35]

Seluruh fitur numerik (original + engineered) kemudian dinormalisasi menggunakan **StandardScaler**:
- Setiap fitur ditransformasikan ke mean 0 dan standar deviasi 1, sehingga skala fitur sebanding dan tidak mendominasi perhitungan jarak.[web:31][web:40]

---

## 🧮 Metode Clustering & Evaluasi

### 1. K-Means Clustering

**K-Means** membagi data menjadi \(k\) klaster dengan meminimalkan jumlah kuadrat jarak tiap titik ke centroid klasternya. Secara umum, jarak yang digunakan adalah **jarak Euclidean**:
\[
d(\mathbf{x}, \mathbf{\mu}) = \sqrt{\sum_{j=1}^{p} (x_j - \mu_j)^2}
\]

Langkah evaluasi:
- Menjalankan K-Means untuk beberapa nilai \(k\) (misal 2–10).
- Mencatat:
  - **Inertia** (Within-Cluster SSE) → dipakai pada **Elbow Method**.
  - **Average Silhouette Score** untuk setiap \(k\).[web:31][web:39][web:40]

**Silhouette coefficient** untuk satu titik \(i\) didefinisikan sebagai:
\[
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
\]
dengan:
- \(a(i)\): rata-rata jarak dari titik \(i\) ke semua titik lain dalam klaster yang sama.
- \(b(i)\): jarak rata-rata terkecil dari titik \(i\) ke semua titik dalam klaster lain.

Nilai \(s(i)\) berada di antara \(-1\) dan \(1\); semakin mendekati 1, semakin baik pemisahan klaster untuk titik tersebut. Rata-rata \(s(i)\) seluruh titik digunakan sebagai **Silhouette Score** model.[web:39][web:40]

Berdasarkan kombinasi **Elbow** (bentuk siku pada plot inertia) dan **Silhouette Score**, dipilih nilai \(k\) yang memberikan trade-off baik antara pemadatan klaster dan pemisahan antar klaster (misalnya \(k = 4\), menyesuaikan hasil aktual).[web:31][web:39]

### 2. Hierarchical Clustering (Agglomerative)

Metode **Agglomerative Clustering** membangun hierarki klaster secara bottom-up:
- Setiap titik awalnya adalah satu klaster.
- Pada tiap langkah, dua klaster paling mirip digabung.
- Proses berlanjut hingga mencapai jumlah klaster yang diinginkan.[web:36][web:37]

Notebook:
- Menggunakan **metode linkage Ward** dan jarak Euclidean.
- Menampilkan **dendrogram** pada subset data untuk memvisualisasikan struktur hierarki dan jarak penggabungan klaster.
- Menetapkan jumlah klaster sama seperti K-Means (mis. \(k\) optimal), sehingga hasilnya dapat dibandingkan dengan segmentasi K-Means.[web:36][web:39]

### 3. DBSCAN

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** mengelompokkan titik berdasarkan kepadatan lokal, dengan dua parameter utama:
- \(\varepsilon\) (eps): radius tetangga.
- `min_samples`: jumlah minimum titik di dalam radius tersebut untuk membentuk klaster ber‑kepadatan tinggi.[web:36][web:40]

Notebook:
- Menjalankan DBSCAN pada subset data (untuk efisiensi) dengan beberapa kombinasi `eps` dan `min_samples`.
- Mengidentifikasi:
  - **Cluster labels** (0, 1, 2, …).
  - **Noise points** (label = −1).
- Menghitung **Silhouette Score** hanya pada titik non‑noise jika jumlah klaster hasil DBSCAN lebih dari satu.
- Memvisualisasikan hasil klaster DBSCAN dalam ruang 2D menggunakan PCA.[web:36][web:40]

---

## 📈 Visualisasi & Analisis Klaster

Berbagai visualisasi digunakan untuk memahami struktur data dan interpretasi klaster:

- **PCA 2D Scatter Plot**  
  - Mengurangi dimensi fitur ke 2 komponen utama (PC1, PC2) dengan PCA.  
  - Mewarangi titik berdasarkan label klaster K-Means / Hierarchical / DBSCAN untuk melihat pemisahan visual.[web:31][web:35]

- **Silhouette Plot (K-Means)**  
  - Menunjukkan distribusi silhouette coefficient per klaster.  
  - Klaster yang baik cenderung memiliki nilai silhouette positif yang cukup tinggi dan relatif seragam.[web:39][web:40]

- **Pairplot (subset fitur + sampel)**  
  - Memeriksa pola klaster di beberapa kombinasi fitur penting (misalnya `BALANCE`, `PURCHASES`, `CASH_ADVANCE`, `CREDIT_LIMIT`).  
  - Membantu melihat apakah klaster terpisah jelas atau saling tumpang tindih di ruang fitur terpilih.[web:31][web:35]

- **Boxplot per Klaster (K-Means)**  
  - Membandingkan distribusi fitur seperti `BALANCE`, `PURCHASES`, `CASH_ADVANCE`, `CREDIT_LIMIT`, `PAYMENTS`, `PRC_FULL_PAYMENT` antar klaster.  
  - Memudahkan identifikasi klaster dengan saldo tinggi, pembelian tinggi, atau perilaku pembayaran berbeda.[web:31][web:33]

- **Dendrogram (Hierarchical)**  
  - Menunjukkan urutan penggabungan klaster dan jarak antar klaster pada tiap level.  
  - Memberi intuisi tentang jumlah klaster alami (cut‑off jarak yang masuk akal).[web:36][web:37]

---

## 🧠 Interpretasi Klaster & Kesimpulan

Berdasarkan hasil clustering (khususnya K-Means sebagai baseline utama) dan ringkasan rata‑rata fitur per klaster, tiap segmen pelanggan dapat diinterpretasikan, misalnya:

- **Cluster 0 – High Balance & Cash Advance Users**  
  - Saldo (`BALANCE`) tinggi, aktivitas `CASH_ADVANCE` dan `CASH_ADVANCE_TRX` tinggi.  
  - Proporsi pembayaran penuh (`PRC_FULL_PAYMENT`) dan rasio `MINIMUM_PAYMENTS` relatif rendah.  
  - Berpotensi sebagai **segmen risiko tinggi** atau *revolvers* yang sering menahan saldo.[web:31][web:33]

- **Cluster 1 – High Spenders with Full Payment**  
  - `PURCHASES`, `PURCHASES_TRX`, dan `CREDIT_LIMIT` tinggi.  
  - `PRC_FULL_PAYMENT` dan `PAYMENTS_TO_BALANCE` tinggi.  
  - Segmen **pelanggan premium / high‑value**, cocok untuk program loyalitas dan penawaran eksklusif.[web:31][web:38]

- **Cluster 2 – Low Activity / Dormant Customers**  
  - Nilai transaksi dan frekuensi sangat rendah di hampir semua metrik.  
  - Target untuk **kampanye aktivasi**, peningkatan engagement, atau penyesuaian limit.[web:31][web:38]

- **Cluster 3 – Installment-Oriented Customers**  
  - `INSTALLMENTS_PURCHASES` dan `PURCHASES_INSTALLMENTS_FREQUENCY` tinggi.  
  - Pola pembayaran cukup stabil namun mungkin tidak penuh.  
  - Cocok untuk **penawaran produk cicilan** dan promosi suku bunga khusus.[web:33][web:38]

(Pola sebenarnya harus disesuaikan dengan hasil tabel profil klaster pada dataset kamu.)

Secara keseluruhan:
- **K-Means** memberikan segmentasi yang stabil dan mudah diinterpretasikan.  
- **Hierarchical Clustering** mendukung pemahaman struktur hubungan antar klaster lewat dendrogram.  
- **DBSCAN** bermanfaat untuk mendeteksi area densitas khusus dan titik‑titik noise, meski pemilihan parameter sensitif terhadap skala dan kepadatan data.[web:33][web:36][web:39]

Pipeline ini dapat dikembangkan lebih lanjut dengan:
- Menambahkan **PCA penuh** sebelum clustering untuk mengurangi dimensi dan noise.  
- Menggunakan metrik evaluasi internal lain (misal **Davies–Bouldin Index**) dan membandingkannya antar metode.  
- Menggabungkan hasil klaster dengan label atau metrik bisnis eksternal (churn, default, revenue) untuk menilai utilitas segmen dalam konteks nyata.[web:32][web:40]
