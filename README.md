# 📂 Machine Learning Midterm 


## 🎯 Tujuan Repository

Repository ini dibuat sebagai kumpulan proyek *machine learning* untuk tugas midterm yang mencakup tiga topik utama:

1. **Klasifikasi** – Fraud Detection (deteksi transaksi penipuan)  
2. **Regresi** – Prediksi nilai target kontinu (tahun rilis / nilai numerik lain)  
3. **Clustering** – Customer Segmentation (pengelompokan pelanggan kartu kredit)

Tujuan utama repository ini adalah:

- Mendemonstrasikan *end-to-end pipeline* machine learning yang rapi, modular, dan reproducible.  
- Mencakup seluruh tahapan standar: EDA, preprocessing, modeling, evaluasi, visualisasi, sampai penyimpanan model & pembuatan submission/prediksi.  
- Menjadi referensi praktis untuk implementasi pipeline ML pada data tabular real-world.

---

## 🧾 Gambaran Singkat Proyek

### 1. Fraud Detection – Klasifikasi

- **Masalah**: Memprediksi apakah sebuah transaksi online termasuk fraud (`isFraud = 1`) atau tidak (`isFraud = 0`).  
- **Dataset**:  
  - `train_transaction.csv` (fitur + label `isFraud`)  
  - `test_transaction.csv` (fitur saja, untuk submission)  
- **Fokus**:
  - Menangani **class imbalance** (fraud vs non-fraud)  
  - Menggunakan model **RandomForestClassifier** dan **XGBoostClassifier**  
  - Menghasilkan file submission `TransactionID, isFraud`.

### 2. Regression – Prediksi Target Kontinu

- **Masalah**: Memprediksi nilai target kontinu dari sekitar 89 fitur numerik (mis. tahun rilis atau skor numerik lain).  
- **Dataset**:  
  - `midterm-regresi-dataset.csv`  
  - Kolom pertama = `target`, kolom berikutnya = `feature_1` … `feature_n`  
- **Fokus**:
  - Preprocessing numerik penuh (imputasi, outlier handling, scaling)  
  - Model **Linear Regression**, **RandomForestRegressor**, **XGBRegressor**  
  - Tuning ringan & opsi stacking sederhana.

### 3. Customer Clustering – Unsupervised Learning

- **Masalah**: Mengelompokkan pelanggan kartu kredit berdasarkan perilaku penggunaan dan pembayaran.  
- **Dataset**:  
  - `clusteringmidterm.csv` dengan fitur seperti `BALANCE`, `PURCHASES`, `CASH_ADVANCE`, `CREDIT_LIMIT`, `PAYMENTS`, dsb.  
- **Fokus**:
  - Clustering dengan **K-Means**, **Hierarchical Clustering (Agglomerative)**, dan **DBSCAN**  
  - Menentukan jumlah klaster optimal (Elbow & Silhouette)  
  - Interpretasi karakteristik tiap klaster (high spender, revolver, dormant, installment user, dll).

---

## 🤖 Model & Metrik yang Digunakan

### 1. Proyek Klasifikasi – Fraud Detection

**Model utama:**

- `RandomForestClassifier`  
- `XGBClassifier` (XGBoost)

**Tahapan penting:**

- Imputasi missing values (median untuk numerik, modus untuk kategorikal)  
- Encoding fitur kategorikal → integer codes  
- Scaling fitur numerik dengan `StandardScaler`  
- Penanganan **imbalanced data** dengan `RandomUnderSampler`  
- Feature selection berbasis *feature importance* RandomForest (top-N fitur)

**Metrik evaluasi:**

- **AUC (Area Under ROC Curve)**  
- **F1-score**  
- **ROC Curve**  
- **Confusion Matrix**  
- **Classification report** (precision, recall, f1 per kelas)

Formula F1-score secara umum:

$$
F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
$$

---

### 2. Proyek Regresi

**Model utama:**

- `LinearRegression` (baseline)  
- `RandomForestRegressor`  
- `XGBRegressor` (XGBoost)  
- (Opsional) `StackingRegressor` yang menggabungkan model terbaik

**Tahapan penting:**

- Imputasi numerik dengan median (`SimpleImputer`)  
- *Clipping* outlier pada percentil 1–99  
- Scaling dengan `RobustScaler` (lebih tahan outlier)  
- Hyperparameter tuning ringan dengan `RandomizedSearchCV`  
- Penyimpanan model terbaik ke file `.joblib` dan contoh inference

**Metrik evaluasi:**

- **MSE (Mean Squared Error)**  
  Rumus:  
  $$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- **RMSE (Root Mean Squared Error)**  
  Rumus:  
  $$\text{RMSE} = \sqrt{\text{MSE}}$$

- **MAE (Mean Absolute Error)**  
  Rumus:  
  $$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} \lvert y_i - \hat{y}_i \rvert$$

- **\(R^2\)** (coefficient of determination) – seberapa besar variasi target yang bisa dijelaskan model.

---

### 3. Proyek Clustering – Customer Segmentation

**Algoritma utama:**

- **K-Means**  
  - Mencari \(k\) klaster dengan meminimalkan jumlah kuadrat jarak ke centroid.  
- **Agglomerative Hierarchical Clustering** (linkage Ward)  
  - Membentuk hierarki klaster, divisualisasikan dengan dendrogram.  
- **DBSCAN**  
  - Clustering berbasis densitas, dapat mendeteksi noise dan bentuk klaster yang tidak beraturan.

**Tahapan penting:**

- Imputasi median → clipping outlier → scaling (`StandardScaler`)  
- Feature engineering rasio (mis. `PURCHASES_TO_LIMIT`, `CASHADV_TO_LIMIT`, dsb.)  
- Penentuan jumlah klaster K-Means dengan:
  - **Elbow Method** (plot inertia)  
  - **Average Silhouette Score**

**Silhouette coefficient** untuk tiap titik \(i\):

$$
s(i) = \frac{b(i) - a(i)}{\max \big(a(i), b(i)\big)}
$$

dengan:
- \(a(i)\): rata-rata jarak ke titik dalam klaster yang sama  
- \(b(i)\): rata-rata jarak minimum ke klaster lain  

Metrik & visual utama:

- Rata-rata **Silhouette Score** per nilai \(k\)  
- Silhouette plot per klaster  
- PCA 2D scatter plot berwarna berdasarkan label klaster  
- Boxplot fitur utama per klaster untuk interpretasi behavior.

---

## 🧭 Navigasi Repository & Notebook

Struktur dan penamaan dapat disesuaikan, namun contoh navigasi yang direkomendasikan:

- `notebooks/`
  - `01_fraud_detection_classification.ipynb`  
    - Pipeline klasifikasi fraud detection (RandomForest + XGBoost + tuning + submission).  
  - `02_regression_midterm.ipynb`  
    - Pipeline regresi lengkap (Linear Regression, RF, XGB, tuning, stacking, saving model).  
  - `03_customer_clustering.ipynb`  
    - Pipeline clustering (K-Means, Hierarchical, DBSCAN, Elbow, Silhouette, visualisasi & interpretasi).

**Cara menggunakan:**

1. Buka repository di GitHub.  
2. Masuk ke folder `notebooks/`.  
3. Pilih notebook sesuai tugas:
   - Klasifikasi → `01_fraud_detection_classification.ipynb`  
   - Regresi → `02_regression_midterm.ipynb`  
   - Clustering → `03_customer_clustering.ipynb`  
4. Klik tombol **Open in Colab** (jika disediakan) atau upload manual ke Google Colab.  
5. Jalankan setiap cell **secara berurutan dari atas ke bawah** untuk mereproduksi seluruh hasil.

---

## 👤 Identitas

- **Nama**  : Elieser Pasaribu  
- **Kelas** : TK-46-04  
- **NIM**   : 1103223209  

Repository ini dibuat sebagai bagian dari tugas/midterm mata kuliah Machine Learning, dengan fokus pada implementasi pipeline end-to-end untuk **klasifikasi**, **regresi**, dan **clustering** pada data tabular.
