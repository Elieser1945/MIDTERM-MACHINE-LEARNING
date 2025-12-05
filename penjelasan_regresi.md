# 🧮 Midterm Regression – End-to-End ML Pipeline

---

## 🎯 Deskripsi Singkat & Tujuan

Proyek ini membangun **pipeline machine learning end-to-end** untuk tugas **regresi**, yaitu memprediksi **nilai target kontinu** (misalnya tahun rilis lagu) dari puluhan fitur numerik.  
Notebook dirancang untuk:

- Dijalankan **step-by-step** di Google Colab  
- Menggunakan komputasi yang **ringan namun lengkap secara konsep**  
- Menghasilkan **model + pipeline** yang siap dipakai untuk *inference* dan pembuatan file prediksi (*submission*)

---

## 📊 Dataset

- Lokasi file di Colab:  
  `/content/midterm-regresi-dataset.csv`

- Karakteristik dataset:
  - Tidak memiliki header  
  - **Kolom pertama**: `target` (nilai kontinu, misalnya tahun rilis)  
  - **Kolom berikutnya**: sekitar 89 fitur numerik (`feature_1`, `feature_2`, ..., `feature_n`)  
  - Semua kolom dikonversi ke tipe `float32` untuk efisiensi memori

- Setelah load, penamaan kolom:
  - `target`  
  - `feature_1`, `feature_2`, …, `feature_n`  

Seluruh fitur diperlakukan sebagai **numerik**, sehingga fokus preprocessing ada pada **imputasi**, **penanganan outlier**, dan **scaling**.

---

## 🧱 Arsitektur & Alur Pipeline

### 1. Preprocessing

Langkah utama pada tahap ini:

- **Pisah fitur dan target**
  - `X`: semua fitur numerik (`feature_*`)  
  - `y`: kolom `target`

- **Imputasi missing values**
  - Menggunakan `SimpleImputer(strategy="median")`  
  - Alasan: median lebih tahan terhadap outlier dibanding mean

- **Handling outlier**
  - Menggunakan transformer kustom yang melakukan *clipping* per fitur  
  - Nilai di-*clip* pada rentang persentil bawah–atas (misalnya 1–99)  
  - Mengurangi pengaruh nilai ekstrem tanpa membuang baris data

- **Scaling**
  - Menggunakan `RobustScaler`  
  - Cocok untuk fitur yang distribusinya miring (*skewed*) dan mengandung outlier

- **Pipeline preprocessing**
  - Seluruh langkah (imputer → clipper → scaler) dibungkus dalam:
    - `Pipeline` (urutan transformasi)  
    - `ColumnTransformer` (menerapkan ke semua fitur numerik)  
  - Memudahkan reuse di semua model dan di tahap inference

---

### 2. Train–Validation Split

- Menggunakan `train_test_split` dengan:
  - `test_size = 0.2` → 80% train, 20% validation  
  - `random_state = 42` untuk **reproducibility**

- Tidak menggunakan `stratify` karena ini **regresi**, bukan klasifikasi.

---

### 3. Modeling

Tiga jenis model utama yang digunakan:

#### 🔹 Baseline – Linear Regression (OLS)

- Pipeline: `preprocess → LinearRegression`  
- Memberikan **baseline sederhana** untuk membandingkan model yang lebih kompleks  
- Berguna untuk melihat apakah hubungan fitur–target cenderung linier

#### 🌲 Tree-based Model 1 – RandomForestRegressor

- Pipeline: `preprocess → RandomForestRegressor`  
- Kelebihan:
  - Mampu memodelkan **hubungan non-linear**  
  - Relatif robust terhadap skala fitur  
  - Menyediakan **feature importance** untuk analisis fitur mana yang paling berpengaruh
- Parameter dibuat **moderate** (jumlah tree dan kedalaman dibatasi) agar runtime tetap wajar di Colab

#### ⚙️ Tree-based Model 2 – XGBRegressor (XGBoost)

- Pipeline: `preprocess → XGBRegressor`  
- Menggunakan:
  - `objective = "reg:squarederror"`  
  - `tree_method = "hist"` agar training lebih cepat  
- Umumnya memiliki performa sangat baik untuk **data tabular** dengan banyak fitur

---

### 4. Hyperparameter Tuning (Ringan)

- Menggunakan **RandomizedSearchCV** pada:
  - `RandomForestRegressor` dan/atau `XGBRegressor`
- Hanya beberapa hyperparameter penting yang di-*search*:
  - Contoh: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, dll.
- `n_iter` dibuat **kecil** agar proses tuning tidak terlalu berat
- Tuning dilakukan di **subset data train** (mis. 30k sampel) untuk menghemat waktu
- Setelah didapat **parameter terbaik**, model di-*refit* pada **full training set** dan dievaluasi di validation set

---

### 5. Ensemble / Stacking (Opsional)

- Menggunakan **StackingRegressor** dengan:
  - Base estimators:
    - RandomForest (hasil tuning)  
    - XGBoost (hasil tuning)  
  - Final estimator: `LinearRegression` sebagai meta-model
- Tujuan:
  - Menggabungkan kelebihan kedua model
  - Melihat apakah **kombinasi model** mampu menurunkan error lebih jauh dibanding model tunggal

---

## 📈 Metrik Evaluasi & Visualisasi

Model dievaluasi pada validation set dengan beberapa metrik regresi utama:

- **MSE (Mean Squared Error)** – rata-rata kuadrat selisih antara nilai aktual dan prediksi  
  \[
  \text{MSE} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  \]

- **RMSE (Root Mean Squared Error)** – akar kuadrat dari MSE, satuan sama dengan target  
  \[
  \text{RMSE} = \sqrt{\text{MSE}}
  \]

- **MAE (Mean Absolute Error)** – rata-rata nilai absolut error  
  \[
  \text{MAE} = \frac{1}{n}\sum_{i=1}^{n} \lvert y_i - \hat{y}_i \rvert
  \]

- **\(R^2\) (coefficient of determination)** – mengukur seberapa besar variasi target yang dapat dijelaskan oleh model (mendekati 1 semakin baik)

Selain angka metrik, notebook juga menampilkan:

- **Residual plot** → scatter antara prediksi dan residual (\(y_{\text{true}} - y_{\text{pred}}\)) untuk mengecek pola error  
- **Predicted vs Actual** → scatter antara nilai aktual dan prediksi untuk melihat seberapa dekat titik-titik dengan garis identitas

---

## 🧪 Cara Menjalankan di Google Colab

1. Pastikan file dataset:
   - `midterm-regresi-dataset.csv`  
   sudah berada di path:  
   `/content/midterm-regresi-dataset.csv`

2. Jalankan notebook **dari atas ke bawah**:
   - Cell 0 → penjelasan / judul  
   - Cell 1 → install & import library  
   - Cell 2 → load dataset  
   - Cell 3 → EDA singkat  
   - Cell 4 → preprocessing (imputasi, outlier handle, scaling)  
   - Cell 5 → train–validation split  
   - Cell 6–8 → training model (Linear Regression, RandomForest, XGBoost)  
   - Cell 9–11 → tuning & stacking (jika digunakan)  
   - Cell 12–14 → evaluasi akhir, simpan model, inference & submission  
   - Cell 15 → catatan dan tips pengembangan

3. Setelah training selesai:
   - Model terbaik disimpan (misalnya `best_regression_model_*.joblib`)  
   - File contoh prediksi disimpan sebagai `submission_regression_example.csv` dengan format:
     - `Id`  
     - `Prediction`  

---



