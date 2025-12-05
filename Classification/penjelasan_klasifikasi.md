
# Fraud Transaction Classification – End-to-End Pipeline

## 1. Deskripsi Singkat

Proyek ini membangun pipeline machine learning end-to-end untuk mendeteksi apakah sebuah transaksi online berpotensi fraud (`isFraud`) berdasarkan fitur-fitur numerik dan kategorikal dari dataset transaksi.  
Notebook dirancang agar:

- Dapat dijalankan *step-by-step* di Google Colab.  
- Menggunakan komputasi ringan (model dan tuning dibuat sederhana namun tetap lengkap).  
- Menghasilkan file submission dengan format `TransactionID, isFraud`.

## 2. Tujuan Proyek

Tujuan utama:

- Memprediksi probabilitas dan label biner apakah sebuah transaksi adalah fraud (`isFraud = 1`) atau bukan (`isFraud = 0`).  
- Mendesain pipeline yang rapi dan modular mulai dari:
  - Import library  
  - Load data  
  - EDA singkat  
  - Data preprocessing (imputasi, encoding, scaling)  
  - Penanganan class imbalance  
  - Feature engineering & feature selection  
  - Train–test split  
  - Training model ML (Random Forest & XGBoost)  
  - Hyperparameter tuning sederhana  
  - Evaluasi model (AUC, ROC, Confusion Matrix, F1-score, visualisasi)  
  - Generate submission untuk `test_transaction.csv`.  

## 3. Dataset

- Sumber: Google Drive, folder: `MyDrive/Fraud Transaction`.  
- File utama:
  - `train_transaction.csv` – berisi fitur + kolom target `isFraud`.  
  - `test_transaction.csv` – berisi fitur tanpa label, digunakan untuk submission.  
- Kolom penting:
  - `TransactionID` – ID unik per transaksi.  
  - `isFraud` – label target (0: bukan fraud, 1: fraud).  

Dataset memiliki karakteristik tipikal data fraud:

- Jumlah baris besar (~600 ribu baris).  
- Fitur campuran numerik dan kategorikal.  
- Distribusi kelas tidak seimbang (fraud jauh lebih sedikit daripada non-fraud).

## 4. Arsitektur & Pipeline

Secara ringkas, pipeline yang dibangun terdiri dari:

1. **Preprocessing**  
   - Imputasi missing values:
     - Fitur numerik: median.  
     - Fitur kategorikal: modus (most frequent).  
   - Encoding kategori:
     - Menggunakan integer codes dengan `.astype('category').cat.codes` agar cepat dan hemat memori.  
   - Scaling:
     - `StandardScaler` untuk fitur numerik agar skala lebih seragam sebelum training model.

2. **Penanganan Class Imbalance**  
   - Menggunakan **Random UnderSampling** untuk menyeimbangkan kelas dengan cara mengurangi jumlah kelas mayoritas.  
   - Pendekatan ini dipilih karena:
     - Jauh lebih cepat daripada SMOTE pada dataset besar.  
     - Cukup baik untuk baseline dan tugas yang fokus pada pipeline lengkap.

3. **Feature Engineering & Selection**  
   - Contoh fitur turunan sederhana (jika kolom tersedia):
     - `Amt_per_card1 = TransactionAmt / card1`.  
   - Feature selection:
     - Menggunakan **RandomForestClassifier** kecil untuk menghitung feature importance.  
     - Memilih **top-N fitur** (misal 60 fitur teratas) untuk mengurangi dimensi dan mempercepat training.

4. **Modeling**  
   - **Model 1 – RandomForestClassifier**
     - Ensemble berbasis pohon keputusan, cocok untuk data tabular dan robust terhadap skala fitur.  
   - **Model 2 – XGBoostClassifier**
     - Gradient boosting berbasis tree dengan performa baik untuk masalah fraud detection dan klasifikasi tabular.  

5. **Hyperparameter Tuning (Ringan)**  
   - Menggunakan **RandomizedSearchCV** pada XGBoost dengan:
     - Jumlah iterasi kecil (`n_iter` kecil).  
     - Parameter penting saja (mis. `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`).  
   - Tuning dilakukan pada **subset data** agar runtime tetap cepat.

6. **Evaluasi**  
   - Metrik:
     - **AUC (Area Under ROC Curve)**  
     - **F1-score**  
     - Confusion Matrix  
     - Classification report (precision, recall, f1 per kelas)  
   - Visualisasi:
     - ROC curve untuk semua model (RandomForest, XGBoost, XGBoost Tuned).  
     - Heatmap Confusion Matrix.

7. **Submission**  
   - Menggunakan model terbaik (berdasarkan AUC).  
   - Memprediksi `isFraud` untuk `test_transaction.csv`.  
   - Menyimpan file `submission_fraud_detection.csv` dengan kolom:  
     - `TransactionID`  
     - `isFraud` (0/1, berdasarkan threshold 0.5).  

## 5. Struktur Notebook (Per Cell)

Notebook disusun agar mudah diikuti secara bertahap di Google Colab:

1. **Cell 1 – Import Library**  
   - Install dan import library utama: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `matplotlib`, `seaborn`.  
   - Set random seed untuk reproducibility.

2. **Cell 2 – Load Dataset dari Google Drive**  
   - Mount Google Drive.  
   - Membaca `train_transaction.csv` dan `test_transaction.csv` dari folder `MyDrive/Fraud Transaction`.  
   - Menampilkan shape dan beberapa baris pertama (head).

3. **Cell 3 – EDA (Exploratory Data Analysis) Ringan**  
   - Cek `info()` dan persentase missing values (top 20 kolom).  
   - Melihat distribusi target `isFraud` (countplot dan persentase).  
   - Menghitung korelasi sederhana antara subset fitur numerik dan `isFraud` (heatmap kecil) untuk mencegah komputasi berat.

4. **Cell 4 – Preprocessing**  
   - Pisahkan `X` (fitur) dan `y` (`isFraud`).  
   - Deteksi kolom numerik dan kategorikal.  
   - Imputasi median untuk numerik, modus untuk kategorikal.  
   - Encoding kategori menjadi integer codes.  
   - Scaling numerik dengan `StandardScaler`.  

5. **Cell 5 – Menangani Imbalance**  
   - Menggunakan `RandomUnderSampler` untuk menyeimbangkan kelas.  
   - Mencetak distribusi kelas sebelum dan sesudah undersampling.

6. **Cell 6 – Feature Engineering & Selection**  
   - Menambahkan fitur turunan sederhana (jika kolom ada).  
   - Menggunakan RandomForest kecil hanya untuk menghitung feature importance.  
   - Memilih top-N fitur paling penting untuk training model utama.

7. **Cell 7 – Train-Test Split**  
   - Split data menjadi **train** dan **validation** (80/20) dengan `train_test_split`.  
   - Menggunakan `stratify=y` agar proporsi kelas di train/valid tetap seimbang.

8. **Cell 8 – Model ML 1: RandomForest**  
   - Melatih `RandomForestClassifier` dengan parameter moderat (jumlah tree terbatas, depth dibatasi).  
   - Menghitung AUC dan F1 di validation set.  

9. **Cell 9 – Model ML 2: XGBoost**  
   - Melatih `XGBClassifier` dengan konfigurasi ringan (`n_estimators` dan `max_depth` moderat, `tree_method='hist'`).  
   - Menghitung AUC dan F1 di validation set.  

10. **Cell 10 – Hyperparameter Tuning Sederhana**  
    - Menggunakan `RandomizedSearchCV` untuk XGBoost pada subset data (misal 50k sampel).  
    - Menampilkan `best_params_` dan skor CV terbaik.  
    - Melatih ulang XGBoost dengan parameter terbaik di seluruh data train.  

11. **Cell 11 – Evaluasi & Visualisasi**  
    - Menggambar ROC curve untuk:
      - RandomForest  
      - XGBoost  
      - XGBoost Tuned  
    - Menampilkan Confusion Matrix (heatmap) dan classification report untuk masing-masing model.

12. **Cell 12 – Perbandingan Performa Model**  
    - Membuat tabel ringkas berisi:
      - AUC  
      - F1-score  
      - Waktu training (detik)  
    - Mengurutkan model berdasarkan AUC tertinggi.

13. **Cell 13 – Generate Submission**  
    - Memilih model dengan AUC terbaik.  
    - Menghasilkan prediksi untuk `test_transaction.csv` (menggunakan fitur yang sudah dipreprocess & terseleksi).  
    - Menyimpan file `submission_fraud_detection.csv` dengan kolom `TransactionID` dan `isFraud`.

## 6. Metrik Evaluasi (Intuitif)

Beberapa metrik utama yang digunakan:

- **AUC (Area Under ROC Curve)**  
  Mengukur kemampuan model membedakan kelas fraud vs non-fraud di berbagai threshold. Semakin mendekati 1, semakin baik pemisahannya.

- **Confusion Matrix**  
  Menampilkan jumlah:
  - True Positive (TP): fraud terdeteksi fraud.  
  - False Positive (FP): bukan fraud tapi diprediksi fraud.  
  - True Negative (TN): bukan fraud dan benar diprediksi bukan fraud.  
  - False Negative (FN): fraud tapi tidak terdeteksi.  

- **Precision & Recall**  
  - Precision: seberapa banyak prediksi fraud yang benar-benar fraud.  
  - Recall: seberapa banyak fraud yang berhasil ditangkap model.  

- **F1-score**  
  Rata-rata harmonik precision dan recall, berguna saat data tidak seimbang.  
  Secara matematis:  
  $$F_1 = 2 \cdot \frac{precision \cdot recall}{precision + recall}$$

## 7. Cara Menjalankan di Google Colab

1. Upload notebook `.ipynb` ke Google Colab.  
2. Pastikan file:
   - `train_transaction.csv`  
   - `test_transaction.csv`  
   sudah berada di folder `MyDrive/Fraud Transaction` di Google Drive akunmu.  
3. Jalankan setiap cell **berurutan dari atas ke bawah**:
   - Cell 1 → import library.  
   - Cell 2 → load data.  
   - …  
   - Cell 13 → generate submission.  
4. Setelah Cell 13, download `submission_fraud_detection.csv` untuk dikumpulkan / di-upload ke sistem penilaian.

## 8. Pertimbangan Desain & Trade-off Kinerja

- **Mengutamakan runtime cepat**  
  - Menghindari model sangat berat (misal `n_estimators` sangat besar atau SMOTE penuh di 600k baris).  
  - Menggunakan undersampling dan subset data saat tuning untuk mengurangi waktu komputasi.  

- **Akurasi vs Kecepatan**  
  - Undersampling bisa menurunkan informasi pada kelas mayoritas, tetapi sangat mengurangi waktu training.  
  - Hyperparameter tuning dibatasi (iterasi sedikit dan ruang parameter sempit) untuk menjaga runtime tetap wajar.  

- **Skalabilitas**  
  - Pendekatan ini dapat diperluas:
    - Menambah model lain (LightGBM, CatBoost).  
    - Mengganti undersampling dengan SMOTE atau kombinasi oversampling + undersampling jika resource memadai.  
    - Menggunakan cross-validation penuh untuk evaluasi yang lebih stabil.

## 9. Pengembangan Lanjutan

Beberapa ide pengembangan untuk versi berikutnya:

- Menambahkan **deep learning** (misalnya MLP sederhana) untuk membandingkan dengan model tree-based.  
- Menggunakan **SMOTE** atau teknik imbalance lain (SMOTEENN, SMOTETomek) pada subset data yang di-*chunk* agar tetap efisien.  
- Menambahkan lebih banyak **feature engineering** spesifik domain pembayaran/finansial.  
- Menggunakan **explainability tools** (SHAP, permutation importance) untuk menganalisis fitur apa yang paling berkontribusi pada prediksi fraud.  
- Menerapkan **threshold tuning** (bukan hanya 0.5) untuk mengoptimalkan metrik tertentu (mis. recall atau F1 di kelas fraud).

## 10. Kredit

- Proyek ini dibuat untuk keperluan tugas akademik / midterm terkait deteksi fraud menggunakan machine learning.  

