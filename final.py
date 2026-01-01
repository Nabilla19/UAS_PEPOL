import numpy as np
import pandas as pd
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# Import hanya metrik evaluasi yang dibutuhkan (Akurasi & Presisi)
from sklearn.metrics import accuracy_score, classification_report, precision_score


# ==========================================
# 1. KONFIGURASI DATASET
# ==========================================
FOLDER_DATASET = "./dataset"  # Pastikan folder ini ada dan berisi file audio
TARGET_PER_KELAS = 50         # Jumlah sampel per kelas
SKENARIO_SPLIT = [0.3, 0.2, 0.1] # Skenario Uji: 30%, 20%, 10%


# Array untuk menampung data
X_data = [] 
y_label = [] 
filenames = [] 


print("=== MEMULAI PROSES PREPROCESSING & EKSTRAKSI ===")


# ==========================================
# 2. FUNGSI PREPROCESSING & EKSTRAKSI
# ==========================================
def process_audio(jenis_kucing, kode_label):
    print(f"--> Memproses kelas: {jenis_kucing}...")
    
    for i in range(1, TARGET_PER_KELAS + 1):
        # Asumsi nama file: persia1.wav, minecoon1.wav, dst.
        filename = f"{jenis_kucing}{i}.wav"
        filepath = os.path.join(FOLDER_DATASET, filename)
        
        try:
            # A. LOAD AUDIO
            # sr=None agar sampling rate mengikuti file asli
            audio, sr = librosa.load(filepath, sr=None)
            
            # B. SILENCE REMOVAL (PREPROCESSING)
            # Memotong bagian diam (hening) di bawah 20 desibel
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
            
            # C. EKSTRAKSI FITUR MFCC
            # Mengambil 13 koefisien MFCC
            mfcc = librosa.feature.mfcc(y=audio_trimmed, sr=sr, n_mfcc=13)
            
            # D. NORMALISASI (RATA-RATA)
            # Mengambil rata-rata dari setiap koefisien agar dimensi data seragam
            mfcc_mean = np.mean(mfcc.T, axis=0)
            
            # Simpan data ke list
            X_data.append(mfcc_mean)
            y_label.append(kode_label)
            filenames.append(filename)
            
        except Exception as e:
            print(f"[ERROR] Gagal memproses: {filename}. Error: {e}")


# Jalankan Fungsi (Label 1 = Persia, Label 0 = Maine Coon)
process_audio("persia", 1)     
process_audio("minecoon", 0)  


# Konversi List ke Numpy Array agar bisa diproses Scikit-Learn
X = np.array(X_data)
y = np.array(y_label)


print(f"\nTotal Data Berhasil Diekstrak: {len(X)} sampel.")


# ==========================================
# 3. SIMPAN DATASET GABUNGAN KE CSV
# ==========================================
print("\n[INFO] Menyimpan Matriks Dataset Lengkap...")
nama_kolom = [f"Fitur_MFCC_{i+1}" for i in range(13)]
df_dataset = pd.DataFrame(X, columns=nama_kolom)
df_dataset['Label_Kelas'] = y
df_dataset['Nama_Kelas'] = df_dataset['Label_Kelas'].apply(lambda x: 'Persia' if x == 1 else 'Maine Coon')
df_dataset.insert(0, 'Nama_File', filenames) # Taruh nama file di kolom pertama


# Simpan ke CSV untuk bahan Tabel di Laporan
df_dataset.to_csv("Laporan_Tabel_Dataset_Gabungan.csv", index=False)
print("--> File 'Laporan_Tabel_Dataset_Gabungan.csv' BERHASIL dibuat!")


# ==========================================
# 4. IMPLEMENTASI KLASIFIKASI SVM
# ==========================================
print("\n" + "="*40)
print("=== MULAI KLASIFIKASI SVM ===")
print("="*40)


for test_size in SKENARIO_SPLIT:
    # Hitung persentase untuk penamaan (Contoh: 0.3 -> 70:30)
    latih_pct = int((1 - test_size) * 100)
    uji_pct = int(test_size * 100)
    
    print(f"\n>>> SKENARIO: {latih_pct}% Data Latih : {uji_pct}% Data Uji")
    
    # A. SPLIT DATA
    # random_state=42 agar hasil konsisten setiap kali di-run
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # B. TRAINING MODEL SVM
    # kernel='rbf' sesuai metodologi
    svm_model = SVC(kernel='rbf', C=100, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)
    
    # C. TESTING / PREDIKSI
    y_pred = svm_model.predict(X_test)
    
    # D. HITUNG EVALUASI (HANYA AKURASI & PRESISI)
    # Menggunakan average='macro' agar rata-rata adil untuk kedua kelas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)


    # Tampilkan Hasil di Terminal
    print(f"1. Akurasi   : {acc * 100:.2f}%")
    print(f"2. Presisi   : {prec * 100:.2f}%")


    # E. SIMPAN HASIL DETIL KE CSV (Untuk Tabel Analisis Per Item)
    df_hasil = pd.DataFrame({
        'Label Asli': y_test,
        'Prediksi SVM': y_pred
    })
    
    # Mapping angka ke teks biar bagus di laporan
    map_label = {1: 'Persia', 0: 'Maine Coon'}
    df_hasil['Label Asli'] = df_hasil['Label Asli'].map(map_label)
    df_hasil['Prediksi SVM'] = df_hasil['Prediksi SVM'].map(map_label)
    
    # Kolom Status (Benar/Salah)
    df_hasil['Status'] = np.where(df_hasil['Label Asli'] == df_hasil['Prediksi SVM'], 'Benar', 'Salah')
    
    # Simpan file per skenario
    nama_file_csv = f"Laporan_Hasil_Klasifikasi_{latih_pct}_{uji_pct}.csv"
    df_hasil.to_csv(nama_file_csv, index_label="No Data")
    print(f"--> Detail prediksi disimpan ke: '{nama_file_csv}'")


print("\n" + "="*50)
print("PROSES SELESAI.")
print("Silakan cek folder proyek untuk melihat file CSV yang terbentuk.")
