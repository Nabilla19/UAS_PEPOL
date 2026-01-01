import numpy as np
import pandas as pd
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. KONFIGURASI DATASET
# ==========================================
FOLDER_DATASET = "./dataset"  # Pastikan folder ini ada
TARGET_PER_KELAS = 50         # Jumlah sampel per kelas
SKENARIO_SPLIT = [0.3, 0.2, 0.1] # 30%, 20%, 10% data uji

# Array untuk menampung data
X_data = [] 
y_label = [] 
filenames = [] # Menyimpan nama file untuk laporan

print("=== MEMULAI PROSES PREPROCESSING & EKSTRAKSI ===")

# ==========================================
# 2. FUNGSI PREPROCESSING & EKSTRAKSI
# ==========================================
def process_audio(jenis_kucing, kode_label):
    print(f"--> Memproses kelas: {jenis_kucing}...")
    
    for i in range(1, TARGET_PER_KELAS + 1):
        filename = f"{jenis_kucing}{i}.wav"
        filepath = os.path.join(FOLDER_DATASET, filename)
        
        try:
            # A. LOAD AUDIO
            audio, sr = librosa.load(filepath, sr=None)
            
            # B. SILENCE REMOVAL (PREPROCESSING)
            # Memotong bagian diam (hening) di bawah 20 desibel
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
            
            # C. EKSTRAKSI FITUR MFCC
            mfcc = librosa.feature.mfcc(y=audio_trimmed, sr=sr, n_mfcc=13)
            
            # D. NORMALISASI (RATA-RATA)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            
            # Simpan data
            X_data.append(mfcc_mean)
            y_label.append(kode_label)
            filenames.append(filename)
            
        except Exception as e:
            print(f"[ERROR] Gagal: {filename}. Pastikan file ada!")

# Jalankan Fungsi (Label 1 = Persia, Label 0 = Maine Coon)
process_audio("persia", 1)     
process_audio("minecoon", 0)  

# Konversi ke Numpy Array
X = np.array(X_data)
y = np.array(y_label)

print(f"\nTotal Data: {len(X)} sampel berhasil diekstrak.")

# ==========================================
# 3. SIMPAN DATASET GABUNGAN KE CSV (UNTUK LAPORAN)
# ==========================================
print("\n[INFO] Menyimpan Matriks Dataset Lengkap...")
nama_kolom = [f"Fitur_{i+1}" for i in range(13)]
df_dataset = pd.DataFrame(X, columns=nama_kolom)
df_dataset['Label_Kelas'] = y
df_dataset['Nama_Kelas'] = df_dataset['Label_Kelas'].apply(lambda x: 'Persia' if x == 1 else 'Maine Coon')
df_dataset.insert(0, 'Nama_File', filenames) # Taruh nama file di depan

# Simpan ke CSV
df_dataset.to_csv("Laporan_Tabel_Dataset_Gabungan.csv", index=False)
print("--> File 'Laporan_Tabel_Dataset_Gabungan.csv' BERHASIL dibuat! (Pakai ini untuk Tabel 2.2 & 2.3)")


# ==========================================
# 4. IMPLEMENTASI KLASIFIKASI SVM
# ==========================================
print("\n=== MULAI KLASIFIKASI SVM ===")

for test_size in SKENARIO_SPLIT:
    # Hitung persentase untuk nama file
    latih_pct = int((1 - test_size) * 100)
    uji_pct = int(test_size * 100)
    
    print(f"\nSkenario: {latih_pct}% Latih : {uji_pct}% Uji")
    
    # A. SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # B. TRAINING MODEL SVM
    svm_model = SVC(kernel='rbf', C=100, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)
    
    # C. TESTING / PREDIKSI
    y_pred = svm_model.predict(X_test)
    
    # D. HITUNG AKURASI
    accuracy = accuracy_score(y_test, y_pred)
    print(f"--> Akurasi: {accuracy * 100:.2f}%")

    # [TAMBAHAN BARU] Tampilkan Precision, Recall, F1-Score
    print("\nLaporan Klasifikasi Lengkap:")
    print(classification_report(y_test, y_pred, target_names=['Maine Coon', 'Persia']))
    
    # E. SIMPAN HASIL DETIL KE CSV (UNTUK LAPORAN)
    # Membuat tabel perbandingan Label Asli vs Prediksi
    df_hasil = pd.DataFrame({
        'Label Asli': y_test,
        'Prediksi SVM': y_pred
    })
    
    # Mapping angka ke teks biar bagus di laporan
    map_label = {1: 'Persia', 0: 'Maine Coon'}
    df_hasil['Label Asli'] = df_hasil['Label Asli'].map(map_label)
    df_hasil['Prediksi SVM'] = df_hasil['Prediksi SVM'].map(map_label)
    
    # Cek Status Benar/Salah
    df_hasil['Status'] = np.where(df_hasil['Label Asli'] == df_hasil['Prediksi SVM'], 'Benar', 'Salah')
    
    # Simpan file
    nama_file_csv = f"Laporan_Hasil_Klasifikasi_{latih_pct}_{uji_pct}.csv"
    df_hasil.to_csv(nama_file_csv, index_label="No Data")
    print(f"--> File '{nama_file_csv}' BERHASIL dibuat! (Pakai ini untuk Tabel Hasil Klasifikasi)")

print("\n" + "="*50)
print("PROSES SELESAI SEMUA. SILAKAN CEK FOLDER PROYEK UNTUK FILE CSV LAPORAN.")
