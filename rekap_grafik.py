import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. INPUT DATA HASIL PENGUJIAN (EDIT DISINI)
# ==========================================
# Masukkan angka dari hasil run terminalmu!
data_hasil = {
    'Akurasi':   [93.33, 95.00, 90.00],  # Ganti dengan angka aslimu
    'Presisi':   [94.74, 96.15, 92.86]   # Ganti dengan angka aslimu
}

labels_skenario = ['70 : 30', '80 : 20', '90 : 10']
warna_bar = ['#4c72b0', '#55a868', '#c44e52'] # Biru, Hijau, Merah

# ==========================================
# 2. FUNGSI PEMBUAT GRAFIK OTOMATIS
# ==========================================
def buat_grafik(judul, data_list, label_y, nama_file_simpan):
    plt.figure(figsize=(8, 6)) # Ukuran gambar
    
    # Membuat Bar Chart dengan warna beda-beda
    bars = plt.bar(labels_skenario, data_list, color=warna_bar, width=0.6, zorder=3)
    
    # Judul dan Label Sumbu
    plt.title(judul, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Skenario Pembagian Data (Latih : Uji)', fontsize=12, labelpad=10)
    plt.ylabel(label_y, fontsize=12, labelpad=10)
    
    # Batas sumbu Y
    plt.ylim(0, 115)
    
    # Grid garis putus-putus di belakang
    plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    
    # Menampilkan Angka di Atas Batang
    for bar in bars:
        tinggi = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., tinggi + 1.5,
                 f'{tinggi}%', 
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(nama_file_simpan, dpi=300)
    print(f"[BERHASIL] Grafik tersimpan: {nama_file_simpan}")
    plt.show()

# ==========================================
# 3. EKSEKUSI PEMBUATAN 2 GRAFIK
# ==========================================

# A. Buat Grafik Akurasi
buat_grafik(
    judul='Grafik Perbandingan Akurasi Klasifikasi SVM\n(Persia vs Maine Coon)',
    data_list=data_hasil['Akurasi'],
    label_y='Akurasi (%)',
    nama_file_simpan='grafik_akurasi_final.png'
)

# B. Buat Grafik Presisi
buat_grafik(
    judul='Grafik Perbandingan Presisi Klasifikasi SVM\n(Persia vs Maine Coon)',
    data_list=data_hasil['Presisi'],
    label_y='Presisi (%)',
    nama_file_simpan='grafik_presisi_final.png'
)
