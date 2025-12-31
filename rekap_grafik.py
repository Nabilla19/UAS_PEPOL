import matplotlib.pyplot as plt

# ==========================================
# 1. DATA HASIL PENGUJIAN (REAL)
# ==========================================
# Data ini diambil dari hasil run terminal kamu sebelumnya
skenario = ['70 : 30', '80 : 20', '90 : 10']
akurasi = [93.33, 95.00, 90.00] 

# ==========================================
# 2. KONFIGURASI GRAFIK
# ==========================================
plt.figure(figsize=(8, 5)) # Ukuran gambar (Lebar x Tinggi)

# Membuat Bar Chart dengan warna berbeda tiap batang
warna_bar = ['#4c72b0', '#55a868', '#c44e52'] # Biru, Hijau, Merah
bars = plt.bar(skenario, akurasi, color=warna_bar, width=0.6, zorder=3)

# Judul dan Label Sumbu
plt.title('Grafik Perbandingan Akurasi Klasifikasi SVM\n(Persia vs Maine Coon)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Skenario Pembagian Data (Latih : Uji)', fontsize=12, labelpad=10)
plt.ylabel('Akurasi (%)', fontsize=12, labelpad=10)

# Mengatur batas sumbu Y agar tampilan rapi (0 sampai 110)
plt.ylim(0, 110)

# ==========================================
# 3. MENAMPILKAN ANGKA DI ATAS BATANG
# ==========================================
for bar in bars:
    tinggi = bar.get_height()
    # Menulis angka persentase tepat di atas batang
    plt.text(bar.get_x() + bar.get_width()/2., tinggi + 2,
             f'{tinggi}%', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# ==========================================
# 4. FINISHING & SIMPAN
# ==========================================
# Menambah garis bantu horizontal (grid) di belakang batang
plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

plt.tight_layout() # Merapikan margin otomatis

# Simpan ke file gambar
nama_file = 'grafik_akurasi_final.png'
plt.savefig(nama_file, dpi=300) 

print(f"BERHASIL! Gambar grafik tersimpan dengan nama: {nama_file}")
print("Silakan buka folder proyek dan masukkan gambar tersebut ke Laporan Word.")
plt.show()
