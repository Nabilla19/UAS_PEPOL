import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. INPUT DATA HASIL PENGUJIAN (EDIT DISINI)
# ==========================================
# Masukkan angka dari hasil run terminalmu!
# Urutan: [Skenario 70:30, Skenario 80:20, Skenario 90:10]

data_hasil = {
    'Akurasi':   [93.33, 95.00, 90.00],  # Ganti dengan angka aslimu
    'Presisi':   [94.74, 96.15, 92.86]   # Ganti dengan angka aslimu
}

labels_skenario = ['70 : 30', '80 : 20', '90 : 10']

# ==========================================
# 2. KONFIGURASI GRAFIK GROUPED BAR
# ==========================================
plt.figure(figsize=(9, 6)) # Ukuran gambar

x = np.arange(len(labels_skenario))  # Lokasi label sumbu X
width = 0.35  # Lebar batang (lebih lebar karena cuma 2 batang)

# Membuat 2 batang berdampingan
rects1 = plt.bar(x - width/2, data_hasil['Akurasi'], width, label='Akurasi', color='#4c72b0') # Biru
rects2 = plt.bar(x + width/2, data_hasil['Presisi'], width, label='Presisi', color='#55a868') # Hijau

# ==========================================
# 3. LABEL DAN JUDUL
# ==========================================
plt.ylabel('Nilai Persentase (%)', fontsize=12)
plt.xlabel('Skenario Pembagian Data', fontsize=12, labelpad=10)
plt.title('Perbandingan Akurasi & Presisi Model SVM', fontsize=14, fontweight='bold', pad=20)
plt.xticks(x, labels_skenario, fontsize=11)
plt.ylim(0, 115)  # Batas atas sumbu Y supaya angka tidak kepotong
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2) # Posisi legenda di atas
plt.grid(axis='y', linestyle='--', alpha=0.3)

# ==========================================
# 4. FUNGSI MENAMPILKAN ANGKA DI ATAS BATANG
# ==========================================
def autolabel(rects):
    """Menempelkan label angka di atas setiap batang"""
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # Jarak teks dari batang
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# ==========================================
# 5. SIMPAN DAN TAMPILKAN
# ==========================================
plt.tight_layout()
nama_file = 'grafik_akurasi_presisi_final.png'
plt.savefig(nama_file, dpi=300)

print(f"BERHASIL! Grafik tersimpan sebagai: {nama_file}")
plt.show()
