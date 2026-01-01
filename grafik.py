import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# KONFIGURASI FILE
# ==========================================
# Ganti nama file ini sesuai dengan file yang ada di laptopmu
file_persia = "./dataset/persia1.wav"      # Contoh 1 file persia
file_mainecoon = "./dataset/minecoon1.wav" # Contoh 1 file maine coon

# Fungsi untuk membuat dan menyimpan grafik
def simpan_grafik_sinyal(file_path, judul, nama_output, lakukan_trim=False):
    try:
        # 1. Load Audio
        y, sr = librosa.load(file_path, sr=None)
        
        # 2. Proses Silence Removal (Jika diminta)
        if lakukan_trim:
            # top_db=20 berarti memotong suara di bawah desibel tertentu (dianggap hening)
            y, _ = librosa.effects.trim(y, top_db=20) 
            judul = f"{judul} (Setelah Silence Removal)"
        
        # 3. Plotting (Menggambar Grafik)
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr, color='blue')
        plt.title(judul)
        plt.xlabel("Waktu (detik)")
        plt.ylabel("Amplitudo")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 4. Simpan gambar
        plt.savefig(nama_output, dpi=300) # Simpan jadi file gambar
        print(f"[BERHASIL] Gambar disimpan: {nama_output}")
        plt.close() # Tutup plot biar ram gak penuh
        
    except Exception as e:
        print(f"[ERROR] Gagal memproses {file_path}: {e}")

# ==========================================
# EKSEKUSI PEMBUATAN GAMBAR
# ==========================================

print("Sedang membuat grafik...")

# 1. GAMBAR DATA AWAL (RAW DATA)
simpan_grafik_sinyal(file_persia, "Sinyal Asli - Persia", "grafik_persia_awal.png", lakukan_trim=False)
simpan_grafik_sinyal(file_mainecoon, "Sinyal Asli - Maine Coon", "grafik_mainecoon_awal.png", lakukan_trim=False)

# 2. GAMBAR HASIL PREPROCESSING (SILENCE REMOVAL)
simpan_grafik_sinyal(file_persia, "Sinyal Persia", "grafik_persia_preprocessed.png", lakukan_trim=True)
simpan_grafik_sinyal(file_mainecoon, "Sinyal Maine Coon", "grafik_mainecoon_preprocessed.png", lakukan_trim=True)

print("Selesai! Cek folder proyeck kamu.")
