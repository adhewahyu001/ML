import cv2
import os

# Buka video
video_path = 'Sheep pregnancy scanning_ Non pregnant ewe'  # Ganti dengan path video Anda
cap = cv2.VideoCapture(video_path)

# Mengecek apakah video berhasil dibuka
if not cap.isOpened():
    print("Gagal membuka video")
    exit()

# Buat folder untuk menyimpan gambar jika belum ada
output_folder = 'non-pregnant'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print("Folder", output_folder, "telah dibuat")

# Looping untuk membaca setiap frame
frame_count = 0
while True:
    ret, frame = cap.read()

    # Keluar loop jika tidak ada frame lagi
    if not ret:
        break

    # Simpan setiap frame sebagai gambar
    nama_file = os.path.join(output_folder, "frame_" + str(frame_count) + ".jpg")
    cv2.imwrite(nama_file, frame)

    frame_count += 1

# Tutup video
cap.release()
cv2.destroyAllWindows()

print("Proses selesai. Total frame yang diambil:", frame_count)
