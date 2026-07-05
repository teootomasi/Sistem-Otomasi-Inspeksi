import os

# PAKSA TENSORFLOW MENGGUNAKAN MODE LEGACY KERAS UNTUK COMPATIBILITY MODEL LAMA
os.environ['TF_USE_LEGACY_KERAS'] = '1'

"""
=============================================================================
APLIKASI INSPEKSI VISUAL OTOMATIS - BOTOL PLASTIK (VERSI KAMERA LAPTOP)
=============================================================================
Aplikasi ini dijalankan LANGSUNG DI LAPTOP (bukan di Colab/Kaggle) karena
membutuhkan akses ke webcam laptop Anda secara langsung.

CARA KERJA (3 tahap, dilakukan berurutan):

  TAHAP 1 - KUMPULKAN DATA
    Ambil foto botol Anda sendiri lewat webcam: beberapa foto kondisi
    NORMAL, beberapa foto kondisi CACAT (goresan/retak/dsb yang Anda buat
    sendiri di botol contoh, atau botol yang memang sudah cacat).
    Semakin banyak & bervariasi fotonya (sudut, jarak, cahaya), semakin
    bagus hasil deteksinya nanti.

  TAHAP 2 - LATIH MODEL
    Model CNN dilatih menggunakan foto-foto yang sudah dikumpulkan.

  TAHAP 3 - DETEKSI REAL-TIME
    Kamera menyala, arahkan ke botol, hasil "NORMAL" / "CACAT" beserta
    tingkat keyakinan (%) akan muncul langsung di layar secara live.

-----------------------------------------------------------------------------
CARA MENJALANKAN DI LAPTOP:

1. Install Python 3.9-3.12 (jika belum ada): https://www.python.org/downloads/
   Saat install, centang "Add Python to PATH".

2. Buka Command Prompt / Terminal, install library yang dibutuhkan:
       pip install opencv-python tensorflow numpy

3. Simpan file ini di sebuah folder, misalnya: aplikasi_inspeksi_botol.py

4. Jalankan:
       python aplikasi_inspeksi_botol.py

5. Ikuti menu yang muncul di terminal (pilih 1, 2, 3, 4, atau 5).

Sebuah jendela kamera akan terbuka saat mengumpulkan data / deteksi.
Tekan tombol sesuai instruksi yang tampil di jendela tersebut.
=============================================================================
"""

import sys
import cv2
import numpy as np

DATASET_DIR = "dataset_botol"
MODEL_PATH = "model_inspeksi_botol.keras"
IMG_SIZE = 96
KAMERA_INDEX = 1  # ganti ke 1 jika laptop punya lebih dari satu kamera dan yang terbuka salah


# =========================================================================
# UTILITAS
# =========================================================================
def pastikan_folder_dataset():
    os.makedirs(f"{DATASET_DIR}/normal", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/cacat", exist_ok=True)


def hitung_data():
    n_normal = len(os.listdir(f"{DATASET_DIR}/normal")) if os.path.exists(f"{DATASET_DIR}/normal") else 0
    n_cacat = len(os.listdir(f"{DATASET_DIR}/cacat")) if os.path.exists(f"{DATASET_DIR}/cacat") else 0
    return n_normal, n_cacat


def buka_kamera():
    # Mengganti nomor indeks menjadi nama file video Anda secara spesifik
    nama_file_video = "video_uji.mp4"
    cap = cv2.VideoCapture(nama_file_video)
    if not cap.isOpened():
        print(f"\n[GAGAL] Video '{nama_file_video}' tidak ditemukan.")
        return None
    return cap


def preprocessing_frame(frame):
    """Resize, grayscale, normalisasi - sama seperti preprocessing pada laporan Bab III/IV."""
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    norm = gray.astype("float32") / 255.0
    return norm[..., np.newaxis]


# =========================================================================
# TAHAP 1 - KUMPULKAN DATA DARI KAMERA
# =========================================================================
def mode_kumpulkan_data(label):
    pastikan_folder_dataset()
    folder = f"{DATASET_DIR}/{label}"
    existing = len(os.listdir(folder))
    cap = buka_kamera()
    if cap is None:
        return

    print(f"\n=== KUMPULKAN DATA: {label.upper()} ===")
    print("  [SPASI] = simpan foto ini      [Q] = selesai & kembali ke menu\n")

    count = existing
    while True:
        ret, frame = cap.read()

        # PERBAIKAN: Jika video habis, putar ulang dari awal (Looping otomatis)
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        tampil = frame.copy()
        cv2.putText(tampil, f"Mode: KUMPULKAN DATA - {label.upper()}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(tampil, f"Sudah tersimpan: {count} foto", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(tampil, "[SPASI]=simpan  [Q]=selesai", (10, tampil.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow("Kumpulkan Data - Inspeksi Botol", tampil)

        # PERBAIKAN: Mengubah jeda dari 1ms menjadi 30ms agar video berjalan stabil
        key = cv2.waitKey(30) & 0xFF

        if key == ord(' '):
            count += 1
            fname = f"{folder}/{label}_{count:04d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"  Tersimpan: {fname}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total data '{label}' sekarang: {count} foto.")


# =========================================================================
# TAHAP 2 - LATIH MODEL
# =========================================================================
def muat_dataset():
    X, y = [], []
    for label_idx, folder in enumerate(["normal", "cacat"]):  # 0=normal, 1=cacat
        path = f"{DATASET_DIR}/{folder}"
        if not os.path.exists(path):
            continue
        for fname in os.listdir(path):
            img = cv2.imread(os.path.join(path, fname))
            if img is None:
                continue
            X.append(preprocessing_frame(img))
            y.append(label_idx)
    return np.array(X), np.array(y)


def bangun_model():
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def mode_latih_model():
    n_normal, n_cacat = hitung_data()
    print(f"\n=== LATIH MODEL ===")
    print(f"Data tersedia -> Normal: {n_normal} foto | Cacat: {n_cacat} foto")

    if n_normal < 15 or n_cacat < 15:
        print("\n[PERINGATAN] Data masih terlalu sedikit (disarankan minimal ~20-30 foto per")
        print("kategori, diambil dari beberapa sudut/jarak/cahaya berbeda) agar model bisa")
        print("belajar dengan baik. Silakan kumpulkan data lebih banyak dulu (menu 1 & 2).")
        lanjut = input("Tetap lanjut latih dengan data yang ada sekarang? (y/n): ").strip().lower()
        if lanjut != "y":
            return

    print("\nMemuat dataset ...")
    X, y = muat_dataset()
    if len(X) == 0:
        print("[GAGAL] Tidak ada data ditemukan. Kumpulkan data dulu lewat menu 1 & 2.")
        return

    print(f"Total data: {len(X)} foto. Memulai pelatihan model ...\n")
    model = bangun_model()
    model.fit(X, y, epochs=25, batch_size=8, validation_split=0.2, verbose=2)

    model.save(MODEL_PATH)
    print(f"\nModel berhasil disimpan sebagai '{MODEL_PATH}'.")
    print("Silakan lanjut ke menu 4 (Deteksi Real-Time).")


# =========================================================================
# TAHAP 3 - DETEKSI REAL-TIME
# =========================================================================
def mode_deteksi_realtime():
    if not os.path.exists(MODEL_PATH):
        print("\n[GAGAL] Model belum ada. Lakukan menu 1, 2, lalu 3 (kumpulkan data & latih) dulu.")
        return

    import tensorflow as tf
    print("\nMemuat model ...")
    model = tf.keras.models.load_model(MODEL_PATH)

    cap = buka_kamera()
    if cap is None:
        return

    print("\n=== DETEKSI REAL-TIME ===")
    print("Arahkan kamera ke botol. Tekan [Q] untuk keluar.\n")

    while True:
        ret, frame = cap.read()

        # PERBAIKAN: Deteksi juga otomatis looping jika video uji habis
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        inp = preprocessing_frame(frame)[np.newaxis, ...]
        prob = float(model.predict(inp, verbose=0)[0][0])  # 0..1, mendekati 1 = cacat

        if prob >= 0.5:
            label = "CACAT"
            keyakinan = prob
            warna = (0, 0, 255)  # merah (BGR)
        else:
            label = "NORMAL"
            keyakinan = 1 - prob
            warna = (0, 200, 0)  # hijau (BGR)

        tampil = frame.copy()
        h, w = tampil.shape[:2]
        cv2.rectangle(tampil, (0, 0), (w - 1, h - 1), warna, 8)  # border indikator status
        cv2.rectangle(tampil, (0, 0), (330, 60), warna, -1)
        cv2.putText(tampil, f"{label} ({keyakinan * 100:.1f}%)", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(tampil, "[Q] = keluar", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow("Deteksi Real-Time - Inspeksi Botol", tampil)

        # PERBAIKAN: Mengubah jeda menjadi 30ms agar pemutaran video deteksi mulus
        if (cv2.waitKey(30) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# =========================================================================
# MENU UTAMA
# =========================================================================
def tampilkan_menu():
    n_normal, n_cacat = hitung_data()
    model_ada = "SUDAH ADA" if os.path.exists(MODEL_PATH) else "belum dilatih"
    print("\n" + "=" * 60)
    print(" SISTEM INSPEKSI VISUAL OTOMATIS - BOTOL PLASTIK (Kamera Laptop)")
    print("=" * 60)
    print(f" Data terkumpul  : Normal = {n_normal} foto | Cacat = {n_cacat} foto")
    print(f" Status model    : {model_ada}")
    print("-" * 60)
    print(" 1. Kumpulkan foto botol NORMAL (buka kamera)")
    print(" 2. Kumpulkan foto botol CACAT  (buka kamera)")
    print(" 3. Latih model dari data yang terkumpul")
    print(" 4. Deteksi Real-Time (buka kamera + model)")
    print(" 5. Keluar")
    print("=" * 60)


def main():
    pastikan_folder_dataset()
    while True:
        tampilkan_menu()
        pilihan = input("Pilih menu (1-5): ").strip()

        if pilihan == "1":
            mode_kumpulkan_data("normal")
        elif pilihan == "2":
            mode_kumpulkan_data("cacat")
        elif pilihan == "3":
            mode_latih_model()
        elif pilihan == "4":
            mode_deteksi_realtime()
        elif pilihan == "5":
            print("Selesai. Sampai jumpa!")
            sys.exit(0)
        else:
            print("Pilihan tidak dikenali, coba lagi.")


if __name__ == "__main__":
    main()