Python
import os
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile

DATASET_DIR = "dataset_botol"
MODEL_PATH = "model_inspeksi_botol.keras"
IMG_SIZE = 96

# Membuat folder dataset otomatis jika belum ada
os.makedirs(f"{DATASET_DIR}/normal", exist_ok=True)
os.makedirs(f"{DATASET_DIR}/cacat", exist_ok=True)


def hitung_data():
    """Fungsi untuk menghitung jumlah koleksi foto sampel di folder dataset."""
    n_normal = len(os.listdir(f"{DATASET_DIR}/normal")) if os.path.exists(f"{DATASET_DIR}/normal") else 0
    n_cacat = len(os.listdir(f"{DATASET_DIR}/cacat")) if os.path.exists(f"{DATASET_DIR}/cacat") else 0
    return n_normal, n_cacat


def preprocessing_frame(frame):
    """Fungsi standarisasi gambar sebelum dimasukkan ke dalam Otak AI CNN."""
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    norm = gray.astype("float32") / 255.0
    return norm[..., np.newaxis]


# --- KONFIGURASI HALAMAN WEB STREAMLIT ---
st.set_page_config(page_title="Inspeksi Visual Botol", layout="wide")

st.title("🏭 Sistem Inspeksi Visual Otomatis - Botol Plastik")
st.write("Aplikasi web berbasis AI untuk mendeteksi kecacatan pada lini produksi botol.")

# --- SIDEBAR (PANEL STATUS SISTEM) ---
st.sidebar.header("📊 Status Sistem")
n_normal, n_cacat = hitung_data()
st.sidebar.metric("Foto Botol Normal", f"{n_normal} gambar")
st.sidebar.metric("Foto Botol Cacat", f"{n_cacat} gambar")

model_ada = os.path.exists(MODEL_PATH)
if model_ada:
    st.sidebar.success("🧠 Model AI: SIAP DIGUNAKAN")
else:
    st.sidebar.warning("⚠️ Model AI: BELUM DILATIH")

# --- PEMBAGIAN MENU TAB ---
tab1, tab2, tab3 = st.tabs(["📁 1 & 2. Kumpulkan Data", "⚙️ 3. Latih Model AI", "🔍 4. Deteksi Real-Time"])

# --- TAB 1: PENGUMPULAN DATA ---
with tab1:
    st.header("Ambil Sampel Data dari Kamera")
    st.write("Gunakan tombol di bawah untuk membuka kamera laptop dan merekam gambar sampel botol Anda.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📸 Ambil Data NORMAL", use_container_width=True):
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Kamera laptop tidak terdeteksi!")
            else:
                st.info(
                    "Jendela kamera dibuka. Hadapkan botol NORMAL, lalu tekan [SPASI] di keyboard untuk mengambil foto. Tekan [Q] jika sudah selesai.")
                count = n_normal
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    tampil = frame.copy()
                    cv2.putText(tampil, f"Simpan NORMAL (Tersimpan: {count})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)
                    cv2.imshow("Kumpulkan Data - Inspeksi Botol", tampil)
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord(' '):
                        count += 1
                        cv2.imwrite(f"{DATASET_DIR}/normal/normal_{count:04d}.jpg", frame)
                    elif key == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
                st.rerun()

    with col2:
        if st.button("🚨 Ambil Data CACAT", use_container_width=True):
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Kamera laptop tidak terdeteksi!")
            else:
                st.info(
                    "Jendela kamera dibuka. Hadapkan botol CACAT, lalu tekan [SPASI] di keyboard untuk mengambil foto. Tekan [Q] jika sudah selesai.")
                count = n_cacat
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    tampil = frame.copy()
                    cv2.putText(tampil, f"Simpan CACAT (Tersimpan: {count})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)
                    cv2.imshow("Kumpulkan Data - Inspeksi Botol", tampil)
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord(' '):
                        count += 1
                        cv2.imwrite(f"{DATASET_DIR}/cacat/cacat_{count:04d}.jpg", frame)
                    elif key == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
                st.rerun()

# --- TAB 2: PELATIHAN MODEL CNN ---
with tab2:
    st.header("Pelatihan Otak Model (CNN)")
    st.write("Sistem akan melatih Convolutional Neural Network (CNN) berdasarkan kategori foto.")

    if st.button("🚀 Mulai Training Model Sekarang", type="primary"):
        if n_normal < 5 or n_cacat < 5:
            st.error(
                f"Data belum cukup! Anda baru memiliki Normal: {n_normal}, Cacat: {n_cacat}. Kumpulkan minimal masing-masing 5-15 foto di Tab 1.")
        else:
            with st.spinner("Sedang melatih model AI... Mohon tunggu..."):
                X, y = [], []
                for label_idx, folder in enumerate(["normal", "cacat"]):
                    path = f"{DATASET_DIR}/{folder}"
                    for fname in os.listdir(path):
                        img = cv2.imread(os.path.join(path, fname))
                        if img is not None:
                            X.append(preprocessing_frame(img))
                            y.append(label_idx)
                X, y = np.array(X), np.array(y)

                from tensorflow.keras import layers, models

                model = models.Sequential([
                    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
                    layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                    layers.MaxPooling2D((2, 2)),
                    layers.Flatten(),
                    layers.Dense(32, activation="relu"),
                    layers.Dense(1, activation="sigmoid"),
                ])
                model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

                model.fit(X, y, epochs=15, batch_size=8, verbose=0)
                model.save(MODEL_PATH)
                st.success("🎉 Pelatihan Berhasil Selesai! Model diperbarui.")
                st.rerun()

# --- TAB 3: DETEKSI REAL-TIME & UPLOAD FILE ---
with tab3:
    st.header("Pengujian Deteksi Visual AI")

    if not model_ada:
        st.error("Silakan lakukan pengumpulan data dan jalankan menu latihan di Tab 2 terlebih dahulu.")
    else:
        # Pilihan Mode Input menggunakan Radio Button
        mode_input = st.radio(
            "🔌 Pilih Sumber Input Deteksi:",
            ["🎥 Kamera Real-Time (Laptop)", "📁 Upload File (Foto atau Video Offline)"],
            horizontal=True
        )

        # Inisialisasi Session State status pemrosesan video/kamera
        if "proses_aktif" not in st.session_state:
            st.session_state.proses_aktif = False

        # --- MODE 1: KAMERA REAL-TIME ---
        if mode_input == "🎥 Kamera Real-Time (Laptop)":
            st.write("Klik tombol di bawah ini untuk mengaktifkan pemindaian langsung lewat kamera.")

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("👁️ Nyalakan Kamera Inspeksi", use_container_width=True, type="primary"):
                    st.session_state.proses_aktif = True
            with col_btn2:
                if st.button("⏹️ Matikan Kamera", use_container_width=True):
                    st.session_state.proses_aktif = False

            if st.session_state.proses_aktif:
                model = tf.keras.models.load_model(MODEL_PATH)
                cap = cv2.VideoCapture(0)
                frame_placeholder = st.empty()

                while st.session_state.proses_aktif:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Gagal membaca kamera. Pastikan tidak terkunci aplikasi lain.")
                        break

                    inp = preprocessing_frame(frame)[np.newaxis, ...]
                    prob = float(model.predict(inp, verbose=0)[0][0])

                    if prob >= 0.5:
                        label, warna, keyakinan = "CACAT", (0, 0, 255), prob
                    else:
                        label, warna, keyakinan = "NORMAL", (0, 200, 0), 1 - prob

                    tampil = frame.copy()
                    h, w = tampil.shape[:2]
                    cv2.rectangle(tampil, (0, 0), (w - 1, h - 1), warna, 8)
                    cv2.rectangle(tampil, (0, 0), (320, 50), warna, -1)
                    cv2.putText(tampil, f"{label} ({keyakinan * 100:.1f}%)", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2)

                    tampil_rgb = cv2.cvtColor(tampil, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(tampil_rgb, channels="RGB", use_container_width=True)

                cap.release()
                frame_placeholder.empty()

        # --- MODE 2: UPLOAD FILE (FOTO ATAU VIDEO) ---
        elif mode_input == "📁 Upload File (Foto atau Video Offline)":
            st.write("Unggah file video (`.mp4`, `.avi`) atau file foto (`.jpg`, `.jpeg`, `.png`) dari botol Anda.")

            uploaded_file = st.file_uploader("Pilih file foto atau video...",
                                             type=["mp4", "avi", "mov", "jpg", "jpeg", "png"])

            if uploaded_file is not None:
                file_ext = uploaded_file.name.split(".")[-1].lower()
                is_image = file_ext in ["jpg", "jpeg", "png"]

                if is_image:
                    # --- PROSES INPUT FOTO ---
                    model = tf.keras.models.load_model(MODEL_PATH)

                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, 1)

                    inp = preprocessing_frame(frame)[np.newaxis, ...]
                    prob = float(model.predict(inp, verbose=0)[0][0])

                    if prob >= 0.5:
                        label, warna, keyakinan = "CACAT", (0, 0, 255), prob
                    else:
                        label, warna, keyakinan = "NORMAL", (0, 200, 0), 1 - prob

                    tampil = frame.copy()
                    h, w = tampil.shape[:2]
                    cv2.rectangle(tampil, (0, 0), (w - 1, h - 1), warna, 8)
                    cv2.rectangle(tampil, (0, 0), (320, 50), warna, -1)
                    cv2.putText(tampil, f"{label} ({keyakinan * 100:.1f}%)", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2)

                    tampil_rgb = cv2.cvtColor(tampil, cv2.COLOR_BGR2RGB)
                    st.image(tampil_rgb, channels="RGB", use_container_width=True)
                    st.success(f"Hasil Analisis Foto: Botol dideteksi {label}!")

                else:
                    # --- PROSES INPUT VIDEO ---
                    col_vbtn1, col_vbtn2 = st.columns(2)
                    with col_vbtn1:
                        if st.button("🚀 Mulai Analisis Video", use_container_width=True, type="primary"):
                            st.session_state.proses_aktif = True
                    with col_vbtn2:
                        if st.button("⏹️ Hentikan Analisis", use_container_width=True):
                            st.session_state.proses_aktif = False

                    if st.session_state.proses_aktif:
                        model = tf.keras.models.load_model(MODEL_PATH)

                        tfile = tempfile.NamedTemporaryFile(delete=False)
                        tfile.write(uploaded_file.read())

                        cap = cv2.VideoCapture(tfile.name)
                        frame_placeholder = st.empty()

                        while cap.isOpened() and st.session_state.proses_aktif:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            inp = preprocessing_frame(frame)[np.newaxis, ...]
                            prob = float(model.predict(inp, verbose=0)[0][0])

                            if prob >= 0.5:
                                label, warna, keyakinan = "CACAT", (0, 0, 255), prob
                            else:
                                label, warna, keyakinan = "NORMAL", (0, 200, 0), 1 - prob

                            tampil = frame.copy()
                            h, w = tampil.shape[:2]
                            cv2.rectangle(tampil, (0, 0), (w - 1, h - 1), warna, 8)
                            cv2.rectangle(tampil, (0, 0), (320, 50), warna, -1)
                            cv2.putText(tampil, f"{label} ({keyakinan * 100:.1f}%)", (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8, (255, 255, 255), 2)

                            tampil_rgb = cv2.cvtColor(tampil, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(tampil_rgb, channels="RGB", use_container_width=True)

                        cap.release()
                        tfile.close()
                        os.unlink(tfile.name)
                        st.session_state.proses_aktif = False
                        st.success("🏁 Pemrosesan file video selesai sepenuhnya!")Python
import os
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile

DATASET_DIR = "dataset_botol"
MODEL_PATH = "model_inspeksi_botol.keras"
IMG_SIZE = 96

# Membuat folder dataset otomatis jika belum ada
os.makedirs(f"{DATASET_DIR}/normal", exist_ok=True)
os.makedirs(f"{DATASET_DIR}/cacat", exist_ok=True)


def hitung_data():
    """Fungsi untuk menghitung jumlah koleksi foto sampel di folder dataset."""
    n_normal = len(os.listdir(f"{DATASET_DIR}/normal")) if os.path.exists(f"{DATASET_DIR}/normal") else 0
    n_cacat = len(os.listdir(f"{DATASET_DIR}/cacat")) if os.path.exists(f"{DATASET_DIR}/cacat") else 0
    return n_normal, n_cacat


def preprocessing_frame(frame):
    """Fungsi standarisasi gambar sebelum dimasukkan ke dalam Otak AI CNN."""
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    norm = gray.astype("float32") / 255.0
    return norm[..., np.newaxis]


# --- KONFIGURASI HALAMAN WEB STREAMLIT ---
st.set_page_config(page_title="Inspeksi Visual Botol", layout="wide")

st.title("🏭 Sistem Inspeksi Visual Otomatis - Botol Plastik")
st.write("Aplikasi web berbasis AI untuk mendeteksi kecacatan pada lini produksi botol.")

# --- SIDEBAR (PANEL STATUS SISTEM) ---
st.sidebar.header("📊 Status Sistem")
n_normal, n_cacat = hitung_data()
st.sidebar.metric("Foto Botol Normal", f"{n_normal} gambar")
st.sidebar.metric("Foto Botol Cacat", f"{n_cacat} gambar")

model_ada = os.path.exists(MODEL_PATH)
if model_ada:
    st.sidebar.success("🧠 Model AI: SIAP DIGUNAKAN")
else:
    st.sidebar.warning("⚠️ Model AI: BELUM DILATIH")

# --- PEMBAGIAN MENU TAB ---
tab1, tab2, tab3 = st.tabs(["📁 1 & 2. Kumpulkan Data", "⚙️ 3. Latih Model AI", "🔍 4. Deteksi Real-Time"])

# --- TAB 1: PENGUMPULAN DATA ---
with tab1:
    st.header("Ambil Sampel Data dari Kamera")
    st.write("Gunakan tombol di bawah untuk membuka kamera laptop dan merekam gambar sampel botol Anda.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📸 Ambil Data NORMAL", use_container_width=True):
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Kamera laptop tidak terdeteksi!")
            else:
                st.info(
                    "Jendela kamera dibuka. Hadapkan botol NORMAL, lalu tekan [SPASI] di keyboard untuk mengambil foto. Tekan [Q] jika sudah selesai.")
                count = n_normal
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    tampil = frame.copy()
                    cv2.putText(tampil, f"Simpan NORMAL (Tersimpan: {count})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)
                    cv2.imshow("Kumpulkan Data - Inspeksi Botol", tampil)
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord(' '):
                        count += 1
                        cv2.imwrite(f"{DATASET_DIR}/normal/normal_{count:04d}.jpg", frame)
                    elif key == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
                st.rerun()

    with col2:
        if st.button("🚨 Ambil Data CACAT", use_container_width=True):
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Kamera laptop tidak terdeteksi!")
            else:
                st.info(
                    "Jendela kamera dibuka. Hadapkan botol CACAT, lalu tekan [SPASI] di keyboard untuk mengambil foto. Tekan [Q] jika sudah selesai.")
                count = n_cacat
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    tampil = frame.copy()
                    cv2.putText(tampil, f"Simpan CACAT (Tersimpan: {count})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)
                    cv2.imshow("Kumpulkan Data - Inspeksi Botol", tampil)
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord(' '):
                        count += 1
                        cv2.imwrite(f"{DATASET_DIR}/cacat/cacat_{count:04d}.jpg", frame)
                    elif key == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
                st.rerun()

# --- TAB 2: PELATIHAN MODEL CNN ---
with tab2:
    st.header("Pelatihan Otak Model (CNN)")
    st.write("Sistem akan melatih Convolutional Neural Network (CNN) berdasarkan kategori foto.")

    if st.button("🚀 Mulai Training Model Sekarang", type="primary"):
        if n_normal < 5 or n_cacat < 5:
            st.error(
                f"Data belum cukup! Anda baru memiliki Normal: {n_normal}, Cacat: {n_cacat}. Kumpulkan minimal masing-masing 5-15 foto di Tab 1.")
        else:
            with st.spinner("Sedang melatih model AI... Mohon tunggu..."):
                X, y = [], []
                for label_idx, folder in enumerate(["normal", "cacat"]):
                    path = f"{DATASET_DIR}/{folder}"
                    for fname in os.listdir(path):
                        img = cv2.imread(os.path.join(path, fname))
                        if img is not None:
                            X.append(preprocessing_frame(img))
                            y.append(label_idx)
                X, y = np.array(X), np.array(y)

                from tensorflow.keras import layers, models

                model = models.Sequential([
                    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
                    layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                    layers.MaxPooling2D((2, 2)),
                    layers.Flatten(),
                    layers.Dense(32, activation="relu"),
                    layers.Dense(1, activation="sigmoid"),
                ])
                model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

                model.fit(X, y, epochs=15, batch_size=8, verbose=0)
                model.save(MODEL_PATH)
                st.success("🎉 Pelatihan Berhasil Selesai! Model diperbarui.")
                st.rerun()

# --- TAB 3: DETEKSI REAL-TIME & UPLOAD FILE ---
with tab3:
    st.header("Pengujian Deteksi Visual AI")

    if not model_ada:
        st.error("Silakan lakukan pengumpulan data dan jalankan menu latihan di Tab 2 terlebih dahulu.")
    else:
        # Pilihan Mode Input menggunakan Radio Button
        mode_input = st.radio(
            "🔌 Pilih Sumber Input Deteksi:",
            ["🎥 Kamera Real-Time (Laptop)", "📁 Upload File (Foto atau Video Offline)"],
            horizontal=True
        )

        # Inisialisasi Session State status pemrosesan video/kamera
        if "proses_aktif" not in st.session_state:
            st.session_state.proses_aktif = False

        # --- MODE 1: KAMERA REAL-TIME ---
        if mode_input == "🎥 Kamera Real-Time (Laptop)":
            st.write("Klik tombol di bawah ini untuk mengaktifkan pemindaian langsung lewat kamera.")

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("👁️ Nyalakan Kamera Inspeksi", use_container_width=True, type="primary"):
                    st.session_state.proses_aktif = True
            with col_btn2:
                if st.button("⏹️ Matikan Kamera", use_container_width=True):
                    st.session_state.proses_aktif = False

            if st.session_state.proses_aktif:
                model = tf.keras.models.load_model(MODEL_PATH)
                cap = cv2.VideoCapture(0)
                frame_placeholder = st.empty()

                while st.session_state.proses_aktif:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Gagal membaca kamera. Pastikan tidak terkunci aplikasi lain.")
                        break

                    inp = preprocessing_frame(frame)[np.newaxis, ...]
                    prob = float(model.predict(inp, verbose=0)[0][0])

                    if prob >= 0.5:
                        label, warna, keyakinan = "CACAT", (0, 0, 255), prob
                    else:
                        label, warna, keyakinan = "NORMAL", (0, 200, 0), 1 - prob

                    tampil = frame.copy()
                    h, w = tampil.shape[:2]
                    cv2.rectangle(tampil, (0, 0), (w - 1, h - 1), warna, 8)
                    cv2.rectangle(tampil, (0, 0), (320, 50), warna, -1)
                    cv2.putText(tampil, f"{label} ({keyakinan * 100:.1f}%)", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    tampil_rgb = cv2.cvtColor(tampil, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(tampil_rgb, channels="RGB", use_container_width=True)

                cap.release()
                frame_placeholder.empty()

        # --- MODE 2: UPLOAD FILE (FOTO ATAU VIDEO) ---
        elif mode_input == "📁 Upload File (Foto atau Video Offline)":
            st.write("Unggah file video (`.mp4`, `.avi`) atau file foto (`.jpg`, `.jpeg`, `.png`) dari botol Anda.")

            uploaded_file = st.file_uploader("Pilih file foto atau video...", type=["mp4", "avi", "mov", "jpg", "jpeg", "png"])

            if uploaded_file is not None:
                file_ext = uploaded_file.name.split(".")[-1].lower()
                is_image = file_ext in ["jpg", "jpeg", "png"]

                if is_image:
                    # --- PROSES INPUT FOTO ---
                    model = tf.keras.models.load_model(MODEL_PATH)

                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, 1)

                    inp = preprocessing_frame(frame)[np.newaxis, ...]
                    prob = float(model.predict(inp, verbose=0)[0][0])

                    if prob >= 0.5:
                        label, warna, keyakinan = "CACAT", (0, 0, 255), prob
                    else:
                        label, warna, keyakinan = "NORMAL", (0, 200, 0), 1 - prob

                    tampil = frame.copy()
                    h, w = tampil.shape[:2]
                    cv2.rectangle(tampil, (0, 0), (w - 1, h - 1), warna, 8)
                    cv2.rectangle(tampil, (0, 0), (320, 50), warna, -1)
                    cv2.putText(tampil, f"{label} ({keyakinan * 100:.1f}%)", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    tampil_rgb = cv2.cvtColor(tampil, cv2.COLOR_BGR2RGB)
                    st.image(tampil_rgb, channels="RGB", use_container_width=True)
                    st.success(f"Hasil Analisis Foto: Botol dideteksi {label}!")

                else:
                    # --- PROSES INPUT VIDEO ---
                    col_vbtn1, col_vbtn2 = st.columns(2)
                    with col_vbtn1:
                        if st.button("🚀 Mulai Analisis Video", use_container_width=True, type="primary"):
                            st.session_state.proses_aktif = True
                    with col_vbtn2:
                        if st.button("⏹️ Hentikan Analisis", use_container_width=True):
                            st.session_state.proses_aktif = False

                    if st.session_state.proses_aktif:
                        model = tf.keras.models.load_model(MODEL_PATH)

                        tfile = tempfile.NamedTemporaryFile(delete=False)
                        tfile.write(uploaded_file.read())

                        cap = cv2.VideoCapture(tfile.name)
                        frame_placeholder = st.empty()

                        while cap.isOpened() and st.session_state.proses_aktif:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            inp = preprocessing_frame(frame)[np.newaxis, ...]
                            prob = float(model.predict(inp, verbose=0)[0][0])

                            if prob >= 0.5:
                                label, warna, keyakinan = "CACAT", (0, 0, 255), prob
                            else:
                                label, warna, keyakinan = "NORMAL", (0, 200, 0), 1 - prob

                            tampil = frame.copy()
                            h, w = tampil.shape[:2]
                            cv2.rectangle(tampil, (0, 0), (w - 1, h - 1), warna, 8)
                            cv2.rectangle(tampil, (0, 0), (320, 50), warna, -1)
                            cv2.putText(tampil, f"{label} ({keyakinan * 100:.1f}%)", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                            tampil_rgb = cv2.cvtColor(tampil, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(tampil_rgb, channels="RGB", use_container_width=True)

                        cap.release()
                        tfile.close()
                        os.unlink(tfile.name)
                        st.session_state.proses_aktif = False
                        st.success("🏁 Pemrosesan file video selesai sepenuhnya!")