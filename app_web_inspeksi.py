import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os

# Konfigurasi
DATASET_DIR = "dataset_botol"
MODEL_PATH = "model_inspeksi_botol.keras"
IMG_SIZE = 96


# Fungsi Pendukung
def preprocessing_frame(frame):
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    norm = gray.astype("float32") / 255.0
    return norm[..., np.newaxis]


# UI Utama
st.set_page_config(page_title="Inspeksi Visual Botol", layout="wide")
st.title("🏭 Sistem Inspeksi Visual Otomatis")

tab1, tab2, tab3 = st.tabs(["📁 1. Kumpulkan Data", "⚙️ 2. Latih Model", "🔍 3. Deteksi"])

with tab1:
    st.header("Kumpulkan Data")
    tipe = st.radio("Pilih kategori:", ["normal", "cacat"])
    foto_input = st.camera_input("Ambil foto botol")
    if foto_input:
        st.success("Foto berhasil diambil!")
        # (Tambahkan logika penyimpanan file di sini)

with tab2:
    st.header("Latih Model")
    if st.button("Mulai Training"):
        st.info("Proses training akan berjalan di sini...")

with tab3:
    st.header("Deteksi Real-Time")
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        camera_file = st.camera_input("Arahkan kamera ke botol")
        if camera_file:
            # Konversi byte ke gambar OpenCV
            bytes_data = camera_file.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            # Deteksi
            inp = preprocessing_frame(cv2_img)[np.newaxis, ...]
            prob = float(model.predict(inp, verbose=0)[0][0])

            if prob >= 0.5:
                st.error(f"Hasil: CACAT ({prob * 100:.1f}%)")
            else:
                st.success(f"Hasil: NORMAL ({(1 - prob) * 100:.1f}%)")
    else:
        st.warning("Model belum ditemukan.")