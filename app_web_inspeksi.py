import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import gdown

# --- KONFIGURASI ---
DATASET_DIR = "dataset_botol"
MODEL_PATH = "model_inspeksi_botol.keras"
IMG_SIZE = 96
# GANTI ID_FILE_ANDA dengan ID file dari Google Drive
DRIVE_FILE_ID = 'MASUKKAN_ID_FILE_GOOGLE_DRIVE_ANDA_DI_SINI'

# --- FUNGSI DOWNLOAD MODEL ---
def download_model_dari_drive(file_id, output_path):
    if not os.path.exists(output_path):
        with st.spinner("Sedang mengunduh model dari Google Drive..."):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output_path, quiet=False)
            st.success("Model berhasil diunduh!")

# --- INISIALISASI ---
os.makedirs(f"{DATASET_DIR}/normal", exist_ok=True)
os.makedirs(f"{DATASET_DIR}/cacat", exist_ok=True)

st.set_page_config(page_title="Inspeksi Visual Botol", layout="wide")
st.title("🏭 Sistem Inspeksi Visual Otomatis")

# Unduh model otomatis jika file belum ada
if DRIVE_FILE_ID != 'MASUKKAN_ID_FILE_GOOGLE_DRIVE_ANDA_DI_SINI':
    download_model_dari_drive(DRIVE_FILE_ID, MODEL_PATH)

# --- FUNGSI PREPROCESSING ---
def preprocessing_frame(frame):
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    norm = gray.astype("float32") / 255.0
    return norm[..., np.newaxis]

# --- TAB ANTARMUKA ---
tab1, tab2, tab3 = st.tabs(["📁 1. Kumpulkan Data", "⚙️ 2. Latih Model", "🔍 3. Deteksi"])

with tab1:
    st.header("Kumpulkan Data")
    tipe = st.radio("Pilih kategori:", ["normal", "cacat"])
    foto_input = st.camera_input("Ambil foto botol")
    if foto_input:
        st.success("Foto berhasil diambil!")

with tab2:
    st.header("Latih Model")
    st.info("Fitur training di Cloud memerlukan waktu. Pastikan data sudah cukup.")
    if st.button("Mulai Training"):
        st.warning("Proses training akan memakan waktu, harap tunggu...")

with tab3:
    st.header("Deteksi Real-Time")
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        camera_file = st.camera_input("Arahkan kamera ke botol")
        if camera_file:
            bytes_data = camera_file.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            inp = preprocessing_frame(cv2_img)[np.newaxis, ...]
            prob = float(model.predict(inp, verbose=0)[0][0])
            if prob >= 0.5:
                st.error(f"Hasil: CACAT ({prob * 100:.1f}%)")
            else:
                st.success(f"Hasil: NORMAL ({(1-prob) * 100:.1f}%)")
    else:
        st.warning("Model belum ditemukan. Pastikan ID Google Drive benar.")