import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import gdown

# --- KONFIGURASI ---
MODEL_PATH = "model_inspeksi_botol.keras"
# Ganti dengan ID file Google Drive Anda
DRIVE_FILE_ID = 'MASUKKAN_ID_FILE_GOOGLE_DRIVE_ANDA_DI_SINI'

st.set_page_config(page_title="Sistem Inspeksi Botol", layout="wide")
st.title("🏭 Sistem Inspeksi Visual Otomatis")


# --- FUNGSI DOWNLOAD OTOMATIS ---
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        st.info("Mengunduh model dari penyimpanan cloud...")
        url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)


# --- INITIALIZATION ---
if 'ok_count' not in st.session_state: st.session_state.ok_count = 0
if 'ng_count' not in st.session_state: st.session_state.ng_count = 0

# --- PROSES INSPEKSI ---
try:
    model = load_model_from_drive()

    col1, col2 = st.columns([2, 1])
    frame_placeholder = col1.empty()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Preprocessing
        img_res = cv2.resize(frame, (96, 96))
        img_norm = img_res.reshape(1, 96, 96, 1) / 255.0

        # Deteksi (CNN)
        pred = model.predict(img_norm, verbose=0)
        hasil = "NORMAL" if pred[0][0] < 0.5 else "CACAT"

        # Counter
        if hasil == "NORMAL":
            st.session_state.ok_count += 1
        else:
            st.session_state.ng_count += 1

        # Update HMI
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        col2.metric("Botol OK", st.session_state.ok_count)
        col2.metric("Botol CACAT", st.session_state.ng_count)

        if cv2.waitKey(100) & 0xFF == ord('q'): break
    cap.release()
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
    st.info("Pastikan ID Google Drive benar dan file diset 'Anyone with the link'.")