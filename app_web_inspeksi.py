import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import gdown
import time

MODEL_PATH = "model_inspeksi_botol.keras"
DRIVE_FILE_ID = '15tS5XwWmiE0qDNq6ydParHAa_0sYO2AB'

st.set_page_config(page_title="Sistem Inspeksi Botol", layout="wide")
st.title("🏭 Sistem Inspeksi Visual Otomatis")


# --- FUNGSI DOWNLOAD & VERIFIKASI ---
@st.cache_resource
def prepare_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Model belum ditemukan, memulai unduhan dari Cloud...")
        url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)

        # Tambahkan jeda/verifikasi
        time.sleep(2)

    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        return None


# --- PROSES UTAMA ---
model = prepare_model()

if model is not None:
    if 'ok_count' not in st.session_state: st.session_state.ok_count = 0
    if 'ng_count' not in st.session_state: st.session_state.ng_count = 0

    col1, col2 = st.columns([2, 1])
    frame_placeholder = col1.empty()

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Proses deteksi
        img_res = cv2.resize(frame, (96, 96))
        img_norm = img_res.reshape(1, 96, 96, 1) / 255.0
        pred = model.predict(img_norm, verbose=0)

        hasil = "NORMAL" if pred[0][0] < 0.5 else "CACAT"
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
else:
    st.error("Gagal memuat model. Pastikan file di Google Drive dapat diakses publik.")