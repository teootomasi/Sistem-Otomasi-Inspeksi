import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import gdown
import time

# --- KONFIGURASI ---
MODEL_PATH = "model_inspeksi_botol.keras"
DRIVE_FILE_ID = '1SHwoXXkCyn5FN9c5ll0b-rqsc_NRSbGo'  # ID Baru Anda

st.set_page_config(page_title="Sistem Inspeksi Visual", layout="wide")
st.title("🏭 Sistem Inspeksi Visual Otomatis")


# --- LOAD MODEL (Dengan Verifikasi) ---
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Mengunduh model dari Google Drive...")
        url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
        time.sleep(2)  # Jeda untuk memastikan file tertulis

    return tf.keras.models.load_model(MODEL_PATH)


# --- PROSES UTAMA ---
try:
    model = load_my_model()

    # Inisialisasi Counter
    if 'ok_count' not in st.session_state: st.session_state.ok_count = 0
    if 'ng_count' not in st.session_state: st.session_state.ng_count = 0

    col1, col2 = st.columns([2, 1])
    frame_placeholder = col1.empty()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Preprocessing & Prediksi
        img_res = cv2.resize(frame, (96, 96))
        img_norm = img_res.reshape(1, 96, 96, 1) / 255.0
        pred = model.predict(img_norm, verbose=0)

        # Logika Keputusan
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
except Exception as e:
    st.error(f"Terjadi error: {e}")