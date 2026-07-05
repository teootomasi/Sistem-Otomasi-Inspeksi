import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import gdown

# --- KONFIGURASI ---
MODEL_PATH = "model_inspeksi_botol.keras"
# ID dari Google Drive Anda
DRIVE_FILE_ID = '15tS5XwWmiE0qDNq6ydParHAa_0sYO2AB'

st.set_page_config(page_title="Sistem Inspeksi Botol", layout="wide")
st.title("🏭 Sistem Inspeksi Visual Otomatis")


# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        st.info("Mengunduh model dari cloud, harap tunggu...")
        url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)


# --- INISIALISASI SESSION ---
if 'ok_count' not in st.session_state: st.session_state.ok_count = 0
if 'ng_count' not in st.session_state: st.session_state.ng_count = 0

# --- PROSES UTAMA ---
try:
    model = load_model_from_drive()
    col1, col2 = st.columns([2, 1])
    frame_placeholder = col1.empty()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Preprocessing (Resize & Normalisasi)
        img_res = cv2.resize(frame, (96, 96))
        img_norm = img_res.reshape(1, 96, 96, 1) / 255.0

        # 2. Deteksi CNN
        pred = model.predict(img_norm, verbose=0)
        hasil = "NORMAL" if pred[0][0] < 0.5 else "CACAT"

        # 3. Counter & HMI
        if hasil == "NORMAL":
            st.session_state.ok_count += 1
        else:
            st.session_state.ng_count += 1

        # 4. Update Tampilan
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        col2.metric("Botol OK", st.session_state.ok_count)
        col2.metric("Botol CACAT", st.session_state.ng_count)

        if cv2.waitKey(100) & 0xFF == ord('q'): break
    cap.release()
except Exception as e:
    st.error(f"Sistem Error: {e}")