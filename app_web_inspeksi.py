import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import gdown

# --- KONFIGURASI ---
MODEL_PATH = "model_inspeksi_botol.keras"
DRIVE_FILE_ID = 'MASUKKAN_ID_FILE_GOOGLE_DRIVE_ANDA_DI_SINI'

# --- INISIALISASI SESSION STATE (Untuk HMI/Counter) ---
if 'ok_count' not in st.session_state: st.session_state.ok_count = 0
if 'ng_count' not in st.session_state: st.session_state.ng_count = 0


# --- FUNGSI DOWNLOAD & MODEL ---
def download_model(file_id, output_path):
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)


if DRIVE_FILE_ID != 'MASUKKAN_ID_FILE_GOOGLE_DRIVE_ANDA_DI_SINI':
    download_model(DRIVE_FILE_ID, MODEL_PATH)

st.title("🏭 Sistem Inspeksi Visual Otomatis")

# --- ALUR SISTEM (Sesuai Bab IV.3) ---
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)

    # Layout HMI
    col1, col2 = st.columns([2, 1])
    frame_placeholder = col1.empty()

    # Kamera
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. PREPROCESSING (Resize & Normalize)
        img_res = cv2.resize(frame, (96, 96))
        img_norm = img_res.reshape(1, 96, 96, 1) / 255.0

        # 2. KLASIFIKASI (CNN Model)
        pred = model.predict(img_norm, verbose=0)
        hasil = "NORMAL" if pred[0][0] < 0.5 else "CACAT"

        # 3. PENGAMBILAN KEPUTUSAN & COUNTER
        if hasil == "NORMAL":
            st.session_state.ok_count += 1
        else:
            st.session_state.ng_count += 1

        # 4. UPDATE HMI (Tampilan Real-time)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        col2.metric("Botol OK", st.session_state.ok_count)
        col2.metric("Botol CACAT", st.session_state.ng_count)

        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
else:
    st.warning("Model belum terunduh. Periksa ID Google Drive Anda.")