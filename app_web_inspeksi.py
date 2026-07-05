import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os

# --- KONFIGURASI ---
# Pastikan file 'model_inspeksi_botol.keras' sudah ada di folder yang sama
MODEL_PATH = "model_inspeksi_botol.keras"

# --- HMI INITIALIZATION ---
st.set_page_config(page_title="Sistem Inspeksi Botol", layout="wide")
st.title("🏭 Sistem Inspeksi Visual Otomatis")

if 'ok_count' not in st.session_state: st.session_state.ok_count = 0
if 'ng_count' not in st.session_state: st.session_state.ng_count = 0

# --- PROSES MODEL ---
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)

    col1, col2 = st.columns([2, 1])
    frame_placeholder = col1.empty()

    # Buka Kamera
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. PREPROCESSING (Sesuai Rancangan)
        img_res = cv2.resize(frame, (96, 96))
        img_norm = img_res.reshape(1, 96, 96, 1) / 255.0

        # 2. DETEKSI (CNN)
        pred = model.predict(img_norm, verbose=0)
        hasil = "NORMAL" if pred[0][0] < 0.5 else "CACAT"

        # 3. KEPUTUSAN & COUNTER
        if hasil == "NORMAL":
            st.session_state.ok_count += 1
        else:
            st.session_state.ng_count += 1

        # 4. TAMPILKAN DI HMI (Update UI)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        col2.metric("Botol OK", st.session_state.ok_count)
        col2.metric("Botol CACAT", st.session_state.ng_count)

        # Delay kecil untuk stabilitas loop
        if cv2.waitKey(100) & 0xFF == ord('q'): break
    cap.release()
else:
    st.error(f"Error: File '{MODEL_PATH}' tidak ditemukan di folder GitHub Anda!")
    st.info("Pastikan Anda sudah mengupload model ke GitHub.")