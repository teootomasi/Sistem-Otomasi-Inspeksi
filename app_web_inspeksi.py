import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os

# --- KONFIGURASI ---
MODEL_PATH = "model_inspeksi_botol.keras"
VIDEO_FILE = "video_uji.mp4"

st.set_page_config(page_title="Sistem Inspeksi Botol", layout="wide")
st.title("🏭 Sistem Inspeksi Visual Otomatis")

# 1. Pastikan model ada (Harus sudah ada di GitHub)
if not os.path.exists(MODEL_PATH):
    st.error(f"File {MODEL_PATH} tidak ditemukan di folder! Silakan upload file model ke GitHub Anda.")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)

# 2. Pilihan Input
input_type = st.radio("Pilih Sumber Input:", ["Video Uji (Cloud/Lokal)", "Webcam (Hanya Lokal)"])

# 3. Logika Kamera/Video
if input_type == "Webcam (Hanya Lokal)":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(VIDEO_FILE)

frame_placeholder = st.empty()
if 'ok_count' not in st.session_state: st.session_state.ok_count = 0
if 'ng_count' not in st.session_state: st.session_state.ng_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop jika video habis
        continue

    # Deteksi
    img = cv2.resize(frame, (96, 96))
    img_norm = img.reshape(1, 96, 96, 1) / 255.0
    pred = model.predict(img_norm, verbose=0)

    label = "NORMAL" if pred[0][0] < 0.5 else "CACAT"
    if label == "NORMAL":
        st.session_state.ok_count += 1
    else:
        st.session_state.ng_count += 1

    # Tampilan
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")
    st.write(f"### Status Terakhir: {label}")
    st.metric("Botol OK", st.session_state.ok_count)
    st.metric("Botol CACAT", st.session_state.ng_count)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()