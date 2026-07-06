import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os

st.set_page_config(page_title="Sistem Inspeksi Botol", layout="wide")
st.title("🏭 Sistem Inspeksi Visual Otomatis")

# 1. Load Model
MODEL_PATH = "model_inspeksi_botol.keras"
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    st.error(f"File {MODEL_PATH} tidak ditemukan! Pastikan sudah di-upload ke GitHub.")
    st.stop()

# 2. Pemilihan Input
input_mode = st.radio("Pilih Mode:", ["Video Uji (Cloud)", "Webcam (Lokal)"])

if input_mode == "Webcam (Lokal)":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture("video_uji.mp4")

frame_placeholder = st.empty()

# 3. Loop Utama dengan struktur yang benar
while cap.isOpened():
    ret, frame = cap.read()

    # Jika gagal membaca (frame habis atau kamera mati), loop kembali ke awal
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # 4. Perbaikan ValueError: Konversi ke Grayscale & Resize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (96, 96))
    img_norm = img.reshape(1, 96, 96, 1) / 255.0

    # Prediksi
    pred = model.predict(img_norm, verbose=0)
    label = "NORMAL" if pred[0][0] < 0.5 else "CACAT"

    # Tampilan
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")
    st.write(f"### Status: {label}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()