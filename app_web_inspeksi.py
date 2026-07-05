import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os

st.set_page_config(page_title="Sistem Inspeksi Botol", layout="wide")
st.title("🏭 Sistem Inspeksi Visual Otomatis")

# 1. Pastikan model sudah ada (model sudah di-push ke GitHub)
MODEL_PATH = "model_inspeksi_botol.keras"
if not os.path.exists(MODEL_PATH):
    st.error(f"Error: {MODEL_PATH} tidak ditemukan. Pastikan file sudah di-upload ke GitHub!")
    st.stop()

# 2. Load model langsung
model = tf.keras.models.load_model(MODEL_PATH)

# 3. Pilihan Input (Dual Mode)
input_mode = st.radio("Pilih Mode:", ["Video Uji (Cloud)", "Webcam (Lokal)"])

if input_mode == "Webcam (Lokal)":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture("video_uji.mp4")

frame_placeholder = st.empty()

# 4. Loop Utama
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop jika video habis
        continue

    img = cv2.resize(frame, (96, 96))
    img_norm = img.reshape(1, 96, 96, 1) / 255.0
    pred = model.predict(img_norm, verbose=0)

    label = "NORMAL" if pred[0][0] < 0.5 else "CACAT"

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")
    st.write(f"### Status: {label}")

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()