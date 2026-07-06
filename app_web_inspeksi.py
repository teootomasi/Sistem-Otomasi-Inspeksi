import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os

# --- KONFIGURASI ---
st.set_page_config(page_title="Sistem Inspeksi Botol", layout="wide")
st.title("🏭 Sistem Inspeksi Visual Otomatis")

# --- LOAD MODEL ---
MODEL_PATH = "model_inspeksi_botol.keras"
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    st.error(f"File {MODEL_PATH} tidak ditemukan!")
    st.stop()

# --- SIDEBAR & KONTROL ---
mode = st.sidebar.radio("Pilih Mode:", ["Video Uji (Cloud)", "Webcam (Lokal)"])
stop_btn = st.sidebar.button("STOP APLIKASI")

if stop_btn:
    st.warning("Aplikasi dihentikan.")
    st.stop()

# --- SETUP KAMERA/VIDEO ---
if mode == "Webcam (Lokal)":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture("video_uji.mp4")

# --- LAYOUT TAMPILAN ---
col1, col2 = st.columns([2, 1])
frame_placeholder = col1.empty()
status_placeholder = col2.empty()

# --- LOOP UTAMA ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # PREPROCESSING CNN
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray, (96, 96))
    img_norm = img_resized.reshape(1, 96, 96, 1) / 255.0

    # PREDIKSI
    pred = model.predict(img_norm, verbose=0)
    label = "NORMAL" if pred[0][0] < 0.5 else "CACAT"
    confidence = (1 - pred[0][0]) if label == "NORMAL" else pred[0][0]

    # UPDATE TAMPILAN (Tidak menumpuk)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")

    with status_placeholder.container():
        st.subheader("Hasil Inspeksi")
        if label == "NORMAL":
            st.success(f"### Status: {label}")
        else:
            st.error(f"### Status: {label}")
        st.metric("Tingkat Keyakinan", f"{confidence*100:.2f}%")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()