import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import time

st.set_page_config(page_title="Sistem Inspeksi", layout="wide")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_inspeksi_botol.keras")


model = load_model()

st.title("🏭 Sistem Inspeksi Visual Otomatis")
mode = st.sidebar.radio("Input:", ["Video Uji", "Live Webcam"])
start_btn = st.sidebar.button("▶ START")

col1, col2 = st.columns([2, 1])
frame_placeholder = col1.empty()

if start_btn:
    cap = cv2.VideoCapture(0 if mode == "Live Webcam" else "video_uji.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- PERBAIKAN TOTAL: GRAYSCALE ---
        # 1. Resize
        img = cv2.resize(frame, (96, 96))
        # 2. Convert ke Grayscale (Penting untuk model 1-channel)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 3. Normalisasi
        img_input = img_gray.astype('float32') / 255.0
        # 4. Reshape menjadi (1, 96, 96, 1)
        img_input = img_input.reshape(1, 96, 96, 1)

        # --- PREDIKSI ---
        prediction = model.predict(img_input, verbose=0)
        pred = float(prediction[0][0])
        status = "CACAT" if pred > 0.5 else "NORMAL"

        # --- TAMPILAN ---
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)
        col2.metric("Status Terakhir", status)

        time.sleep(0.1)  # Dibuat lebih cepat agar tidak lag
    cap.release()