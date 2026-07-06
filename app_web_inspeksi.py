import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import time
import os

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
status_placeholder = col2.empty()

if start_btn:
    cap = cv2.VideoCapture(0 if mode == "Live Webcam" else "video_uji.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- PREPROCESSING AMAN ---
        img = cv2.resize(frame, (96, 96))
        # Mengubah ke format float32
        img_input = img.astype('float32') / 255.0
        # Memastikan dimensi (1, 96, 96, 3) untuk RGB
        img_input = np.expand_dims(img_input, axis=0)

        # --- PREDIKSI ---
        try:
            prediction = model.predict(img_input, verbose=0)
            # Mengambil nilai tunggal
            pred = float(prediction[0][0])

            status = "CACAT" if pred > 0.5 else "NORMAL"
            conf = (pred * 100) if status == "CACAT" else ((1 - pred) * 100)

            # --- TAMPILAN ---
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            status_placeholder.metric("Status Terakhir", status, f"{conf:.1f}%")

        except Exception as e:
            st.error(f"Error Model: {e}")
            break

        time.sleep(0.5)
    cap.release()