import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
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
status_placeholder = col2.empty()

if start_btn:
    cap = cv2.VideoCapture(0 if mode == "Live Webcam" else "video_uji.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- PERBAIKAN DIMENSI (INPUT SHAPE) ---
        # 1. Resize ke ukuran yang sama saat model dilatih (96x96)
        img = cv2.resize(frame, (96, 96))

        # 2. Pastikan input memiliki 3 Channel (RGB) - Tambahkan cv2.cvtColor jika perlu
        # Jika model dilatih dengan RGB:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. Normalisasi
        img_input = img_rgb.astype('float32') / 255.0

        # 4. Tambahkan dimensi batch (1, 96, 96, 3)
        img_input = np.expand_dims(img_input, axis=0)

        # --- PREDIKSI ---
        try:
            prediction = model.predict(img_input, verbose=0)
            pred = float(prediction[0][0])

            status = "CACAT" if pred > 0.5 else "NORMAL"

            # --- TAMPILAN ---
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            status_placeholder.metric("Status Terakhir", status)

            time.sleep(0.5)
        except Exception as e:
            st.error(f"Error Model: {e}")
            st.write("Detail input:", img_input.shape)  # Untuk debugging
            break

    cap.release()