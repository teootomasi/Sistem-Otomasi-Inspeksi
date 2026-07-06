import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
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

if start_btn:
    # Cek apakah video ada di folder
    if mode == "Video Uji" and not os.path.exists("video_uji.mp4"):
        st.error("File 'video_uji.mp4' tidak ditemukan di folder GitHub!")
    else:
        cap = cv2.VideoCapture(0 if mode == "Live Webcam" else "video_uji.mp4")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.info("Video Selesai atau Tidak Terbaca")
                break

            # Prediksi
            img = cv2.resize(frame, (96, 96))
            img_input = np.expand_dims(img.astype('float32') / 255.0, axis=0)
            pred = float(model.predict(img_input, verbose=0)[0][0])
            status = "CACAT" if pred > 0.5 else "NORMAL"

            # Tampilan
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            col2.metric("Status Terakhir", status)

            time.sleep(0.5)
        cap.release()