import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import time
import os

# Konfigurasi Layout
st.set_page_config(page_title="Sistem Inspeksi Botol", layout="wide")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_inspeksi_botol.keras")


model = load_model()


# Fungsi Simpan Log
def save_log(status):
    file_path = "laporan_produksi.csv"
    new_data = pd.DataFrame({"Waktu": [time.strftime("%H:%M:%S")], "Status": [status]})
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        pd.concat([df, new_data], ignore_index=True).to_csv(file_path, index=False)
    else:
        new_data.to_csv(file_path, index=False)


# UI Sidebar
st.sidebar.title("Kontrol Sistem")
mode = st.sidebar.radio("Input:", ["Video Uji", "Live Webcam"])
start_btn = st.sidebar.button("▶ START")

st.title("🏭 Sistem Inspeksi Visual Otomatis")

# Membuat 2 Sheet (Tabs)
tab1, tab2 = st.tabs(["Monitoring Live", "Laporan Produksi"])

with tab1:
    col1, col2 = st.columns([2, 1])
    frame_placeholder = col1.empty()
    status_placeholder = col2.empty()

    if start_btn:
        cap = cv2.VideoCapture(0 if mode == "Live Webcam" else "video_uji.mp4")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Preprocessing (Grayscale sesuai model Anda)
            img = cv2.resize(frame, (96, 96))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_input = img_gray.reshape(1, 96, 96, 1).astype('float32') / 255.0

            # Prediksi
            pred = float(model.predict(img_input, verbose=0)[0][0])
            status = "CACAT" if pred > 0.5 else "NORMAL"

            # Tampilan UI
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            status_placeholder.metric("Status Terakhir", status)

            save_log(status)
            time.sleep(0.1)
        cap.release()

with tab2:
    st.subheader("📋 Data Log Produksi")
    if os.path.exists("laporan_produksi.csv"):
        df = pd.read_csv("laporan_produksi.csv")
        st.dataframe(df.tail(20), use_container_width=True)