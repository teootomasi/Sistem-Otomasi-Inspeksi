import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import time
import os

# --- KONFIGURASI ---
st.set_page_config(page_title="Sistem Inspeksi Botol", layout="wide")


# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_inspeksi_botol.keras")


model = load_model()


# --- FUNGSI LOG ---
def save_log(new_data):
    file_path = "laporan_produksi.csv"
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        pd.concat([df_existing, new_data], ignore_index=True).to_csv(file_path, index=False)
    else:
        new_data.to_csv(file_path, index=False)


# --- UI SIDEBAR ---
st.sidebar.title("Kontrol Sistem")
mode = st.sidebar.radio("Input:", ["Video Uji", "Live Webcam"])
start_btn = st.sidebar.button("▶ START")
stop_btn = st.sidebar.button("⏹ STOP")

# --- UI UTAMA ---
st.title("🏭 Sistem Inspeksi Visual Otomatis")
tab1, tab2 = st.tabs(["Monitoring Live", "Laporan Produksi"])

with tab1:
    col1, col2 = st.columns([2, 1])
    frame_placeholder = col1.empty()
    status_placeholder = col2.empty()

    if start_btn:
        cap = cv2.VideoCapture(0 if mode == "Live Webcam" else "video_uji.mp4")
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret: break

            # Prediksi
            img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 96))
            img_norm = img.reshape(1, 96, 96, 1) / 255.0
            pred = model.predict(img_norm, verbose=0)
            status = "NORMAL" if pred[0][0] < 0.5 else "CACAT"

            # Tampilan Kamera
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            status_placeholder.metric("Status Terakhir", status)

            # Save CSV
            new_log = pd.DataFrame({"Waktu": [time.strftime("%H:%M:%S")], "Status": [status]})
            save_log(new_log)

            time.sleep(0.1)
        cap.release()

with tab2:
    st.subheader("📋 Laporan Produksi")
    if os.path.exists("laporan_produksi.csv"):
        df = pd.read_csv("laporan_produksi.csv")
        st.dataframe(df.tail(20), use_container_width=True)