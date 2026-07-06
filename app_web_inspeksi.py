import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import time
import os

# Konfigurasi Layout
st.set_page_config(page_title="Smart Factory Inspection", layout="wide")


# Load Model - Pastikan file model ada di folder yang sama
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_inspeksi_botol.keras")


model = load_model()


# Fungsi Simpan Data
def save_log(new_data):
    file_path = "laporan_produksi.csv"
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        pd.concat([df_existing, new_data], ignore_index=True).to_csv(file_path, index=False)
    else:
        new_data.to_csv(file_path, index=False)


# Sidebar
st.sidebar.title("Kontrol Sistem")
mode = st.sidebar.radio("Input:", ["Video Uji", "Live Webcam"])
start_btn = st.sidebar.button("▶ START")
stop_btn = st.sidebar.button("⏹ STOP")

st.title("🏭 Sistem Inspeksi Visual Otomatis")
col1, col2 = st.columns([2, 1])
frame_placeholder = col1.empty()
status_placeholder = col2.empty()

if start_btn:
    cap = cv2.VideoCapture(0 if mode == "Live Webcam" else "video_uji.mp4")

    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret: break

        # Preprocessing: Sesuaikan dengan input model Anda (Gray/RGB)
        # Jika model dilatih dengan gambar berwarna, hapus cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(frame, (96, 96))
        img_norm = img.reshape(1, 96, 96, 3) / 255.0  # Gunakan 3 untuk RGB

        pred = model.predict(img_norm, verbose=0)[0][0]

        # LOGIKA PREDIKSI:
        # Jika model output 0-1, sesuaikan thresholdnya.
        # Jika masih salah, tukar posisi "NORMAL" dan "CACAT"
        status = "CACAT" if pred > 0.5 else "NORMAL"
        conf = (pred * 100) if status == "CACAT" else ((1 - pred) * 100)

        # Update UI
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        status_placeholder.metric("Status Terakhir", status, f"{conf:.2f}%")

        # Log ke CSV
        new_log = pd.DataFrame({"Waktu": [time.strftime("%H:%M:%S")], "Status": [status], "Conf": [f"{conf:.2f}%"]})
        save_log(new_log)

        # Jeda 0.5 detik agar sistem responsif tapi tidak terlalu berat
        time.sleep(0.5)

    cap.release()

# Tab Laporan
st.divider()
if os.path.exists("laporan_produksi.csv"):
    st.subheader("📋 Laporan Produksi")
    df = pd.read_csv("laporan_produksi.csv")
    st.dataframe(df.tail(10), use_container_width=True)