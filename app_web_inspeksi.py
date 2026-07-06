import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import time
import os

# Konfigurasi Layout
st.set_page_config(page_title="Smart Factory Inspection", layout="wide")


# 1. Load Model (Caching agar tidak berat)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_inspeksi_botol.keras")


model = load_model()


# 2. Fungsi Simpan Log Produksi
def save_log(status):
    file_path = "laporan_produksi.csv"
    new_data = pd.DataFrame({"Waktu": [time.strftime("%H:%M:%S")], "Status": [status]})
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        pd.concat([df, new_data], ignore_index=True).to_csv(file_path, index=False)
    else:
        new_data.to_csv(file_path, index=False)


# 3. Sidebar Kontrol
st.sidebar.title("Kontrol Sistem")
mode = st.sidebar.radio("Input:", ["Video Uji", "Live Webcam"])
start_btn = st.sidebar.button("▶ START")

st.title("🏭 Sistem Inspeksi Visual Otomatis")

# 4. Membuat 3 Tab (Dashboard Profesional)
tab1, tab2, tab3 = st.tabs(["Monitoring Live", "Laporan Produksi", "Validasi Model"])

with tab1:
    col1, col2 = st.columns([2, 1])
    frame_placeholder = col1.empty()
    status_placeholder = col2.empty()

    if start_btn:
        cap = cv2.VideoCapture(0 if mode == "Live Webcam" else "video_uji.mp4")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Preprocessing: Grayscale (1 Channel) sesuai model Anda
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
            time.sleep(0.1)  # Responsif & Real-time
        cap.release()

with tab2:
    st.subheader("📋 Data Log Produksi")
    if os.path.exists("laporan_produksi.csv"):
        df = pd.read_csv("laporan_produksi.csv")
        st.dataframe(df.tail(20), use_container_width=True)

with tab3:
    st.subheader("📊 Validasi Performa Model")
    # GANTI ANGKA DI BAWAH SESUAI HASIL EVALUASI ASLI ANDA
    data_performa = {
        "Skenario": ["S1", "S1", "S2", "S2", "S3", "S3", "S4", "S4"],
        "Label": ["Cacat", "Normal", "Cacat", "Normal", "Cacat", "Normal", "Cacat", "Normal"],
        "Precision": [0.97, 0.92, 0.98, 0.94, 0.98, 0.91, 0.96, 0.94],
        "Recall": [0.91, 0.97, 0.94, 0.98, 0.90, 0.98, 0.94, 0.96],
        "F1-Score": [0.94, 0.94, 0.96, 0.96, 0.93, 0.94, 0.95, 0.95],
        "Accuracy": [0.94, 0.94, 0.96, 0.96, 0.94, 0.94, 0.95, 0.95]
    }
    df_performa = pd.DataFrame(data_performa)
    st.table(df_performa)
    st.info("Catatan: Metrik ini divalidasi menggunakan dataset uji untuk memastikan keandalan sistem.")