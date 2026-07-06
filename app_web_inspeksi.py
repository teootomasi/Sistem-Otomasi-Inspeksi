import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import time
import os

st.set_page_config(page_title="Smart Factory Inspection", layout="wide")


@st.cache_resource
def load_model():
    # Pastikan model Anda sudah benar-benar dilatih dengan dataset cacat yang cukup
    return tf.keras.models.load_model("model_inspeksi_botol.keras")


model = load_model()


# --- FUNGSI PREDIKSI YANG DIPERBAIKI ---
def get_prediction(frame):
    img = cv2.resize(frame, (96, 96))  # Sesuaikan dengan input size model Anda
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_norm = img_gray.reshape(1, 96, 96, 1) / 255.0

    pred = model.predict(img_norm, verbose=0)[0][0]

    # ADJUST THRESHOLD DI SINI
    # Jika model Anda output-nya 0 = Normal, 1 = Cacat
    # Jika masih salah, coba balik logika ini (pred > 0.5)
    status = "CACAT" if pred > 0.5 else "NORMAL"
    return status, pred


# --- SIDEBAR ---
st.sidebar.title("Kontrol Sistem")
mode = st.sidebar.radio("Input:", ["Video Uji", "Live Webcam"])
start_btn = st.sidebar.button("▶ START")
stop_btn = st.sidebar.button("⏹ STOP")

st.title("🏭 Sistem Inspeksi Visual Otomatis")

# Layout Utama
col1, col2 = st.columns([2, 1])
frame_placeholder = col1.empty()
status_placeholder = col2.empty()

if start_btn:
    cap = cv2.VideoCapture(0 if mode == "Live Webcam" else "video_uji.mp4")

    # Gunakan counter untuk mengatur kecepatan inspeksi (1 detik per produk)
    frame_count = 0
    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        # Inspeksi hanya dilakukan setiap 30 frame (asumsi 30fps = 1 detik)
        # Ubah angka 30 sesuai kecepatan video Anda
        if frame_count % 30 == 0:
            status, conf = get_prediction(frame)
            status_placeholder.metric("Status Terakhir", status, f"Conf: {conf:.2f}")

            # Log Data
            new_log = pd.DataFrame({"Waktu": [time.strftime("%H:%M:%S")], "Status": [status]})
            if os.path.exists("laporan_produksi.csv"):
                pd.concat([pd.read_csv("laporan_produksi.csv"), new_log]).to_csv("laporan_produksi.csv", index=False)
            else:
                new_log.to_csv("laporan_produksi.csv", index=False)

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        # Hapus time.sleep agar video tidak patah-patah
    cap.release()