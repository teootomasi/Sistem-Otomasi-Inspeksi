import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import time

st.set_page_config(page_title="Sistem Inspeksi Botol", layout="wide")

# --- INISIALISASI STATE ---
if 'history' not in st.session_state:
    st.session_state.history = []


# --- LOAD MODEL ---
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("model_inspeksi_botol.keras")


model = load_my_model()

# --- SIDEBAR KONTROL (PERMANEN) ---
st.sidebar.title("Kontrol Sistem")
mode = st.sidebar.radio("Pilih Input:", ["Video Uji", "Webcam"])
start = st.sidebar.button("▶ START")
stop = st.sidebar.button("⏹ STOP")

# --- TAMPILAN UTAMA ---
st.title("🏭 Sistem Inspeksi Visual Otomatis")
col_view, col_report = st.columns([2, 1])

frame_placeholder = col_view.empty()
status_display = col_view.empty()

# --- LOGIKA OPERASIONAL ---
if start:
    cap = cv2.VideoCapture(0 if mode == "Webcam" else "video_uji.mp4")

    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret: break

        # Preprocessing
        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 96))
        img_norm = img.reshape(1, 96, 96, 1) / 255.0

        # Prediksi
        pred = model.predict(img_norm, verbose=0)
        label = "NORMAL" if pred[0][0] < 0.5 else "CACAT"
        conf = (1 - pred[0][0]) if label == "NORMAL" else pred[0][0]

        # Update Visual Feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        status_display.success(f"### Status Terkini: {label} ({conf * 100:.1f}%)")

        # Simpan ke riwayat (Logika sederhana: hanya simpan jika ada perubahan status)
        if not st.session_state.history or st.session_state.history[-1]['Status'] != label:
            st.session_state.history.append({"Waktu": time.strftime("%H:%M:%S"), "Status": label})

        time.sleep(0.5)  # Kecepatan inspeksi

    cap.release()

# --- AREA LAPORAN (SEPARATE SHEET) ---
with col_report:
    st.subheader("📋 Laporan Inspeksi")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.table(df.tail(10))  # Menampilkan 10 botol terakhir
        st.download_button("Download Laporan CSV", df.to_csv(), "laporan.csv", "text/csv")
    else:
        st.info("Tekan START untuk memulai inspeksi.")