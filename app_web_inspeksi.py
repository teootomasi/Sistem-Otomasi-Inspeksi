import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import time
import os

# --- KONFIGURASI LAYOUT ---
st.set_page_config(page_title="Smart Factory Inspection", layout="wide")

# Styling agar kamera dan dashboard tidak saling "memakan" ruang
st.markdown("""
    <style>
    [data-testid="stImage"] { width: 100% !important; border-radius: 10px; border: 2px solid #333; }
    .css-1r6slp0 { padding: 0 !important; }
    </style>
""", unsafe_allow_html=True)


# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_inspeksi_botol.keras")


model = load_model()


# --- AUTO-SAVE LOGIKA ---
def save_log(new_data):
    file_path = "laporan_produksi.csv"
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        pd.concat([df_existing, new_data]).to_csv(file_path, index=False)
    else:
        new_data.to_csv(file_path, index=False)


# --- SIDEBAR ---
st.sidebar.title("Industrial Control")
mode = st.sidebar.radio("Source:", ["Video Uji", "Live Webcam"])
start_btn = st.sidebar.button("▶ START INSPECTION")
stop_btn = st.sidebar.button("⏹ STOP SYSTEM")

# --- TAMPILAN UTAMA ---
st.title("🏭 Automated Bottle Inspection System")
tab1, tab2 = st.tabs(["Live Vision", "Production Report"])

with tab1:
    col1, col2 = st.columns([1.5, 1])  # Proporsi layar agar kamera tetap lega
    frame_placeholder = col1.empty()
    status_placeholder = col2.empty()

    if start_btn:
        cap = cv2.VideoCapture(0 if mode == "Live Webcam" else "video_uji.mp4")
        # Mengatur resolusi agar pas (640x480 adalah standar laptop)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret: break

            # Preprocessing
            img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 96))
            img_norm = img.reshape(1, 96, 96, 1) / 255.0

            # Prediksi
            pred = model.predict(img_norm, verbose=0)
            status = "NORMAL" if pred[0][0] < 0.5 else "CACAT"

            # Update UI
            # Mengubah frame agar tidak terpotong (mengikuti ratio asli)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, use_column_width=True)

            status_placeholder.markdown(f"""
                ### Inspection Result
                Status: **{status}**
                Confidence: {100 - pred[0][0] * 100:.2f}%
            """)

            # Auto-save ke CSV
            new_log = pd.DataFrame({"Waktu": [time.strftime("%H:%M:%S")], "Status": [status]})
            save_log(new_log)

            time.sleep(1.0)
        cap.release()

with tab2:
    st.subheader("📋 Laporan Produksi Terintegrasi")
    if os.path.exists("laporan_produksi.csv"):
        df = pd.read_csv("laporan_produksi.csv")
        st.dataframe(df.tail(20), use_container_width=True)