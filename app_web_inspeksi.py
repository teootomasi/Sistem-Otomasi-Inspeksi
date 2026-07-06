import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import time

# --- KONFIGURASI ---
st.set_page_config(page_title="Sistem Inspeksi Botol", layout="wide")
st.title("🏭 Sistem Inspeksi Visual Otomatis")

# --- INISIALISASI STATE (Dinamis & Terus Berjalan) ---
if 'log_data' not in st.session_state:
    st.session_state.log_data = pd.DataFrame(columns=["Waktu", "Status", "Keyakinan"])
if 'counter' not in st.session_state:
    st.session_state.counter = {"NORMAL": 0, "CACAT": 0}

# --- LOAD MODEL ---
MODEL_PATH = "model_inspeksi_botol.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# --- SIDEBAR ---
mode = st.sidebar.radio("Pilih Mode:", ["Video Uji (Cloud)", "Webcam (Lokal)"])
if st.sidebar.button("Reset Statistik"):
    st.session_state.log_data = pd.DataFrame(columns=["Waktu", "Status", "Keyakinan"])
    st.session_state.counter = {"NORMAL": 0, "CACAT": 0}

# --- INPUT VIDEO ---
cap = cv2.VideoCapture(0) if mode == "Webcam (Lokal)" else cv2.VideoCapture("video_uji.mp4")

col1, col2 = st.columns([2, 1])
frame_placeholder = col1.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # PREDIKSI
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (96, 96))
    img_norm = img.reshape(1, 96, 96, 1) / 255.0
    pred = model.predict(img_norm, verbose=0)

    label = "NORMAL" if pred[0][0] < 0.5 else "CACAT"
    conf = (1 - pred[0][0]) if label == "NORMAL" else pred[0][0]

    # UPDATE COUNTER
    st.session_state.counter[label] += 1

    # UPDATE LOG (Simpan 10 data terakhir saja)
    new_log = pd.DataFrame(
        {"Waktu": [time.strftime("%H:%M:%S")], "Status": [label], "Keyakinan": [f"{conf * 100:.2f}%"]})
    st.session_state.log_data = pd.concat([new_log, st.session_state.log_data]).head(10)

    # VISUALISASI
    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    with col2.container():
        st.subheader("📊 Monitoring Real-time")
        k1, k2 = st.columns(2)
        k1.metric("Total Normal", st.session_state.counter["NORMAL"])
        k2.metric("Total Cacat", st.session_state.counter["CACAT"])

        st.write("---")
        st.write("**Log Riwayat Inspeksi:**")
        st.dataframe(st.session_state.log_data, use_container_width=True)

    time.sleep(0.1)  # Memberi jeda agar tidak terlalu cepat