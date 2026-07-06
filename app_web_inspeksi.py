while cap.isOpened() and not stop_btn:
    ret, frame = cap.read()
    if not ret: break

    # ... (kode preprocessing & prediksi tetap sama)

    # Update UI & Save
    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
    status_placeholder.metric("Status Terakhir", status, f"{conf:.2%}")

    # Log hanya jika status berubah (atau setiap 5 frame agar tidak terlalu penuh)
    new_log = pd.DataFrame({"Waktu": [time.strftime("%H:%M:%S")], "Status": [status]})
    save_log(new_log)

    # UBAH DARI 1.0 MENJADI 0.1 ATAU 0.05 AGAR SANGAT CEPAT
    time.sleep(0.1)
cap.release()