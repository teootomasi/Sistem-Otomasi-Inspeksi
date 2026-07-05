import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('model_inspeksi_botol.keras')
cap = cv2.VideoCapture(0)

print("Sistem Inspeksi Lokal Aktif. Tekan 'q' untuk keluar.")
while True:
    ret, frame = cap.read()
    if not ret: break

    img = cv2.resize(frame, (96, 96))
    img = img.reshape(1, 96, 96, 1) / 255.0
    pred = model.predict(img, verbose=0)

    label = "NORMAL" if pred[0][0] < 0.5 else "CACAT"
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Inspeksi Lokal', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()