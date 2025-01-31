import streamlit as st
import cv2
import numpy as np
import tempfile

# Judul aplikasi
st.title("Real-Time Face Detection with OpenCV")

# Menggunakan file Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Fungsi untuk deteksi wajah di gambar
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Menggambar kotak di sekitar wajah
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image

# Fungsi untuk menangkap video dan memproses frame-by-frame
def capture_video():
    # Mengakses kamera pertama
    cap = cv2.VideoCapture(0)

    # Menangkap video selama 5 detik atau lebih
    stframe = st.empty()  # Placeholder untuk menampilkan gambar

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi wajah
        frame = detect_faces(frame)

        # Menampilkan frame di Streamlit
        stframe.image(frame, channels="BGR", use_column_width=True)

        # Membatasi waktu atau kondisi untuk keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Menutup kamera setelah selesai
    cap.release()
    cv2.destroyAllWindows()

# Tombol untuk memulai video stream
if st.button('Start Video Stream'):
    capture_video()
