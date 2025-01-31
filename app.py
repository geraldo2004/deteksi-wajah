import streamlit as st
import cv2
import numpy as np

# Judul aplikasi
st.title("Face Detection from Uploaded Video")

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

# Upload video
video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if video_file is not None:
    # Membaca file video
    file_bytes = np.asarray(bytearray(video_file.read()), dtype=np.uint8)
    cap = cv2.VideoCapture(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR))

    stframe = st.empty()  # Placeholder untuk menampilkan frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi wajah
        frame = detect_faces(frame)

        # Menampilkan frame di Streamlit
        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()
