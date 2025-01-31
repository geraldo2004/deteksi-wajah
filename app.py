import streamlit as st
import cv2
import numpy as np

# Judul aplikasi
st.title("Face Detection from Webcam Input")

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

# Input kamera (menggunakan webcam)
image = st.camera_input("Take a picture")

if image is not None:
    # Mengonversi gambar dari BytesIO ke format OpenCV
    img = np.array(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    # Deteksi wajah
    img = detect_faces(img)

    # Menampilkan gambar dengan wajah yang terdeteksi
    st.image(img, channels="BGR")
