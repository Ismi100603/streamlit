import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Judul aplikasi
st.title("Aplikasi Klasifikasi Jenis Tanah Gambut")

# Input gambar
uploaded_file = st.file_uploader("Upload gambar tanah gambut", type=["jpg", "png", "jpeg"])

# Load model CNN yang telah disimpan
model = load_model("cnn_tanah_gambut.h5", compile=False)
classes = ["Fibrik", "Hemik", "Saprik"]  # Sesuaikan dengan label dataset

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah
    st.image(uploaded_file, caption="Gambar yang diunggah", use_container_width=True)

    # Baca gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (224, 224))  # Sesuaikan dengan input model
    img = img / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Tambahkan batch dimension

    # Prediksi
    prediction = model.predict(img)
    predicted_class = classes[np.argmax(prediction)]

    # Tampilkan hasil prediksi
    st.write(f"**Hasil Prediksi:** {predicted_class}")
