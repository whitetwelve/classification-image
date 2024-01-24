import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import uuid



# Load the trained TensorFlow model
model_path = "models/animal-image-classification2.h5"
model = tf.keras.models.load_model(model_path)

# Load the labels
class_labels = open("labels.txt", "r").readlines()
st.title("Klasifikasi Hewan")

st.write("Karna keterbatasan data saat ini hanya support foto gajah, harimau, singa dan zebra untuk tingkat prediksi yang memuaskan.")
# Add an option to choose between image upload and camera capture
option = st.sidebar.selectbox("Pilih opsi yang ada ..", ("Upload Image", "Capture from Camera"))

if option == "Upload Image":
    uploaded_image = st.file_uploader("Masukkan gambar (jpg,png,jpeg) only!", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        image = np.array(image.resize((190, 190))) / 255.0
        image = np.expand_dims(image, axis=0)

        # Make predictions
        prediction = model.predict(image)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]
        confidence = prediction[0][predicted_class_index]

        st.write(f"Ini adalah hewan: {predicted_class}")
        st.write(f"Akurasi (%): {confidence:.2f}")  # Display confidence with 2 decimal places

elif option == "Capture from Camera":
    st.write("Camera Capture Mode")

    # Open the camera
    cap = cv2.VideoCapture(0)  # 0 indicates the default camera

    if not cap.isOpened():
        st.write("Error: Could not open camera.")
        st.stop()

    stop_camera_button_key = "stop_camera_button_" + str(uuid.uuid4())  # Membuat UUID sebagai kunci unik

    while True:
        ret, frame = cap.read()

        if not ret:
            st.write("Error: Could not read frame from camera.")
            break

        # Display the captured frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB", use_column_width=True, caption="Camera Capture")

        # Preprocess the captured frame
        resized_frame = cv2.resize(frame_rgb, (190, 190)) / 255.0
        image = np.expand_dims(resized_frame, axis=0)

        # Make predictions
        prediction = model.predict(image)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]
        confidence = prediction[0][predicted_class_index]

        st.write(f"Ini adalah hewan: {predicted_class}")
        st.write(f"Akurasi (%): {confidence:.2f}")  # Display confidence with 2 decimal places
        
        if st.button("Stop Camera", key=stop_camera_button_key):
            break

    # Release the camera
    cap.release()