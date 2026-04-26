import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json


model = tf.keras.models.load_model("model.h5")

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}


st.title("Potato Leaf Disease Detection App")
st.write("Upload a potato leaf image and get prediction")


uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Show image
    img = image.load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Result
    st.success(f"Prediction: {labels[predicted_class]}")