import streamlit as st
import joblib
from PIL import Image
import numpy as np
import os

# Load trained model and preprocessing - use cache not to reload it every time?
@st.cache()
def load_model_and_preprocessing():
    model = joblib.load(os.path.dirname(os.path.dirname(os.getcwd()))+'/ml_logic'+'/model.py')
    preprocessing = joblib.load(os.path.dirname(os.path.dirname(os.getcwd()))+'/ml_logic'+'/preprocessing.py')
    return model, preprocessing

model, preprocessing = load_model_and_preprocessing()

# Function to upload and save images
def save_uploaded_image(uploaded_img):
    UPLOAD_FOLDER = os.getcwd() + '/uploaded_images'
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    if uploaded_img is not None:
        image_path = os.path.join(UPLOAD_FOLDER, uploaded_img.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_img.getbuffer())
        st.success("Image saved successfully!")
        return image_path

# File uploader
uploaded_img = st.file_uploader("Upload Image", type=['jpg', 'png'])

# Save uploaded image
if uploaded_img is not None:
    image_path = save_uploaded_image(uploaded_img)
    st.image(image_path, caption="Uploaded Image", use_column_width=True)

# Check if the upload worked
if uploaded_img is not None:
    # Preprocess the uploaded image
    processed_image = preprocessing.preprocess(uploaded_img)

    # Make predictions
    prediction = model.predict(processed_image)

    # Display the prediction
    st.write("Prediction:", prediction)
