import streamlit as st
from io import BytesIO
from PIL import Image
import requests
import io
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from project_waste_sorter.ml_logic.registry import load_model
from project_waste_sorter.params import *
from project_waste_sorter.api.api_file import *


# Title
st.title("Waste Sorter Smart Bin")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Function to make prediction request to backend
def predict(image):
    endpoint = "http://localhost:8000/predict/"
    files = {"file": image}
    response = requests.post(endpoint, files=files)
    return response.json()

# Display prediction results
if uploaded_file is not None:
    # Preprocess image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')

    # Make prediction request
    prediction = predict(image_bytes)

    # Display prediction results
    st.write("Prediction:", prediction)
