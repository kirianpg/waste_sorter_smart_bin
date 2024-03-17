import streamlit as st
from io import BytesIO
from PIL import Image
import requests
import io


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
    #image = Image.open(uploaded_file)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    #image_bytes = io.BytesIO()
    #image.save(image_bytes, format='PNG')

    # Make prediction request
    prediction = predict(uploaded_file)

    # Display prediction results
    st.write("Prediction:", prediction)
