import streamlit as st
import joblib
from PIL import Image
import numpy as np

# Load trained model and preprocessing - use cache not to reload it every time?
@st.cache()
def load_model_and_preprocessing():
    model = joblib.load('model_path')
    preprocessing = joblib.load('preprocessing_path')
    return model, preprocessing

model, preprocessing = load_model_and_preprocessing()

# Formatting of interface
st.markdown("""# This is your new SMART BIN!
## It will take care of your waste management by classifying your waste automatically.
The waste classifications in our MVP are:""")

# Function to preprocess new uploaded image
def preprocess_image(image):
    processed_image = preprocessing.preprocess(image)
    return processed_image

# File Upload - do we only want to allo jpg or also others?
uploaded_img = st.file_uploader("Upload Image", type=['jpg'])

# Check if the upload worked
if uploaded_img is not None:
    # Display the uploaded image
    image = Image.open(uploaded_img)
    st.image(image, caption='Uploaded Image')

    # Process the uploaded image
    processed_image = preprocess_image(image)

    # Make predictions
    prediction = model.predict(processed_image)

    # Display the prediction
    st.write("Prediction:", prediction)
