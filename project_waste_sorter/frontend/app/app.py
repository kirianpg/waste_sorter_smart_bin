import streamlit as st
from io import BytesIO
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from project_waste_sorter.params import *
from project_waste_sorter.interface.main import pred

#Michael we need to make sure the get_model() function returns the prediction
from project_waste_sorter.interface.main_local import get_model


def pred_streamlit(user_input, model):
    """
    Make a prediction using the latest trained model, using a single image as input
    """
    st.title("Waste Classification App")
    uploaded_image = Image.open(BytesIO(user_input))

    # Resize to 224 x 224
    uploaded_image = uploaded_image.resize((224, 224))

    # Convert the image pixels to a numpy array
    image = img_to_array(uploaded_image)

    # Reshape data for the model
    image = image.reshape((1, 224, 224, 3))

    # Prepare the image for the VGG model
    image = preprocess_input(image)

    # Run prediction
    model_prediction = model.predict(image)

    # Formatting the result
    # We only need the max prediction
    #predictions_with_categories = [(INDEX_TO_CATEGORIES[i], prob) for i, prob in enumerate(predictions[0])]

    #predictions_with_categories.sort(key=lambda x: x[1], reverse=True)

    return st.write("Prediction:", model_prediction)
