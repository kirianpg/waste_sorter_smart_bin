import streamlit as st
from io import BytesIO
from PIL import Image
import requests
import io


# Title
st.title(":green[Hi! I'm the Waste Sorter Smart Bin! Ready to be a recycling pro?]")

# First image to decorate the web app
col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image("https://m.media-amazon.com/images/I/71wY9yPVYUL._AC_SX695_.jpg")

with col3:
    st.write(' ')

# Explain what the app is for
'''Recycling is essential if we want to help in the climate crisis. Countries and cities are improving their recycling protocoles and increasing their ressources so that
citizens are able to recycle a larger amount of garbage and do it in a more efficient way. As a consequence, the number
of different recycling bins are increasing rapidly in some cities like in Barcelona,
where you have 5 different containers for plastic/metal, paper, glass, organic and trash.

However, recycling protocoles might vary significantly from one city to the other. That's why citizens and, specially, tourists, might need some help
when trying to classify their garbage.
'''

# Second image to decorate the web app
col4, col5, col6 = st.columns(3)
with col4:
    st.write(' ')

with col5:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR1h2FMlGjWQj7bjsPi01R200MvTFsiy7yN8w&usqp=CAU")

with col6:
    st.write(' ')

st.markdown("""
            #### Did you ever spent 20 minutes infront of a bunch of recycling bins wondering where to throw your chewing gum wrapper?

            ## 💡 We have the solution for you! 💡

            ##### ➡️ Our marvelous app will tell you where to throw your garbage depending on the city you are! 👏👏👏
            """)

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
    prob = round(100*prediction['result'][1],0)
    bin = prediction['result'][0]

    # Display prediction results
    st.write(f"I'm {prob}% sure this goes to the {bin} bin!")

    st.title(":green[Let's go recycle this garbage!]")
