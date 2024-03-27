import streamlit as st
# from io import BytesIO
# from PIL import Image
import requests
# import io
import pandas as pd
import plotly.express as px
import os
from bins_locations import *


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
    endpoint = "https://waste-sorter-service-tuoydlr2ga-ew.a.run.app/predict/"
    files = {"file": image}
    response = requests.post(endpoint, files=files)
    return response.json()

# Function to get geodata bases
def get_geodata(class_file):
    'Returns the df for a certain garbage class'

    # Path to geodata files
    path_to_data = os.path.join("project_waste_sorter", "frontend", "app", "recycling_points_data", class_file)

    # Specific structure and preprocessing for Lyon DB => if other cities, we will have to homogenize all DB
    df = pd.read_csv(path_to_data, sep=';', index_col='idemplacementsilo')
    df['lon'] = df['lon'].apply(lambda x: float(x.replace(",",".")))
    df['lat'] = df['lat'].apply(lambda x: float(x.replace(",",".")))

    return df

# Make and display prediction results
if uploaded_file is not None:
    # Preprocess image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Make prediction request
    prediction = predict(uploaded_file)
    prob = round(100*prediction['result'][1],0)
    image_class = prediction['result'][0]

    # Display prediction results
    st.write(f"I'm {prob}% sure that this goes to the {image_class} bin!")

    # Select your city
    st.markdown('<h4 style="color: green;">Let me take you to the proper bin ! 🤝 </h4>', unsafe_allow_html=True)
    city = st.selectbox("Where do you live?", [None, 'Lyon', 'Other city'])

    if city is not None:
        RP_message, bins, bin_images, points = custom_policies[city]

        # message
        st.write(f"The recycling policy is the following in {city} : ")
        st.text(RP_message)

        # action
        bin = bins[image_class]
        st.write(f":{bin}[🫵  You should throw this piece of garbage to the {bin} bin! 🫵 ]")
        st.image(os.path.join("project_waste_sorter", "frontend", "app", "images",bin_images[image_class]))

        # map
        if points is not None:
            df = get_geodata(points[image_class])
            st.write(f"You can go to one of the following recycling points for {image_class} : ")
            fig = px.scatter_mapbox(df, lat="lat", lon="lon", zoom=3, text='adresse')
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig)

        st.title(":green[Let's go recycle this garbage!]")
