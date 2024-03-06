import streamlit as st

import numpy as np
import pandas as pd

st.markdown("""# This is your new SMART BIN!
## It will take care of your waste management by classifying your waste automatically.
The waste classifications in our MVP are:""")

# Allow user to upload a JPG image
uploaded_file = st.file_uploader("Upload a JPG image", type=["jpg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
