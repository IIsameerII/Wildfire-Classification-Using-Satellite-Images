import streamlit as st
from PIL import Image
import torch
from torch import nn
import torchvision
from going_modular.utils import load_model
from pathlib import Path
from going_modular import data_setup, engine, predictions

st.title("Use Custom Images for Wildfire Classification")

# Guidance on how to use the Demo.
st.header('Getting Started')
st.write('You can upload your own satellite image and run a prediction on the model to classify the area as wildfire prone or not. Before uploading an image, please read the directions below so you can get the best possible predictions from the model. \n 1)	Take an image from Bing Maps since, because it is less cluttered with store names, street names, etc. If they are cluttered you can unselect "details" in the option.  Click here to go to [Bing maps](https://share.streamlit.io/mesmith027/streamlit_webapps/main/MC_pi/streamlit_app.py) \n 2)	Use the satellite terrain while taking a snip of the map. \n 3)	Take an image of about an acre roughly, preferably with 100 meters elevation. This will improve the model performance as the dataset also has image of about 1 acre having a elevation of 100 meters.')

# Model prediction header
st.header("Model Prediction")

# Get Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.warning(f'Prediction will run on {device.upper()}')

# Get the image using file_uploader widget
image = st.file_uploader(label='Upload a sattelite image',
                 accept_multiple_files=False)

if image!=None:

    # Open the image
    image = Image.open(image).convert('RGB')

    # Display the image
    st.image(image)

    going_

