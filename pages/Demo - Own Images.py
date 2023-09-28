import streamlit as st
from PIL import Image
import torch
from torch import nn
import torchvision
from going_modular.utils import load_model
from pathlib import Path
from going_modular import data_setup, engine

st.title("Predict with your own images")

# Get Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.warning(f'Prediction will run on {device.upper()}')

# Get the image using file_uploader widget
image = st.file_uploader(label='Upload a sattelite image',
                 accept_multiple_files=False)

# Classnames for our repository
class_names = ['No Wildfire','Wildfire']

# Initialize Model
model_loaded = torchvision.models.efficientnet_b0().to(device)
model_loaded.eval()

model_loaded.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=2)).to(device) # Hardcoded the class names

model_loaded = load_model(model_loaded, Path(r"model\EfficientNet_b0-Wildfire_Classifier.pt"))

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # "DEFAULT" = best available

# Get transforms from weights (these are the transforms used to train a particular or obtain a particular set of weights)
automatic_transforms = weights.transforms()

if image!=None:

    # Open the image
    image = Image.open(image).convert('RGB')

    # Display the image
    st.image(image)

    # Transfrom the image
    transformed_image = automatic_transforms(image)

    # Unsqueeze the image
    pred_image = torch.unsqueeze(transformed_image,dim=0)
    pred_image = pred_image.to(device)

    # For debugging and understanding
    # st.info(image_unsqueeze.shape)
    with torch.inference_mode():
        # Get logits for forward pass
        y_logits = model_loaded(pred_image)

        # Get pred
        y_pred_prob = torch.argmax(y_logits,dim=1).item()

    st.info(y_logits)
    st.header(f'Prediction : {class_names[y_pred_prob]}')

