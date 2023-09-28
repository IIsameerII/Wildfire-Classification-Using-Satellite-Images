import streamlit as st
from pathlib import Path
import torch
from going_modular.data_setup import create_dataset_valid
from going_modular.predictions import predict_single_image

st.set_page_config(page_title='Use Validation Images')
st.title('Use Validation Images for Prediction')

st.write('Running the model is very simple! Adjust the slider to get the predictions you want and then click "Run Prediciton".')

# Create a dataset
valid_dir = Path(r'validation_dataset\valid')

# This function will create a dataset
valid_dataset = create_dataset_valid(valid_dir)

num_images = int(st.slider(label='Number of Predictions',min_value=1,max_value=100,value=3))

# This button will check it is pressed then start what is in the loop
if st.button(label='Run Prediction',use_container_width=True):
    for image in range(0,num_images):
        rand_idx = torch.randint(low=0,high=len(valid_dataset),size=[1,1]).item()
        X,y = valid_dataset[rand_idx]

        st.image(X)
        
        pred_class = predict_single_image(X,y)
        if pred_class[1] == True:
            st.success(f'Predicted Class: "{pred_class[0]}" is True!')
        else:
            st.error(f'Predicted Class: "{pred_class[0]}" is False.')
    