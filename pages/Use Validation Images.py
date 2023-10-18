import streamlit as st
from pathlib import Path
import torch
from going_modular.data_setup import create_dataset_valid
from going_modular.predictions import predict_single_image

# Classnames on which the model is trained on
# Classnames for our repository
class_names = ['No Wildfire','Wildfire']

st.set_page_config(page_title='Use Validation Images')
st.title('Use Validation Images for Prediction')

st.write('Running the model is very simple! Adjust the slider to get the predictions you want and then click "Run Prediciton".')

# Get Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.warning(f'Prediction will run on {device.upper()}')


# Create a dataset
valid_dir = Path(r'./validation_dataset/valid')

# This function will create a dataset
valid_dataset = create_dataset_valid(valid_dir)

num_images = int(st.slider(label='Number of Predictions',min_value=1,max_value=100,value=3))

# This button will check it is pressed then start what is in the loop
if st.button(label='Run Prediction',use_container_width=True):
    with st.spinner("Prediction Running...Please Wait.."):

        for image in range(0,num_images):
            
            st.markdown("""---""")
            # Select an image and its corresponding label randomly using random indexing
            rand_idx = torch.randint(low=0,high=len(valid_dataset),size=[1,1]).item()
            X,y = valid_dataset[rand_idx]

            pred_class = predict_single_image(X,y)

             # A Streamlit Container Widget
            with st.container():

                # 2 columns
                col1,col2 = st.columns(2)

                # The first column will show the image
                with col1:
                    st.image(X ,caption=f'Ground Truth: {class_names[y]} | Predicted Class: {pred_class[0]}')

                # The second column will show a if the predicted class matches with the ground truth
                with col2:
                    if pred_class[1] == True:
                        st.success(f'✅ Predicted Class: "{pred_class[0]}" is True! ')
                    else:
                        st.error(f'❌ Predicted Class: "{pred_class[0]}" is False.')
            
            
            
