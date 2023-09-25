import streamlit as st
from PIL import Image

st.header('Wildfire Classification and Analysis Using Satellite Images')
st.markdown('<div style="text-align: justify;">Wildfires are a devastating natural disaster that poses significant threats to both human life and the environment. These catastrophic events, often driven by factors like climate change and human activity, have been increasing in frequency and intensity in recent years. Therefore, it is crucial to develop innovative approaches to monitor, predict, and mitigate wildfires. One such approach is the use of satellite images for wildfire classification, a project that holds immense importance and offers numerous benefits.</div>', unsafe_allow_html=True)
st.write('')
st.markdown('<div style="text-align: justify;">In conclusion, the project of wildfire classification using satellite images is of paramount importance due to its potential to enhance early detection, situational awareness, prediction, and environmental monitoring of wildfires. The benefits it offers, including improved public safety, cost reduction, scalability, and global coverage, make it a valuable tool in the fight against wildfires. Moreover, this project aligns with broader efforts to address the increasing challenges posed by climate change and its impacts on wildfire activity. By harnessing the power of satellite technology, we can take significant steps toward mitigating the devastating effects of wildfires and protecting both human communities and the environment.</div>', unsafe_allow_html=True)

img = Image.open(r'C:\Users\SameerAhamed\Documents\GitHub\Wildfire-Classification\pages\images\fire-fighters-at-forest.jpg')
st.image(img)
