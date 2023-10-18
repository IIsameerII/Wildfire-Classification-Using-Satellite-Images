# Wildfire-Classification-Using-Satellite-Images

To run the project without setting up go to [Streamlit (wildfire-classification.streamlit.app)](https://wildfire-classification.streamlit.app/)

## Overview

This project focuses on classifying approximately 1 acre of land as wildfire-prone or not, using satellite images. We employ an EfficientNet B0 model for this classification, leveraging a dataset obtained from Kaggle. Additionally, we have created a user-friendly UI application using Streamlit to facilitate interaction with the model.

## Model Details

The classification model used in this project is EfficientNet B0. EfficientNet is a convolutional neural network architecture known for its high efficiency and accuracy. It's particularly well-suited for image classification tasks and offers a good balance between model size and performance. The use of EfficientNet B0 ensures accurate wildfire classification in our project.

## Dataset

The dataset used for training and testing the model was sourced from Kaggle. It is crucial for building and fine-tuning machine learning models, and it plays a significant role in the effectiveness of our wildfire classification system.

## User Interface

We have created a user-friendly Streamlit application that allows users to interact with the wildfire classification model. The application consists of three main pages:

### Introduction (First Page)

This page provides an introduction to the project, explaining its objectives and the importance of wildfire classification. Users can gain a clear understanding of the project's goals here.

### Image Classification (Second Page)

On the second page, users can upload their own satellite images for classification. The model will predict whether the uploaded images depict wildfire-prone areas or not. This feature enables users to apply the model to real-world scenarios.

### Model Validation (Third Page)

The third page offers a set of validation images that users can use to test the model's performance. This is a valuable resource for understanding how well the model classifies wildfire-prone areas and can be used for demonstration and educational purposes.

### Project Importance

Wildfire Classification using Satellite Images is important for several reasons:

1. **Early Detection:** Satellite images enable the early detection of wildfire-prone areas. This early warning can help authorities and communities prepare for potential wildfires, including evacuations and resource allocation.
2. **Resource Allocation:** By identifying areas at risk of wildfires, resources like firefighting teams, equipment, and suppression materials can be strategically allocated to mitigate the impact of wildfires more effectively.
3. **Environmental Conservation:** Wildfires have a significant impact on the environment, including damage to ecosystems, loss of biodiversity, and air quality degradation. Accurate classification allows for timely responses to protect natural habitats and reduce environmental damage.
4. **Human Safety:** Wildfires pose a serious threat to human lives and property. Timely classification provides the opportunity to evacuate people from high-risk areas and reduce the loss of life and property.
5. **Insurance and Risk Assessment:** Insurance companies can use wildfire classification data to assess and price policies for properties in wildfire-prone areas, which can contribute to more accurate risk assessment and fairer premiums.
6. **Urban Planning:** Knowing which areas are at risk of wildfires is crucial for urban planning and development. It can influence decisions regarding building codes, infrastructure, and zoning to minimize the risk to human settlements.
7. **Climate Change Monitoring:** Climate change is increasing the frequency and intensity of wildfires. Monitoring these changes through satellite imagery and classification helps in understanding the broader impact of climate change and devising strategies to mitigate its effects.
8. **Natural Resource Management:** Wildfires can affect the availability of natural resources such as timber, water sources, and agricultural lands. Accurate classification assists in managing and protecting these resources.
9. **Scientific Research:** Wildfire classification data is valuable for scientific research on fire behavior, ecological impact, and climate modeling. It contributes to our understanding of wildfire dynamics and can lead to better fire management strategies.
10. **Public Awareness:** By making wildfire risk information accessible to the public, individuals and communities can take proactive steps to reduce their vulnerability and better prepare for potential wildfires.
11. **Policy and Legislation:** Governments can use wildfire classification data to inform and strengthen policies and regulations related to wildfire management, land use, and disaster preparedness.

In summary, Wildfire Classification using Satellite Images is crucial for proactive risk management, environmental protection, public safety, and informed decision-making at various levels, from individual homeowners to government agencies and insurers. It plays a vital role in mitigating the devastating impact of wildfires on both human and natural systems.

## Acknowledgments

I would like to extend my gratitude to Daniel Bourke for his contribution to the project. The *going_modular* directory structure of the project was inspired by Daniel Bourke's [PyTorch for Deep Learning in 2023: Zero to Mastery ](https://www.udemy.com/course/pytorch-for-deep-learning/)course and has been implemented to enhance project organization and maintainability.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies.
3. Run the Streamlit application to start using the wildfire classification model.
