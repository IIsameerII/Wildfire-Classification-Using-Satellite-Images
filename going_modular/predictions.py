"""
Utility functions to make predictions.

Main reference for code creation: https://www.learnpytorch.io/06_pytorch_transfer_learning/#6-make-predictions-on-images-from-the-test-set 
"""
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn

from typing import List, Tuple
from going_modular.utils import load_model

from PIL import Image
from pathlib import Path

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"


#==================================I WANT THE CODE TO INITIALIZE FOR STREAMLIT APP====================

# Classnames for our repository
class_names = ['No Wildfire','Wildfire']

# Initialize Model
model_loaded = torchvision.models.efficientnet_b0().to(device)
model_loaded.eval()

model_loaded.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=2)).to(device) # Hardcoded the class names

model_loaded = load_model(model_loaded, Path(r"./model/EfficientNet_b0-Wildfire_Classifier.pt"))

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # "DEFAULT" = best available

# Get transforms from weights (these are the transforms used to train a particular or obtain a particular set of weights)
automatic_transforms = weights.transforms()

#====================================================END==================================


# Predict on a target image with a target model
# Function created in: https://www.learnpytorch.io/06_pytorch_transfer_learning/#6-make-predictions-on-images-from-the-test-set
def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    device: torch.device = device,
):
    """Predicts on a target image with a target model.

    Args:
        model (torch.nn.Module): A trained (or untrained) PyTorch model to predict on an image.
        class_names (List[str]): A list of target classes to map predictions to.
        image_path (str): Filepath to target image to predict on.
        image_size (Tuple[int, int], optional): Size to transform target image to. Defaults to (224, 224).
        transform (torchvision.transforms, optional): Transform to perform on image. Defaults to None which uses ImageNet normalization.
        device (torch.device, optional): Target device to perform prediction on. Defaults to device.
    """

    # Open image
    img = Image.open(image_path)

    # Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    ### Predict on image ###

    # Make sure the model is on the target device
    model.to(device)

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img)
    plt.title(
        f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}"
    )
    plt.axis(False)

def predict_single_image(image,y_label=None): # The y_label is optional

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

        # Predicted class
        predicted_class = class_names[y_pred_prob]

    if y_label == None:
        return predicted_class
    else:
        if predicted_class == y_label:
            return [predicted_class,True]
        else:
            return [predicted_class,True]
    



    


