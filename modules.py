import torch
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
from PIL import Image
import streamlit as st

# Transforms for input images
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_names = ['acne', 'eczema', 'melanoma & moles', 'psoriasis']

def get_classes_len() -> int:
    return len(class_names)

def visualize_model_predictions(model,img_path):
    was_training = model.training
    model.eval()

    img = Image.open(img_path)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        
        st.write(class_names[preds[0]])

        model.train(mode=was_training)