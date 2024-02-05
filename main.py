# Author: Neelesh Chevuri

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import streamlit as st
from modules import get_classes_len, visualize_model_predictions


device = "cpu"

model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
model_ft.fc = nn.Linear(num_ftrs, get_classes_len())

model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load('dermalyze-ai-resnet-ft-4_exp.pth', map_location=torch.device(device)))
model_ft.eval()

uploaded_img = st.file_uploader("Upload an Image! (jpg)")

if uploaded_img is not None:
    visualize_model_predictions(model_ft, img_path=uploaded_img)

#visualize_model_predictions(
 #   model_ft,
  #  img_path='S_0917_acne_M1080444.2e16d0ba.fill-920x613.jpg'
#)

