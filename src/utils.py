import os
from PIL import Image
import torch
from torchvision import transforms

def load_image(image_file):
    """
    Loads an image from a file path or file-like object.
    """
    try:
        img = Image.open(image_file).convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def get_transform():
    """
    Returns the standard transform for the model.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
