 
from tqdm import tqdm
from PIL import Image
import torch
import os
import numpy as np
 
from transformers import CLIPProcessor, CLIPModel
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
 
def calculate_clip_I(image1, image2):

    inputs1 = processor(images=image1, return_tensors="pt").to(device)
    inputs2 = processor(images=image2, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features1 = model.get_image_features(**inputs1)
        image_features2 = model.get_image_features(**inputs2)

    image_features1 /= image_features1.norm(dim=-1, keepdim=True)
    image_features2 /= image_features2.norm(dim=-1, keepdim=True)

    similarity = torch.matmul(image_features1, image_features2.T).cpu().numpy()[0][0]
    
    return similarity