import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score


def calculate_fid(real_images_folder, generated_images_folder):

    fid_value = fid_score.calculate_fid_given_paths(
        paths=[real_images_folder, generated_images_folder],
        batch_size=50,
        device="cuda",
        dims=2048,
    )
    
    return fid_value