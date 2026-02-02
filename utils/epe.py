import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic = True

from torchvision.models.optical_flow import Raft_Large_Weights

weights = Raft_Large_Weights.DEFAULT
transforms = weights.transforms()


def preprocess(source_batch, target_batch):
    source_batch = F.resize(source_batch, size=[480, 832], antialias=False)
    target_batch = F.resize(target_batch, size=[480, 832], antialias=False)
    return transforms(source_batch, target_batch)

from torchvision.models.optical_flow import raft_large

# If you can, run this example on a GPU, it will be a lot faster.
device = "cuda" if torch.cuda.is_available() else "cpu"

model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
model = model.eval()

def calculate_epe(img1_batch, img2_batch):
    # img [N, C, H, W]
    
    # first calculate the op of img1 and img2
    img1_source, img1_target = preprocess(img1_batch[:-1], img1_batch[1:])
    img2_source, img2_target = preprocess(img2_batch[:-1], img2_batch[1:])
    
    # op
    img1_flows = model(img1_source.to(device).contiguous(), img1_target.to(device).contiguous())[-1]  # [N, 2, H, W]
    img2_flows = model(img2_source.to(device).contiguous(), img2_target.to(device).contiguous())[-1]
    
    # epe
    diff = img1_flows - img2_flows  
    epe = torch.norm(diff, p=2, dim=1)
    mean_epe = epe.mean()
    
    return mean_epe.cpu().numpy()
