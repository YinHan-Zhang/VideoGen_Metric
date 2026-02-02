import torch
import numpy as np
import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization


# img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
# img1 = torch.zeros(1,3,64,64)
# d = loss_fn_vgg(img0, img1)

def calculate_lpips(img1, img2):
    lpips_score = loss_fn_vgg(img1, img2).cpu().numpy()
    return np.squeeze(lpips_score)
