# add
import torch
import numpy as np
from torch import nn


# Pyramid Noise implementation from GeoWizard training code
# https://github.com/fuxiao0719/GeoWizard/blob/5b25910f5ceaecb4f5f3db000153052628611c9d/geowizard/training/training/train_depth_normal.py#L299
def pyramid_noise_like(x, timesteps, discount=0.9):
    b, c, w_ori, h_ori = x.shape 
    u = nn.Upsample(size=(w_ori, h_ori), mode='bilinear')
    noise = torch.randn_like(x)
    scale = 1.5
    for i in range(10):
        r = np.random.random()*scale + scale # Rather than always going 2x, 
        w, h = max(1, int(w_ori/(r**i))), max(1, int(h_ori/(r**i)))
        noise += u(torch.randn(b, c, w, h).to(x)) * (timesteps[...,None,None,None]/1000) * discount**i
        if w==1 or h==1: break # Lowest resolution is 1x1
    return noise/noise.std() # Scaled back to roughly unit variance