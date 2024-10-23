# @GonzaloMartinGarcia

import torch
import random

# Multiresolution nosie from
# https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2?s=31.
def pyramid_noise_like(x, discount=0.9):
    b, c, w, h = x.shape 
    u = torch.nn.Upsample(size=(w, h), mode='bilinear')
    noise = torch.randn_like(x)
    for i in range(10):
        r = random.random()*2+2  
        w, h = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
        noise += u(torch.randn(b, c, w, h).to(x)) * discount**i
        if w==1 or h==1: 
            break 
    return noise / noise.std()