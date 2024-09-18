# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

# @GonzaloMartinGarcia
# This file is a modified version of the original Marigold pipeline file. 
# Based on GeoWizard, we added the option to sample surface normals, marked with # add.

from typing import Dict, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
    DDPMScheduler,
)
from diffusers.utils import BaseOutput
from PIL import Image
from torchvision.transforms.functional import resize, pil_to_tensor
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depths
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)

# add
import random


# add
# Surface Normals Ensamble from the GeoWizard github repository (https://github.com/fuxiao0719/GeoWizard)
def ensemble_normals(input_images:torch.Tensor):
    normal_preds = input_images
    bsz, d, h, w = normal_preds.shape
    normal_preds = normal_preds / (torch.norm(normal_preds, p=2, dim=1).unsqueeze(1)+1e-5)
    phi = torch.atan2(normal_preds[:,1,:,:], normal_preds[:,0,:,:]).mean(dim=0)
    theta = torch.atan2(torch.norm(normal_preds[:,:2,:,:], p=2, dim=1), normal_preds[:,2,:,:]).mean(dim=0)
    normal_pred = torch.zeros((d,h,w)).to(normal_preds)
    normal_pred[0,:,:] = torch.sin(theta) * torch.cos(phi)
    normal_pred[1,:,:] = torch.sin(theta) * torch.sin(phi)
    normal_pred[2,:,:] = torch.cos(theta) 
    angle_error = torch.acos(torch.clip(torch.cosine_similarity(normal_pred[None], normal_preds, dim=1),-0.999, 0.999))
    normal_idx = torch.argmin(angle_error.reshape(bsz,-1).sum(-1))
    return normal_preds[normal_idx], None

# add
# Pyramid nosie from 
#   https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2?s=31
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

class MarigoldDepthOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
        normal_np (`np.ndarray`):
            Predicted normal map, with normal vectors in the range of [-1, 1].
        normal_colored (`PIL.Image.Image`):
            Colorized normal map
    """

    depth_np: np.ndarray
    depth_colored: Union[None, Image.Image]
    uncertainty: Union[None, np.ndarray]
    # add
    normal_np: np.ndarray
    normal_colored: Union[None, Image.Image]


class MarigoldPipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using Marigold: https://marigoldmonodepth.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
    """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler,DDPMScheduler,LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        denoising_steps: int = 10,
        ensemble_size: int = 10,
        processing_res: int = 768,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
        # add
        noise="gaussian",
        normals=False,
    ) -> MarigoldDepthOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            processing_res (`int`, *optional*, defaults to `768`):
                Maximum resolution of processing.
                If set to 0: will not resize at all.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize depth prediction to match input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`, defaults to: `bilinear`.
            denoising_steps (`int`, *optional*, defaults to `10`):
                Number of diffusion denoising steps (DDIM) during inference.
            ensemble_size (`int`, *optional*, defaults to `10`):
                Number of predictions to be ensembled.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            color_map (`str`, *optional*, defaults to `"Spectral"`, pass `None` to skip colorized depth map generation):
                Colormap used to colorize the depth map.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
            noise (`str`, *optional*, defaults to `gaussian`):
                Type of noise to be used for the initial depth map.
                Can be one of `gaussian`, `pyramid`, `zeros`.
            normals (`bool`, *optional*, defaults to `False`):
                If `True`, the pipeline will predict surface normals instead of depth maps.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth_np** (`np.ndarray`) Predicted depth map, with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [3, H, W] and values in [0, 1], None if `color_map` is `None`
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
            - **normal_np** (`np.ndarray`) Predicted normal map, with normal vectors in the range of [-1, 1]
            - **normal_colored** (`PIL.Image.Image`) Colorized normal map
        """

        assert processing_res >= 0
        assert ensemble_size >= 1

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------

        # Convert to torch tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            rgb = pil_to_tensor(input_image) # [H, W, rgb] -> [rgb, H, W]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image.squeeze()
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (
            3 == rgb.dim() and 3 == input_size[0]
        ), f"Wrong input shape {input_size}, expected [rgb, H, W]"

        # Resize image
        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

        # Normalize rgb values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting depth/normal --------------

        # Batch repeated input image
        duplicated_rgb = torch.stack([rgb_norm] * ensemble_size)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        # load iterator
        pred_ls  = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader

        # inference (batched)
        for batch in iterable:
            (batched_img,) = batch
            pred_raw = self.single_infer(
                rgb_in=batched_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                # add
                noise=noise,
                normals=normals,
            )
            pred_ls.append(pred_raw.detach())
        preds = torch.concat(pred_ls, dim=0).squeeze()
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------

        if ensemble_size > 1:   # add
            pred, pred_uncert = ensemble_normals(preds) if normals else ensemble_depths(preds, **(ensemble_kwargs or {}))
        else:
            pred = preds
            pred_uncert = None

        # ----------------- Post processing -----------------

        if normals:
            # add
            # Normalizae normal vectors to unit length
            pred /= (torch.norm(pred, p=2, dim=0, keepdim=True)+1e-5)
        else:
            # Scale relative prediction to [0, 1]
            min_d = torch.min(pred)
            max_d = torch.max(pred)
            if max_d == min_d:
                pred = torch.zeros_like(pred)
            else:
                pred = (pred - min_d) / (max_d - min_d)
            
        # Resize back to original resolution
        if match_input_res:
            pred = resize(
                pred if normals else pred.unsqueeze(0),
                (input_size[-2],input_size[-1]),
                interpolation=resample_method,
                antialias=True,
            ).squeeze()

        # Convert to numpy
        pred = pred.cpu().numpy()

        # Process prediction for visualization
        if not normals:
            # add
            pred = pred.clip(0, 1)
            if color_map is not None:
                colored = colorize_depth_maps(
                    pred, 0, 1, cmap=color_map
                ).squeeze()  # [3, H, W], value in (0, 1)
                colored = (colored * 255).astype(np.uint8)
                colored_hwc = chw2hwc(colored)
                colored_img = Image.fromarray(colored_hwc)
            else:
                colored_img = None
        else:
            pred = pred.clip(-1.0, 1.0)
            colored = (((pred+1)/2) * 255).astype(np.uint8)
            colored_hwc = chw2hwc(colored)
            colored_img = Image.fromarray(colored_hwc)

        
        return MarigoldDepthOutput(
            depth_np       = pred if not normals else None,
            depth_colored  = colored_img if not normals else None,
            uncertainty    = pred_uncert,
            # add
            normal_np      = pred if normals else None,
            normal_colored = colored_img if normals else None,
        )


    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        num_inference_steps: int,
        show_pbar: bool,
        # add
        noise="gaussian",
        normals=False,
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
            noise (`str`, *optional*, defaults to `gaussian`):
                Type of noise to be used for the initial depth map.
                Can be one of `gaussian`, `pyramid`, `zeros`.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        device = self.device
        rgb_in = rgb_in.to(device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]
        
        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)

        # add
        # Initial prediction
        latent_shape = rgb_latent.shape
        if noise == "gaussian":
            latent = torch.randn(
                latent_shape,
                device=device,
                dtype=self.dtype,
            )
        elif noise == "pyramid":
            latent = pyramid_noise_like(rgb_latent).to(device) # [B, 4, h, w]
        elif noise == "zeros":
            latent = torch.zeros(
                latent_shape,
                device=device,
                dtype=self.dtype,
            )
        else:
            raise ValueError(f"Unknown noise type: {noise}")

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        )  # [B, 2, 1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            
            unet_input = torch.cat(
                [rgb_latent, latent], dim=1
            )  # this order is important

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_step = self.scheduler.step(
                noise_pred, t, latent
            )
        
            latent = scheduler_step.prev_sample
        
        if normals:
            # add
            # decode and normalize normal vectors
            normal = self.decode_normal(latent)
            normal /= (torch.norm(normal, p=2, dim=1, keepdim=True)+1e-5)
            return normal
        else:      
            # decode and normalize depth map          
            depth = self.decode_depth(latent)
            depth = torch.clip(depth, -1.0, 1.0)
            depth = (depth + 1.0) / 2.0
            return depth


    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent
    

    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean
    
    # add
    def decode_normal(self, normal_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode normal latent into normal map.

        Args:
            normal_latent (`torch.Tensor`):
                normal latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """ 
        # scale latent
        normal_latent = normal_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(normal_latent)
        normal = self.vae.decoder(z)
        return normal
