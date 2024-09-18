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
# The following code is built upon Marigold's run.py, and was adapted to include some new settings
# and normals estimation. All additions made are marked with a # add.

import argparse
import logging
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

# add
from marigold import MarigoldPipeline
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import random

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

# add
# Code is from Marigold's util/seed_all.py
def seed_all(seed: int = 0):
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="GonzaloMG/marigold-e2e-ft-depth", # add
        help="Checkpoint path or hub name.",
    )

    parser.add_argument(
        "--input_rgb_dir",
        type=str,
        required=True,
        help="Path to the input image folder.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=1,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed. For the original (DDIM) version, it's recommended to use 10-50 steps, while for LCM 1-4 steps.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`. Default: `bilinear`",
    )

    # depth map colormap
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )

    # other settings
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducibility seed. Set to `None` for unseeded inference.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    parser.add_argument(
        "--apple_silicon",
        action="store_true",
        help="Flag of running on Apple Silicon.",
    )

    # add 
    parser.add_argument(
        "--noise",
        type=str,
        default='zeros',
        choices=["gaussian", "pyramid", "zeros"],
    )
    parser.add_argument(
        "--modality",
        type=str,
        default='depth',
        choices=["depth", "normals"],
    )
    parser.add_argument(
        "--timestep_spacing",
        type=str,
        default='trailing',
        choices=["trailing", "leading"],
    ) 

    args = parser.parse_args()

    # add
    noise = args.noise
    modality = args.modality
    timestep_spacing = args.timestep_spacing
    normals = True if modality == 'normals' else False

    checkpoint_path = args.checkpoint
    input_rgb_dir = args.input_rgb_dir
    output_dir = args.output_dir

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    apple_silicon = args.apple_silicon
    if apple_silicon and 0 == batch_size:
        batch_size = 1  # set default batchsize

    # -------------------- Preparation --------------------
    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with denoise_steps = {denoise_steps}, ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res}, seed = {seed}; "
        f"color_map = {color_map}."
    )

    # Random seed
    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    # Output directory
    os.makedirs(output_dir, exist_ok=True)
    if modality == 'normals':
        # add
        output_dir_color = os.path.join(output_dir, "normal_colored")
        output_dir_npy = os.path.join(output_dir, "normal_npy")
    else:
        output_dir_color = os.path.join(output_dir, "depth_colored")
        output_dir_npy = os.path.join(output_dir, "depth_npy")
        output_dir_tif = os.path.join(output_dir, "depth_bw")
        os.makedirs(output_dir_tif, exist_ok=True)
    
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps:0")
        else:
            device = torch.device("cpu")
            logging.warning("MPS is not available. Running on CPU will be slow.")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    rgb_filename_list = glob(os.path.join(input_rgb_dir, "*"))
    rgb_filename_list = [
        f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    rgb_filename_list = sorted(rgb_filename_list)
    n_images = len(rgb_filename_list)
    if n_images > 0:
        logging.info(f"Found {n_images} images")
    else:
        logging.error(f"No image found in '{input_rgb_dir}'")
        exit(1)

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.info(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None

    # add
    # load Model
    logging.info(f"Loading Model: {checkpoint_path}")
    unet         = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet")   
    vae          = AutoencoderKL.from_pretrained(checkpoint_path, subfolder="vae")  
    text_encoder = CLIPTextModel.from_pretrained(checkpoint_path, subfolder="text_encoder")  
    tokenizer    = CLIPTokenizer.from_pretrained(checkpoint_path, subfolder="tokenizer") 
    scheduler    = DDIMScheduler.from_pretrained(checkpoint_path, timestep_spacing=timestep_spacing, subfolder="scheduler") 
    pipe = MarigoldPipeline.from_pretrained(pretrained_model_name_or_path = checkpoint_path,
                                            unet=unet, 
                                            vae=vae, 
                                            scheduler=scheduler, 
                                            text_encoder=text_encoder, 
                                            tokenizer=tokenizer, 
                                            variant=variant, 
                                            torch_dtype=dtype, 
                                            )

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass  # run without xformers

    pipe = pipe.to(device)
    pipe.unet.eval()

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        os.makedirs(output_dir, exist_ok=True)

        for rgb_path in tqdm(rgb_filename_list, desc="Estimating depth", leave=True):
            # Read input image
            input_image = Image.open(rgb_path)

            # Predict depth or normals
            pipe_out = pipe(
                input_image,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=batch_size,
                color_map=color_map,
                show_progress_bar=True, 
                resample_method=resample_method,
                # add
                normals     = normals,
                noise     = noise,
            )

            # add
            pred: np.ndarray = pipe_out.normal_np if normals else pipe_out.depth_np
            pred_colored: Image.Image = pipe_out.normal_colored if normals else pipe_out.depth_colored

            # Save prediction as npy
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            pred_name_base = rgb_name_base + "_pred"
            npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
            if os.path.exists(npy_save_path):
                logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
            np.save(npy_save_path, pred)

            # Save prediction as colorized image
            colored_save_path = os.path.join(
                output_dir_color, f"{pred_name_base}_colored.png"
            )
            if os.path.exists(colored_save_path):
                logging.warning(
                    f"Existing file: '{colored_save_path}' will be overwritten"
                )
            pred_colored.save(colored_save_path)

            if not normals:
                 # Save depth as 16-bit uint grey scale png
                depth_to_save = (pred * 65535.0).astype(np.uint16)
                png_save_path = os.path.join(output_dir_tif, f"{pred_name_base}.png")
                if os.path.exists(png_save_path):
                    logging.warning(f"Existing file: '{png_save_path}' will be overwritten")
                Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")

