import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

import time
import cv2
from PIL import Image

# add
from pathlib import Path
import argparse
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

import sys

# add
from DSINE.utils import utils, visualize as vis_utils
from DSINE.projects.dsine import config
from DSINE.projects.baseline_normal.dataloader import *
from Marigold.marigold import MarigoldPipeline
from GeoWizard.geowizard.models.geowizard_pipeline import DepthNormalEstimationPipeline


def test(args, model, test_loader, device):

    # add
    denoise_steps   = args.denoise_steps
    ensemble_size   = args.ensemble_size
    processing_res  = args.processing_res
    resample_method = args.resample_method
    noise           = args.noise
    iterator = 0 

    with torch.no_grad():
        total_normal_errors = None

        for data_dict in tqdm(test_loader):

            # add
            # for GeoWizard, set default domain if not provided
            if args.domain is None:
                if (data_dict['dataset_name'][0] == 'nyuv2') or (data_dict['dataset_name'][0] == 'ibims') or (data_dict['dataset_name'][0] == 'scannet'):
                    domain = "indoor"
                elif (data_dict['dataset_name'][0] == 'sintel') or (data_dict['dataset_name'][0] == 'oasis'):
                    domain = "outdoor"
                else:
                    raise Exception('invalid dataset name')
            else:
                domain = args.domain

            #↓↓↓↓
            #NOTE: forward pass
            img = data_dict['img'].to(device)
            
            # Normalize RGB
            img_min = img.min().item()
            img_max = img.max().item()
            img = (img-img_min)/(img_max-img_min) * 255.0
            img = img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            # To PIL
            img = Image.fromarray(img)
            
            # add
            # forward pass
            if args.model_type == "geowizard":
                pipe_out = model(img,
                                denoising_steps = denoise_steps,
                                ensemble_size   = ensemble_size,
                                processing_res=processing_res,
                                match_input_res = True,
                                color_map = "Spectral",
                                show_progress_bar = False,
                                domain = domain,
                                noise = noise
                                )
                norm_out = pipe_out.normal_np
                pred_norm = torch.from_numpy(norm_out).permute(2,0,1).unsqueeze(0).to(device)
            else:
                pipe_out = model(img,
                                denoising_steps=denoise_steps,
                                ensemble_size=ensemble_size,
                                processing_res=processing_res,
                                match_input_res=True,
                                color_map=None,
                                show_progress_bar=False,
                                resample_method=resample_method,
                                batch_size=0,
                                noise       = noise,
                                normals     = True,
                                )
                norm_out = pipe_out.normal_np
                pred_norm = torch.from_numpy(norm_out).unsqueeze(0).to(device)
            #↑↑↑↑

            if 'normal' in data_dict.keys():
                gt_norm = data_dict['normal'].to(device)
                gt_norm_mask = data_dict['normal_mask'].to(device)

                pred_error = utils.compute_normal_error(pred_norm, gt_norm)
                if total_normal_errors is None:
                    total_normal_errors = pred_error[gt_norm_mask]
                else:
                    total_normal_errors = torch.cat((total_normal_errors, pred_error[gt_norm_mask]), dim=0)

            iterator += 1

        if total_normal_errors is not None:
            metrics = utils.compute_normal_metrics(total_normal_errors)
            print("mean median rmse 5 7.5 11.25 22.5 30")
            print("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" % (
                metrics['mean'], metrics['median'], metrics['rmse'],
                metrics['a1'], metrics['a2'], metrics['a3'], metrics['a4'], metrics['a5']))
            
            # add
            # save the metrics in a txt file
            metrics_path = os.path.join(args.output_dir, 'test', args.dataset_name_test, 'metrics.txt')
            print("metrics_path", metrics_path)
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, 'w') as f:
                f.write("Normal Estimation Metrics:\n")
                f.write(f"Metrics at iteration {iterator}\n")
                f.write("mean median rmse 5 7.5 11.25 22.5 30\n")
                f.write("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n" % (
                    metrics['mean'], metrics['median'], metrics['rmse'],
                    metrics['a1'], metrics['a2'], metrics['a3'], metrics['a4'], metrics['a5']))
            print(f"Metrics saved to {metrics_path}")
        else:
            print("No normal errors to compute metrics.")


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


if __name__ == '__main__':
    device = torch.device('cuda')
    args = config.get_args(test=True)

    # add
    # seed evaluation
    seed = args.seed
    if seed is None:
        import time
        seed = int(time.time())
    seed_all(seed)

    # add
    # Load Model
    checkpoint_path =  args.ckpt_path
    dtype = torch.float32
    print(f"Loading model from {checkpoint_path}")
    if args.model_type == "geowizard":
        vae = AutoencoderKL.from_pretrained(checkpoint_path, subfolder='vae')
        if args.timestep_spacing is not None:
            scheduler = DDIMScheduler.from_pretrained(checkpoint_path, timestep_spacing=args.timestep_spacing, subfolder='scheduler')
        else:
            scheduler = DDIMScheduler.from_pretrained(checkpoint_path, subfolder='scheduler')
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(checkpoint_path, subfolder="image_encoder")
        feature_extractor = CLIPImageProcessor.from_pretrained(checkpoint_path, subfolder="feature_extractor")
        unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet")
        model = DepthNormalEstimationPipeline(vae=vae,
                                    image_encoder=image_encoder,
                                    feature_extractor=feature_extractor,
                                    unet=unet,
                                    scheduler=scheduler)        
    elif args.model_type == "marigold":
        variant = None
        unet         = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet")   
        vae          = AutoencoderKL.from_pretrained(checkpoint_path, subfolder="vae")  
        text_encoder = CLIPTextModel.from_pretrained(checkpoint_path, subfolder="text_encoder")  
        tokenizer    = CLIPTokenizer.from_pretrained(checkpoint_path, subfolder="tokenizer") 
        scheduler    = DDIMScheduler.from_pretrained(checkpoint_path, timestep_spacing=args.timestep_spacing, subfolder="scheduler") 
        model = MarigoldPipeline.from_pretrained(pretrained_model_name_or_path = checkpoint_path,
                                                unet=unet, 
                                                vae=vae, 
                                                scheduler=scheduler, 
                                                text_encoder=text_encoder, 
                                                tokenizer=tokenizer, 
                                                variant=variant, 
                                                torch_dtype=dtype, 
                                                )
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    try:
        model.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers
    model = model.to(device)
    model.unet.eval()
    #↑↑↑↑
 
    # test the model
    if args.mode == 'benchmark':
        # do not resize/crop the images when benchmarking
        args.input_height = args.input_width = 0
        args.data_augmentation_same_fov = 0
        
        if args.eval_data == "all":
            data_to_evaluate = [('nyuv2', 'test'), 
                                ('scannet', 'test'),
                                ('ibims', 'ibims'),
                                ('sintel', 'sintel'),
                                #('oasis', 'val')
                                ]
        elif args.eval_data == "nyuv2":
            data_to_evaluate = [('nyuv2', 'test')]
        elif args.eval_data == "scannet":
            data_to_evaluate = [('scannet', 'test')]
        elif args.eval_data == "ibims":
            data_to_evaluate = [('ibims', 'ibims')]
        elif args.eval_data == "sintel":
            data_to_evaluate = [('sintel', 'sintel')]
        # elif args.eval_data == "oasis":
        #     data_to_evaluate = [('oasis', 'val')]
        print(f"Testing on {data_to_evaluate}")

        for dataset_name, split in data_to_evaluate:

            args.dataset_name_test = dataset_name
            args.test_split = split
            test_loader = TestLoader(args).data

            test(args, model, test_loader, device)


