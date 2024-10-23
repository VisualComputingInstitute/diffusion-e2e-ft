# A reimplemented version in public environments by Xiao Fu and Mu Hu

# @GonzaloMartinGarcia
# Training code for the end-to-end fine-tuned GeoWizard Model from 
# 'Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think'.
# This training code is a modified version of the original GeoWizard training code,
# https://github.com/fuxiao0719/GeoWizard/blob/main/geowizard/training/training/train_depth_normal.py.
# Modifications have been marked with the comment # add.

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import logging
import tqdm

import sys

from accelerate import Accelerator
import numpy as np
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.utils import ProjectConfiguration, set_seed
import shutil

from diffusers import DDPMScheduler, AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import  is_wandb_available # check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import accelerate

# add
if is_wandb_available():
    import wandb
sys.path.append(os.getcwd())
from GeoWizard.geowizard.models.geowizard_pipeline import DepthNormalEstimationPipeline
from GeoWizard.geowizard.models.unet_2d_condition import UNet2DConditionModel
from torch.optim.lr_scheduler import LambdaLR
from training.util.lr_scheduler import IterExponential
from training.util.loss import ScaleAndShiftInvariantLoss, AngularLoss
from training.dataloaders.load import MixedDataLoader, Hypersim, VirtualKITTI2

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="GeoWizard")

    # add
    # End-to-end fine-tuned Settings
    parser.add_argument(
        "--e2e_ft",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--noise_type",
        choices=["zeros", "pyramid", "gaussian"],
    )
    parser.add_argument(
        "--lr_total_iter_length",
        type=int,
        default=20000,
    )
    
    # GeoWizard Settings
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/model-finetuned", # add
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=2, # add
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=None # add
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=20000, # add
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16, # add
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5, # add
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--class_embedding_lr_mult",
        type=float,
        default=10,
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="exponential", # add
        help=(
            'The scheduler type to use. Also choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int, 
        default=100, # add
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", 
        action="store_true", 
        help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    # using EMA for improving the generalization
    parser.add_argument(
        "--use_ema", 
        action="store_true", 
        help="Whether to use EMA model."
    )
    # dataloaderes
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1, 
        help="For distributed training: local_rank"
    )
    # how many steps csave a checkpoints
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=20000, # add
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    # using xformers for efficient training 
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", 
        action="store_true", 
        help="Whether or not to use xformers."
    )
    # noise offset?::: #TODO HERE
    parser.add_argument(
        "--noise_offset", 
        type=float, 
        default=0, 
        help="The scale of noise offset."
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="e2e-ft-diffusion", # add
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    # get the local rank
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

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
    
def main():

    ''' ------------------------Configs Preparation----------------------------'''
    # give the args parsers
    args = parse_args()
    # save  the tensorboard log files
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # tell the gradient_accumulation_steps, mix precison, and tensorboard
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True) # only the main process show the logs

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Doing I/O at the main proecss
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # add
    # Save training arguments in a txt file
    args_dict = vars(args)
    args_str = '\n'.join(f"{key}: {value}" for key, value in args_dict.items())
    args_path = os.path.join(args.output_dir, "arguments.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args_path, 'w') as file:
        file.write(args_str)
    
    ''' ------------------------Non-NN Modules Definition----------------------------'''
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
    sd_image_variations_diffusers_path = 'lambdalabs/sd-image-variations-diffusers'
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(sd_image_variations_diffusers_path, subfolder="image_encoder")
    feature_extractor = CLIPImageProcessor.from_pretrained(sd_image_variations_diffusers_path, subfolder="feature_extractor")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder='vae')
    # add
    # no modification are made to the UNet since we fine-tune GeoWizard.
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # using EMA
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    # Freeze vae and set unet to trainable.
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.train() # only make the unet-trainable        

    # using xformers for efficient attentions.
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            logger.info("use xformers to speed up", main_process_only=True)

        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model
                
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # using checkpoint for saving the memories
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # how many cards did we use: accelerator.num_processes
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    params, params_class_embedding = [], []
    for name, param in unet.named_parameters():
        if 'class_embedding' in name:
            params_class_embedding.append(param)
        else:
            params.append(param)

    # optimizer settings
    optimizer = optimizer_cls(
        [
            {"params": params, "lr": args.learning_rate},
            {"params": params_class_embedding, "lr": args.learning_rate * args.class_embedding_lr_mult}
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # get the training dataset
    with accelerator.main_process_first():

        # add 
        # Load datasets
        hypersim_root_dir = "data/hypersim/processed"
        vkitti_root_dir   = "data/virtual_kitti_2"
        train_dataset_hypersim = Hypersim(root_dir=hypersim_root_dir, transform=True)
        train_dataset_vkitti   = VirtualKITTI2(root_dir=vkitti_root_dir, transform=True)
        train_dataloader_vkitti   = torch.utils.data.DataLoader(train_dataset_vkitti,   shuffle=True, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers)
        train_dataloader_hypersim = torch.utils.data.DataLoader(train_dataset_hypersim, shuffle=True, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers)
        train_loader = MixedDataLoader(train_dataloader_hypersim, train_dataloader_vkitti, split1=9, split2=1)

    # because the optimizer not optimized every time, so we need to calculate how many steps it optimizes,
    # it is usually optimized by 
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # add
    # Scheduler
    if args.lr_scheduler == "exponential":
        lr_func      = IterExponential(total_iter_length = args.lr_total_iter_length*accelerator.num_processes, final_ratio = 0.01, warmup_steps = args.lr_warmup_steps*accelerator.num_processes)
        lr_scheduler = LambdaLR(optimizer= optimizer, lr_lambda=lr_func)
    elif args.lr_scheduler == "constant":
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
        )
    else:
        raise ValueError(f"Unknown lr_scheduler {args.lr_scheduler}")

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    clip_image_mean = torch.as_tensor(feature_extractor.image_mean)[:,None,None].to(accelerator.device, dtype=torch.float32)
    clip_image_std  = torch.as_tensor(feature_extractor.image_std)[:,None,None].to(accelerator.device, dtype=torch.float32)

    # We need to initialize the trackers we use, and also store our configuration.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)
    
    # Here is the DDP training: actually is 4
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # add
    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset_vkitti)+len(train_dataset_hypersim)}") # add
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    # Progress bar
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # add 
    # Init task specific losses
    ssi_loss          = ScaleAndShiftInvariantLoss()
    angular_loss_norm = AngularLoss()
    # Init loss dictionary for logging
    loss_logger = { "ssi": 0.0,             # depth level loss
                    "ssi_count": 0.0,
                    "normals_angular": 0.0, # normals level loss
                    "normals_angular_count": 0.0
                    }
     
    # add
    # Get noise scheduling parameters for later conversion from a parameterized prediction into clean latent.
    alpha_prod = noise_scheduler.alphas_cumprod.to(accelerator.device, dtype=weight_dtype)
    beta_prod  = 1 - alpha_prod
    
    # Training Loop
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train() 
        train_loss = 0.0
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(unet):
                
                # add
                # RGB
                image_data_resized = batch['rgb'].to(accelerator.device,dtype=weight_dtype)
                # Depth
                depth_resized_normalized = batch['depth'].to(accelerator.device,dtype=weight_dtype)
                # Validation mask
                val_mask = batch["val_mask"].to(accelerator.device)
                invalid_mask  = ~val_mask
                latent_mask   = ~torch.max_pool2d(invalid_mask.float(), 8, 8).bool()
                latent_mask   = latent_mask.repeat((2, 4, 1, 1)).detach() 
                # Surface normals
                normal_resized = batch['normals'].to(accelerator.device,dtype=weight_dtype)*-1 # GeoWizard trains on inverted normals!

                # Compute CLIP image embeddings
                imgs_in_proc = TF.resize((image_data_resized +1)/2, 
                    (feature_extractor.crop_size['height'], feature_extractor.crop_size['width']), 
                    interpolation=InterpolationMode.BICUBIC, 
                    antialias=True
                )
                # do the normalization in float32 to preserve precision
                imgs_in_proc = ((imgs_in_proc.float() - clip_image_mean) / clip_image_std).to(weight_dtype)        
                imgs_embed = image_encoder(imgs_in_proc).image_embeds.unsqueeze(1).to(weight_dtype)

                # encode latents
                with torch.no_grad():
                    if args.e2e_ft:
                        # add
                        # When E2E FT, we only need to encode the RGB image
                        h_batch = vae.encoder(image_data_resized)
                        moments_batch = vae.quant_conv(h_batch)
                        mean_batch, _ = torch.chunk(moments_batch, 2, dim=1)
                        rgb_latents   = mean_batch * vae.config.scaling_factor
                        depth_latents, normal_latents = torch.zeros_like(rgb_latents), torch.zeros_like(rgb_latents) # dummy latents
                    else:
                        h_batch = vae.encoder(torch.cat((image_data_resized, depth_resized_normalized, normal_resized), dim=0).to(weight_dtype))
                        moments_batch = vae.quant_conv(h_batch)
                        mean_batch, _ = torch.chunk(moments_batch, 2, dim=1)
                        batch_latents = mean_batch * vae.config.scaling_factor
                        rgb_latents, depth_latents, normal_latents = torch.chunk(batch_latents, 3, dim=0)
                    geo_latents = torch.cat((depth_latents, normal_latents), dim=0)

                # here is the setting batch size, in our settings, it can be 1.0
                bsz = rgb_latents.shape[0]
            
                # add
                # Sample timesteps
                if args.e2e_ft:
                    # Set timesteps to the first denoising step
                    timesteps = torch.ones((bsz,), device=depth_latents.device).repeat(2) * (noise_scheduler.config.num_train_timesteps-1)
                    timesteps = timesteps.long()
                else:
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=depth_latents.device).repeat(2)
                    timesteps = timesteps.long()

                # add
                # Sample noise
                if args.noise_type == "zeros":
                    noise = torch.zeros_like(geo_latents).to(accelerator.device)
                elif args.noise_type == "pyramid":
                    noise = pyramid_noise_like(geo_latents, timesteps).to(accelerator.device)
                elif args.noise_type == "gaussian":
                    noise = torch.randn_like(geo_latents).to(accelerator.device)
                else:
                    raise ValueError(f"Unknown noise type {args.noise_type}")
                
                # add
                # Add noise to the depth latents
                if args.e2e_ft:
                    noisy_geo_latents = noise # no ground truth when single step fine-tuning
                else:
                    # add noise to the depth lantents
                    noisy_geo_latents = noise_scheduler.add_noise(geo_latents, noise, timesteps)

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(geo_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                batch_imgs_embed = imgs_embed.repeat((2, 1, 1))  # [B*2, 1, 768]

                # hybrid hierarchical switcher 
                geo_class = torch.tensor([[0, 1], [1, 0]], dtype=weight_dtype, device=accelerator.device)
                geo_embedding = torch.cat([torch.sin(geo_class), torch.cos(geo_class)], dim=-1).repeat_interleave(bsz, 0)

                # add
                # Domain class
                if batch["domain"][0] == 'indoor':
                    domain_class = torch.tensor([[1., 0., 0]], device=accelerator.device, dtype=weight_dtype)
                elif batch["domain"][0] == 'outdoor':
                    domain_class = torch.tensor([[0., 1., 0]], device=accelerator.device, dtype=weight_dtype)
                else:
                    raise ValueError(f"Unknown domain {batch['domain'][0]}")
                domain_class = domain_class.repeat(bsz, 1)

                domain_embedding = torch.cat([torch.sin(domain_class), torch.cos(domain_class)], dim=-1).repeat(2,1).to(accelerator.device)
                class_embedding = torch.cat((geo_embedding, domain_embedding), dim=-1)

                # predict the noise residual and compute the loss.
                unet_input = torch.cat((rgb_latents.repeat(2,1,1,1), noisy_geo_latents), dim=1)

                noise_pred = unet(unet_input, 
                                timesteps, 
                                encoder_hidden_states=batch_imgs_embed,
                                class_labels=class_embedding).sample #[B, 4, h, w]
                
                # add
                # Compute loss
                loss = torch.tensor(0.0, device=accelerator.device, requires_grad=True)
                if latent_mask.any():
                    if not args.e2e_ft:
                        # Diffusion loss
                        loss = F.mse_loss(noise_pred[latent_mask].float(), target[latent_mask].float(), reduction="mean")
                    else:
                        # End-to-end task specific fine-tuning loss
                        # Convert parameterized prediction into latent prediction.
                        # Code is based on the DDIM code from diffusers,
                        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py.
                        alpha_prod_t = alpha_prod[timesteps].view(-1, 1, 1, 1)
                        beta_prod_t  =  beta_prod[timesteps].view(-1, 1, 1, 1)
                        if noise_scheduler.config.prediction_type == "v_prediction":
                            current_latent_estimate = (alpha_prod_t**0.5) * noisy_geo_latents - (beta_prod_t**0.5) * noise_pred
                        elif noise_scheduler.config.prediction_type == "epsilon":
                            current_latent_estimate = (noisy_geo_latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
                        else:
                            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                        # clip or threshold prediction (only here for completeness, not used by SD2 or our models with v_prediction)
                        if noise_scheduler.config.thresholding:
                            pred_original_sample = noise_scheduler._threshold_sample(pred_original_sample)
                        elif noise_scheduler.config.clip_sample:
                            pred_original_sample = pred_original_sample.clamp(
                                -noise_scheduler.config.clip_sample_range, noise_scheduler.config.clip_sample_range
                            )
                        # Decode the latent estimate
                        current_latent_estimate = current_latent_estimate / vae.config.scaling_factor
                        z = vae.post_quant_conv(current_latent_estimate)
                        current_estimate = vae.decoder(z)
                        current_depth_estimate, current_normal_estimate = torch.chunk(current_estimate, 2, dim=0)
                        # Process depth and get GT
                        current_depth_estimate = current_depth_estimate.mean(dim=1, keepdim=True) 
                        current_depth_estimate = torch.clamp(current_depth_estimate,-1,1) 
                        depth_ground_truth = batch["metric"].to(device=accelerator.device, dtype=weight_dtype)
                        # Process normals and get GT
                        norm = torch.norm(current_normal_estimate, p=2, dim=1, keepdim=True) + 1e-5
                        current_normal_estimate = current_normal_estimate / norm
                        current_normal_estimate = torch.clamp(current_normal_estimate,-1,1) 
                        normal_ground_truth = batch["normals"].to(device=accelerator.device, dtype=weight_dtype) * -1 # GeoWizard trains on inverted normals!
                        # Compute task-specific loss             
                        estimation_loss = 0
                        depth_scale  = 0.5 # ssi loss is roughly 2x the angular loss
                        normal_scale = 1.0
                        # Scale and shift invariant loss
                        estimation_loss_ssi = ssi_loss(current_depth_estimate, depth_ground_truth, val_mask)
                        if not torch.isnan(estimation_loss_ssi).any():
                            estimation_loss = estimation_loss + (estimation_loss_ssi*depth_scale)
                            loss_logger["ssi"] += estimation_loss_ssi.detach().item()   
                            loss_logger["ssi_count"] += 1
                        # Angular loss
                        estimation_loss_ang_norm = angular_loss_norm(current_normal_estimate, normal_ground_truth, val_mask)
                        if not torch.isnan(estimation_loss_ang_norm).any():
                            estimation_loss = estimation_loss + (estimation_loss_ang_norm*normal_scale)
                            loss_logger["normals_angular"] += estimation_loss_ang_norm.detach().item()   
                            loss_logger["normals_angular_count"] += 1
                        loss = loss + estimation_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                # add
                accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

                train_loss = 0.0

                # add
                # logg depth and normals losses separately
                for key in list(loss_logger.keys()):
                    if "_count" not in key:
                        count_key = key + "_count"
                        if loss_logger[count_key] != 0:
                            # compute avg
                            loss_logger[key] /= loss_logger[count_key]
                            # log loss
                            loss_name = key + "_loss"
                            accelerator.log({loss_name: loss_logger[key]}, step=global_step)
                # set all losses to 0
                for key in list(loss_logger.keys()):
                    loss_logger[key] = 0.0
                
                # saving the checkpoints
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            
            # Log loss and learning rate for progress bar
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # Stop training
            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            # validation each epoch by calculate the epe and the visualization depth
            if args.use_ema:    
                # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())

            if args.use_ema:
                # Switch back to the original UNet parameters.
                ema_unet.restore(unet.parameters())

    # add        
    # Create GeoWizard pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="scheduler", 
            timestep_spacing="trailing" # set scheduler timestep spacing to trailing for later inference.
        )
        pipeline = DepthNormalEstimationPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=vae,
                unet=unet,
                scheduler=scheduler,
                image_encoder=image_encoder,
                feature_extractor=feature_extractor
        )
        logger.info(f"Saving pipeline to {args.output_dir}")
        pipeline.save_pretrained(args.output_dir)
    logger.info(f"Finished training.")

    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__=="__main__":
    main()
