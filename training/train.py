# @GonzaloMartinGarcia
# Training code for 'Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think'.
# This training code is a modified version of the original text-to-image SD training code from the HuggingFace Inc. Team,
# https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py.
  
import argparse
import logging
import math
import os
import shutil

import accelerate
import datasets
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from torch.optim.lr_scheduler import LambdaLR
from dataloaders.load import *
from util.noise import pyramid_noise_like
from util.loss import ScaleAndShiftInvariantLoss, AngularLoss
from util.unet_prep import replace_unet_conv_in
from util.lr_scheduler import IterExponential

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")
logger = get_logger(__name__, log_level="INFO")

#############
# Arguments 
#############

def parse_args():
    parser = argparse.ArgumentParser(description="Training code for 'Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think'.")
    # Our settings:
    parser.add_argument(
        "--modality",
        type=str,
        choices=["depth", "normals"],
        required=True,
    )
    parser.add_argument(
        "--noise_type", 
        type=str,
        default=None, # If left as None, Stable Diffusion checkpoints can be trained without altering the input channels (i.e., only 4 input channels for the RGB input).
        choices=["zeros", "gaussian", "pyramid"],
        help="If left as None, Stable Diffusion checkpoints can be trained without altering the input channels (i.e., only 4 input channels for RGB input)."
    )
    parser.add_argument(
        "--lr_exp_warmup_steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--lr_total_iter_length",
        type=int,
        default=20000,
    )
    # Stable diffusion training settings
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/model-finetuned",
        required=True,
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
        default=2, 
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=15,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
        required=True,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
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
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
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
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1,
        help="For distributed training: local_rank"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=20000,
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
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", 
        action="store_true", 
        help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="e2e-ft-diffusion",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

########################
# VAE Helper Functions
########################

# Apply VAE Encoder to image
def encode_image(vae, image):
    h = vae.encoder(image)
    moments = vae.quant_conv(h)
    latent, _ = torch.chunk(moments, 2, dim=1)
    return latent

# Apply VAE Decoder to latent
def decode_image(vae, latent):
    z = vae.post_quant_conv(latent)
    image = vae.decoder(z)
    return image

##########################
# MAIN Training Function
##########################

def main():
    args = parse_args()

    # Init accelerator and logger
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Save training arguments in a .txt file
    if accelerator.is_main_process:
        args_dict = vars(args)
        args_str = '\n'.join(f"{key}: {value}" for key, value in args_dict.items())
        args_path = os.path.join(args.output_dir, "arguments.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(args_path, 'w') as file:
            file.write(args_str)
    if args.noise_type is None:
        logger.warning("Noise type is `None`. This setting is only meant for checkpoints without image conditioning, such as Stable Diffusion.")

    # Load model components
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant)
    vae          = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant)
    unet         = UNet2DConditionModel.from_pretrained( args.pretrained_model_name_or_path, subfolder="unet", revision=None)
    if args.noise_type is not None:
        # Double UNet input layers if necessary
        if unet.config['in_channels'] != 8:
            replace_unet_conv_in(unet, repeat=2)
            logger.info("Unet conv_in layer is replaced for RGB-depth or RGB-normals input")

    # Freeze or set model components to training mode
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # Use xformers for efficient attention
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Diffusers model loading and saving functions 
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()
        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Gradient checkpointing
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Optimizer
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Learning rate scheduler
    lr_func      = IterExponential(total_iter_length = args.lr_total_iter_length*accelerator.num_processes, final_ratio = 0.01, warmup_steps = args.lr_exp_warmup_steps*accelerator.num_processes)
    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    # Training datasets
    hypersim_root_dir = "data/hypersim/processed"
    vkitti_root_dir   = "data/virtual_kitti_2"
    train_dataset_hypersim = Hypersim(root_dir=hypersim_root_dir, transform=True)
    train_dataset_vkitti   = VirtualKITTI2(root_dir=vkitti_root_dir, transform=True)
    train_dataloader_vkitti   = torch.utils.data.DataLoader(train_dataset_vkitti,   shuffle=True, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers)
    train_dataloader_hypersim = torch.utils.data.DataLoader(train_dataset_hypersim, shuffle=True, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers)
    train_dataloader = MixedDataLoader(train_dataloader_hypersim, train_dataloader_vkitti, split1=9, split2=1)

    # Prepare everything with `accelerator` (Move to GPU)
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Mixed precision and weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
        unet.to(dtype=weight_dtype)
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
        unet.to(dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)    
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Calculate number of training steps and epochs
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset_vkitti)+len(train_dataset_hypersim)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Resume training from checkpoint
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
        disable=not accelerator.is_local_main_process,) 

    # Init task specific losses
    ssi_loss           = ScaleAndShiftInvariantLoss()
    angular_loss_norm  = AngularLoss()

    # Pre-compute empty text CLIP encoding
    empty_token    = tokenizer([""], padding="max_length", truncation=True, return_tensors="pt").input_ids
    empty_token    = empty_token.to(accelerator.device)
    empty_encoding = text_encoder(empty_token, return_dict=False)[0]
    empty_encoding = empty_encoding.to(accelerator.device)

    # Get noise scheduling parameters for later conversion from a parameterized prediction into latent.
    alpha_prod = noise_scheduler.alphas_cumprod.to(accelerator.device, dtype=weight_dtype)
    beta_prod  = 1 - alpha_prod
 
    # Training Loop
    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info(f"At Epoch {epoch}:")
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):     
                
                # RGB latent
                rgb_latents = encode_image(vae, batch["rgb"].to(device=accelerator.device, dtype=weight_dtype))
                rgb_latents = rgb_latents * vae.config.scaling_factor
         
                # Validity mask
                val_mask = batch["val_mask"].bool().to(device=accelerator.device)

                # Set timesteps to the first denoising step
                timesteps = torch.ones((rgb_latents.shape[0],), device=rgb_latents.device) * (noise_scheduler.config.num_train_timesteps-1) # 999
                timesteps = timesteps.long()
                
                # Sample noisy latent
                if (args.noise_type is None) or (args.noise_type == "zeros"):
                    noisy_latents = torch.zeros_like(rgb_latents).to(accelerator.device)
                elif args.noise_type == "pyramid":
                    noisy_latents = pyramid_noise_like(rgb_latents).to(accelerator.device)
                elif args.noise_type == "gaussian":
                    noisy_latents = torch.randn_like(rgb_latents).to(accelerator.device)
                else:
                    raise ValueError(f"Unknown noise type {args.noise_type}")

                # Generate UNet prediction
                encoder_hidden_states = empty_encoding.repeat(len(batch["rgb"]), 1, 1)
                unet_input = (
                    torch.cat((rgb_latents, noisy_latents), dim=1).to(accelerator.device)
                    if args.noise_type is not None
                    else rgb_latents
                )   
                model_pred = unet(unet_input, timesteps, encoder_hidden_states, return_dict=False)[0]

                # End-to-end fine-tuning 
                loss = torch.tensor(0.0, device=accelerator.device, requires_grad=True)
                if val_mask.any():

                    # Convert parameterized prediction into latent prediction.
                    # Code is based on the DDIM code from diffusers,
                    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py.
                    alpha_prod_t = alpha_prod[timesteps].view(-1, 1, 1, 1)
                    beta_prod_t  =  beta_prod[timesteps].view(-1, 1, 1, 1)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        current_latent_estimate = (alpha_prod_t**0.5) * noisy_latents - (beta_prod_t**0.5) * model_pred
                    elif noise_scheduler.config.prediction_type == "epsilon":
                        current_latent_estimate = (noisy_latents - beta_prod_t ** (0.5) * model_pred) / alpha_prod_t ** (0.5)
                    elif noise_scheduler.config.prediction_type == "sample":
                        current_latent_estimate = model_pred
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    # clip and threshold prediction (only here for completeness, not used by SD2 or our models with v_prediction)
                    if noise_scheduler.config.thresholding:
                        pred_original_sample = noise_scheduler._threshold_sample(pred_original_sample)
                    elif noise_scheduler.config.clip_sample:
                        pred_original_sample = pred_original_sample.clamp(
                            -noise_scheduler.config.clip_sample_range, noise_scheduler.config.clip_sample_range
                        )
                    
                    # Decode latent prediction
                    current_latent_estimate = current_latent_estimate / vae.config.scaling_factor
                    current_estimate = decode_image(vae, current_latent_estimate)

                    # Post-process predicted images and retrieve ground truth
                    if args.modality == "depth":
                        current_estimate = current_estimate.mean(dim=1, keepdim=True) 
                        current_estimate = torch.clamp(current_estimate,-1,1) 
                        ground_truth = batch["metric"].to(device=accelerator.device, dtype=weight_dtype)
                    elif args.modality == "normals":
                        norm = torch.norm(current_estimate, p=2, dim=1, keepdim=True) + 1e-5
                        current_estimate = current_estimate / norm
                        current_estimate = torch.clamp(current_estimate,-1,1)
                        ground_truth = batch["normals"].to(device=accelerator.device, dtype=weight_dtype)
                    else:
                        raise ValueError(f"Unknown modality {args.modality}")

                    # Compute task-specific loss   
                    estimation_loss = 0
                    if args.modality == "depth":              
                        estimation_loss_ssi = ssi_loss(current_estimate, ground_truth, val_mask)
                        if not torch.isnan(estimation_loss_ssi).any():
                            estimation_loss = estimation_loss + estimation_loss_ssi
                    elif args.modality == "normals":
                        estimation_loss_ang_norm = angular_loss_norm(current_estimate, ground_truth, val_mask)
                        if not torch.isnan(estimation_loss_ang_norm).any():
                            estimation_loss = estimation_loss + estimation_loss_ang_norm
                    else:
                        raise ValueError(f"Unknown modality {args.modality}")
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
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                train_loss = 0.0
                # Save model checkpoint 
                if global_step % args.checkpointing_steps == 0:
                    logger.info(f"Entered Saving Code at global step {global_step} checkpointing_steps {args.checkpointing_steps}")
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

            # Break training
            if global_step >= args.max_train_steps:
                break     
    
    # Create SD pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="scheduler", 
            timestep_spacing="trailing", # set scheduler timestep spacing to trailing for later inference.
            revision=args.revision, 
            variant=args.variant
        )
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            revision=args.revision,
            variant=args.variant,
        )
        logger.info(f"Saving pipeline to {args.output_dir}")
        pipeline.save_pretrained(args.output_dir)
    
    logger.info(f"Finished training.")

    accelerator.end_training()

if __name__ == "__main__":
    main()