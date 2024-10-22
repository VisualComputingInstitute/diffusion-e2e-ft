#!/bin/bash 

accelerate launch GeoWizard/geowizard/training/train_depth_normal.py \
  --pretrained_model_name_or_path "lemonaddie/geowizard" \
  --e2e_ft \
  --noise_type="zeros" \
  --max_train_steps 20000 \
  --checkpointing_steps 20000 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing \
  --learning_rate 3e-5 \
  --lr_total_iter_length 20000 \
  --lr_warmup_steps 100 \
  --mixed_precision="no" \
  --output_dir "model-finetuned/geowizard_e2e_ft" \
  --enable_xformers_memory_efficient_attention \
  "$@"