#!/bin/bash 

accelerate launch training/train.py \
  --pretrained_model_name_or_path "prs-eth/marigold-v1-0" \
  --modality "depth" \
  --noise_type "zeros" \
  --max_train_steps 20000 \
  --checkpointing_steps 20000 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing \
  --learning_rate 3e-05 \
  --lr_total_iter_length 20000 \
  --lr_exp_warmup_steps 100 \
  --mixed_precision "no" \
  --output_dir "model-finetuned/marigold_e2e_ft_depth" \
  --enable_xformers_memory_efficient_attention \
  "$@"