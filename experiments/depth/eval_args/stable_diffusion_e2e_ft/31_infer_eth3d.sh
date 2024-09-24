#!/usr/bin/env bash
set -e
set -x

# add
checkpoint="stable-diffusion-e2e-ft-depth"
checkpoint_path="GonzaloMG/$checkpoint"

# add
python Marigold/infer.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path" \
    --base_data_dir="data/marigold_eval" \
    --processing_res 756 \
    --dataset_config Marigold/config/dataset/data_eth3d.yaml \
    --output_dir="experiments/depth/marigold/$checkpoint/eth3d/prediction" \
    --resample_method bilinear \
    --model_type "marigold"