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
    --processing_res 0 \
    --dataset_config Marigold/config/dataset/data_scannet_val.yaml \
    --output_dir="experiments/depth/marigold/$checkpoint/scannet/prediction" \
    --model_type "marigold"