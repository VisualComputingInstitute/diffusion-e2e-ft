#!/usr/bin/env bash
set -e
set -x

# add
checkpoint="geowizard-e2e-ft"
checkpoint_path="GonzaloMG/$checkpoint"

# add
python Marigold/infer.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path" \
    --base_data_dir="data/marigold_eval" \
    --processing_res 640 \
    --dataset_config Marigold/config/dataset/data_diode_all.yaml \
    --output_dir="experiments/depth/marigold/$checkpoint/diode/prediction" \
    --resample_method bilinear \
    --model_type "geowizard"