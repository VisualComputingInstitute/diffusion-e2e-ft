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
    --processing_res 0 \
    --dataset_config Marigold/config/dataset/data_kitti_eigen_test.yaml \
    --output_dir="experiments/depth/marigold/$checkpoint/kitti_eigen_test/prediction" \
    --model_type "geowizard"