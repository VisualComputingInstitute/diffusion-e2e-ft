#!/usr/bin/env bash
set -e
set -x

# add
checkpoint="geowizard-e2e-ft"

# add
python Marigold/eval.py \
    --base_data_dir="data/marigold_eval" \
    --dataset_config Marigold/config/dataset/data_kitti_eigen_test.yaml \
    --alignment least_square \
    --prediction_dir experiments/depth/marigold/$checkpoint/kitti_eigen_test/prediction \
    --output_dir experiments/depth/marigold/$checkpoint/kitti_eigen_test/eval_metric