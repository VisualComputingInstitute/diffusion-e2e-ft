#!/usr/bin/env bash
set -e
set -x

# add
checkpoint="marigold-e2e-ft-depth"

# add
python Marigold/eval.py \
    --base_data_dir="data/marigold_eval" \
    --dataset_config Marigold/config/dataset/data_diode_all.yaml \
    --alignment least_square \
    --prediction_dir experiments/depth/marigold/$checkpoint/diode/prediction \
    --output_dir experiments/depth/marigold/$checkpoint/diode/eval_metric