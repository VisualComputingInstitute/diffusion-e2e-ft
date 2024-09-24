#!/usr/bin/env bash

set -e
set -x

# add
bash experiments/depth/eval_args/geowizard_e2e_ft/11_infer_nyu.sh
bash experiments/depth/eval_args/geowizard_e2e_ft/12_eval_nyu.sh
bash experiments/depth/eval_args/geowizard_e2e_ft/21_infer_kitti.sh
bash experiments/depth/eval_args/geowizard_e2e_ft/22_eval_kitti.sh
bash experiments/depth/eval_args/geowizard_e2e_ft/41_infer_scannet.sh
bash experiments/depth/eval_args/geowizard_e2e_ft/42_eval_scannet.sh
bash experiments/depth/eval_args/geowizard_e2e_ft/51_infer_diode.sh
bash experiments/depth/eval_args/geowizard_e2e_ft/52_eval_diode.sh
bash experiments/depth/eval_args/geowizard_e2e_ft/31_infer_eth3d.sh
bash experiments/depth/eval_args/geowizard_e2e_ft/32_eval_eth3d.sh