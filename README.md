# Fine-Tuning Image-Conditional Diffusion Models

[[`Paper`](https://arxiv.org/abs/2409.11355)] [[`Project Page`](https://gonzalomartingarcia.github.io/diffusion-e2e-ft/)] [[`HF demo depth`](https://huggingface.co/spaces/GonzaloMG/marigold-e2e-ft-depth)] [[`HF demo normals`](https://huggingface.co/spaces/GonzaloMG/marigold-e2e-ft-normals)] [[`BibTeX`](#-Citation)]

<img src="assets/teaser_images.png" width="600" alt="Teaser Images">

## 🔧 Setup
Tested with Python 3.10.
 1. Clone repository:

```bash
git clone https://github.com/VisualComputingInstitute/diffusion-e2e-ft.git
cd diffusion-e2e-ft
```

2. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🤖 Models

The following checkpoints are available for inference. Note that the Marigold (Depth) and GeoWizard (Depth & Normals) diffusion estimators are the official checkpoints provided by their respective authors and were not trained by us. Following the Marigold training regimen, we have trained a Marigold diffusion estimator for normals.

"E2E FT" denotes models we have fine-tuned end-to-end on task-specific losses, either starting from the pretrained diffusion estimator or directly from Stable Diffusion.
Since the fine-tuned models are single-step deterministic models, the noise should always be `zeros` and the ensemble size and number of inference steps should always be `1`. 

| Models                        |  Diffusion Estimator          |  Stable Diffusion + E2E FT                  | Diffusion Estimator + E2E FT        |
|-------------------------------|-------------------------------|---------------------------------------------|-------------------------------------|
| **Marigold (Depth)**          | `prs-eth/marigold-depth-v1-0` | `GonzaloMG/stable-diffusion-e2e-ft-depth`   | `GonzaloMG/marigold-e2e-ft-depth`   |
| **Marigold (Normals)**        | `GonzaloMG/marigold-normals`  | `GonzaloMG/stable-diffusion-e2e-ft-normals` | `GonzaloMG/marigold-e2e-ft-normals` |
| **GeoWizard (Depth&Normals)** | `lemonaddie/geowizard`        | N/A                                         |  `GonzaloMG/geowizard-e2e-ft`       |

## 🏃 Inference

1. Marigold checkpoints:
```bash
python Marigold/run.py \
    --checkpoint="GonzaloMG/marigold-e2e-ft-depth" \
    --modality depth \
    --input_rgb_dir="input" \
    --output_dir="output/marigold_ft"
```

```bash
python Marigold/run.py \
    --checkpoint="GonzaloMG/marigold-e2e-ft-normals" \
    --modality normals \
    --input_rgb_dir="input" \
    --output_dir="output/marigold_ft"
```

| Argument                | Description |
|-------------------------|-------------|
| `--checkpoint`            | Hugging Face model path. |
| `--modality`              | Output modality; `depth` or `normals`. |
| `--input_rgb_dir`         | Path to the input images. |
| `--output_dir`            | Path to the output depth or normal images. |
| `--denoise_steps`         | Number of inference steps; default `1` for E2E FT models. |
| `--ensemble_size`         | Number of samples for ensemble; default `1` for E2E FT models. |
| `--timestep_spacing`      | Defines how timesteps are distributed; `trailing` or `leading`; default `trailing` for the fixed inference schedule. |
| `--noise`                 | Noise types; `gaussian`, `pyramid`, or `zeros`; default `zeros` for E2E FT models. |
| `--processing_res`        | Resolution the model uses for generation; `0` for matching the RGB input resolution; default `768`. |
| `--output_processing_res` | If `True`, the generated image is not resized to match the RGB input resolution; default `False`. |
| `--half_precision`        | If `True`, operations are performed in half precision; default `False`. |
| `--seed`                  | Sets the seed. |
| `--batch_size`            | Batched inference when ensembling; default `1`. |
| `--resample_method`       | Resampling method used for resizing the RGB input and generated output; `bilinear`, `bicubic`, or `nearest`; default `bilinear`. |


2. GeoWizard checkpoints:

```bash
python GeoWizard/run_infer.py \
    --pretrained_model_path="GonzaloMG/geowizard-e2e-ft" \
    --domain indoor \
    --input_dir="input" \
    --output_dir="output/geowizard_ft"
```

| Argument                | Description |
|-------------------------|-------------|
| `--pretrained_model_path` | Hugging Face model path. |
| `--domain`                | Domain with respect to the RGB input; `indoor`, `outdoor`, or `object`. |
| `--input_dir`             | Path to the input images. |
| `--output_dir`            | Path to the output depth and normal images. |
| `--denoise_steps`         | Number of inference steps; default `1` for E2E FT models. |
| `--ensemble_size`         | Number of samples for ensemble; default `1` for E2E FT models. |
| `--timestep_spacing`      | Defines how timesteps are distributed; `trailing` or `leading`; default `trailing` for the fixed inference schedule. |
| `--noise`                 | Noise types; `gaussian`, `pyramid`, or `zeros`; default `zeros` for E2E FT models. |
| `--processing_res`        | Resolution the model uses for generation; `0` for matching the RGB input resolution; default `768`. |
| `--output_processing_res` | If `True`, the generated image is not resized to match the RGB input resolution; default `False`.  |
| `--half_precision`        | If `True`, operations are performed in half precision; default `False`. |
| `--seed`                  | Sets the seed.  |

By using the correct `trailing` timestep spacing, it is possible to sample single to few-step depth maps and surface normals from diffusion estimators. These samples will be blurry but become sharper by increasing the number of inference steps, e.g., from `10` to `50`. Metrics can be improved by increasing the ensemble size, e.g., to `10`. Since diffusion estimators are probabilistic models, the noise setting can be adjusted to either `gaussian` noise or multiresolution `pyramid` noise.

Our single-step deterministic E2E FT models outperform the previously mentioned diffusion estimators.

## 📋 Performance

| Depth Method                | Inference Time   | NYUv2 AbsRel↓ | KITTI AbsRel↓  | ETH3D AbsRel↓| ScanNet AbsRel↓ | DIODE AbsRel↓  |
|-----------------------------|------------------|---------------|----------------|--------------|-----------------|----------------|
| Stable Diffusion + E2E FT   | **121ms**        | 5.4           | **9.6**        | 6.4          | **5.8**         |  30.3          |
| Marigold + E2E FT           | **121ms**        | **5.2**       | **9.6**        | **6.2**      | **5.8**         | **30.2**       |
| GeoWizard + E2E FT          | **254ms**        | **5.6**       | 9.8            | 6.3          | 5.9             | 30.6           |


| Normals Method            | Inference Time | NYUv2 Mean↓  | ScanNet Mean↓ | iBims-1 Mean↓ | Sintel Mean↓  |
|---------------------------|----------------|--------------|---------------|---------------|---------------|
| Stable Diffusion + E2E FT | **121ms**      | 16.5         | 15.3          | 16.1          | 33.5          |
| Marigold + E2E FT         | **121ms**      | 16.2         | **14.7**      | **15.8**      | 33.5          |
| GeoWizard + E2E FT        | **254ms**      | **16.1**     | **14.7**      | 16.2          | **33.4**      |

Inference time is for a single 576x768-pixel image, evaluated on an NVIDIA RTX 4090 GPU.

## 📊 Evaluation

We utilize the official [Marigold](https://github.com/prs-eth/Marigold) evaluation pipeline to evaluate the affine-invariant depth estimation checkpoints, and we use the official [DSINE](https://github.com/baegwangbin/DSINE) evaluation pipeline to evaluate the surface normal estimation checkpoints. The code has been streamlined to exclude unnecessary parts, and changes have been marked.


### Depth

The Marigold evaluation datasets can be downloaded to `data/marigold_eval/` at the root of the project using the following snippet:
```bash
wget -r -np -nH --cut-dirs=4 -R "index.html*" -P data/marigold_eval/ https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/
```
After downloading, the folder structure should look as follows:
```
data
└── marigold_eval
    ├── diode
    │   └── diode_val.tar
    ├── eth3d
    │   └── eth3d.tar
    ├── kitti
    │   └── kitti_eigen_split_test.tar
    ├── nyuv2
    │   └── nyu_labeled_extracted.tar
    └── scannet
        └── scannet_val_sampled_800_1.tar
```

Run the `0_infer_eval_all.sh` script to evaluate the desired model on all datasets.

```bash
./experiments/depth/eval_args/marigold_e2e_ft/0_infer_eval_all.sh
./experiments/depth/eval_args/stable_diffusion_e2e_ft/0_infer_eval_all.sh
./experiments/depth/eval_args/geowizard_e2e_ft/0_infer_eval_all.sh
```

The evaluation results for the selected model are located in the `experiments/depth/marigold` directory. For a given dataset, the script first performs the necessary inference, storing the estimations in a `prediction` folder. Later, these depth maps are aligned and evaluated against the ground truth. Metrics and evaluation settings are available as `.txt` files.

```
<model>
└── <dataset>
    ├── arguments.txt
    ├── eval_metric
    │   └── eval_metrics-least_square.txt
    └── prediction
```


### Normals

The [DSINE evaluation datasets (`dsine_eval.zip`)](https://drive.google.com/drive/folders/1t3LMJIIrSnCGwOEf53Cyg0lkSXd3M4Hm) should be extracted into the `data` folder at the root of the project.
The folder structure should look as follows:
```
data
└── dsine_eval
   ├── ibims
   ├── nyuv2
   ├── oasis
   ├── scannet
   ├── sintel
   └── vkitti
```

The folder `experiments/normals/eval_args` contains evaluation setting `.txt` files for each `<model>`.

```bash
python -m DSINE.projects.dsine.test \
          experiments/normals/eval_args/<model>.txt \
          --mode benchmark
```

Evaluation results are saved in the `experiments/normals/dsine` folder. This includes the used settings (`params.txt`) and the metrics for each `<dataset>` (`metrics.txt`).

```
dsine
  └── <model-type/model>
      ├── log
      │   └── params.txt
      └── test
          └── <dataset>
              └── metrics.txt
```

## 🎓 Citation

If you use our work in your research, please use the following BibTeX entry.

```
@article{martingarcia2024diffusione2eft,
  title   = {Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think},
  author  = {Gonzalo Martin Garcia and Karim Abou Zeid and Christian Schmidt and Daan de Geus and Alexander Hermans and Bastian Leibe},
  journal = {arXiv preprint arXiv:2409.11355},
  year    = {2024}
}
```
