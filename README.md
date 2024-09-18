# Fine-Tuning Image-Conditional Diffusion Models

[[`Paper`](https://arxiv.org/abs/2409.11355)] [[`HF demo`](https://huggingface.co/spaces/GonzaloMG/diffusion-e2e-ft)] [[`BibTeX`](#-Citation)]

<img src="assets/teaser_images.png" width="600">

## 🔧 Setup
Tested with Python 3.10.
 1. Clone repository:

```bash
git clone https://github.com/VisualComputingInstitute/stable-e2e-ft.git
cd stable-e2e-ft
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
<!--| Marigold + `trailing`       | **121ms**        | 5.7           | 10.8           | 6.9          | 6.6             |  31.1          |
| GeoWizard + `trailing`      | **254ms**        | 5.8           | 13.3           | 7.8          | 6.2             |  32            |-->


| Normals Method            | Inference Time | NYUv2 Mean↓  | ScanNet Mean↓ | iBims-1 Mean↓ | Sintel Mean↓  |
|---------------------------|----------------|--------------|---------------|---------------|---------------|
| Stable Diffusion + E2E FT | **121ms**      | 16.5         | 15.3          | 16.1          | 33.5          |
| Marigold + E2E FT         | **121ms**      | 16.2         | **14.7**      | **15.8**      | 33.5          |
| GeoWizard + E2E FT        | **254ms**      | **16.1**     | **14.7**      | 16.2          | **33.4**      |
<!-- | Marigold + `trailing`     | **121ms**      | 18.8         | 17.7          | 18.4          | 39.1          |
| GeoWizard + `trailing`    | **254ms**      | 17.0         | 15.5          | 18.3          | 35.9          |-->

Inference time is for a single 576x768-pixel image, evaluated on an NVIDIA RTX 4090 GPU.

## 🎓 Citation

If you use our work in your research, please use the following BibTeX entry.

```
@article{garcia2024diffusione2eft,
  title   = {Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think},
  author  = {Gonzalo Martin Garcia and Karim Abou Zeid and Christian Schmidt and Daan de Geus and Alexander Hermans and Bastian Leibe},
  journal = {arXiv preprint arXiv:2409.11355},
  year    = {2024}
}
```
