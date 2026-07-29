# LapSTFDiff: Laplacian-Guided Progressive Diffusion for Remote Sensing Spatiotemporal Fusion

This repository contains the code used for remote sensing spatiotemporal fusion
(STF) experiments, including the proposed `LapSTFDiff` model and the comparison
methods retained in this release.

LapSTFDiff formulates multispectral STF as a conditional generative diffusion
problem. It combines a change-aware conditional denoising network, Ada-UNet,
with an inference-stage Laplacian-guided progressive sampler. Ada-UNet extracts
multi-temporal change priors from coarse observations and a fine-resolution
reference image, uses FiLM-based feature modulation for conditional interaction,
and introduces an Adaptive Multiscale Detail Refiner (AdaMDR) to preserve
high-frequency structural details. The progressive sampler reorganizes reverse
diffusion into a coarse-to-fine trajectory, recovering low-frequency radiometric
structure before refining local textures and boundaries.

## Paper Scope

The paper evaluates LapSTFDiff on Landsat-MODIS STF datasets:

- `CIA`: Coleambally Irrigation Area benchmark.
- `LGC`: Lower Gwydir Catchment benchmark.
- `ML`: a mixed-agricultural dataset covering McLean County, Illinois.

The experiments use common reconstruction and spectral metrics, including
RMSE, MAE, PSNR, SSIM, ERGAS, CC, SAM, and UIQI, and also include downstream
land-cover classification validation on the ML dataset.

## Repository Layout

```text
config/                 Experiment configuration files
scripts/                Dataset preparation and formatting utilities
setting/                Shell setting helpers
src/
  inferencer/           Inference loops
  logger/               Logging and metric trackers
  metrics/              Reconstruction and spectral metrics
  model/                LapSTFDiff and comparison model implementations
  trainer/              Training loops
  utils/                General utilities
tools/
  analysis/             Figure, error-map, scatter, and box-plot utilities
  classifier/           Downstream land-cover classification utilities
  efficient_stat/       Complexity and inference-time helpers
  inference/            Inference entry scripts
  train/                Training entry scripts
```

The proposed method is organized under:

```text
src/model/LapSTFDiff/
src/trainer/LapSTFDiff*.py
src/inferencer/LapSTFDiff*.py
config/LapSTFDiff/
tools/train/train_LapSTFDiff*.py
tools/inference/test_LapSTFDiff*.py
```

## Installation

Create a Python environment, install PyTorch for your CUDA version, then install
the remaining dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

The project code historically uses the package namespace `src`. The editable
install keeps that import pattern intact.

## Data and Checkpoints

This release intentionally does not include datasets, pretrained checkpoints,
generated results, paper assets, caches, or large binary files.

Expected dataset paths in configs follow this pattern:

```text
data/spatio_temporal_fusion/<DATASET>/private_data/syy_setting-9/<split>
```

Typical image keys are:

```text
Landsat_01, Landsat_02, MODIS_01, MODIS_02
```

Before running an experiment, adjust each config file so that `data_root` and
checkpoint paths point to your local files.

Many configs import dataset helpers from `src.data.*`. If your checkout does
not include `src/data/`, restore the data-loading package from the full project
before running training or inference scripts.

## Training LapSTFDiff

Example command for the CIA configuration:

```bash
python tools/train/train_LapSTFDiff_lap.py \
  --congfig_path config/LapSTFDiff/lap/syy_setting-9/CIA/mult_4_128/config_x1_x2_x3.py
```

The command-line argument is intentionally named `--congfig_path` to remain
compatible with the original scripts.

## Inference

Example command:

```bash
python tools/inference/test_LapSTFDiff_lap.py \
  --congfig_path config/LapSTFDiff/lap/syy_setting-9/CIA/inference/inferency_8.py
```

For inference configs, update:

```python
checkpoint_path_x1_x2_x3
checkpoint_path_x2_x3
checkpoint_path_x3
```

to match the checkpoints produced by your training runs.

## Metrics and Analysis

Offline metric and visualization utilities are kept under `tools/analysis/` and
`tools/offline_metric_cal.py`. They assume generated predictions are saved under
`results/` and ground-truth images are available under `data/`. Adjust the paths
inside each script before use.

Downstream land-cover classification utilities are kept under
`tools/classifier/`. The final full-image evaluation entry is
`tools/classifier/evaluate_all_methods.py`; it produces class visualization
PNGs, confidence maps, confusion matrices, and classification metrics such as
OA, Kappa, macro-F1, and mIoU for LapSTFDiff and the comparison methods.

## Notes for Release Users

- The codebase is an experiment-oriented research repository, so configurations
  preserve the original naming conventions where possible.
- Training and inference outputs are written under `results/` by default.
- Large data, checkpoints, generated images, and paper figures are ignored by
  `.gitignore`.
- Some comparison methods require additional third-party dependencies beyond the
  common setup, depending on which script is executed.

## Citation

If this repository is useful for your work, please cite the corresponding paper
once the final bibliographic information is available:

```bibtex
@article{zhao2026lapstfdiff,
  title   = {LapSTFDiff: Laplacian-Guided Progressive Diffusion for Remote Sensing Spatiotemporal Fusion},
  author  = {Zhao, Jiazhen},
  journal = {TBD},
  year    = {2026}
}
```
