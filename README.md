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

## Datasets

The paper evaluates LapSTFDiff on Landsat-MODIS STF datasets:

- `CIA`: Coleambally Irrigation Area benchmark.
- `LGC`: Lower Gwydir Catchment benchmark.
- `ML`: a mixed-agricultural dataset covering McLean County, Illinois.

The experiments use common reconstruction and spectral metrics, including
RMSE, MAE, PSNR, SSIM, ERGAS, CC, SAM, and UIQI, and also include downstream
land-cover classification validation on the **ML** dataset.

[Data download](https://drive.google.com/file/d/1cJWPX89Gpmn_aepAIRdEmXuflTwtkOUH/view?usp=drive_link)

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
  inference/            Inference entry scripts
  train/                Training entry scripts
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
