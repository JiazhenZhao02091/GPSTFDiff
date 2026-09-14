# GPSTFDiff: Gaussian Pyramid Guided Progressive Diffusion for Remote Sensing Spatiotemporal Fusion

This repository contains the code used for remote sensing spatiotemporal fusion
(STF) experiments, including the proposed `GPSTFDiff` model and the comparison
methods retained in this release.

## Datasets

The paper evaluates GPSTFDiff on Landsat-MODIS STF datasets:

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
  model/                GPSTFDiff and comparison model implementations
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
@article{zhao2026gpstfdiff,
  title   = {GPSTFDiff: Gaussian Pyramid Guided Progressive Diffusion for Remote Sensing Spatiotemporal Fusion},
  author  = {Zhao, Jiazhen},
  journal = {TBD},
  year    = {2026}
}
```
