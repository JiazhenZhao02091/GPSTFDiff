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

## Running Experiments

Run commands from the repository root. GPU selection follows the environment;
scripts do not override `CUDA_VISIBLE_DEVICES`.

```bash
CUDA_VISIBLE_DEVICES=0 python -m tools.train.train_GPSTFDiff \
  --congfig_path config/GPSTFDiff/syy_setting-9/CIA/config.py

CUDA_VISIBLE_DEVICES=0 python -m tools.inference.test_GPSTFDiff_lap \
  --congfig_path config/GPSTFDiff/syy_setting-9/CIA/inference_lap.py
```

The existing argument name is `--congfig_path`. Update checkpoint filenames in
the inference configuration to match your trained or downloaded weights.
The `src/data/` package includes the dataset classes, transforms, and
dataloader helpers used by these configurations. Place the prepared TIFF
datasets in the configured data directories before running these commands.

Configuration paths default to the repository root. Set `GPSTFDIFF_ROOT` to
use another root, or set `GPSTFDIFF_DATA_DIR` and `GPSTFDIFF_RESULTS_DIR` to
override the corresponding directories. The EDCSTFN baseline also accepts
`GPSTFDIFF_AUTOENCODER_CHECKPOINT` for its pretrained autoencoder.

Evaluate prediction folders with explicit inputs:

```bash
python -m tools.offline_metric_cal_by_dict \
  --gt-dir /path/to/Landsat_02 \
  --method GPSTFDiff=/path/to/GPSTFDiff_predictions \
  --method STFDiff=/path/to/STFDiff_predictions \
  --output-dir results/evaluation

python -m tools.k_means --pred-image /path/to/prediction.tif \
  --cdl /path/to/cdl.tif --output-dir results/classification
```

`python -m tools.efficient_stat.param_flops` reports THOP-counted MACs and
parameters for one denoiser forward pass, rather than the entire sampling
process. `python -m tools.efficient_stat.time_cal` measures progressive
sampling on synthetic inputs; use `--device cpu` when CUDA is unavailable.
Both tools accept `--without-adamdr`; unsupported legacy ablation variants
have been removed. Run the metric tests with
`python -m unittest discover -s tests -v`.

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
