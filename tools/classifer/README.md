# crop_unet

This folder contains the U-Net crop classifier used by the downstream classification evaluation.

## Files

- `train.py`: U-Net classifier training entry.
- `config.py`: training config, Landsat/CDL label paths, and checkpoint save path.
- `evaluate_full.py`: full-image downstream classification evaluation entry.
- `evaluate_patch.py`: patch-wise downstream classification evaluation entry.
- `inference_config.py`: inference/evaluation config and label mapping.
- `save_checkpoint/Unet_checkpoint_epochs_1000.pth`: checkpoint used by the current downstream classification scripts.

## Data

The classifier uses the ML Landsat images and CDL labels under the project data directory:

- `data/spatio_temporal_fusion/ML/Crop/Landsat`
- `data/spatio_temporal_fusion/ML/CDL_cropped`
- `data/spatio_temporal_fusion/ML/CDL_patch`

## Run

Run from the project root:

```bash
python crop_unet/train.py
python crop_unet/evaluate_full.py
python crop_unet/evaluate_patch.py
```

Evaluation outputs are written to `results/crop_unet/`.
