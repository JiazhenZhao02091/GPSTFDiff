# Downstream Classifier Utilities

This folder contains the land-cover classification code used for downstream
validation on the ML dataset.

## Files

- `config.py`: shared class mapping, data paths, checkpoint path, and runtime
  settings for the classifier.
- `train_unet.py`: trains the multispectral U-Net classifier with automatically
  estimated class weights from the CDL masks.
- `single_img_inference.py`: runs the trained classifier on one Landsat-style
  image and writes a CDL-ID label map.
- `evaluate_all_methods.py`: final full-image evaluation script for all STF
  methods. It writes class visualizations, confidence maps, confusion matrices,
  and `Metrics_Summary.txt`.
- `evaluate_patch_methods.py`: patch-wise evaluation variant.

## Expected Inputs

The default paths are relative to the repository root:

```text
data/spatio_temporal_fusion/ML/public_processing_data/format_data/crop_118_1654_125_2173/original/Landsat
data/spatio_temporal_fusion/ML/CDL_cropped
checkpoints/classifier/Unet_checkpoint_epochs_1000.pth
results/<method>/...
```

Datasets, checkpoints, and generated result images are not included in this
release. Place the trained classifier checkpoint at the configured
`checkpoints/classifier/` path or update `config.py` before running inference.

## Commands

Train the U-Net classifier:

```bash
python tools/classifier/train_unet.py
```

Run single-image classification:

```bash
python tools/classifier/single_img_inference.py
```

Run final full-image downstream evaluation:

```bash
python tools/classifier/evaluate_all_methods.py
```

Run patch-wise downstream evaluation:

```bash
python tools/classifier/evaluate_patch_methods.py
```
