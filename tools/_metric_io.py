from copy import copy
from pathlib import Path
import re

import numpy as np
import tifffile as tiff
import torch


def image_key(path):
    return re.sub(r'_(?:L|M|save_img|mask)_', '_', Path(path).stem)


def indexed_images(directory):
    directory = Path(directory)
    paths = sorted([*directory.glob('*.tif'), *directory.glob('*.tiff')])
    if not paths:
        raise ValueError(f'No TIFF images found in {directory}')
    index = {}
    for path in paths:
        key = image_key(path)
        if key in index:
            raise ValueError(f'Duplicate image identifier {key} in {directory}')
        index[key] = path
    return index


def match_image_paths(gt_dir_path, pred_dir_path, mask_dir_path=None):
    gt_index = indexed_images(gt_dir_path)
    pred_index = indexed_images(pred_dir_path)
    if gt_index.keys() != pred_index.keys():
        missing = sorted(gt_index.keys() - pred_index.keys())
        extra = sorted(pred_index.keys() - gt_index.keys())
        raise ValueError(f'GT/pred identifiers differ: missing={missing}, extra={extra}')
    if mask_dir_path is None:
        candidates = []
        for name in ('mask', 'Mask'):
            path = Path(gt_dir_path).parent / name
            if path.is_dir() and not any(path.samefile(existing) for existing in candidates):
                candidates.append(path)
        if len(candidates) > 1:
            raise ValueError('Multiple mask directories found; specify mask_dir_path')
        mask_dir_path = candidates[0] if candidates else None
    mask_index = None if mask_dir_path is None else indexed_images(mask_dir_path)
    if mask_index is not None and mask_index.keys() != gt_index.keys():
        raise ValueError('Mask identifiers must match the GT image identifiers')
    return [(gt_index[key], pred_index[key], None if mask_index is None else mask_index[key])
            for key in sorted(gt_index)]


def load_image_tensor(path, normalize_scale=10000.0, layout='HWC'):
    if not np.isfinite(normalize_scale) or normalize_scale <= 0:
        raise ValueError('normalize_scale must be finite and positive')
    if layout not in ('HWC', 'CHW'):
        raise ValueError("layout must be 'HWC' or 'CHW'")
    array = tiff.imread(path).astype(np.float32) / normalize_scale
    if array.ndim == 2:
        array = array[None]
    elif array.ndim == 3 and layout == 'HWC':
        array = array.transpose(2, 0, 1)
    elif array.ndim != 3:
        raise ValueError(f'Expected a 2D or 3D image in {path}')
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)


def load_mask_tensor(path, reference):
    if path is None:
        return None
    array = tiff.imread(path)
    spatial = tuple(reference.shape[-2:])
    if array.ndim == 2:
        array = array[None]
    elif array.ndim == 3:
        if array.shape[:2] == spatial:
            array = array.transpose(2, 0, 1)
        elif array.shape[-2:] != spatial:
            raise ValueError(f'Mask dimensions do not match the image in {path}')
    else:
        raise ValueError(f'Expected a 2D or 3D mask in {path}')
    return torch.from_numpy(np.ascontiguousarray(array > 0)).unsqueeze(0)


def global_metric_value(metric, gt, pred, mask=None):
    reduced_metric = copy(metric)
    if hasattr(reduced_metric, 'is_reduce_channel'):
        reduced_metric.is_reduce_channel = True
    return reduced_metric(gt, pred, mask=mask).nanmean().item()


def mean_defined(values, axis=0):
    values = np.asarray(values)
    valid = ~np.isnan(values)
    count = valid.sum(axis=axis)
    total = np.where(valid, values, 0).sum(axis=axis)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)
