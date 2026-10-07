from pathlib import Path
from src.metrics import *
from src.logger import FusionLogger
import numpy as np
import torch
from tools._metric_io import (
    global_metric_value, load_image_tensor, load_mask_tensor,
    match_image_paths, mean_defined,
)


def offline_evaluation_report(
    gt_dir_path, pred_dir_path, exp_name, metric_list, output_dir,
    output_log_name=None, mask_dir_path=None, normalize_scale=10000.0,
    gt_layout='HWC', pred_layout='HWC',
):
    pairs = match_image_paths(gt_dir_path, pred_dir_path, mask_dir_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_log_name = output_log_name or f'offline_metric_cal_{exp_name}.log'
    txt_logger = FusionLogger(
        logger_name=f'offline_metric_cal_{exp_name}', log_level='INFO',
        log_file=str(output_dir / output_log_name),
    )
    all_results = {metric.__name__: [] for metric in metric_list}
    global_results = {metric.__name__: [] for metric in metric_list}
    with torch.no_grad():
        for i, (gt_path, pred_path, mask_path) in enumerate(pairs):
            gt = load_image_tensor(gt_path, normalize_scale, gt_layout)
            pred = load_image_tensor(pred_path, normalize_scale, pred_layout)
            mask = load_mask_tensor(mask_path, gt)
            if mask is not None and not mask.any():
                txt_logger.warning(f'No valid pixels in {mask_path}; metrics are undefined')
            msg = f'evaluate {i + 1}/{len(pairs)}: {gt_path.name}'
            for metric in metric_list:
                value = metric(gt, pred, mask=mask)
                name = metric.__name__
                all_results[name].append(value.cpu().numpy())
                if value.numel() > 1:
                    global_value = global_metric_value(metric, gt, pred, mask)
                    bands = '\t'.join(f'{v:.4f}' for v in value.tolist())
                    msg += f', {name}: {global_value:.4f} ({bands})'
                else:
                    global_value = value.item()
                    msg += f', {name}: {global_value:.4f}'
                global_results[name].append(global_value)
            txt_logger.info(msg)
    summary = {}
    msg = f'Average over {len(pairs)} images'
    for metric in metric_list:
        name = metric.__name__
        global_value = float(mean_defined(global_results[name]))
        bands = mean_defined(all_results[name])
        msg += f', {name}: {global_value:.4f}'
        if bands.ndim > 0:
            summary[name] = {'global': global_value, 'bands': bands.tolist()}
            msg += ' (' + '\t'.join(f'{v:.4f}' for v in bands.tolist()) + ')'
        else:
            summary[name] = global_value
    txt_logger.info(msg)
    return summary


if __name__ == '__main__':
    gt_dir_path = Path('data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/patch/Landsat_02')
    pred_dir_path = Path('results/GPSTFDiff/syy_setting-9/CIA/ablation/inference_DDPM/patch/imgs/CIA/0/save_img')
    exp_name = 'DDPM_patch'
    metric_list = [
        RMSE(is_reduce_channel=False),
        MAE(is_reduce_channel=False),
        PSNRONE(max_value=1.0, is_reduce_channel=False),
        SSIM(data_range=1.0, is_reduce_channel=False),
        ERGAS(ratio=1.0 / 16.0),
        CC(is_reduce_channel=False),
        SAM(),
        UIQI(is_reduce_channel=False),
    ]
    offline_evaluation_report(
        gt_dir_path, pred_dir_path, exp_name, metric_list, output_dir='.',
        output_log_name=f'offline_metric_cal_{exp_name}.log',
    )
