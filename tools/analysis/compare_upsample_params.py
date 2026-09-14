import argparse
import copy
import csv
import importlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataloader.data_sampler import EpochBasedSampler
from src.metrics import CC, ERGAS, MAE, RMSE, SAM, SSIM, UIQI, PSNRONE
from src.model.GPSTFDiff.diffusion_lap_inferencer import Diffusion as OriginalDiffusion
from src.model.GPSTFDiff.diffusion_lap_inferencer_with_upsample_alpha import (
    Diffusion as FixedDiffusion,
)
from tools.offline_metric_cal import offline_evaluation_report
from tools.offline_metric_cal_by_dict import batch_offline_evaluation_and_report


def config_path_to_module(config_path):
    return config_path.replace("/", ".").replace(".py", "")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_state_dict(ckpt_path, ema_prefix="ema_model.model."):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ema_ckpt = ckpt["ema"]
    state_dict = {}
    for key, value in ema_ckpt.items():
        if key.startswith(ema_prefix):
            state_dict[key[len(ema_prefix) :]] = value
    if not state_dict:
        raise RuntimeError(f"No EMA keys with prefix {ema_prefix!r} in {ckpt_path}")
    return state_dict


def build_model(diffusion_cls, template_model):
    return diffusion_cls(
        model=copy.deepcopy(template_model.model),
        image_size=template_model.image_size,
        num_train_timesteps=template_model.num_train_timesteps,
        sampling_timesteps=template_model.sampling_timesteps,
        loss_type=template_model.loss_type,
        objective=template_model.objective,
        mode=template_model.mode,
        T1=template_model.T1,
        T2=template_model.T2,
        ddim_sampling_eta=template_model.ddim_sampling_eta,
        model_x3=copy.deepcopy(template_model.model_x3),
        model_x2_x3=copy.deepcopy(template_model.model_x2_x3),
        model_x1_x2_x3=copy.deepcopy(template_model.model_x1_x2_x3),
    )


def load_stage_checkpoints(model, config):
    ckpt_x3 = checkpoint_state_dict(config.checkpoint_path_x3)
    ckpt_x2_x3 = checkpoint_state_dict(config.checkpoint_path_x2_x3)
    ckpt_x1_x2_x3 = checkpoint_state_dict(config.checkpoint_path_x1_x2_x3)
    model.model_x3.load_state_dict(ckpt_x3, strict=False)
    model.model_x2_x3.load_state_dict(ckpt_x2_x3, strict=False)
    model.model_x1_x2_x3.load_state_dict(ckpt_x1_x2_x3, strict=False)


def build_test_dataloader(config, data_root):
    if data_root is None:
        return config.test_dataloader
    dataset = config.dataset_cls_func(
        dataset_name="CIA",
        data_root=data_root,
        transform_func_list=config.test_transform_list,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=EpochBasedSampler(dataset=dataset, is_shuffle=False, seed=42),
        num_workers=0,
    )


def denormalize(save_tensor, normalize_scale, normalize_mode):
    save_img = save_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
    if normalize_mode == 1:
        save_img = save_img * normalize_scale
    elif normalize_mode == 2:
        save_img = (save_img + 1.0) / 2.0 * normalize_scale
    save_img = np.clip(save_img, 0, normalize_scale)
    if normalize_scale == 255:
        return save_img.astype(np.uint8)
    if normalize_scale == 1:
        return save_img.astype(np.float32)
    if normalize_scale == 10000:
        return save_img.astype(np.uint16)
    return save_img.astype(np.float32)


def save_name_from_key(key):
    save_name = key.split("-")[-1]
    return save_name[:8] + "_save_img_" + save_name[9:] + ".tif"


def batch_to_inputs(data_per_batch, device):
    return [
        data_per_batch["coarse_img_01"].to(device),
        data_per_batch["coarse_img_02"].to(device),
        data_per_batch["fine_img_01"].to(device),
    ]


def metric_values(metric_list, outputs, metrics_gt):
    values = {}
    outputs_01 = (outputs + 1.0) / 2.0
    metrics_gt_01 = (metrics_gt + 1.0) / 2.0
    for metric in metric_list:
        value = metric(outputs_01, metrics_gt_01)
        values[metric.__name__] = float(value.detach().cpu().item())
    return values


def run_variant(name, diffusion_cls, config, args, device):
    variant_dir = args.output_root / name
    save_dir = variant_dir / "save_img"
    save_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    model = build_model(diffusion_cls, config.model).to(device)
    load_stage_checkpoints(model, config)
    model.eval()

    metric_list = [copy.deepcopy(metric).to(device) for metric in config.metric_list]
    rows = []

    with torch.no_grad():
        for iter_idx, data_per_batch in enumerate(args.test_dataloader):
            if args.max_samples is not None and iter_idx >= args.max_samples:
                break

            inputs = batch_to_inputs(data_per_batch, device)
            metrics_gt = data_per_batch["fine_img_02"].to(device)
            normalize_scale = data_per_batch["normalize_scale"][0].numpy()
            normalize_mode = data_per_batch["normalize_mode"][0].numpy()
            key = data_per_batch["key"][0]

            set_seed(args.seed + iter_idx)
            outputs = model.sample(*inputs)
            loss = float(F.mse_loss(outputs, metrics_gt).detach().cpu().item())
            values = metric_values(metric_list, outputs, metrics_gt)

            save_path = save_dir / save_name_from_key(key)
            tifffile.imwrite(
                save_path,
                denormalize(outputs, normalize_scale, normalize_mode),
            )

            row = {"variant": name, "iter": iter_idx, "key": key, "loss": loss}
            row.update(values)
            row["save_path"] = str(save_path)
            rows.append(row)
            print(f"{name}: {iter_idx + 1} samples", flush=True)

    del model
    torch.cuda.empty_cache()
    return rows


def summarize(rows):
    metric_keys = [
        key
        for key in rows[0].keys()
        if key not in {"variant", "iter", "key", "save_path"}
    ]
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in metric_keys
    }


def write_outputs(output_root, original_rows, fixed_rows, args):
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows = original_rows + fixed_rows
    fieldnames = list(all_rows[0].keys())
    with (output_root / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "config_path": args.config_path,
        "data_root": str(args.data_root) if args.data_root is not None else None,
        "gt_dir": str(args.gt_dir) if args.gt_dir is not None else None,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "original": summarize(original_rows),
        "fixed": summarize(fixed_rows),
    }
    with (output_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    write_pairwise_delta(output_root, original_rows, fixed_rows)
    if args.gt_dir is not None and args.max_samples is None:
        write_offline_metrics(output_root, args.gt_dir)


def write_pairwise_delta(output_root, original_rows, fixed_rows):
    original_by_key = {row["key"]: row for row in original_rows}
    fixed_by_key = {row["key"]: row for row in fixed_rows}
    metric_keys = [
        key
        for key in original_rows[0].keys()
        if key not in {"variant", "iter", "key", "save_path"}
    ]
    rows = []
    for key in sorted(original_by_key):
        if key not in fixed_by_key:
            continue
        row = {"key": key}
        for metric_key in metric_keys:
            row[f"original_{metric_key}"] = original_by_key[key][metric_key]
            row[f"fixed_{metric_key}"] = fixed_by_key[key][metric_key]
            row[f"delta_{metric_key}"] = (
                fixed_by_key[key][metric_key] - original_by_key[key][metric_key]
            )
        rows.append(row)

    with (output_root / "paired_delta.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def offline_metric_list():
    return [
        RMSE(is_reduce_channel=False),
        MAE(is_reduce_channel=False),
        PSNRONE(max_value=1.0, is_reduce_channel=False),
        SSIM(data_range=1.0, is_reduce_channel=False),
        ERGAS(ratio=1.0 / 16.0),
        CC(is_reduce_channel=False),
        SAM(),
        UIQI(is_reduce_channel=False),
    ]


def write_offline_metrics(output_root, gt_dir):
    offline_dir = output_root / "offline_metrics"
    methods_dict = {
        "original": str(output_root / "original" / "save_img"),
        "fixed": str(output_root / "fixed" / "save_img"),
    }
    batch_offline_evaluation_and_report(
        methods_dict=methods_dict,
        gt_dir_path=gt_dir,
        output_dir=offline_dir,
        output_log_name="batch_metrics.txt",
        metric_list=offline_metric_list(),
    )

    single_summary = {}
    for method_name, pred_dir in methods_dict.items():
        single_summary[method_name] = offline_evaluation_report(
            gt_dir_path=gt_dir,
            pred_dir_path=pred_dir,
            exp_name=method_name,
            metric_list=offline_metric_list(),
            output_dir=offline_dir,
            output_log_name=f"{method_name}_offline_metric_cal.log",
        )

    with (offline_dir / "single_method_summary.json").open("w") as f:
        json.dump(single_summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default="config/GPSTFDiff/lap/syy_setting-9/CIA/inference/inferency_10step.py",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("compare/upsample_params"),
    )
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--gt_dir", type=Path, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = importlib.import_module(config_path_to_module(args.config_path))
    args.test_dataloader = build_test_dataloader(config, args.data_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config_path, args.output_root / Path(args.config_path).name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_rows = run_variant("original", OriginalDiffusion, config, args, device)
    fixed_rows = run_variant("fixed", FixedDiffusion, config, args, device)
    write_outputs(args.output_root, original_rows, fixed_rows, args)


if __name__ == "__main__":
    main()
