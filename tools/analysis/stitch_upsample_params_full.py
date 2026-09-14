import argparse
import json
import re
from pathlib import Path

import numpy as np
import tifffile

from src.metrics import CC, ERGAS, MAE, PSNRONE, RMSE, SAM, SSIM, UIQI
from tools.offline_metric_cal import offline_evaluation_report
from tools.offline_metric_cal_by_dict import batch_offline_evaluation_and_report


PATCH_NAME_PATTERN = re.compile(
    r"^(Group_\d+)_save_img_(\d+)_(\d+)_(\d+)_(\d+)\.tif$"
)


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


def full_gt_paths(full_gt_dir):
    paths = sorted(Path(full_gt_dir).glob("*.tif"))
    if not paths:
        raise FileNotFoundError(f"No .tif files found in {full_gt_dir}")
    return paths


def full_shapes_by_group(full_gt_dir):
    shapes = {}
    names = {}
    for gt_path in full_gt_paths(full_gt_dir):
        group = gt_path.name.split("_L_")[0]
        shapes[group] = tifffile.imread(gt_path).shape
        names[group] = gt_path.name
    return shapes, names


def parse_patch_name(path):
    match = PATCH_NAME_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected patch output name: {path.name}")
    group, y0, y1, x0, x1 = match.groups()
    return group, int(y0), int(y1), int(x0), int(x1)


def stitch_variant(patch_save_dir, full_save_dir, full_gt_dir):
    shapes, names = full_shapes_by_group(full_gt_dir)
    canvases = {}
    coverage = {}

    patch_paths = sorted(Path(patch_save_dir).glob("*.tif"))
    if not patch_paths:
        raise FileNotFoundError(f"No patch outputs found in {patch_save_dir}")

    for patch_path in patch_paths:
        group, y0, y1, x0, x1 = parse_patch_name(patch_path)
        if group not in shapes:
            raise KeyError(f"{group} is not present in full GT directory {full_gt_dir}")

        patch = tifffile.imread(patch_path)
        if group not in canvases:
            canvases[group] = np.zeros(shapes[group], dtype=patch.dtype)
            coverage[group] = np.zeros(shapes[group][:2], dtype=np.uint8)

        expected_shape = (y1 - y0, x1 - x0, *shapes[group][2:])
        if patch.shape != expected_shape:
            raise ValueError(
                f"{patch_path} shape {patch.shape} does not match {expected_shape}"
            )

        canvases[group][y0:y1, x0:x1, ...] = patch
        coverage[group][y0:y1, x0:x1] += 1

    full_save_dir = Path(full_save_dir)
    full_save_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for group in sorted(shapes):
        if group not in canvases:
            raise RuntimeError(f"Missing stitched output for {group}")
        if not np.all(coverage[group] == 1):
            raise RuntimeError(f"Patch coverage for {group} is incomplete or overlapping")
        save_path = full_save_dir / names[group]
        tifffile.imwrite(save_path, canvases[group])
        saved.append(str(save_path))
    return saved


def write_offline_metrics(output_root, full_gt_dir):
    offline_dir = Path(output_root) / "offline_metrics"
    methods_dict = {
        "original": str(Path(output_root) / "original" / "save_img"),
        "fixed": str(Path(output_root) / "fixed" / "save_img"),
    }
    batch_offline_evaluation_and_report(
        methods_dict=methods_dict,
        gt_dir_path=full_gt_dir,
        output_dir=offline_dir,
        output_log_name="batch_metrics.txt",
        metric_list=offline_metric_list(),
    )

    single_summary = {}
    for method_name, pred_dir in methods_dict.items():
        single_summary[method_name] = offline_evaluation_report(
            gt_dir_path=full_gt_dir,
            pred_dir_path=pred_dir,
            exp_name=method_name,
            metric_list=offline_metric_list(),
            output_dir=offline_dir,
            output_log_name=f"{method_name}_offline_metric_cal.log",
        )

    with (offline_dir / "single_method_summary.json").open("w") as f:
        json.dump(single_summary, f, indent=2)
    return single_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patch_root",
        type=Path,
        default=Path("compare/upsample_params/normal_config_patch_full/patch"),
    )
    parser.add_argument(
        "--full_root",
        type=Path,
        default=Path("compare/upsample_params/normal_config_patch_full/full"),
    )
    parser.add_argument(
        "--full_gt_dir",
        type=Path,
        default=Path(
            "data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/full/Landsat_02"
        ),
    )
    args = parser.parse_args()

    saved = {}
    for variant in ("original", "fixed"):
        saved[variant] = stitch_variant(
            patch_save_dir=args.patch_root / variant / "save_img",
            full_save_dir=args.full_root / variant / "save_img",
            full_gt_dir=args.full_gt_dir,
        )

    summary = {
        "patch_root": str(args.patch_root),
        "full_gt_dir": str(args.full_gt_dir),
        "stitched_outputs": saved,
        "offline_metrics": write_offline_metrics(args.full_root, args.full_gt_dir),
    }
    args.full_root.mkdir(parents=True, exist_ok=True)
    with (args.full_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
