from pathlib import Path
import json
import sys

import numpy as np
import tifffile as tiff
import torch

SA_STF_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SA_STF_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics import CC, ERGAS, MAE, PSNRONE, RMSE, SAM, SSIM, UIQI


COMPARE_ROOT = REPO_ROOT / "compare"
DATA_ROOT = REPO_ROOT / "data" / "spatio_temporal_fusion"


RUNS = [
    ("CIA", "官方预训练", "full", DATA_ROOT / "CIA/private_data/syy_setting-9/test/full/Landsat_02", COMPARE_ROOT / "SA-STF/results/syy_setting-9/CIA/full/imgs/CIA/0/save_img"),
    ("CIA", "官方预训练", "patch", DATA_ROOT / "CIA/private_data/syy_setting-9/test/patch/Landsat_02", COMPARE_ROOT / "SA-STF/results/syy_setting-9/CIA/patch/imgs/CIA/0/save_img"),
    ("CIA", "自训练 final", "full", DATA_ROOT / "CIA/private_data/syy_setting-9/test/full/Landsat_02", COMPARE_ROOT / "SA-STF/results/syy_setting-9_self_train/CIA/full/imgs/CIA/0/save_img"),
    ("CIA", "自训练 final", "patch", DATA_ROOT / "CIA/private_data/syy_setting-9/test/patch/Landsat_02", COMPARE_ROOT / "SA-STF/results/syy_setting-9_self_train/CIA/patch/imgs/CIA/0/save_img"),
    ("LGC", "官方预训练", "full", DATA_ROOT / "LGC/private_data/syy_setting-9/test/full/Landsat_02", COMPARE_ROOT / "SA-STF/results/syy_setting-9/LGC/full/imgs/LGC/0/save_img"),
    ("LGC", "官方预训练", "patch", DATA_ROOT / "LGC/private_data/syy_setting-9/test/patch/Landsat_02", COMPARE_ROOT / "SA-STF/results/syy_setting-9/LGC/patch/imgs/LGC/0/save_img"),
    ("LGC", "自训练 final", "full", DATA_ROOT / "LGC/private_data/syy_setting-9/test/full/Landsat_02", COMPARE_ROOT / "SA-STF/results/syy_setting-9_self_train/LGC/full/imgs/LGC/0/save_img"),
    ("LGC", "自训练 final", "patch", DATA_ROOT / "LGC/private_data/syy_setting-9/test/patch/Landsat_02", COMPARE_ROOT / "SA-STF/results/syy_setting-9_self_train/LGC/patch/imgs/LGC/0/save_img"),
    ("ML", "AHB 官方预训练迁移 scale1", "full", DATA_ROOT / "ML/private_data/syy_setting-9/test/full/Landsat_02", COMPARE_ROOT / "SA-STF/results/syy_setting-9_ML_official_AHB_weight_scale1/ML/full/imgs/ML/0/save_img"),
    ("ML", "AHB 官方预训练迁移 scale1", "patch", DATA_ROOT / "ML/private_data/syy_setting-9/test/patch/Landsat_02", COMPARE_ROOT / "SA-STF/results/syy_setting-9_ML_official_AHB_weight_scale1/ML/patch/imgs/ML/0/save_img"),
    ("ML", "自训练 scale1 final", "full", DATA_ROOT / "ML/private_data/syy_setting-9/test/full/Landsat_02", COMPARE_ROOT / "SA-STF/results/syy_setting-9_ML_self_train_scale1/ML/full/imgs/ML/0/save_img"),
    ("ML", "自训练 scale1 final", "patch", DATA_ROOT / "ML/private_data/syy_setting-9/test/patch/Landsat_02", COMPARE_ROOT / "SA-STF/results/syy_setting-9_ML_self_train_scale1/ML/patch/imgs/ML/0/save_img"),
]


METRICS = [
    RMSE(is_reduce_channel=False),
    MAE(is_reduce_channel=False),
    PSNRONE(max_value=1.0, is_reduce_channel=False),
    SSIM(data_range=1.0, is_reduce_channel=False),
    ERGAS(ratio=1.0 / 16.0),
    CC(is_reduce_channel=False),
    SAM(),
    UIQI(is_reduce_channel=False),
]


def to_hwc(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.shape[0] in (1, 3, 4, 6) and arr.shape[-1] not in (1, 3, 4, 6):
        return arr.transpose(1, 2, 0)
    return arr


def format_vector(values: list[float]) -> str:
    return "/".join(f"{value:.6f}" for value in values)


def default_scale(dataset: str) -> float:
    return 1.0 if dataset == "ML" else 10000.0


def compute_run(dataset: str, weight_type: str, split: str, gt_dir: Path, pred_dir: Path) -> dict:
    scale = default_scale(dataset)
    gt_paths = sorted(gt_dir.glob("*.tif"))
    pred_paths = sorted(pred_dir.glob("*.tif"))
    if len(gt_paths) != len(pred_paths):
        raise RuntimeError(
            f"count mismatch {dataset} {weight_type} {split}: "
            f"gt={len(gt_paths)} pred={len(pred_paths)}"
        )

    all_results = {metric.__name__: [] for metric in METRICS}
    for gt_path, pred_path in zip(gt_paths, pred_paths):
        gt = to_hwc(tiff.imread(str(gt_path))).astype(np.float32) / scale
        pred = to_hwc(tiff.imread(str(pred_path))).astype(np.float32) / scale
        if gt.shape != pred.shape:
            raise RuntimeError(f"shape mismatch: {gt_path} {gt.shape} vs {pred_path} {pred.shape}")

        gt_tensor = torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0)
        pred_tensor = torch.from_numpy(pred.transpose(2, 0, 1)).unsqueeze(0)
        for metric in METRICS:
            value = metric(gt_tensor, pred_tensor)
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            all_results[metric.__name__].append(np.asarray(value))

    metrics_summary = {}
    for name, values in all_results.items():
        mean_value = np.mean(np.array(values), axis=0)
        metrics_summary[name] = mean_value.reshape(-1).astype(float).tolist()

    return {
        "dataset": dataset,
        "weight_type": weight_type,
        "split": split,
        "num_images": len(gt_paths),
        "scale": scale,
        "metrics": metrics_summary,
    }


def main() -> None:
    rows = [compute_run(*run) for run in RUNS]

    output_json = SA_STF_ROOT / "results" / "sastf_band_metrics.json"
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    output_md = SA_STF_ROOT / "results" / "sastf_band_metrics_table.md"
    lines = [
        "| 数据集 | 权重类型 | split | 图像数 | RMSE(B1-B6) | MAE(B1-B6) | PSNRONE(B1-B6) | SSIM(B1-B6) | CC(B1-B6) | UIQI(B1-B6) | ERGAS | SAM |",
        "|---|---|---|---:|---|---|---|---|---|---|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            "| {dataset} | {weight_type} | {split} | {num_images} | {rmse} | {mae} | "
            "{psnr} | {ssim} | {cc} | {uiqi} | {ergas:.6f} | {sam:.6f} |".format(
                dataset=row["dataset"],
                weight_type=row["weight_type"],
                split=row["split"],
                num_images=row["num_images"],
                rmse=format_vector(metrics["RMSE"]),
                mae=format_vector(metrics["MAE"]),
                psnr=format_vector(metrics["PSNRONE"]),
                ssim=format_vector(metrics["SSIM"]),
                cc=format_vector(metrics["CC"]),
                uiqi=format_vector(metrics["UIQI"]),
                ergas=metrics["ergas"][0],
                sam=metrics["SAM"][0],
            )
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_json)
    print(output_md)


if __name__ == "__main__":
    main()
