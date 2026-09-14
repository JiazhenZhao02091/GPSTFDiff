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


def to_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.shape[0] not in (1, 3, 4, 6) and arr.shape[-1] in (1, 3, 4, 6):
        return arr.transpose(2, 0, 1)
    return arr


def read_tensor(path: Path, scale: float) -> torch.Tensor:
    arr = to_chw(tiff.imread(str(path)).astype(np.float32)) / scale
    return torch.from_numpy(arr).unsqueeze(0)


def as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() > 1:
            return float(value.mean().item())
        return float(value.item())
    arr = np.asarray(value)
    return float(arr.mean())


def normalize_pred_name(pred_name: str) -> str:
    return pred_name.replace("_save_img_.tif", ".tif")


def target_path_for_prediction(gt_dir: Path, pred_path: Path) -> Path:
    sample_id = normalize_pred_name(pred_path.name).removesuffix(".tif")
    if sample_id.count("_") == 1:
        return gt_dir / f"{sample_id}_L_.tif"
    group = "_".join(sample_id.split("_")[:2])
    suffix = "_".join(sample_id.split("_")[2:])
    return gt_dir / f"{group}_L_{suffix}.tif"


def evaluate(gt_dir: Path, pred_dir: Path, gt_scale: float, pred_scale: float) -> dict:
    pred_paths = sorted(pred_dir.glob("*.tif"))
    values = {metric.__name__: [] for metric in METRICS}
    for pred_path in pred_paths:
        gt_path = target_path_for_prediction(gt_dir, pred_path)
        gt = read_tensor(gt_path, gt_scale)
        pred = read_tensor(pred_path, pred_scale)
        for metric in METRICS:
            values[metric.__name__].append(as_float(metric(gt, pred)))
    return {key: float(np.mean(items)) for key, items in values.items()}


def main() -> None:
    data_root = REPO_ROOT / "data" / "spatio_temporal_fusion" / "ML" / "private_data" / "syy_setting-9" / "test"
    compare_root = REPO_ROOT / "compare" / "SA-STF" / "results"
    runs = [
        ("AHB official full", data_root / "full" / "Landsat_02", compare_root / "syy_setting-9_ML_official_AHB_weight/ML/full/imgs/ML/0/save_img"),
        ("AHB official patch", data_root / "patch" / "Landsat_02", compare_root / "syy_setting-9_ML_official_AHB_weight/ML/patch/imgs/ML/0/save_img"),
        ("self final full", data_root / "full" / "Landsat_02", compare_root / "syy_setting-9_ML_self_train/ML/full/imgs/ML/0/save_img"),
        ("self final patch", data_root / "patch" / "Landsat_02", compare_root / "syy_setting-9_ML_self_train/ML/patch/imgs/ML/0/save_img"),
    ]
    output = {}
    for name, gt_dir, pred_dir in runs:
        output[name] = {
            "old_wrong_scale_gt10000_pred10000": evaluate(gt_dir, pred_dir, 10000.0, 10000.0),
            "correct_existing_output_gt1_pred10000": evaluate(gt_dir, pred_dir, 1.0, 10000.0),
        }
    out_path = compare_root / "ml_scale_diagnostic_metrics.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
