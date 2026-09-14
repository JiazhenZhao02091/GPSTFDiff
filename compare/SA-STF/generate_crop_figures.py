"""
Generate SA-STF crop figures aligned with the PPT crop coordinates.

The crop coordinates are taken from tools/analysis/subimg.py,
tools/analysis/sub_errormap.py, and tools/analysis/sub_classifer_map.py.
Only self-trained SA-STF results are used.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from PIL import Image, ImageDraw


SA_STF_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SA_STF_ROOT.parents[1]
DATA_ROOT = REPO_ROOT / "data" / "spatio_temporal_fusion"
OUT_ROOT = SA_STF_ROOT / "analysis_figures" / "crops"


BBox = Tuple[int, int, int, int]  # y_start, y_end, x_start, x_end


DATASETS = {
    "CIA": {
        "scale": 10000.0,
        "gt_dir": DATA_ROOT / "CIA/private_data/syy_setting-9/test/full/Landsat_02",
        "pred_dir": SA_STF_ROOT / "results/syy_setting-9_self_train/CIA/full/imgs/CIA/0/save_img",
        "bboxes": [(1320, 1576, 760, 1016), (820, 1076, 900, 1156)],
    },
    "LGC": {
        "scale": 10000.0,
        "gt_dir": DATA_ROOT / "LGC/private_data/syy_setting-9/test/full/Landsat_02",
        "pred_dir": SA_STF_ROOT / "results/syy_setting-9_self_train/LGC/full/imgs/LGC/0/save_img",
        "bboxes": [(120, 376, 140, 396), (700, 956, 980, 1236)],
    },
    "ML": {
        "scale": 1.0,
        "gt_dir": DATA_ROOT / "ML/private_data/syy_setting-9/test/full/Landsat_02",
        "pred_dir": SA_STF_ROOT / "results/syy_setting-9_ML_self_train_scale1/ML/full/imgs/ML/0/save_img",
        "bboxes": [(676, 932, 608, 864), (206, 462, 1008, 1264)],
    },
}


CLASSIFIER_INPUT = (
    SA_STF_ROOT
    / "classifier_eval/ML_syy_setting-9_scale1/SA-STF_self_train_scale1_final/full"
)
CLASSIFIER_BBOXES: List[BBox] = [
    (560, 944, 520, 904),
    (140, 524, 700, 1084),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def group_key(path: Path) -> str:
    match = re.match(r"^(Group_\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot parse group key from {path}")
    return match.group(1)


def to_hwc(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4, 6) and arr.shape[-1] not in (1, 3, 4, 6):
        return arr.transpose(1, 2, 0)
    return arr


def read_tif(path: Path) -> np.ndarray:
    return to_hwc(tiff.imread(path).astype(np.float32))


def read_reflectance(path: Path, scale: float) -> np.ndarray:
    arr = read_tif(path)
    if scale != 1.0:
        arr = arr / scale
    elif float(np.nanmax(arr)) > 1.5:
        arr = arr / 10000.0
    return np.clip(arr, 0.0, 1.0)


def find_group_file(directory: Path, key: str) -> Path:
    matches = sorted(path for path in directory.glob("*.tif*") if group_key(path) == key)
    if not matches:
        raise FileNotFoundError(f"No file for {key} in {directory}")
    return matches[0]


def stretch_rgb_321(arr: np.ndarray) -> np.ndarray:
    arr = to_hwc(arr).astype(np.float32)
    if arr.shape[-1] >= 4:
        rgb = arr[:, :, [3, 2, 1]]
    elif arr.shape[-1] >= 3:
        rgb = arr[:, :, :3]
    else:
        rgb = np.repeat(arr[:, :, :1], 3, axis=-1)
    if float(np.nanmax(rgb)) <= 2.0:
        rgb = rgb * 10000.0
    lo = np.percentile(rgb, 2, axis=(0, 1), keepdims=True)
    hi = np.percentile(rgb, 98, axis=(0, 1), keepdims=True)
    denom = np.where(hi - lo <= 0, 1.0, hi - lo)
    rgb = (rgb - lo) / denom * 255.0
    return np.clip(np.nan_to_num(rgb), 0, 255).astype(np.uint8)


def crop_array(arr: np.ndarray, bbox: BBox) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    return arr[y0:y1, x0:x1, ...]


def save_marked_image(image: Image.Image, path: Path, bboxes: Iterable[BBox], width: int = 3) -> None:
    marked = image.copy().convert("RGB")
    draw = ImageDraw.Draw(marked)
    for y0, y1, x0, x1 in bboxes:
        draw.rectangle([x0, y0, x1, y1], outline="red", width=width)
    marked.save(path)


def save_rgb_crops(dataset: str, key: str, gt_rgb: np.ndarray, pred_rgb: np.ndarray, bboxes: List[BBox]) -> List[Path]:
    out_dir = OUT_ROOT / "visual_subimages" / dataset / key
    ensure_dir(out_dir)
    outputs: List[Path] = []
    images = {"GT": gt_rgb, "SA-STF_self_train": pred_rgb}
    for name, arr in images.items():
        full = Image.fromarray(arr)
        full_path = out_dir / f"{name}_full_321.png"
        full.save(full_path)
        outputs.append(full_path)
        marked_path = out_dir / f"{name}_full_marked_321.png"
        save_marked_image(full, marked_path, bboxes)
        outputs.append(marked_path)
        for bbox in bboxes:
            y0, y1, x0, x1 = bbox
            patch_path = out_dir / f"{name}_{y0}_{y1}_{x0}_{x1}_patch_321.png"
            Image.fromarray(crop_array(arr, bbox)).save(patch_path)
            outputs.append(patch_path)
    return outputs


def save_error_crops(dataset: str, key: str, error: np.ndarray, bboxes: List[BBox]) -> List[Path]:
    out_dir = OUT_ROOT / "error_maps" / dataset / key
    ensure_dir(out_dir)
    outputs: List[Path] = []
    vmax = max(float(np.percentile(error, 98)), 1e-6)
    norm = mcolors.PowerNorm(gamma=1.5, vmin=0.0, vmax=vmax)
    mapper = cm.ScalarMappable(cmap="OrRd", norm=norm)
    mapper.set_array([])

    full_rgba = mapper.to_rgba(error, bytes=True)
    full_img = Image.fromarray(full_rgba)
    full_path = out_dir / "SA-STF_self_train_full_errormap.png"
    full_img.save(full_path)
    outputs.append(full_path)

    marked_path = out_dir / "SA-STF_self_train_full_marked_errormap.png"
    save_marked_image(full_img, marked_path, bboxes)
    outputs.append(marked_path)

    for bbox in bboxes:
        y0, y1, x0, x1 = bbox
        patch_rgba = mapper.to_rgba(crop_array(error, bbox), bytes=True)
        patch_path = out_dir / f"SA-STF_self_train_{y0}_{y1}_{x0}_{x1}_patch_errormap.png"
        Image.fromarray(patch_rgba).save(patch_path)
        outputs.append(patch_path)

    fig, ax = plt.subplots(figsize=(0.6, 5))
    cbar = fig.colorbar(mapper, cax=ax)
    cbar.ax.tick_params(labelsize=14)
    cbar_path = out_dir / f"colorbar_{key}.png"
    fig.savefig(cbar_path, bbox_inches="tight", dpi=600, facecolor="white")
    plt.close(fig)
    outputs.append(cbar_path)
    return outputs


def save_classifier_crops() -> List[Path]:
    outputs: List[Path] = []
    if not CLASSIFIER_INPUT.exists():
        return outputs

    out_root = OUT_ROOT / "classification" / "ML" / "SA-STF_self_train_scale1_final" / "full"
    for group_dir in sorted(path for path in CLASSIFIER_INPUT.iterdir() if path.is_dir() and path.name.startswith("Group_")):
        out_group = out_root / group_dir.name
        ensure_dir(out_group)
        for src_path in sorted(group_dir.glob("*.png")):
            image = Image.open(src_path).convert("RGBA")
            prefix = src_path.stem
            full_path = out_group / f"{prefix}_full.png"
            image.save(full_path)
            outputs.append(full_path)

            marked_path = out_group / f"{prefix}_full_marked.png"
            save_marked_image(image, marked_path, CLASSIFIER_BBOXES, width=4)
            outputs.append(marked_path)
            for idx, bbox in enumerate(CLASSIFIER_BBOXES, start=1):
                y0, y1, x0, x1 = bbox
                patch = image.crop((x0, y0, x1, y1))
                patch_path = out_group / f"{prefix}_patch_{idx}_{y0}_{y1}_{x0}_{x1}.png"
                patch.save(patch_path)
                outputs.append(patch_path)
    return outputs


def generate_dataset_crops(dataset: str, cfg: Dict) -> List[Path]:
    outputs: List[Path] = []
    for gt_path in sorted(cfg["gt_dir"].glob("*.tif*")):
        key = group_key(gt_path)
        pred_path = find_group_file(cfg["pred_dir"], key)
        gt_raw = read_tif(gt_path)
        pred_raw = read_tif(pred_path)
        gt_ref = read_reflectance(gt_path, cfg["scale"])
        pred_ref = read_reflectance(pred_path, cfg["scale"])
        if gt_ref.shape != pred_ref.shape:
            raise ValueError(f"Shape mismatch for {dataset} {key}: {gt_ref.shape} vs {pred_ref.shape}")
        error = np.mean(np.abs(gt_ref - pred_ref), axis=-1)

        outputs.extend(
            save_rgb_crops(
                dataset=dataset,
                key=key,
                gt_rgb=stretch_rgb_321(gt_raw),
                pred_rgb=stretch_rgb_321(pred_raw),
                bboxes=cfg["bboxes"],
            )
        )
        outputs.extend(save_error_crops(dataset=dataset, key=key, error=error, bboxes=cfg["bboxes"]))
    return outputs


def write_index(outputs: List[Path]) -> Path:
    index_path = OUT_ROOT / "README.md"
    ensure_dir(OUT_ROOT)
    lines = [
        "# SA-STF crop figures",
        "",
        "Generated with `compare/SA-STF/generate_crop_figures.py`.",
        "Only self-trained SA-STF results are included.",
        "",
        "Crop coordinates match the PPT scripts:",
        "",
        "- visual/error crops: `tools/analysis/subimg.py` and `tools/analysis/sub_errormap.py`",
        "- classification crops: `tools/analysis/sub_classifer_map.py`",
        "",
    ]
    for path in sorted(outputs):
        lines.append(f"- `{path.relative_to(SA_STF_ROOT)}`")
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path


def main() -> None:
    outputs: List[Path] = []
    for dataset, cfg in DATASETS.items():
        outputs.extend(generate_dataset_crops(dataset, cfg))
    outputs.extend(save_classifier_crops())
    index_path = write_index(outputs)
    print(f"Generated {len(outputs)} files")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
