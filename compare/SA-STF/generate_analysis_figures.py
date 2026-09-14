import json
import math
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


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.unicode_minus"] = False


SA_STF_ROOT = Path(__file__).resolve().parent
COMPARE_ROOT = SA_STF_ROOT.parent
REPO_ROOT = SA_STF_ROOT.parents[1]
DATA_ROOT = REPO_ROOT / "data" / "spatio_temporal_fusion"
OUT_ROOT = SA_STF_ROOT / "analysis_figures"


DATASETS = {
    "CIA": {
        "scale": 10000.0,
        "full_gt": DATA_ROOT / "CIA/private_data/syy_setting-9/test/full/Landsat_02",
        "patch_gt": DATA_ROOT / "CIA/private_data/syy_setting-9/test/patch/Landsat_02",
        "methods": {
            "Official": {
                "full": SA_STF_ROOT / "results/syy_setting-9/CIA/full/imgs/CIA/0/save_img",
                "patch": SA_STF_ROOT / "results/syy_setting-9/CIA/patch/imgs/CIA/0/save_img",
                "summary_full": SA_STF_ROOT / "results/syy_setting-9/summary_all_full.json",
                "summary_patch": SA_STF_ROOT / "results/syy_setting-9/summary_all_patch.json",
                "summary_dataset": "CIA",
            },
            "Self-train": {
                "full": SA_STF_ROOT / "results/syy_setting-9_self_train/CIA/full/imgs/CIA/0/save_img",
                "patch": SA_STF_ROOT / "results/syy_setting-9_self_train/CIA/patch/imgs/CIA/0/save_img",
                "summary_full": SA_STF_ROOT / "results/syy_setting-9_self_train/summary_CIA_full.json",
                "summary_patch": SA_STF_ROOT / "results/syy_setting-9_self_train/summary_CIA_patch.json",
                "summary_dataset": "CIA",
            },
        },
    },
    "LGC": {
        "scale": 10000.0,
        "full_gt": DATA_ROOT / "LGC/private_data/syy_setting-9/test/full/Landsat_02",
        "patch_gt": DATA_ROOT / "LGC/private_data/syy_setting-9/test/patch/Landsat_02",
        "methods": {
            "Official": {
                "full": SA_STF_ROOT / "results/syy_setting-9/LGC/full/imgs/LGC/0/save_img",
                "patch": SA_STF_ROOT / "results/syy_setting-9/LGC/patch/imgs/LGC/0/save_img",
                "summary_full": SA_STF_ROOT / "results/syy_setting-9/summary_all_full.json",
                "summary_patch": SA_STF_ROOT / "results/syy_setting-9/summary_all_patch.json",
                "summary_dataset": "LGC",
            },
            "Self-train": {
                "full": SA_STF_ROOT / "results/syy_setting-9_self_train/LGC/full/imgs/LGC/0/save_img",
                "patch": SA_STF_ROOT / "results/syy_setting-9_self_train/LGC/patch/imgs/LGC/0/save_img",
                "summary_full": SA_STF_ROOT / "results/syy_setting-9_self_train/summary_LGC_full.json",
                "summary_patch": SA_STF_ROOT / "results/syy_setting-9_self_train/summary_LGC_patch.json",
                "summary_dataset": "LGC",
            },
        },
    },
    "ML": {
        "scale": 1.0,
        "full_gt": DATA_ROOT / "ML/private_data/syy_setting-9/test/full/Landsat_02",
        "patch_gt": DATA_ROOT / "ML/private_data/syy_setting-9/test/patch/Landsat_02",
        "methods": {
            "AHB-official": {
                "full": SA_STF_ROOT / "results/syy_setting-9_ML_official_AHB_weight_scale1/ML/full/imgs/ML/0/save_img",
                "patch": SA_STF_ROOT / "results/syy_setting-9_ML_official_AHB_weight_scale1/ML/patch/imgs/ML/0/save_img",
                "summary_full": SA_STF_ROOT / "results/syy_setting-9_ML_official_AHB_weight_scale1/summary_ML_full.json",
                "summary_patch": SA_STF_ROOT / "results/syy_setting-9_ML_official_AHB_weight_scale1/summary_ML_patch.json",
                "summary_dataset": "ML",
            },
            "Self-train": {
                "full": SA_STF_ROOT / "results/syy_setting-9_ML_self_train_scale1/ML/full/imgs/ML/0/save_img",
                "patch": SA_STF_ROOT / "results/syy_setting-9_ML_self_train_scale1/ML/patch/imgs/ML/0/save_img",
                "summary_full": SA_STF_ROOT / "results/syy_setting-9_ML_self_train_scale1/summary_ML_full.json",
                "summary_patch": SA_STF_ROOT / "results/syy_setting-9_ML_self_train_scale1/summary_ML_patch.json",
                "summary_dataset": "ML",
            },
        },
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_hwc(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.shape[0] in (1, 3, 4, 6) and arr.shape[-1] not in (1, 3, 4, 6):
        return arr.transpose(1, 2, 0)
    return arr


def read_reflectance(path: Path, scale: float) -> np.ndarray:
    arr = to_hwc(tiff.imread(path).astype(np.float32))
    if scale != 1.0:
        arr = arr / scale
    elif float(np.nanmax(arr)) > 1.5:
        arr = arr / 10000.0
    return np.clip(arr, 0.0, 1.0)


def group_key(path: Path) -> str:
    match = re.match(r"^(Group_\d+)", path.name)
    if not match:
        return path.stem
    return match.group(1)


def patch_key(path: Path) -> str:
    name = path.stem
    name = name.replace("_L_", "_")
    name = name.replace("_save_img_", "_")
    name = name.rstrip("_")
    return name


def find_by_group(pred_dir: Path, key: str) -> Path:
    matches = sorted(pred_dir.glob(f"{key}*.tif*"))
    if not matches:
        raise FileNotFoundError(f"missing prediction for {key}: {pred_dir}")
    return matches[0]


def stretch_rgb(img: np.ndarray, bands: Tuple[int, int, int] = (2, 1, 0)) -> np.ndarray:
    rgb = img[:, :, list(bands)]
    out = np.zeros_like(rgb, dtype=np.float32)
    for idx in range(3):
        band = rgb[:, :, idx]
        lo, hi = np.percentile(band[np.isfinite(band)], [2, 98])
        if hi <= lo:
            out[:, :, idx] = np.clip(band, 0, 1)
        else:
            out[:, :, idx] = np.clip((band - lo) / (hi - lo), 0, 1)
    return out


def mean_abs_error(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(gt - pred), axis=2)


def sample_band_pairs(gt: np.ndarray, pred: np.ndarray, band: int, n: int, rng: np.random.Generator):
    x = gt[:, :, band].reshape(-1)
    y = pred[:, :, band].reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if len(x) > n:
        idx = rng.choice(len(x), size=n, replace=False)
        return x[idx], y[idx], x, y
    return x, y, x, y


def scalar_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    diff = pred - gt
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))
    psnr = float(20.0 * np.log10(1.0 / max(rmse, 1e-12)))
    g = gt.reshape(-1)
    p = pred.reshape(-1)
    if np.std(g) < 1e-8 or np.std(p) < 1e-8:
        cc = float("nan")
    else:
        cc = float(np.corrcoef(g, p)[0, 1])
    dot = np.sum(gt * pred, axis=2)
    ng = np.linalg.norm(gt, axis=2)
    npred = np.linalg.norm(pred, axis=2)
    cos = np.clip(dot / np.maximum(ng * npred, 1e-12), -1, 1)
    sam = float(np.mean(np.arccos(cos)))
    return {"RMSE": rmse, "MAE": mae, "PSNR": psnr, "CC": cc, "SAM": sam}


def load_summary(path: Path, dataset: str) -> Dict[str, float]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        return rows
    for row in rows:
        if row.get("dataset") == dataset:
            return row
    raise RuntimeError(f"dataset {dataset} not found in {path}")


def plot_error_and_visual_grids(dataset: str, cfg: Dict) -> List[Path]:
    outputs = []
    out_dir = OUT_ROOT / "error_maps" / dataset
    visual_dir = OUT_ROOT / "visual_comparison" / dataset
    ensure_dir(out_dir)
    ensure_dir(visual_dir)
    for gt_path in sorted(cfg["full_gt"].glob("*.tif*")):
        key = group_key(gt_path)
        gt = read_reflectance(gt_path, cfg["scale"])
        preds = {
            method: read_reflectance(find_by_group(method_cfg["full"], key), cfg["scale"])
            for method, method_cfg in cfg["methods"].items()
        }
        errors = {method: mean_abs_error(gt, pred) for method, pred in preds.items()}
        vmax = max(np.percentile(error, 98) for error in errors.values())
        norm = mcolors.PowerNorm(gamma=1.5, vmin=0, vmax=max(vmax, 1e-6))
        cmap = "OrRd"

        individual_base = out_dir / "individual"
        for method, error in errors.items():
            method_dir = individual_base / method
            ensure_dir(method_dir)
            fig, ax = plt.subplots(figsize=(6, 6 * error.shape[0] / error.shape[1]))
            ax.imshow(error, cmap=cmap, norm=norm)
            ax.axis("off")
            fig.subplots_adjust(0, 0, 1, 1)
            out_path = method_dir / f"errormap_{key}.png"
            fig.savefig(out_path, dpi=450, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            outputs.append(out_path)

        cols = 1 + len(preds) * 2
        fig, axes = plt.subplots(1, cols, figsize=(3.3 * cols, 3.6))
        axes[0].imshow(stretch_rgb(gt))
        axes[0].set_title("GT", fontweight="bold")
        axes[0].axis("off")
        col = 1
        for method, pred in preds.items():
            axes[col].imshow(stretch_rgb(pred))
            axes[col].set_title(method, fontweight="bold")
            axes[col].axis("off")
            col += 1
            axes[col].imshow(errors[method], cmap=cmap, norm=norm)
            axes[col].set_title(f"{method} error", fontweight="bold")
            axes[col].axis("off")
            col += 1
        fig.tight_layout()
        out_path = out_dir / f"Comparison_{key}.png"
        fig.savefig(out_path, dpi=350, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)

        fig_cbar, ax_cbar = plt.subplots(figsize=(0.55, 4.5))
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig_cbar.colorbar(sm, cax=ax_cbar)
        cbar.ax.tick_params(labelsize=11)
        out_path = out_dir / f"colorbar_{key}.png"
        fig_cbar.savefig(out_path, dpi=350, bbox_inches="tight", facecolor="white")
        plt.close(fig_cbar)
        outputs.append(out_path)

        fig, axes = plt.subplots(2, 1 + len(preds), figsize=(3.4 * (1 + len(preds)), 6.6))
        axes[0, 0].imshow(stretch_rgb(gt))
        axes[0, 0].set_title("GT RGB", fontweight="bold")
        axes[0, 0].axis("off")
        axes[1, 0].axis("off")
        for idx, (method, pred) in enumerate(preds.items(), start=1):
            axes[0, idx].imshow(stretch_rgb(pred))
            axes[0, idx].set_title(f"{method} RGB", fontweight="bold")
            axes[0, idx].axis("off")
            axes[1, idx].imshow(errors[method], cmap=cmap, norm=norm)
            axes[1, idx].set_title(f"{method} abs error", fontweight="bold")
            axes[1, idx].axis("off")
        fig.tight_layout()
        out_path = visual_dir / f"Visual_Analysis_{key}.png"
        fig.savefig(out_path, dpi=350, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def plot_scatter(dataset: str, cfg: Dict) -> List[Path]:
    outputs = []
    out_dir = OUT_ROOT / "scatter" / dataset
    ensure_dir(out_dir)
    rng = np.random.default_rng(42)
    bands = [("RED", 2, "red"), ("NIR", 3, "blue")]
    for gt_path in sorted(cfg["full_gt"].glob("*.tif*")):
        key = group_key(gt_path)
        gt = read_reflectance(gt_path, cfg["scale"])
        fig, axes = plt.subplots(1, len(cfg["methods"]), figsize=(5 * len(cfg["methods"]), 4.8))
        if len(cfg["methods"]) == 1:
            axes = [axes]
        for ax, (method, method_cfg) in zip(axes, cfg["methods"].items()):
            pred = read_reflectance(find_by_group(method_cfg["full"], key), cfg["scale"])
            for band_name, band_idx, color in bands:
                x, y, x_all, y_all = sample_band_pairs(gt, pred, band_idx, 25000, rng)
                ax.scatter(y, x, c=color, s=2, alpha=0.4, edgecolors="none", label=band_name)
                if len(x_all) > 10 and np.var(y_all) > 1e-8:
                    m, b = np.polyfit(y_all.astype(np.float64), x_all.astype(np.float64), 1)
                    ax.plot([0, 1], [b, m + b], color=color, lw=1.6)
            ax.plot([0, 1], [0, 1], "k--", lw=1.1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal", "box")
            ax.set_title(method, fontweight="bold")
            ax.set_xlabel("Prediction")
            ax.set_ylabel("Ground Truth")
            ax.legend(loc="upper left", frameon=False, fontsize=9, markerscale=4)
        fig.suptitle(f"{dataset} {key} RED/NIR scatter", y=1.02, fontweight="bold")
        fig.tight_layout()
        out_path = out_dir / f"{key}_red_nir_scatter.png"
        fig.savefig(out_path, dpi=350, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)

        fig, axes = plt.subplots(2, len(cfg["methods"]), figsize=(5 * len(cfg["methods"]), 9.5))
        if len(cfg["methods"]) == 1:
            axes = axes[:, None]
        sc_ref = None
        for col, (method, method_cfg) in enumerate(cfg["methods"].items()):
            pred = read_reflectance(find_by_group(method_cfg["full"], key), cfg["scale"])
            for row, (band_name, band_idx, _) in enumerate(bands):
                ax = axes[row, col]
                x, y, _, _ = sample_band_pairs(gt, pred, band_idx, 35000, rng)
                hist, xedges, yedges = np.histogram2d(y, x, bins=150, range=[[0, 1], [0, 1]])
                ix = np.clip(np.digitize(y, xedges) - 1, 0, 149)
                iy = np.clip(np.digitize(x, yedges) - 1, 0, 149)
                z = hist[ix, iy]
                order = np.argsort(z)
                sc_ref = ax.scatter(y[order], x[order], c=z[order], s=2, cmap="jet", edgecolors="none")
                ax.plot([0, 1], [0, 1], "k--", lw=1.1)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect("equal", "box")
                ax.set_title(f"{method} {band_name}", fontweight="bold")
                ax.set_xlabel("Prediction")
                ax.set_ylabel("Ground Truth")
        fig.tight_layout()
        out_path = out_dir / f"{key}_density_scatter.png"
        fig.savefig(out_path, dpi=350, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)
        if sc_ref is not None:
            fig_cbar, ax_cbar = plt.subplots(figsize=(0.55, 4.5))
            fig_cbar.colorbar(plt.cm.ScalarMappable(norm=sc_ref.norm, cmap=sc_ref.cmap), cax=ax_cbar)
            out_path = out_dir / f"colorbar_{key}.png"
            fig_cbar.savefig(out_path, dpi=350, bbox_inches="tight")
            plt.close(fig_cbar)
            outputs.append(out_path)
    return outputs


def iter_patch_pairs(cfg: Dict, method_cfg: Dict) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    scale = cfg["scale"]
    pred_by_key = {patch_key(path): path for path in method_cfg["patch"].glob("*.tif*")}
    for gt_path in sorted(cfg["patch_gt"].glob("*.tif*")):
        key = patch_key(gt_path)
        pred_path = pred_by_key.get(key)
        if pred_path is None:
            continue
        gt = read_reflectance(gt_path, scale)
        pred = read_reflectance(pred_path, scale)
        if gt.shape == pred.shape and float(np.max(gt)) > 1e-6:
            yield gt, pred


def plot_patch_boxplots(dataset: str, cfg: Dict) -> List[Path]:
    out_dir = OUT_ROOT / "boxplots"
    ensure_dir(out_dir)
    metric_names = ["RMSE", "CC", "SAM"]
    results = {metric: {} for metric in metric_names}
    for method, method_cfg in cfg["methods"].items():
        values = {metric: [] for metric in metric_names}
        for gt, pred in iter_patch_pairs(cfg, method_cfg):
            metrics = scalar_metrics(gt, pred)
            for metric in metric_names:
                if np.isfinite(metrics[metric]):
                    values[metric].append(metrics[metric])
        for metric in metric_names:
            results[metric][method] = values[metric]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, metric in zip(axes, metric_names):
        data = [results[metric][method] for method in cfg["methods"]]
        ax.boxplot(
            data,
            tick_labels=list(cfg["methods"].keys()),
            patch_artist=True,
            widths=0.5,
            showfliers=False,
            boxprops=dict(facecolor="white", color="black", linewidth=1.2),
            medianprops=dict(color="darkorange", linewidth=2.0),
            whiskerprops=dict(color="black", linewidth=1.2),
            capprops=dict(color="black", linewidth=1.2),
        )
        ax.set_ylabel(metric)
        ax.set_xlabel("Method")
        ax.set_title(metric, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{dataset} patch metric distribution", fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = out_dir / f"{dataset}_patch_metric_boxplot.png"
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)
    return [out_path]


def plot_metric_bars(dataset: str, cfg: Dict) -> List[Path]:
    out_dir = OUT_ROOT / "metric_bars"
    ensure_dir(out_dir)
    metrics = ["PSNRONE", "SSIM", "RMSE", "MAE", "CC", "SAM", "UIQI", "ergas"]
    rows = []
    labels = []
    for method, method_cfg in cfg["methods"].items():
        for split, summary_key in [("full", "summary_full"), ("patch", "summary_patch")]:
            row = load_summary(method_cfg[summary_key], method_cfg["summary_dataset"])
            rows.append([float(row[m]) for m in metrics])
            labels.append(f"{method}\n{split}")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    x = np.arange(len(rows))
    for ax, metric_idx in zip(axes, range(len(metrics))):
        values = [row[metric_idx] for row in rows]
        colors = ["#5277A3", "#7BA6C8", "#B55A4A", "#D98F75"][: len(values)]
        ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(metrics[metric_idx], fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{dataset} SA-STF summary metrics", fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = out_dir / f"{dataset}_summary_metric_bars.png"
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)
    return [out_path]


def plot_band_heatmaps() -> List[Path]:
    out_dir = OUT_ROOT / "band_heatmaps"
    ensure_dir(out_dir)
    rows = json.loads((SA_STF_ROOT / "results/sastf_band_metrics.json").read_text(encoding="utf-8"))
    outputs = []
    label_map = {
        "官方预训练": "Official",
        "自训练 final": "Self-train final",
        "AHB 官方预训练迁移 scale1": "AHB-official scale1",
        "自训练 scale1 final": "Self-train scale1 final",
    }
    for dataset in DATASETS:
        ds_rows = [row for row in rows if row["dataset"] == dataset]
        if not ds_rows:
            continue
        labels = [f"{label_map.get(row['weight_type'], row['weight_type'])}\n{row['split']}" for row in ds_rows]
        metric_specs = [("RMSE", "viridis_r"), ("PSNRONE", "viridis"), ("SSIM", "viridis"), ("CC", "viridis")]
        fig, axes = plt.subplots(1, len(metric_specs), figsize=(5 * len(metric_specs), 0.55 * len(ds_rows) + 2.5))
        for ax, (metric, cmap) in zip(axes, metric_specs):
            data = np.array([row["metrics"][metric] for row in ds_rows], dtype=np.float32)
            im = ax.imshow(data, aspect="auto", cmap=cmap)
            ax.set_title(metric, fontweight="bold")
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_xticks(np.arange(6))
            ax.set_xticklabels([f"B{i}" for i in range(1, 7)])
            for r in range(data.shape[0]):
                for c in range(data.shape[1]):
                    ax.text(c, r, f"{data[r, c]:.3f}", ha="center", va="center", fontsize=7, color="white" if metric != "PSNRONE" else "black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"{dataset} per-band metrics", fontweight="bold", y=1.02)
        fig.tight_layout()
        out_path = out_dir / f"{dataset}_per_band_metric_heatmaps.png"
        fig.savefig(out_path, dpi=350, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def plot_classifier_metrics() -> List[Path]:
    out_dir = OUT_ROOT / "classification"
    ensure_dir(out_dir)
    root = SA_STF_ROOT / "classifier_eval/ML_syy_setting-9_scale1"
    runs = [
        ("AHB-official\nfull", root / "SA-STF_official_AHB_weight_scale1/full/metrics_summary.json"),
        ("AHB-official\npatch", root / "SA-STF_official_AHB_weight_scale1/patch/metrics_summary.json"),
        ("Self-train\nfull", root / "SA-STF_self_train_scale1_final/full/metrics_summary.json"),
        ("Self-train\npatch", root / "SA-STF_self_train_scale1_final/patch/metrics_summary.json"),
    ]
    if not all(path.exists() for _, path in runs):
        return []
    rows = [json.loads(path.read_text(encoding="utf-8")) for _, path in runs]
    labels = [label for label, _ in runs]
    metrics = [("average_oa", "OA"), ("average_kappa", "Kappa"), ("average_f1_macro", "F1 macro"), ("average_miou", "mIoU")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    x = np.arange(len(labels))
    for ax, (key, title) in zip(axes, metrics):
        values = [row[key] for row in rows]
        ax.bar(x, values, color=["#5277A3", "#7BA6C8", "#B55A4A", "#D98F75"], edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("ML classifier evaluation on SA-STF outputs", fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = out_dir / "ML_classifier_metric_bars.png"
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)

    class_names = list(rows[0]["class_f1"].keys())
    data = np.array([[row["class_f1"][name] for name in class_names] for row in rows], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=25, ha="right")
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            ax.text(c, r, f"{data[r, c]:.3f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_title("Class-wise F1", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    fig.tight_layout()
    out_path_2 = out_dir / "ML_classifier_class_f1_heatmap.png"
    fig.savefig(out_path_2, dpi=350, bbox_inches="tight")
    plt.close(fig)
    return [out_path, out_path_2]


def write_index(outputs: List[Path]) -> Path:
    by_parent: Dict[str, List[Path]] = {}
    for path in outputs:
        key = str(path.parent.relative_to(OUT_ROOT))
        by_parent.setdefault(key, []).append(path)
    lines = [
        "# SA-STF analysis figures",
        "",
        "Generated from valid SA-STF syy_setting-9 outputs. ML uses scale1 rerun results only.",
        "",
    ]
    for parent in sorted(by_parent):
        lines.append(f"## {parent}")
        lines.append("")
        for path in sorted(by_parent[parent]):
            lines.append(f"- `{path.relative_to(COMPARE_ROOT)}`")
        lines.append("")
    out_path = OUT_ROOT / "README.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    ensure_dir(OUT_ROOT)
    outputs: List[Path] = []
    for dataset, cfg in DATASETS.items():
        print(f"[{dataset}] scatter")
        outputs.extend(plot_scatter(dataset, cfg))
        print(f"[{dataset}] error and visual grids")
        outputs.extend(plot_error_and_visual_grids(dataset, cfg))
        print(f"[{dataset}] patch boxplots")
        outputs.extend(plot_patch_boxplots(dataset, cfg))
        print(f"[{dataset}] metric bars")
        outputs.extend(plot_metric_bars(dataset, cfg))
    print("[all] per-band heatmaps")
    outputs.extend(plot_band_heatmaps())
    print("[ML] classifier figures")
    outputs.extend(plot_classifier_metrics())
    index_path = write_index(outputs)
    print(index_path)
    print(f"generated {len(outputs)} figures")


if __name__ == "__main__":
    main()
