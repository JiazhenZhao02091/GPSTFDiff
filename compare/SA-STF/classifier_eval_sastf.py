import argparse
import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import tifffile as tiff
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from debug.classfier.unet.inference_config import CONFIG
from scripts.spatio_temparol_fusion.dataset_generation.dataset_config.syy_setting import DATASET_SETTING

CLASS_NAMES = {
    0: "Corn",
    1: "Soybean",
    2: "Grassland",
    3: "Forest",
    4: "Developed",
    255: "Background",
}
VALID_LABELS = [0, 1, 2, 3, 4]
VALID_CLASS_STRINGS = [CLASS_NAMES[i] for i in VALID_LABELS]
CLASS_CMAP = mcolors.ListedColormap(["#90C44B", "#389644", "#EDDEA9", "#0F5724", "#800000"])


class Logger:
    def __init__(self, filename: Path):
        self.terminal = sys.stdout
        self.log = filename.open("a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_year_from_filename(filename: str) -> Optional[str]:
    match = re.search(r"(\d{4})", filename)
    return match.group(1) if match else None


def get_inference_config(location: str, split: str) -> List[Dict[str, str]]:
    triplets = DATASET_SETTING.get(location, {}).get(split, [])
    dataset_split = []
    for idx, triplet in enumerate(triplets):
        target_date = triplet[1] if len(triplet) >= 2 else (triplet[0] if triplet else None)
        if target_date is None:
            continue
        dataset_split.append(
            {
                "group_id": f"Group_{idx + 1:02d}",
                "date": target_date,
                "year": get_year_from_filename(target_date) or target_date.split("-")[0],
            }
        )
    if dataset_split:
        return dataset_split
    return [
        {"group_id": "Group_01", "year": "2021"},
        {"group_id": "Group_02", "year": "2022"},
        {"group_id": "Group_03", "year": "2023"},
        {"group_id": "Group_04", "year": "2023"},
    ]


def generate_legend_log(log_dir: Path) -> None:
    norm = mcolors.Normalize(vmin=0, vmax=4)
    with (log_dir / "color_legend.txt").open("w", encoding="utf-8") as f:
        f.write("=== Class Color Legend (RGB) ===\n")
        for class_id in VALID_LABELS:
            rgba = CLASS_CMAP(norm(class_id))
            r, g, b = [int(x * 255) for x in rgba[:3]]
            f.write(f"ID {class_id:<3} | {CLASS_NAMES[class_id]:<12} | RGB: ({r:>3}, {g:>3}, {b:>3})\n")
        f.write("ID 255 | Background   | RGB: (255, 255, 255) [White]\n")


def save_colored_png(array: np.ndarray, save_path: Path, valid_mask: Optional[np.ndarray] = None) -> None:
    norm = mcolors.Normalize(vmin=0, vmax=4)
    colored_img = CLASS_CMAP(norm(array))
    rgb_img = (colored_img[:, :, :3] * 255).astype(np.uint8)
    if valid_mask is not None:
        rgb_img[~valid_mask] = [255, 255, 255]
    Image.fromarray(rgb_img).save(save_path)


def save_confidence_map(conf_array: np.ndarray, save_path: Path, valid_mask: Optional[np.ndarray] = None) -> None:
    cmap = plt.get_cmap("jet")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    colored_img = cmap(norm(conf_array))
    rgb_img = (colored_img[:, :, :3] * 255).astype(np.uint8)
    if valid_mask is not None:
        rgb_img[~valid_mask] = [255, 255, 255]
    Image.fromarray(rgb_img).save(save_path)


def plot_confusion_matrix(cm: np.ndarray, save_path: Path, title: str) -> None:
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=VALID_CLASS_STRINGS,
        yticklabels=VALID_CLASS_STRINGS,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def preprocess(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] < img.shape[0]:
        img = np.transpose(img, (2, 0, 1))
    elif img.ndim == 2:
        img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)


def load_classifier(device: str) -> torch.nn.Module:
    model = smp.Unet(
        encoder_name="resnet18",
        in_channels=CONFIG["in_channels"],
        classes=CONFIG["num_classes"],
        encoder_weights=None,
    )
    model_path = Path(CONFIG["model_path"])
    if not model_path.exists():
        model_path = REPO_ROOT / model_path
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def resolve_existing_path(path: Optional[Path]) -> Optional[Path]:
    if path is None or path.exists() or path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    return repo_path if repo_path.exists() else path


@torch.no_grad()
def predict_with_confidence(model: torch.nn.Module, img: np.ndarray, device: str):
    x = torch.from_numpy(img).to(device).unsqueeze(0)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    conf, pred = torch.max(probs, dim=1)
    return pred.cpu().numpy()[0], conf.cpu().numpy()[0]


def cdl_to_label(cdl_path: Path):
    cdl_arr = tiff.imread(cdl_path)
    gt_label = np.full_like(cdl_arr, CONFIG["ignore_label"])
    for k, v in CONFIG["id_map"].items():
        gt_label[cdl_arr == k] = v
    return gt_label, gt_label != CONFIG["ignore_label"]


def find_full_pred(pred_dir: Path, group_id: str) -> Optional[Path]:
    candidates = sorted(pred_dir.glob(f"{group_id}*save_img*.tif*"))
    return candidates[0] if candidates else None


def iter_patch_preds(pred_dir: Path, group_id: str) -> Iterable[tuple[Path, str]]:
    pattern = re.compile(rf"^({re.escape(group_id)})_(\d+_\d+_\d+_\d+)_save_img_.*\.tif(?:f)?$")
    for path in sorted(pred_dir.glob(f"{group_id}_*save_img*.tif*")):
        match = pattern.match(path.name)
        if match:
            yield path, match.group(2)


def find_cdl_full(cdl_dir: Path, year: str) -> Optional[Path]:
    candidates = sorted([p for p in cdl_dir.glob("*.tif*") if year in p.name])
    return candidates[0] if candidates else None


def compute_and_store(
    method_name: str,
    mode: str,
    rows: List[Dict[str, float]],
    global_cm: np.ndarray,
    save_dir: Path,
) -> None:
    precision = np.diag(global_cm) / np.maximum(np.sum(global_cm, axis=0), 1)
    recall = np.diag(global_cm) / np.maximum(np.sum(global_cm, axis=1), 1)
    class_f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
    summary = {
        "method": method_name,
        "mode": mode,
        "num_items": len(rows),
        "average_oa": float(np.mean([r["oa"] for r in rows])),
        "average_kappa": float(np.mean([r["kappa"] for r in rows])),
        "average_f1_macro": float(np.mean([r["f1_macro"] for r in rows])),
        "average_miou": float(np.mean([r["miou"] for r in rows])),
        "class_f1": {VALID_CLASS_STRINGS[i]: float(class_f1[i]) for i in range(len(VALID_LABELS))},
        "global_confusion_matrix": global_cm.tolist(),
        "items": rows,
    }
    (save_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "*" * 50,
        f"Global Summary for {method_name.upper()} ({mode}, {len(rows)} items)",
        "*" * 50,
        f"Average OA:      {summary['average_oa']:.4f}",
        f"Average Kappa:   {summary['average_kappa']:.4f}",
        f"Average F1(mac): {summary['average_f1_macro']:.4f}",
        f"Average mIoU:    {summary['average_miou']:.4f}",
        "",
        "[Class-wise F1 Scores]",
    ]
    for name, value in summary["class_f1"].items():
        lines.append(f"  - {name:<12}: {value:.4f}")
    lines.append("*" * 50)
    (save_dir / "Metrics_Summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_confusion_matrix(global_cm, save_dir / f"{method_name}_ConfusionMatrix.png", f"Confusion Matrix - {method_name}")
    print("\n".join(lines), flush=True)


def evaluate_full(args, model, dataset_split) -> None:
    save_dir = args.output_root / args.method_name / "full"
    save_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    global_cm = np.zeros((len(VALID_LABELS), len(VALID_LABELS)), dtype=np.int64)
    for item in dataset_split:
        group_id = item["group_id"]
        pred_path = find_full_pred(args.full_pred_dir, group_id)
        cdl_path = find_cdl_full(args.full_cdl_dir, item["year"])
        if pred_path is None or cdl_path is None:
            print(f"[skip full] {group_id}: pred={pred_path} cdl={cdl_path}", flush=True)
            continue
        gt_label, valid_mask = cdl_to_label(cdl_path)
        p_map, conf_map = predict_with_confidence(model, preprocess(tiff.imread(pred_path)), args.device)
        p = p_map[valid_mask]
        g = gt_label[valid_mask]
        cm = confusion_matrix(g, p, labels=VALID_LABELS)
        global_cm += cm
        intersection = np.diag(cm)
        union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - intersection
        row = {
            "group_id": group_id,
            "year": item["year"],
            "pred_path": str(pred_path),
            "cdl_path": str(cdl_path),
            "oa": float(accuracy_score(g, p)),
            "kappa": float(cohen_kappa_score(g, p)),
            "f1_macro": float(f1_score(g, p, average="macro", zero_division=0)),
            "miou": float(np.mean(intersection / (union + 1e-6))),
        }
        rows.append(row)
        group_dir = save_dir / group_id
        group_dir.mkdir(parents=True, exist_ok=True)
        save_colored_png(gt_label, group_dir / "GT_Class.png", valid_mask)
        save_colored_png(p_map, group_dir / f"{args.method_name}_Class.png", valid_mask)
        save_confidence_map(conf_map, group_dir / f"{args.method_name}_Confidence.png", valid_mask)
        print(f"[full] {group_id} OA={row['oa']:.4f} Kappa={row['kappa']:.4f} mIoU={row['miou']:.4f}", flush=True)
    if rows:
        compute_and_store(args.method_name, "full", rows, global_cm, save_dir)


def evaluate_patch(args, model, dataset_split) -> None:
    save_dir = args.output_root / args.method_name / "patch"
    save_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    global_cm = np.zeros((len(VALID_LABELS), len(VALID_LABELS)), dtype=np.int64)
    for item in dataset_split:
        group_id = item["group_id"]
        year = item["year"]
        for pred_path, coords in iter_patch_preds(args.patch_pred_dir, group_id):
            cdl_path = args.patch_cdl_dir / f"CDL{year}_{coords}.tif"
            if not cdl_path.exists():
                print(f"[skip patch] missing {cdl_path}", flush=True)
                continue
            gt_label, valid_mask = cdl_to_label(cdl_path)
            if not np.any(valid_mask):
                continue
            p_map, conf_map = predict_with_confidence(model, preprocess(tiff.imread(pred_path)), args.device)
            p = p_map[valid_mask]
            g = gt_label[valid_mask]
            cm = confusion_matrix(g, p, labels=VALID_LABELS)
            global_cm += cm
            intersection = np.diag(cm)
            union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - intersection
            row = {
                "group_id": group_id,
                "coords": coords,
                "year": year,
                "pred_path": str(pred_path),
                "cdl_path": str(cdl_path),
                "oa": float(accuracy_score(g, p)),
                "kappa": float(cohen_kappa_score(g, p)),
                "f1_macro": float(f1_score(g, p, average="macro", zero_division=0)),
                "miou": float(np.mean(intersection / (union + 1e-6))),
            }
            rows.append(row)
            if args.save_patch_visuals:
                patch_dir = save_dir / group_id / coords
                patch_dir.mkdir(parents=True, exist_ok=True)
                save_colored_png(gt_label, patch_dir / "GT_Class.png", valid_mask)
                save_colored_png(p_map, patch_dir / f"{args.method_name}_Class.png", valid_mask)
                save_confidence_map(conf_map, patch_dir / f"{args.method_name}_Confidence.png", valid_mask)
        print(f"[patch] {group_id} accumulated_items={len(rows)}", flush=True)
    if rows:
        compute_and_store(args.method_name, "patch", rows, global_cm, save_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SA-STF ML outputs with the repo classifier.")
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--mode", choices=["full", "patch", "both"], default="both")
    parser.add_argument("--full-pred-dir", type=Path)
    parser.add_argument("--patch-pred-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("compare/SA-STF/classifier_eval/ML_syy_setting-9"))
    parser.add_argument("--location", default="ML")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--full-cdl-dir", type=Path, default=Path("data/spatio_temporal_fusion/ML/CDL_cropped"))
    parser.add_argument("--patch-cdl-dir", type=Path, default=Path("data/spatio_temporal_fusion/ML/CDL_patch"))
    parser.add_argument("--save-patch-visuals", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.full_pred_dir = resolve_existing_path(args.full_pred_dir)
    args.patch_pred_dir = resolve_existing_path(args.patch_pred_dir)
    args.full_cdl_dir = resolve_existing_path(args.full_cdl_dir)
    args.patch_cdl_dir = resolve_existing_path(args.patch_cdl_dir)
    args.output_root.mkdir(parents=True, exist_ok=True)
    generate_legend_log(args.output_root)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sys.stdout = Logger(args.output_root / f"classifier_eval_{args.method_name}_{timestamp}.log")
    print(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2, sort_keys=True), flush=True)
    model = load_classifier(args.device)
    dataset_split = get_inference_config(args.location, args.split)
    if args.mode in {"full", "both"}:
        if args.full_pred_dir is None:
            raise ValueError("--full-pred-dir is required for full mode")
        evaluate_full(args, model, dataset_split)
    if args.mode in {"patch", "both"}:
        if args.patch_pred_dir is None:
            raise ValueError("--patch-pred-dir is required for patch mode")
        evaluate_patch(args, model, dataset_split)
    print("classifier evaluation done", flush=True)


if __name__ == "__main__":
    main()
