import argparse
import importlib.util
import json
import types
import os
import re
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import requests
import tifffile as tiff
import torch

SA_STF_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SA_STF_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics import CC, ERGAS, MAE, PSNRONE, RMSE, SAM, SSIM, UIQI


def load_sa_stf_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


diffusion_pkg = types.ModuleType("Diffusion")
diffusion_pkg.__path__ = [str(SA_STF_ROOT / "Diffusion")]
sys.modules["Diffusion"] = diffusion_pkg
load_sa_stf_module("Diffusion.loss_perceptual", SA_STF_ROOT / "Diffusion" / "loss_perceptual.py")
UNet = load_sa_stf_module("Diffusion.Model", SA_STF_ROOT / "Diffusion" / "Model.py").UNet
diffusion_res_module = load_sa_stf_module("Diffusion.diffusion_res", SA_STF_ROOT / "Diffusion" / "diffusion_res.py")


class DummyPerceptualLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, inputs, targets):
        return inputs.new_tensor(0.0)


diffusion_res_module.PerceptualLoss = DummyPerceptualLoss
ResidualDiffusion = diffusion_res_module.ResidualDiffusion


GOOGLE_DRIVE_IDS = {
    "AHB": "1Vu0mU3nivZJ3L7abhpbgFe4KtXnA-uqh",
    "CIA": "1lE4z7dQbo-U2G063MNiM0rTdtVgC9gv9",
    "LGC": "1hkFrM0EzbtTaRHp_skG931JpT2MHEB9i",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.shape[0] not in (1, 3, 4, 6) and arr.shape[-1] in (1, 3, 4, 6):
        arr = arr.transpose(2, 0, 1)
    return arr


def read_tiff_chw(path: Path, max_data: float) -> np.ndarray:
    arr = to_chw(tiff.imread(path).astype(np.float32))
    return arr / max_data * 2.0 - 1.0


def default_scale(dataset: str) -> float:
    return 1.0 if dataset == "ML" else 10000.0


def read_metric_tensor(path: Path, scale: float) -> torch.Tensor:
    arr = to_chw(tiff.imread(path).astype(np.float32)) / scale
    return torch.from_numpy(arr).unsqueeze(0)


def save_tiff_chw(arr: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(save_path, arr.transpose(1, 2, 0).astype(np.float32))


def list_target_groups(data_root: Path, dataset: str, split: str) -> List[Tuple[str, str, str, Path]]:
    target_dir = data_root / dataset / "private_data" / "syy_setting-9" / "test" / split / "Landsat_02"
    paths = sorted(target_dir.glob("Group_*_L_*.tif"))
    if not paths:
        raise FileNotFoundError(f"No target images found under {target_dir}")
    groups = []
    for path in paths:
        match = re.match(r"^(Group_\d+)_L_(.*)\.tif$", path.name)
        if not match:
            continue
        group = match.group(1)
        suffix = match.group(2)
        sample_id = group if suffix == "" else f"{group}_{suffix}"
        groups.append((sample_id, group, suffix, path))
    return groups


def group_paths(data_root: Path, dataset: str, split: str, group: str, suffix: str) -> Dict[str, Path]:
    root = data_root / dataset / "private_data" / "syy_setting-9" / "test" / split
    landsat_name = f"{group}_L_{suffix}.tif"
    modis_name = f"{group}_M_{suffix}.tif"
    paths = {
        "test_lr_path": root / "MODIS_02" / modis_name,
        "test_hr_path": root / "Landsat_02" / landsat_name,
        "test_ref0_path": root / "Landsat_01" / landsat_name,
        "test_refsr0_path": root / "MODIS_01" / modis_name,
        "test_ref1_path": root / "Landsat_03" / landsat_name,
        "test_refsr1_path": root / "MODIS_03" / modis_name,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing group files:\n" + "\n".join(missing))
    return paths


def download_google_drive_file(file_id: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_url = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(base_url, params={"id": file_id}, stream=True, timeout=120)
    response.raise_for_status()
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break
    if token is not None:
        response.close()
        response = session.get(base_url, params={"id": file_id, "confirm": token}, stream=True, timeout=120)
        response.raise_for_status()
    with output_path.open("wb") as f:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
    response.close()


def resolve_weight(dataset: str, weights_dir: Path, weight_path: Path = None, download: bool = False) -> Path:
    if weight_path is not None:
        weight_path = weight_path.expanduser().resolve()
        if not weight_path.exists():
            raise FileNotFoundError(weight_path)
        return weight_path

    candidates = sorted(weights_dir.glob(f"*{dataset}*"))
    candidates = [p for p in candidates if p.suffix.lower() in {".pt", ".pth", ".ckpt"}]
    if candidates:
        return candidates[0]

    output_path = weights_dir / f"{dataset}_SA_STF_official.pt"
    if download:
        if dataset not in GOOGLE_DRIVE_IDS:
            raise KeyError(f"No official Google Drive id configured for dataset {dataset}")
        print(f"Downloading {dataset} pretrained weight to {output_path}")
        download_google_drive_file(GOOGLE_DRIVE_IDS[dataset], output_path)
        return output_path

    raise FileNotFoundError(
        f"No {dataset} weight found in {weights_dir}. "
        f"Run with --download-weights or pass --weight-path."
    )


def build_sampler(weight_path: Path, device: torch.device, args: argparse.Namespace) -> ResidualDiffusion:
    net_model = UNet(
        in_channel=6,
        out_channel=6,
        inner_channel=args.inner_channel,
        norm_groups=32,
        channel_mults=[1, 2, 4, 8, 8],
        dropout=0,
    )
    ckpt = torch.load(str(weight_path), map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    cleaned = {key.replace("module.", ""): value for key, value in ckpt.items()}
    net_model.load_state_dict(cleaned, strict=True)
    net_model.eval()
    sampler = ResidualDiffusion(
        model_x0=net_model,
        total_epoch=args.epoch,
        timesteps=args.timesteps,
        sampling_steps=args.sampling_steps,
        ddim_sampling_eta=args.ddim_eta,
        device=device,
    ).to(device)
    sampler.eval()
    return sampler


def patch_indices(size: int, patch_size: int, patch_stride: int) -> List[int]:
    end = (size - patch_stride) // patch_stride * patch_stride
    indices = [i for i in range(0, end, patch_stride)]
    if (size - patch_stride) % patch_stride != 0:
        indices.append(size - patch_size)
    return sorted(set(max(0, i) for i in indices))


@torch.no_grad()
def infer_one(
    paths: Dict[str, Path],
    sampler: ResidualDiffusion,
    device: torch.device,
    args: argparse.Namespace,
    max_data: float,
) -> np.ndarray:
    lr = read_tiff_chw(paths["test_lr_path"], max_data)
    ref0 = read_tiff_chw(paths["test_ref0_path"], max_data)
    refsr0 = read_tiff_chw(paths["test_refsr0_path"], max_data)
    ref1 = read_tiff_chw(paths["test_ref1_path"], max_data)
    refsr1 = read_tiff_chw(paths["test_refsr1_path"], max_data)
    images = [lr, ref0, ref1, refsr0, refsr1]

    _, height, width = lr.shape
    output = np.zeros_like(ref0, dtype=np.float32)
    patch_stride = args.patch_size // 2
    h_indices = patch_indices(height, args.patch_size, patch_stride)
    w_indices = patch_indices(width, args.patch_size, patch_stride)

    total = len(h_indices) * len(w_indices)
    current = 0
    for row, h_start0 in enumerate(h_indices):
        for col, w_start0 in enumerate(w_indices):
            patches = [
                torch.from_numpy(img[:, h_start0:h_start0 + args.patch_size, w_start0:w_start0 + args.patch_size])
                .unsqueeze(0)
                .to(device)
                .float()
                for img in images
            ]
            result_patch = sampler.sample(patches[0], patches[1], patches[3], patches[2], patches[4]).squeeze(0)

            h_start = h_start0
            h_end = h_start0 + args.patch_size
            w_start = w_start0
            w_end = w_start0 + args.patch_size
            cur_h_start = 0
            cur_h_end = args.patch_size
            cur_w_start = 0
            cur_w_end = args.patch_size

            trim = args.patch_size // 4
            if row != 0:
                h_start += trim
                cur_h_start = trim
            if row != len(h_indices) - 1:
                h_end -= trim
                cur_h_end -= trim
            if col != 0:
                w_start += trim
                cur_w_start = trim
            if col != len(w_indices) - 1:
                w_end -= trim
                cur_w_end -= trim

            output[:, h_start:h_end, w_start:w_end] = (
                result_patch[:, cur_h_start:cur_h_end, cur_w_start:cur_w_end]
                .cpu()
                .numpy()
            )
            current += 1
            print(f"patch {current}/{total}", flush=True)

    return np.clip((output + 1.0) * 0.5, 0.0, 1.0)


def metric_list() -> List[torch.nn.Module]:
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


def tensor_to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() > 1:
            return float(value.mean().item())
        return float(value.item())
    arr = np.asarray(value)
    return float(arr.mean())


def evaluate_pairs(pairs: Iterable[Tuple[Path, Path]], gt_scale: float, pred_scale: float) -> Dict[str, float]:
    metrics = metric_list()
    values = {metric.__name__: [] for metric in metrics}
    for gt_path, pred_path in pairs:
        gt = read_metric_tensor(gt_path, gt_scale)
        pred = read_metric_tensor(pred_path, pred_scale)
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch: {gt_path} {tuple(gt.shape)} vs {pred_path} {tuple(pred.shape)}")
        for metric in metrics:
            values[metric.__name__].append(tensor_to_float(metric(gt, pred)))
    return {key: float(np.mean(item)) for key, item in values.items()}


def run_dataset(dataset: str, args: argparse.Namespace) -> Dict[str, object]:
    data_root = args.data_root.resolve()
    max_data = args.max_data if args.max_data is not None else default_scale(dataset)
    save_scale = args.save_scale if args.save_scale is not None else default_scale(dataset)
    result_root = args.result_root / dataset / args.split
    save_img_dir = result_root / "imgs" / dataset / "0" / "save_img"
    save_img_01_dir = result_root / "imgs" / dataset / "0" / "save_img_01"
    log_dir = result_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    weight_path = resolve_weight(dataset, args.weights_dir, args.weight_path, args.download_weights)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"dataset={dataset} device={device} weight={weight_path}")
    sampler = build_sampler(weight_path, device, args)

    groups = list_target_groups(data_root, dataset, args.split)
    if args.limit is not None:
        groups = groups[: args.limit]

    pairs = []
    for index, (sample_id, group, suffix, target_path) in enumerate(groups, start=1):
        print(f"[{dataset}] group {index}/{len(groups)} {sample_id}", flush=True)
        paths = group_paths(data_root, dataset, args.split, group, suffix)
        pred01 = infer_one(paths, sampler, device, args, max_data)

        scaled_path = save_img_dir / f"{sample_id}_save_img_.tif"
        raw_path = save_img_01_dir / f"{sample_id}_save_img_01_.tif"
        save_tiff_chw(pred01 * save_scale, scaled_path)
        save_tiff_chw(pred01, raw_path)
        pairs.append((target_path, scaled_path))

    summary = evaluate_pairs(pairs, gt_scale=max_data, pred_scale=save_scale)
    summary.update(
        {
            "dataset": dataset,
            "split": args.split,
            "num_images": len(pairs),
            "max_data": max_data,
            "save_scale": save_scale,
            "weight_path": str(weight_path),
            "save_img_dir": str(save_img_dir),
            "save_img_01_dir": str(save_img_01_dir),
        }
    )
    summary_path = log_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SA-STF on CIA/LGC syy_setting-9 without touching repo results.")
    parser.add_argument("--dataset", choices=["CIA", "LGC", "ML", "all"], default="all")
    parser.add_argument("--split", choices=["full", "patch"], default="full")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "spatio_temporal_fusion")
    parser.add_argument("--result-root", type=Path, default=SA_STF_ROOT / "results" / "syy_setting-9")
    parser.add_argument("--weights-dir", type=Path, default=SA_STF_ROOT / "weights")
    parser.add_argument("--weight-path", type=Path)
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--inner-channel", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--max-data", type=float)
    parser.add_argument("--save-scale", type=float)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.result_root.mkdir(parents=True, exist_ok=True)
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    datasets = ["CIA", "LGC"] if args.dataset == "all" else [args.dataset]
    summaries = [run_dataset(dataset, args) for dataset in datasets]
    (args.result_root / f"summary_{args.dataset}_{args.split}.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
