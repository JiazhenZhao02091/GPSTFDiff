import argparse
import importlib.util
import json
import os
import random
import re
import sys
import types
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import DataLoader, Dataset

SA_STF_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SA_STF_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_sa_stf_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


diffusion_pkg = types.ModuleType("Diffusion")
diffusion_pkg.__path__ = [str(SA_STF_ROOT / "Diffusion")]
sys.modules["Diffusion"] = diffusion_pkg
loss_module = load_sa_stf_module("Diffusion.loss_perceptual", SA_STF_ROOT / "Diffusion" / "loss_perceptual.py")
UNet = load_sa_stf_module("Diffusion.Model", SA_STF_ROOT / "Diffusion" / "Model.py").UNet
diffusion_res_module = load_sa_stf_module("Diffusion.diffusion_res", SA_STF_ROOT / "Diffusion" / "diffusion_res.py")


class DummyPerceptualLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, inputs, targets):
        return inputs.new_tensor(0.0)


def set_perceptual_loss(kind: str) -> None:
    if kind == "dummy":
        diffusion_res_module.PerceptualLoss = DummyPerceptualLoss
    else:
        diffusion_res_module.PerceptualLoss = loss_module.PerceptualLoss


ResidualDiffusion = diffusion_res_module.ResidualDiffusion


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
    return to_chw(tiff.imread(path).astype(np.float32)) / max_data


def default_max_data(dataset: str) -> float:
    return 1.0 if dataset == "ML" else 10000.0


def parse_landsat_name(path: Path) -> Tuple[str, str]:
    match = re.match(r"^(Group_\d+)_L_(.*)\.tif$", path.name)
    if not match:
        raise ValueError(f"Unexpected Landsat filename: {path.name}")
    return match.group(1), match.group(2)


def split_paths(data_root: Path, dataset: str, split: str, group: str, suffix: str) -> Dict[str, Path]:
    root = data_root / dataset / "private_data" / "syy_setting-9" / split
    landsat_name = f"{group}_L_{suffix}.tif"
    modis_name = f"{group}_M_{suffix}.tif"
    paths = {
        "LR": root / "MODIS_02" / modis_name,
        "HR": root / "Landsat_02" / landsat_name,
        "Ref0": root / "Landsat_01" / landsat_name,
        "RefSR0": root / "MODIS_01" / modis_name,
        "Ref1": root / "Landsat_03" / landsat_name,
        "RefSR1": root / "MODIS_03" / modis_name,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing training files:\n" + "\n".join(missing))
    return paths


class SyySetting9Dataset(Dataset):
    def __init__(self, data_root: Path, dataset: str, split: str, max_data: float):
        self.data_root = data_root
        self.dataset = dataset
        self.split = split
        self.max_data = max_data
        target_dir = data_root / dataset / "private_data" / "syy_setting-9" / split / "Landsat_02"
        target_paths = sorted(target_dir.glob("Group_*_L_*.tif"))
        if not target_paths:
            raise FileNotFoundError(target_dir)
        self.samples = []
        for target_path in target_paths:
            group, suffix = parse_landsat_name(target_path)
            self.samples.append(split_paths(data_root, dataset, split, group, suffix))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        paths = self.samples[index]
        item = {
            key: torch.from_numpy(read_tiff_chw(path, self.max_data) * 2.0 - 1.0)
            for key, path in paths.items()
        }
        if random.random() < 0.5:
            item = {key: torch.flip(value, dims=[2]) for key, value in item.items()}
        item["Index"] = torch.tensor(index, dtype=torch.long)
        return item


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def train_one_dataset(dataset: str, args: argparse.Namespace) -> Dict[str, object]:
    set_perceptual_loss(args.perceptual)
    max_data = args.max_data if args.max_data is not None else default_max_data(dataset)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    run_root = args.output_root / dataset
    ckpt_dir = run_root / "checkpoints"
    log_dir = run_root / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"

    train_set = SyySetting9Dataset(args.data_root.resolve(), dataset, "train", max_data)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    net_model = UNet(
        in_channel=6,
        out_channel=6,
        inner_channel=args.inner_channel,
        norm_groups=32,
        channel_mults=[1, 2, 4, 8, 8],
        dropout=0,
    )
    if args.resume:
        net_model.load_state_dict(torch.load(args.resume, map_location=device))

    optimizer = torch.optim.Adam(net_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 120, 160], gamma=0.5)
    trainer = ResidualDiffusion(
        model_x0=net_model,
        total_epoch=args.epochs,
        timesteps=args.timesteps,
        sampling_steps=args.sampling_steps,
        ddim_sampling_eta=args.ddim_eta,
        device=device,
    ).to(device)

    metadata = {
        "dataset": dataset,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_samples": len(train_set),
        "perceptual": args.perceptual,
        "max_data": max_data,
        "output_root": str(run_root),
    }
    (run_root / "train_config.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(metadata, sort_keys=True) + "\n")

    global_step = 0
    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            batch = move_batch(batch, device)
            loss, loss_feature = trainer(batch, epoch)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), args.grad_clip)
            optimizer.step()
            global_step += 1

            msg = (
                f"dataset={dataset} epoch={epoch + 1}/{args.epochs} "
                f"batch={batch_index}/{len(train_loader)} step={global_step} "
                f"loss={loss.item():.6f} loss_lr_feature={loss_feature.item():.6f}"
            )
            print(msg, flush=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(msg + "\n")
            if args.max_steps is not None and global_step >= args.max_steps:
                break

        scheduler.step()
        torch.save(net_model.state_dict(), ckpt_dir / "latest.pt")
        if (epoch + 1) % args.save_every == 0:
            torch.save(net_model.state_dict(), ckpt_dir / f"ckpt_SA_STF_{epoch + 1}.pt")
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    torch.save(net_model.state_dict(), ckpt_dir / "final.pt")
    metadata["global_step"] = global_step
    (run_root / "train_summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SA-STF on syy_setting-9 without GDAL.")
    parser.add_argument("--dataset", choices=["CIA", "LGC", "ML", "all"], default="all")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "spatio_temporal_fusion")
    parser.add_argument("--output-root", type=Path, default=SA_STF_ROOT / "train_runs" / "syy_setting-9")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--inner-channel", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--max-data", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--perceptual", choices=["vgg", "dummy"], default="vgg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    datasets = ["CIA", "LGC"] if args.dataset == "all" else [args.dataset]
    summaries = [train_one_dataset(dataset, args) for dataset in datasets]
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / f"summary_{args.dataset}.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
